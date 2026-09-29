"""TinyNeRF trainer matched to the C++ implementation in app/main.cpp.

The numerical recipe (network, encoding order, ray chunking, near/far, holdout,
and the LCG that picks training views) is the same on both sides. Weight
initialisation still follows each runtime's RNG, so the two runs are not
expected to produce bit-identical images.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SEED = 9458
L_EMBED = 6
SAMPLES_PER_RAY = 64
RAY_CHUNK = 4096
HOLDOUT_INDEX = 45
WARMUP_STEPS = 5
LEARNING_RATE = 5e-4
NEAR = 2.0
FAR = 6.0
ORBIT_FRAMES = 30
ORBIT_RESOLUTION = 200


def lcg_next(state):
    return (state * 1664525 + 1013904223) & 0xFFFFFFFF


def training_indices(n_images, n_iters, holdout):
    state = SEED
    indices = []
    for _ in range(n_iters):
        state = lcg_next(state)
        idx = state % n_images
        while idx == holdout:
            state = lcg_next(state)
            idx = state % n_images
        indices.append(idx)
    return indices


def sync_device():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def current_allocated():
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory()
    return 0


def driver_allocated():
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory()
    return 0


class TinyNeRF(torch.nn.Module):
    def __init__(self, D=8, W=256, L=L_EMBED):
        super().__init__()
        in_dim = 3 + 3 * 2 * L
        self.hidden = torch.nn.ModuleList(
            [torch.nn.Linear(in_dim if i == 0 else W, W) for i in range(D)]
        )
        self.output = torch.nn.Linear(W, 4)
        self.L = L

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = torch.relu_(layer(h))
        return self.output(h)


def init_xavier(model):
    """Glorot uniform weights and zero biases, matching the TinyNeRF TensorFlow init.

    PyTorch's default Kaiming init drives every density to zero within a few
    dozen steps on this scene, and ReLU then blocks any recovery.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)


class Renderer:
    def __init__(self, model, height, width, focal, device):
        self.model = model
        self.H = height
        self.W = width
        self.focal = focal
        self.device = device
        ii, jj = torch.meshgrid(
            torch.arange(width, device=device, dtype=torch.float32),
            torch.arange(height, device=device, dtype=torch.float32),
            indexing="xy",
        )
        self.dirs_cam = torch.stack(
            (
                (ii - width * 0.5) / focal,
                -(jj - height * 0.5) / focal,
                -torch.ones_like(ii),
            ),
            dim=-1,
        )
        self.freqs = torch.pow(
            2.0,
            torch.arange(model.L, device=device, dtype=torch.float32),
        )

    def positional_encoding(self, x):
        scaled = x[:, None, :] * self.freqs[None, :, None]
        interleaved = torch.stack((torch.sin(scaled), torch.cos(scaled)), dim=2).flatten(1)
        return torch.cat((x, interleaved), dim=-1)

    def rays_for_pose(self, c2w):
        rotation = c2w[:3, :3]
        translation = c2w[:3, 3]
        directions = torch.sum(self.dirs_cam[..., None, :] * rotation, dim=-1)
        directions = F.normalize(directions, dim=-1)
        origins = translation.view(1, 1, 3).expand(self.H, self.W, 3)
        return origins, directions

    def composite(self, rgb, density, z_vals):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        tail = torch.full(
            (z_vals.shape[0], 1), 1.0e10, device=z_vals.device, dtype=z_vals.dtype
        )
        dists = torch.cat((dists, tail), dim=-1)
        alpha = 1.0 - torch.exp(-density * dists)
        transmittance = torch.cumprod(1.0 - alpha + 1.0e-10, dim=-1)
        ones = torch.ones((z_vals.shape[0], 1), device=z_vals.device, dtype=z_vals.dtype)
        transmittance = torch.cat((ones, transmittance[..., :-1]), dim=-1)
        weights = transmittance * alpha
        return torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)

    def render_rays(self, origins, directions, near, far, n_samples):
        z_vals = torch.linspace(near, far, n_samples, device=origins.device, dtype=origins.dtype)
        points = origins[:, None, :] + directions[:, None, :] * z_vals.view(1, n_samples, 1)
        raw = self.model(self.positional_encoding(points.reshape(-1, 3)))
        raw = raw.view(origins.shape[0], n_samples, 4)
        rgb = torch.sigmoid(raw[..., :3])
        density = torch.relu(raw[..., 3])
        z = z_vals.view(1, n_samples).expand(origins.shape[0], n_samples)
        return self.composite(rgb, density, z)

    def render(self, c2w, near=NEAR, far=FAR, n_samples=SAMPLES_PER_RAY, ray_chunk=RAY_CHUNK):
        origins, directions = self.rays_for_pose(c2w)
        n_rays = self.H * self.W
        origins = origins.reshape(n_rays, 3)
        directions = directions.reshape(n_rays, 3)
        image = torch.empty((n_rays, 3), device=origins.device, dtype=origins.dtype)
        for start in range(0, n_rays, ray_chunk):
            end = min(start + ray_chunk, n_rays)
            image[start:end] = self.render_rays(
                origins[start:end], directions[start:end], near, far, n_samples
            )
        return image.view(self.H, self.W, 3)


def train_step(renderer, optimizer, pose, target, peaks):
    n_rays = renderer.H * renderer.W
    inv_n = 1.0 / n_rays
    optimizer.zero_grad(set_to_none=True)
    origins, directions = renderer.rays_for_pose(pose)
    origins = origins.reshape(n_rays, 3)
    directions = directions.reshape(n_rays, 3)
    target_flat = target.reshape(n_rays, 3)
    total = torch.zeros((), device=target.device)
    for start in range(0, n_rays, RAY_CHUNK):
        end = min(start + RAY_CHUNK, n_rays)
        count = end - start
        rgb = renderer.render_rays(
            origins[start:end], directions[start:end], NEAR, FAR, SAMPLES_PER_RAY
        )
        peaks[0] = max(peaks[0], current_allocated())
        peaks[1] = max(peaks[1], driver_allocated())
        loss = F.mse_loss(rgb, target_flat[start:end]) * (count * inv_n)
        total = total + loss.detach()
        loss.backward()
    optimizer.step()
    peaks[0] = max(peaks[0], current_allocated())
    peaks[1] = max(peaks[1], driver_allocated())
    return total.detach()


def save_image(tensor, path):
    from PIL import Image

    array = (
        tensor.detach()
        .to(torch.float32)
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)


def spherical_pose(azimuth_deg, elevation_deg, radius):
    phi = elevation_deg * math.pi / 180.0
    theta = azimuth_deg * math.pi / 180.0
    cphi, sphi = math.cos(phi), math.sin(phi)
    ctheta, stheta = math.cos(theta), math.sin(theta)
    translation = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, radius], [0.0, 0.0, 0.0, 1.0]]
    )
    phi_mat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cphi, -sphi, 0.0],
            [0.0, sphi, cphi, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    theta_mat = torch.tensor(
        [
            [ctheta, 0.0, -stheta, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [stheta, 0.0, ctheta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    flip = torch.tensor(
        [[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    c2w = flip @ theta_mat @ phi_mat @ translation
    return c2w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("iters", nargs="?", type=int, default=1000)
    parser.add_argument("--no-orbit", action="store_true")
    args = parser.parse_args()

    args.output_path.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    torch.manual_seed(SEED)
    if device.type == "mps":
        torch.mps.manual_seed(SEED)

    images = torch.load(args.data_path / "images.pt", weights_only=False).to(device).float()
    poses = torch.load(args.data_path / "poses.pt", weights_only=False).to(device).float()
    focal = float(torch.load(args.data_path / "focal.pt", weights_only=False).item())
    n_images, height, width, _ = images.shape
    if HOLDOUT_INDEX >= n_images:
        raise SystemExit(f"Holdout index {HOLDOUT_INDEX} is outside the dataset ({n_images} images).")

    print(f"Images shape: {tuple(images.shape)}")
    print(f"Poses shape: {tuple(poses.shape)}")
    print(f"Focal length: {focal}")

    target_gt = images[HOLDOUT_INDEX]
    save_image(target_gt, args.output_path / "holdout_gt.png")
    torch.save(target_gt.detach().cpu(), args.output_path / "holdout_gt.pt")

    model = TinyNeRF()
    init_xavier(model)
    model = model.to(device)
    renderer = Renderer(model, height, width, focal, device)
    # foreach/fused off so the update matches LibTorch Adam, which has neither switch.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, foreach=False, fused=False)

    indices = training_indices(n_images, args.iters, HOLDOUT_INDEX)
    print("First training indices:", " ".join(str(i) for i in indices[:8]))

    peaks = [0, 0]
    step_ms = []
    last_loss = None
    train_start = time.perf_counter()
    for i, img_idx in enumerate(indices):
        step_start = time.perf_counter()
        last_loss = train_step(renderer, optimizer, poses[img_idx], images[img_idx], peaks)
        sync_device()
        step_ms.append((time.perf_counter() - step_start) * 1000.0)
        if i % 50 == 0 or i + 1 == args.iters:
            print(f"Iteration {i + 1}/{args.iters}, loss: {last_loss.item():.6f}, step_ms: {step_ms[-1]:.1f}")
    train_wall_s = time.perf_counter() - train_start

    timed = step_ms[WARMUP_STEPS:] if len(step_ms) > WARMUP_STEPS else step_ms
    median_step = sorted(timed)[len(timed) // 2] if timed else 0.0
    mean_step = sum(timed) / len(timed) if timed else 0.0

    with torch.inference_mode():
        holdout_start = time.perf_counter()
        prediction = renderer.render(poses[HOLDOUT_INDEX])
        sync_device()
        holdout_render_ms = (time.perf_counter() - holdout_start) * 1000.0
        holdout_mse = float(F.mse_loss(prediction, target_gt).item())
        holdout_psnr = -10.0 * torch.log10(torch.tensor(max(holdout_mse, 1e-12))).item()
        save_image(prediction, args.output_path / "holdout_pred.png")
        torch.save(prediction.detach().cpu(), args.output_path / "holdout_pred.pt")
        torch.save(model.state_dict(), args.output_path / "model.pt")
        print(f"Holdout PSNR: {holdout_psnr:.3f} dB")

        orbit_wall_s = 0.0
        if not args.no_orbit:
            focal_hd = focal * (ORBIT_RESOLUTION / width)
            renderer_hd = Renderer(model, ORBIT_RESOLUTION, ORBIT_RESOLUTION, focal_hd, device)
            orbit_start = time.perf_counter()
            for frame in range(ORBIT_FRAMES):
                azimuth = frame * 360.0 / ORBIT_FRAMES
                pose = spherical_pose(azimuth, -30.0, 2.1).to(device)
                rendered = renderer_hd.render(pose)
                save_image(rendered, frames_dir / f"frame_{frame}.png")
            sync_device()
            orbit_wall_s = time.perf_counter() - orbit_start

    metrics = {
        "implementation": "python",
        "device": str(device),
        "torch": torch.__version__,
        "build": "interpreter",
        "iters": args.iters,
        "warmup_steps_excluded_from_step_stats": WARMUP_STEPS,
        "holdout_index": HOLDOUT_INDEX,
        "ray_chunk": RAY_CHUNK,
        "samples_per_ray": SAMPLES_PER_RAY,
        "encoding_frequencies": L_EMBED,
        "mlp": "8x256",
        "init": "xavier_uniform_zero_bias",
        "lr": LEARNING_RATE,
        "near": NEAR,
        "far": FAR,
        "depth_jitter": False,
        "train_wall_s": train_wall_s,
        "median_step_ms": median_step,
        "mean_step_ms": mean_step,
        "peak_allocated_bytes": peaks[0],
        "peak_driver_bytes": peaks[1],
        "holdout_mse": holdout_mse,
        "holdout_psnr_db": holdout_psnr,
        "holdout_render_ms": holdout_render_ms,
        "orbit_frames": ORBIT_FRAMES,
        "orbit_resolution": ORBIT_RESOLUTION,
        "orbit_wall_s": orbit_wall_s,
        "first_indices": indices[:8],
    }
    (args.output_path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"Train wall: {train_wall_s:.2f} s, median step: {median_step:.1f} ms, "
        f"peak allocated: {peaks[0]} bytes"
    )
    print(f"Wrote {args.output_path / 'metrics.json'}")


if __name__ == "__main__":
    main()
