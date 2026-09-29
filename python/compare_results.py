"""Score the C++ and Python holdout renders and build a side-by-side figure."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def ssim_rgb(a, b):
    """Mean channel SSIM. a and b are float tensors [H, W, 3] in [0, 1]."""
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    coords = torch.arange(11, dtype=torch.float32) - 5
    kernel = torch.exp(-(coords**2) / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11).contiguous()
    mu_a = F.conv2d(a, window, padding=5, groups=3)
    mu_b = F.conv2d(b, window, padding=5, groups=3)
    sigma_a = F.conv2d(a * a, window, padding=5, groups=3) - mu_a**2
    sigma_b = F.conv2d(b * b, window, padding=5, groups=3) - mu_b**2
    sigma_ab = F.conv2d(a * b, window, padding=5, groups=3) - mu_a * mu_b
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    )
    return float(score.mean())


def psnr(a, b):
    mse = float(F.mse_loss(a, b))
    return mse, -10.0 * torch.log10(torch.tensor(max(mse, 1e-12))).item()


def load_rgb(path):
    return torch.load(path, weights_only=False).to(torch.float32).clamp(0, 1)


def panel(paths, labels, out_path):
    images = [Image.open(p).convert("RGB") for p in paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_h = 28
    canvas = Image.new("RGB", (width * len(images), height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        x = i * width + (width - image.width) // 2
        canvas.paste(image, (x, label_h + (height - image.height) // 2))
        draw.text((i * width + 8, 6), label, fill=(0, 0, 0))
    canvas.save(out_path)


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    cpp = json.loads((root / "cpp" / "metrics.json").read_text())
    python = json.loads((root / "python" / "metrics.json").read_text())
    gt = load_rgb(root / "cpp" / "holdout_gt.pt")
    cpp_pred = load_rgb(root / "cpp" / "holdout_pred.pt")
    py_pred = load_rgb(root / "python" / "holdout_pred.pt")

    cpp_mse, cpp_psnr = psnr(cpp_pred, gt)
    py_mse, py_psnr = psnr(py_pred, gt)
    cpp_ssim = ssim_rgb(cpp_pred, gt)
    py_ssim = ssim_rgb(py_pred, gt)

    panel(
        [
            root / "cpp" / "holdout_gt.png",
            root / "cpp" / "holdout_pred.png",
            root / "python" / "holdout_pred.png",
        ],
        [
            "Holdout ground truth",
            f"C++  {cpp_psnr:.2f} dB",
            f"Python  {py_psnr:.2f} dB",
        ],
        root / "holdout_comparison.png",
    )

    frame_ids = [0, 7, 15, 22]
    for name in ("cpp", "python"):
        frames = [root / name / "frames" / f"frame_{i}.png" for i in frame_ids]
        if all(path.exists() for path in frames):
            panel(
                frames,
                [f"{name} frame {i}" for i in frame_ids],
                root / f"{name}_orbit_strip.png",
            )

    summary = {
        "cpp_holdout_mse": cpp_mse,
        "cpp_holdout_psnr_db": cpp_psnr,
        "cpp_holdout_ssim": cpp_ssim,
        "python_holdout_mse": py_mse,
        "python_holdout_psnr_db": py_psnr,
        "python_holdout_ssim": py_ssim,
        "cpp_metrics": cpp,
        "python_metrics": python,
    }
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k.endswith("ssim") or "psnr" in k or "mse" in k}, indent=2))


if __name__ == "__main__":
    main()
