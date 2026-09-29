# Neural Radiance Fields in C++

A C++ implementation of TinyNeRF, a minimal Neural Radiance Field. Built with LibTorch.

Includes minimal adaptation of [NERF, ECCV 2020](https://arxiv.org/abs/2003.08934): positional encoding, MLP predicting RGB and density, compositing with classical volume rendering equation. <span style="color:red;">No view-dependent color or hierarchical (coarse/fine) sampling.</span>

## Steps
1. Loads RGB images, camera-to-world poses, and a focal length.
2. Trains an 8-layer MLP (width 256) mapping positionally encoded 3D points to RGB + density.
3. Trains on every view except a held-out one, then renders that held-out view and scores it with PSNR.
4. Renders a 30-frame orbit around the scene at 200x200.
5. Writes a `metrics.json` with step times, peak device memory, and quality numbers.

## Dataset
Official Lego scene used in TinyNeRF (`tiny_nerf_data.npz`).

## Project layout

```
app/main.cpp                 Training loop, timing, metrics
include/  src/               Model, renderer, utilities
python/train_tinynerf.py     Matched Python trainer
python/compare_results.py    PSNR/SSIM scoring and comparison figures
data/                        LibTorch tensors: images.pt, poses.pt, focal.pt
tiny_nerf_data.npz           Original NumPy dump of the Lego scene
results/                     Recorded run outputs and EVALUATION.md
```

## Requirements

- CMake 3.5+
- C++17 compiler
- [LibTorch](https://pytorch.org/get-started/locally/) 2.9.1

Download the CPU build of LibTorch for your platform and unpack it so that `third_party/libtorch/lib` exists:

```bash
mkdir -p third_party
curl -L -o libtorch.zip \
  https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.9.1.zip
unzip -q libtorch.zip -d third_party && rm libtorch.zip
```

## Build

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$(pwd)/third_party/libtorch"
cmake --build build-release
```

## Data

The executable expects a directory containing three pickle-serialized tensors:

| File | Contents |
| --- | --- |
| `images.pt` | `float` RGB images, shape `[N, H, W, 3]`, values in `[0, 1]` |
| `poses.pt` | Camera-to-world matrices, shape `[N, 4, 4]` |
| `focal.pt` | Scalar focal length in pixels |

`data/` already has these, converted from `tiny_nerf_data.npz`. To regenerate:

```python
import numpy as np
import torch

data = np.load("tiny_nerf_data.npz")
torch.save(torch.from_numpy(data["images"]).float(), "data/images.pt",
           _use_new_zipfile_serialization=False)
torch.save(torch.from_numpy(data["poses"]).float(), "data/poses.pt",
           _use_new_zipfile_serialization=False)
torch.save(torch.from_numpy(data["focal"]).float(), "data/focal.pt",
           _use_new_zipfile_serialization=False)
```

The C++ loader uses `torch::pickle_load`, so the old (non-zip) `torch.save` format is required.

## Run

```bash
./build-release/torch_ray <data_path> <output_path> [iters] [--no-orbit]
```

`iters` defaults to 1000. `--no-orbit` skips the final turntable, which is useful when only training time matters.

```bash
./build-release/torch_ray data results/cpp
python python/train_tinynerf.py data results/python
python python/compare_results.py
```

The Python trainer takes the same positional arguments and the same `--no-orbit` flag.

Written to the output directory by both implementations:

| File | Contents |
| --- | --- |
| `holdout_gt.png` / `.pt` | Held-out ground-truth view (pose index 45) |
| `holdout_pred.png` / `.pt` | Rendered prediction of that same pose after training |
| `frames/frame_0.png` ... `frame_29.png` | Final 200x200 orbit |
| `metrics.json` | Step times, peak memory, holdout MSE/PSNR, render and orbit timings |

`train_tinynerf.py` additionally writes `model.pt`. `compare_results.py` reads `results/cpp/` and `results/python/`, then writes `results/comparison.json`, `results/holdout_comparison.png`, and the two orbit strips.

## Training defaults

These live at the top of `app/main.cpp` and are mirrored at the top of `python/train_tinynerf.py`:

| Parameter | Value |
| --- | --- |
| Iterations | 1000 |
| Seed | 9458 |
| Optimizer | Adam, learning rate `5e-4` |
| Samples per ray | 64, no depth jitter |
| Positional encoding frequencies | 6 (39 input channels, raw xyz included) |
| MLP depth / width | 8 / 256 |
| Initialisation | Glorot uniform, zero bias |
| Ray chunk | 4,096 rays, backward per chunk |
| Near / far | 2.0 / 6.0 |
| Held-out pose index | 45 |
| Warmup steps excluded from step statistics | 5 |
| Final orbit | 30 frames at 200x200, radius 2.1, elevation -30 degrees |

