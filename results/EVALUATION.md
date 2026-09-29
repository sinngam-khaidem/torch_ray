# TinyNeRF evaluation: C++ versus Python

Both programs trained the same TinyNeRF on the Lego scene (100 views, 100×100) for 1000 Adam steps and rendered a held-out view plus a 30-frame orbit. View 45 was held out of training. The view order, network, encoding, sample count, near/far bounds, chunk size, and initialisation are the same in both.

One training fix was required before either run would learn. PyTorch’s default Kaiming initialisation drove every density negative by about step 20. ReLU then blocked recovery and the render stayed black (holdout PSNR stuck at 12.1 dB, the score of a black image). Both sides now use Glorot uniform weights and zero biases, which is the initialisation used by the original TensorFlow TinyNeRF. With that change, the first 100 logged losses match to the printed digits.

## Setup

| Item | Value |
| --- | --- |
| Scene | TinyNeRF Lego, holdout index 45 |
| MLP | 8 × 256, ReLU, linear output of 4 |
| Encoding | 6 frequencies, raw xyz included |
| Samples per ray | 64, no depth jitter, near 2, far 6 |
| Optimiser | Adam, learning rate 5e-4, one step per image |
| Ray chunk | 4,096 rays, backward after each chunk |
| Iterations | 1,000, seed 9458 |
| Device | Apple MPS |
| C++ | LibTorch 2.9.1, Release build |
| Python | PyTorch 2.6.0 |

Step time includes a device synchronisation, so it measures the GPU step rather than just the launch. The first 5 steps are excluded from the median and the mean. Image writes are outside the training clock.

## Quantitative

| | C++ | Python |
| --- | ---: | ---: |
| Holdout MSE | 0.004071 | 0.003946 |
| Holdout PSNR | 23.90 dB | 24.04 dB |
| Holdout SSIM | 0.820 | 0.840 |
| Median step | 2313 ms | 1049 ms |
| Mean step | 2455 ms | 1067 ms |
| Training wall time | 2456 s | 1067 s |
| Peak allocated memory | 2.086 GiB | 2.084 GiB |
| Peak Metal driver pool | 4.47 GiB | 4.11 GiB |
| Holdout render (100×100) | 1633 ms | 411 ms |
| Orbit, 30×200×200 | 209 s | 50 s |

Python is 2.2× faster per training step and about 4× faster at inference. Peak allocated memory, sampled while a chunk’s graph is alive, differs by 2.7 MiB. The graph, not the language wrapper, sets the footprint.

The 0.14 dB PSNR gap and 0.02 SSIM gap are small next to the gap from ground truth. Logged losses agree through iteration 101 and then drift by a few thousandths. LibTorch 2.9.1 and PyTorch 2.6.0 are not the same build, so part of the speed difference can be the MPS kernels rather than C++ versus Python itself.

## Qualitative

The held-out photograph is a sharp yellow Lego bulldozer on a studded base. Both reconstructions recover that object: cab, bucket, tracks, and base are in the right place, and the background stays black. Fine studs and the small markings on the body are soft. Neither render is a sharp novel view after 1000 steps.

The two predictions are hard to tell apart at 100×100. Python keeps slightly more contrast on the base, which matches its higher PSNR and SSIM. Side by side: `results/holdout_comparison.png`.

The orbit is a new camera path (radius 2.1, elevation −30°, focal length scaled to 200×200), not a training pose. Both turntables show the same bulldozer from the side and the front, and both are blurrier than the held-out training view. Empty background stays empty. Corresponding frames, for example `frame_8.png`, show the same pose and the same softness.

## Where the outputs are

- `results/cpp/` and `results/python/`: `holdout_gt.png`, `holdout_pred.png`, `metrics.json`, `frames/frame_0.png` … `frame_29.png`
- `results/holdout_comparison.png`: ground truth, C++ prediction, Python prediction
- `results/cpp_orbit_strip.png`, `results/python_orbit_strip.png`
