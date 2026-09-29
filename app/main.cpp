#include "models.h"
#include "renderer.h"
#include "utils.h"

#include <ATen/detail/MPSHooksInterface.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

namespace {

constexpr int kSeed = 9458;
constexpr int kNumEncodingFunctions = 6;
constexpr int kDepthSamplesPerRay = 64;
constexpr int kRayChunk = 4096;
constexpr int kHoldoutIndex = 45;
constexpr int kWarmupSteps = 5;
constexpr float kLearningRate = 5e-4f;
constexpr float kNear = 2.0f;
constexpr float kFar = 6.0f;
constexpr int kDefaultIters = 1000;
constexpr int kOrbitFrames = 30;
constexpr int kOrbitResolution = 200;

uint32_t lcg_next(uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return state;
}

std::vector<int> training_indices(int n_images, int n_iters, int holdout) {
  uint32_t state = kSeed;
  std::vector<int> indices;
  indices.reserve(n_iters);
  for (int i = 0; i < n_iters; ++i) {
    int idx = static_cast<int>(lcg_next(state) % static_cast<uint32_t>(n_images));
    while (idx == holdout) {
      idx = static_cast<int>(lcg_next(state) % static_cast<uint32_t>(n_images));
    }
    indices.push_back(idx);
  }
  return indices;
}

void sync_device() {
  if (torch::mps::is_available()) {
    at::detail::getMPSHooks().deviceSynchronize();
  }
}

size_t mps_current_bytes() {
  if (!torch::mps::is_available()) {
    return 0;
  }
  return at::detail::getMPSHooks().getCurrentAllocatedMemory();
}

size_t mps_driver_bytes() {
  if (!torch::mps::is_available()) {
    return 0;
  }
  return at::detail::getMPSHooks().getDriverAllocatedMemory();
}

void save_tensor(const torch::Tensor& tensor, const std::filesystem::path& path) {
  const auto bytes = torch::pickle_save(tensor.detach().to(torch::kCPU));
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void note_memory(size_t& peak_current, size_t& peak_driver) {
  peak_current = std::max(peak_current, mps_current_bytes());
  peak_driver = std::max(peak_driver, mps_driver_bytes());
}

torch::Tensor train_step(Renderer& renderer, torch::optim::Adam& optimizer,
                         const torch::Tensor& pose, const torch::Tensor& target,
                         size_t& peak_current, size_t& peak_driver) {
  const int n_rays = renderer.height() * renderer.width();
  const float inv_n = 1.0f / static_cast<float>(n_rays);
  optimizer.zero_grad(true);

  auto rays = renderer.rays_for_pose(pose);
  auto origins = rays.origins.reshape({n_rays, 3});
  auto directions = rays.directions.reshape({n_rays, 3});
  auto target_flat = target.reshape({n_rays, 3});
  auto total_loss = torch::zeros({}, target.options());

  for (int start = 0; start < n_rays; start += kRayChunk) {
    const int end = std::min(start + kRayChunk, n_rays);
    const int count = end - start;
    auto rgb = renderer.render_rays(origins.slice(0, start, end),
                                    directions.slice(0, start, end), kNear, kFar,
                                    kDepthSamplesPerRay);
    note_memory(peak_current, peak_driver);
    auto loss = torch::mse_loss(rgb, target_flat.slice(0, start, end)) *
                (static_cast<float>(count) * inv_n);
    total_loss = total_loss + loss.detach();
    loss.backward();
  }

  optimizer.step();
  note_memory(peak_current, peak_driver);
  return total_loss.detach();
}

double median_after_warmup(std::vector<double> samples) {
  if (samples.size() <= static_cast<size_t>(kWarmupSteps)) {
    return samples.empty() ? 0.0 : samples.back();
  }
  samples.erase(samples.begin(), samples.begin() + kWarmupSteps);
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double mean_after_warmup(const std::vector<double>& samples) {
  if (samples.size() <= static_cast<size_t>(kWarmupSteps)) {
    return 0.0;
  }
  const double sum = std::accumulate(samples.begin() + kWarmupSteps, samples.end(), 0.0);
  return sum / static_cast<double>(samples.size() - kWarmupSteps);
}

void write_metrics(const std::filesystem::path& path, const std::string& device,
                   int iters, double train_wall_s, double median_step_ms,
                   double mean_step_ms, size_t peak_current, size_t peak_driver,
                   double holdout_mse, double holdout_psnr, double holdout_render_ms,
                   double orbit_wall_s, const std::vector<int>& indices) {
  std::ofstream out(path);
  out << std::setprecision(10);
  out << "{\n";
  out << "  \"implementation\": \"cpp\",\n";
  out << "  \"device\": \"" << device << "\",\n";
  out << "  \"libtorch\": \"" << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR
      << "." << TORCH_VERSION_PATCH << "\",\n";
  out << "  \"build\": \"Release\",\n";
  out << "  \"iters\": " << iters << ",\n";
  out << "  \"warmup_steps_excluded_from_step_stats\": " << kWarmupSteps << ",\n";
  out << "  \"holdout_index\": " << kHoldoutIndex << ",\n";
  out << "  \"ray_chunk\": " << kRayChunk << ",\n";
  out << "  \"samples_per_ray\": " << kDepthSamplesPerRay << ",\n";
  out << "  \"encoding_frequencies\": " << kNumEncodingFunctions << ",\n";
  out << "  \"mlp\": \"8x256\",\n";
  out << "  \"init\": \"xavier_uniform_zero_bias\",\n";
  out << "  \"lr\": " << kLearningRate << ",\n";
  out << "  \"near\": " << kNear << ",\n";
  out << "  \"far\": " << kFar << ",\n";
  out << "  \"depth_jitter\": false,\n";
  out << "  \"train_wall_s\": " << train_wall_s << ",\n";
  out << "  \"median_step_ms\": " << median_step_ms << ",\n";
  out << "  \"mean_step_ms\": " << mean_step_ms << ",\n";
  out << "  \"peak_allocated_bytes\": " << peak_current << ",\n";
  out << "  \"peak_driver_bytes\": " << peak_driver << ",\n";
  out << "  \"holdout_mse\": " << holdout_mse << ",\n";
  out << "  \"holdout_psnr_db\": " << holdout_psnr << ",\n";
  out << "  \"holdout_render_ms\": " << holdout_render_ms << ",\n";
  out << "  \"orbit_frames\": " << kOrbitFrames << ",\n";
  out << "  \"orbit_resolution\": " << kOrbitResolution << ",\n";
  out << "  \"orbit_wall_s\": " << orbit_wall_s << ",\n";
  out << "  \"first_indices\": [";
  const int shown = std::min(8, static_cast<int>(indices.size()));
  for (int i = 0; i < shown; ++i) {
    if (i) out << ", ";
    out << indices[i];
  }
  out << "]\n}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::filesystem::path data_path, output_path;
  if (!parse_arguments(argc, argv, data_path, output_path)) {
    return 1;
  }
  const int num_iters = (argc >= 4) ? std::stoi(argv[3]) : kDefaultIters;
  const bool render_orbit = !(argc >= 5 && std::string(argv[4]) == "--no-orbit");
  std::filesystem::create_directories(output_path);
  std::filesystem::create_directories(output_path / "frames");

  std::cout << "LibTorch version: " << TORCH_VERSION_MAJOR << "."
            << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH << std::endl;

  set_manual_seed(kSeed);
  torch::Device device = get_device();
  std::cout << "Using device: " << device << std::endl;

  torch::Tensor images = load_tensor(data_path / "images.pt").to(device);
  torch::Tensor poses = load_tensor(data_path / "poses.pt").to(device);
  const float focal_length = load_focal(data_path / "focal.pt");
  const int n_images = static_cast<int>(images.size(0));
  const int height = static_cast<int>(images.size(1));
  const int width = static_cast<int>(images.size(2));

  if (kHoldoutIndex >= n_images) {
    std::cerr << "Holdout index " << kHoldoutIndex << " is outside the dataset ("
              << n_images << " images)." << std::endl;
    return 1;
  }

  std::cout << "Images shape: " << images.sizes() << std::endl;
  std::cout << "Poses shape: " << poses.sizes() << std::endl;
  std::cout << "Focal length: " << focal_length << std::endl;

  auto target_gt = images[kHoldoutIndex];
  save_image(target_gt, output_path / "holdout_gt.png");
  save_tensor(target_gt, output_path / "holdout_gt.pt");

  TinyNeRFModel model(device, kNumEncodingFunctions, 8, 256);
  Renderer renderer(model, height, width, focal_length, device);
  torch::optim::Adam optimizer(model.parameters(),
                               torch::optim::AdamOptions(kLearningRate));

  const auto indices = training_indices(n_images, num_iters, kHoldoutIndex);
  std::cout << "First training indices:";
  for (int i = 0; i < std::min(8, num_iters); ++i) {
    std::cout << " " << indices[i];
  }
  std::cout << std::endl;

  size_t peak_current = 0;
  size_t peak_driver = 0;
  std::vector<double> step_ms;
  step_ms.reserve(num_iters);
  torch::Tensor last_loss;

  const auto train_start = std::chrono::steady_clock::now();
  for (int i = 0; i < num_iters; ++i) {
    const auto step_start = std::chrono::steady_clock::now();
    last_loss = train_step(renderer, optimizer, poses[indices[i]],
                           images[indices[i]], peak_current, peak_driver);
    sync_device();
    const auto step_end = std::chrono::steady_clock::now();
    step_ms.push_back(
        std::chrono::duration<double, std::milli>(step_end - step_start).count());

    if (i % 50 == 0 || i + 1 == num_iters) {
      std::cout << "Iteration " << i + 1 << "/" << num_iters
                << ", loss: " << last_loss.item<float>()
                << ", step_ms: " << step_ms.back() << std::endl;
    }
  }
  const auto train_end = std::chrono::steady_clock::now();
  const double train_wall_s =
      std::chrono::duration<double>(train_end - train_start).count();

  torch::InferenceMode inference;
  const auto holdout_start = std::chrono::steady_clock::now();
  auto prediction = renderer.render(poses[kHoldoutIndex], kNear, kFar,
                                    kDepthSamplesPerRay, kRayChunk);
  sync_device();
  const double holdout_render_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                holdout_start)
          .count();

  const double holdout_mse =
      torch::mse_loss(prediction, target_gt).item<double>();
  const double holdout_psnr = -10.0 * std::log10(std::max(holdout_mse, 1e-12));
  save_image(prediction, output_path / "holdout_pred.png");
  save_tensor(prediction, output_path / "holdout_pred.pt");
  std::cout << "Holdout PSNR: " << holdout_psnr << " dB" << std::endl;

  double orbit_wall_s = 0.0;
  if (render_orbit) {
    const float focal_hd =
        focal_length *
        (static_cast<float>(kOrbitResolution) / static_cast<float>(width));
    Renderer renderer_hd(model, kOrbitResolution, kOrbitResolution, focal_hd, device);
    const auto orbit_start = std::chrono::steady_clock::now();
    render_and_save_orbit_views(renderer_hd, kOrbitFrames, output_path / "frames",
                                2.1f, kNear, kFar, kDepthSamplesPerRay, kRayChunk);
    sync_device();
    orbit_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                 orbit_start)
                       .count();
  }

  const double median_step = median_after_warmup(step_ms);
  const double mean_step = mean_after_warmup(step_ms);
  write_metrics(output_path / "metrics.json", device.str(), num_iters, train_wall_s,
                median_step, mean_step, peak_current, peak_driver, holdout_mse,
                holdout_psnr, holdout_render_ms, orbit_wall_s, indices);

  std::cout << "Train wall: " << train_wall_s << " s, median step: " << median_step
            << " ms, peak allocated: " << peak_current << " bytes" << std::endl;
  std::cout << "Wrote " << (output_path / "metrics.json") << std::endl;
  return 0;
}
