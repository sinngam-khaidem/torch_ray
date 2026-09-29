#include "renderer.h"

using namespace torch::indexing;

Renderer::Renderer(TinyNeRFModel& model, int H, int W, float focal_length,
                   const torch::Device& device)
    : model_(model),
      H_(H),
      W_(W),
      focal_length_(focal_length),
      device_(device) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  auto i_coords = torch::arange(W_, opts);
  auto j_coords = torch::arange(H_, opts);
  auto grid = torch::meshgrid({i_coords, j_coords}, "xy");
  auto ii = grid[0];
  auto jj = grid[1];

  dirs_cam_ = torch::stack({(ii - W_ * 0.5f) / focal_length_,
                            -(jj - H_ * 0.5f) / focal_length_,
                            -torch::ones_like(ii)},
                           -1);

  freqs_ = torch::pow(
      2.0f, torch::arange(model_.L_embed_, opts));
}

torch::Tensor Renderer::positional_encoding(const torch::Tensor& x) const {
  // x: [N, 3] -> [N, 3 + 6L] with sin/cos interleaved per frequency,
  // matching [x, sin(2^0 x), cos(2^0 x), sin(2^1 x), cos(2^1 x), ...].
  auto scaled = x.unsqueeze(1) * freqs_.view({1, -1, 1});  // [N, L, 3]
  auto interleaved =
      torch::stack({torch::sin(scaled), torch::cos(scaled)}, 2).flatten(1);
  return torch::cat({x, interleaved}, -1);
}

Renderer::Rays Renderer::rays_for_pose(
    const torch::Tensor& tform_cam2world) const {
  auto c2w = tform_cam2world.to(device_);
  auto rotation = c2w.index({Slice(0, 3), Slice(0, 3)});
  auto translation = c2w.index({Slice(0, 3), 3});

  auto directions =
      torch::sum(dirs_cam_.unsqueeze(-2) * rotation, -1);
  directions = torch::nn::functional::normalize(
      directions, torch::nn::functional::NormalizeFuncOptions().dim(-1));

  auto origins = translation.view({1, 1, 3}).expand({H_, W_, 3});
  return {origins, directions};
}

torch::Tensor Renderer::composite(const torch::Tensor& rgb,
                                  const torch::Tensor& density,
                                  const torch::Tensor& z_vals) const {
  auto dists = z_vals.index({"...", Slice(1, None)}) -
               z_vals.index({"...", Slice(None, -1)});
  auto tail = torch::full({z_vals.size(0), 1}, 1.0e10, z_vals.options());
  dists = torch::cat({dists, tail}, -1);

  auto alpha = 1.0f - torch::exp(-density * dists);
  auto transmittance = torch::cumprod(1.0f - alpha + 1.0e-10, -1);
  auto ones = torch::ones({z_vals.size(0), 1}, z_vals.options());
  transmittance = torch::cat(
      {ones, transmittance.index({"...", Slice(None, -1)})}, -1);

  auto weights = transmittance * alpha;
  return torch::sum(weights.unsqueeze(-1) * rgb, -2);
}

torch::Tensor Renderer::render_rays(const torch::Tensor& origins,
                                    const torch::Tensor& directions,
                                    float near, float far,
                                    int n_samples) const {
  auto z_vals = torch::linspace(near, far, n_samples, origins.options());
  auto points = origins.unsqueeze(1) +
                directions.unsqueeze(1) * z_vals.view({1, n_samples, 1});

  auto embedded = positional_encoding(points.reshape({-1, 3}));
  auto raw = model_.forward(embedded).view({origins.size(0), n_samples, 4});

  auto rgb = torch::sigmoid(raw.index({"...", Slice(0, 3)}));
  auto density = torch::relu(raw.index({"...", 3}));
  auto z = z_vals.view({1, n_samples}).expand({origins.size(0), n_samples});
  return composite(rgb, density, z);
}

torch::Tensor Renderer::render(const torch::Tensor& tform_cam2world, float near,
                               float far, int n_samples, int ray_chunk) const {
  auto rays = rays_for_pose(tform_cam2world);
  const int n_rays = H_ * W_;
  auto origins = rays.origins.reshape({n_rays, 3});
  auto directions = rays.directions.reshape({n_rays, 3});
  auto image = torch::empty({n_rays, 3}, origins.options());

  for (int start = 0; start < n_rays; start += ray_chunk) {
    const int end = std::min(start + ray_chunk, n_rays);
    auto rgb = render_rays(origins.slice(0, start, end),
                           directions.slice(0, start, end), near, far,
                           n_samples);
    image.slice(0, start, end).copy_(rgb);
  }

  return image.view({H_, W_, 3});
}
