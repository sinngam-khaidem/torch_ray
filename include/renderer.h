#ifndef RENDERER_H_
#define RENDERER_H_

#include <torch/torch.h>

#include "models.h"

class Renderer {
 public:
  struct Rays {
    torch::Tensor origins;     // [H, W, 3]
    torch::Tensor directions;  // [H, W, 3]
  };

  Renderer(TinyNeRFModel& model, int H, int W, float focal_length,
           const torch::Device& device);

  Rays rays_for_pose(const torch::Tensor& tform_cam2world) const;

  // origins and directions are [N, 3]. Returns RGB [N, 3].
  torch::Tensor render_rays(const torch::Tensor& origins,
                            const torch::Tensor& directions, float near,
                            float far, int n_samples) const;

  // Full image [H, W, 3]. Caller should be under inference / no-grad so
  // chunk results can be copied into one buffer.
  torch::Tensor render(const torch::Tensor& tform_cam2world, float near = 2.0f,
                       float far = 6.0f, int n_samples = 64,
                       int ray_chunk = 4096) const;

  int height() const { return H_; }
  int width() const { return W_; }

 private:
  torch::Tensor positional_encoding(const torch::Tensor& x) const;
  torch::Tensor composite(const torch::Tensor& rgb, const torch::Tensor& density,
                          const torch::Tensor& z_vals) const;

  TinyNeRFModel& model_;
  int H_;
  int W_;
  float focal_length_;
  torch::Device device_;
  torch::Tensor dirs_cam_;  // [H, W, 3], camera-space ray directions
  torch::Tensor freqs_;     // [L]
};

#endif  // RENDERER_H_
