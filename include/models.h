#ifndef MODELS_H_
#define MODELS_H_

#include <torch/torch.h>

#include <vector>

class TinyNeRFModel : public torch::nn::Module {
 public:
  TinyNeRFModel(const torch::Device& device = torch::kCPU, int L_embed = 6,
                int D = 8, int W = 256);

  torch::Tensor forward(const torch::Tensor& x);
  int L_embed_;

 private:
  std::vector<torch::nn::Linear> hidden_;
  torch::nn::Linear output_{nullptr};
  torch::Device device_;
};

#endif  // MODELS_H_
