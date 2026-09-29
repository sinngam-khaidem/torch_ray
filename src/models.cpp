#include "models.h"

#include <torch/nn/init.h>

TinyNeRFModel::TinyNeRFModel(const torch::Device& device, int L_embed, int D,
                             int W)
    : L_embed_(L_embed), device_(device) {
  const int input_dim = 3 + 3 * 2 * L_embed_;

  for (int i = 0; i < D; ++i) {
    const int in_features = (i == 0) ? input_dim : W;
    auto layer = torch::nn::Linear(in_features, W);
    register_module("hidden_" + std::to_string(i), layer);
    hidden_.push_back(layer);
  }

  output_ = register_module("output", torch::nn::Linear(W, 4));

  // Glorot uniform + zero bias. PyTorch's default Kaiming init collapses
  // density through ReLU on this scene; this matches the TensorFlow TinyNeRF init.
  for (auto& layer : hidden_) {
    torch::nn::init::xavier_uniform_(layer->weight);
    torch::nn::init::zeros_(layer->bias);
  }
  torch::nn::init::xavier_uniform_(output_->weight);
  torch::nn::init::zeros_(output_->bias);

  this->to(device_);
}

torch::Tensor TinyNeRFModel::forward(const torch::Tensor& x) {
  torch::Tensor h = x;
  for (auto& layer : hidden_) {
    h = layer->forward(h);
    h = torch::relu_(h);
  }
  return output_->forward(h);
}
