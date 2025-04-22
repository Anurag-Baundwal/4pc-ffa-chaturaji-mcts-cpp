#pragma once

#include <torch/torch.h>
#include "types.h" // Include basic types if needed

namespace chaturaji_cpp {

// --- Residual Block ---
struct ResBlockImpl : torch::nn::Module {
    ResBlockImpl(int channels);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
};
TORCH_MODULE(ResBlock); // Creates ResBlock class from ResBlockImpl


// --- Main Network ---
struct ChaturajiNNImpl : torch::nn::Module {
    ChaturajiNNImpl();

    // Returns policy logits (before softmax) and value estimate
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

private:
    // Input convolutional layer
    // Input channels updated to 33
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr};

    // Residual blocks
    torch::nn::Sequential resblocks_; // Use Sequential for simplicity

    // Policy head
    torch::nn::Conv2d policy_conv_{nullptr};
    torch::nn::BatchNorm2d policy_bn_{nullptr};
    torch::nn::Linear policy_fc_{nullptr};

    // Value head
    torch::nn::Conv2d value_conv_{nullptr};
    torch::nn::BatchNorm2d value_bn_{nullptr};
    torch::nn::Linear value_fc1_{nullptr};
    torch::nn::Linear value_fc2_{nullptr};
};
TORCH_MODULE(ChaturajiNN); // Creates ChaturajiNN class from ChaturajiNNImpl

} // namespace chaturaji_cpp