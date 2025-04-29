#pragma once

#include <torch/torch.h>
#include <vector>   // For vector
#include <utility>  // For std::pair
#include "types.h" // Include basic types (EvaluationRequest, EvaluationResult)

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
    // Input shape: [B, C, H, W]
    // Output shapes: { [B, 4096], [B, 1] }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    // --- Batched Evaluation Interface ---
    /**
     * @brief Evaluates a batch of board states provided as EvaluationRequests.
     * Internally calls the forward method. Handles tensor stacking and result unpacking.
     *
     * @param requests Vector of evaluation requests. Assumes state tensors are ready (e.g., on CPU).
     * @param device The target device (CPU/CUDA) to perform inference on.
     * @return Vector of evaluation results, corresponding to the input requests. Results are on CPU.
     */
    std::vector<EvaluationResult> evaluate_batch(
        const std::vector<EvaluationRequest>& requests,
        torch::Device device
    );
    // --- END Batched Evaluation Interface ---

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