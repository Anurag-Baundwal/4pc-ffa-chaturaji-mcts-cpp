#include "model.h"
#include <vector>
#include <stdexcept> // For runtime_error

namespace chaturaji_cpp {

// --- Configuration Constants ---
// Network Body
const int NUM_RES_BLOCKS = 6;
const int NUM_CHANNELS = 64;

// Policy Head
const int POLICY_HEAD_CONV_CHANNELS = 12;

// Value Head
const int VALUE_HEAD_CONV_CHANNELS = 2;
const int VALUE_FC_HIDDEN_CHANNELS = 96;
// --- End Configuration Constants ---

// --- ResBlock Implementation ---
ResBlockImpl::ResBlockImpl(int channels) :
    conv1_(torch::nn::Conv2dOptions(channels, channels, /*kernel_size=*/3).padding(1)),
    bn1_(channels),
    conv2_(torch::nn::Conv2dOptions(channels, channels, /*kernel_size=*/3).padding(1)),
    bn2_(channels)
{
    // Register modules
    register_module("conv1", conv1_);
    register_module("bn1", bn1_);
    register_module("conv2", conv2_);
    register_module("bn2", bn2_);
}

torch::Tensor ResBlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x;
    x = torch::relu(bn1_(conv1_(x)));
    x = bn2_(conv2_(x));
    x += residual;
    x = torch::relu(x);
    return x;
}


// --- ChaturajiNN Implementation ---
ChaturajiNNImpl::ChaturajiNNImpl() :
    // Input layer (33 input channels, NUM_CHANNELS output channels)
    conv1_(torch::nn::Conv2dOptions(33, NUM_CHANNELS, /*kernel_size=*/3).padding(1)),
    bn1_(NUM_CHANNELS),

    // Policy head
    policy_conv_(torch::nn::Conv2dOptions(NUM_CHANNELS, POLICY_HEAD_CONV_CHANNELS, /*kernel_size=*/1)),
    policy_bn_(POLICY_HEAD_CONV_CHANNELS),
    policy_fc_(torch::nn::LinearOptions(POLICY_HEAD_CONV_CHANNELS * 8 * 8, 4096)),

    // Value head
    value_conv_(torch::nn::Conv2dOptions(NUM_CHANNELS, VALUE_HEAD_CONV_CHANNELS, /*kernel_size=*/1)), // NUM_CHANNELS in, 1 out
    value_bn_(VALUE_HEAD_CONV_CHANNELS),
    value_fc1_(torch::nn::LinearOptions(VALUE_HEAD_CONV_CHANNELS * 8 * 8, VALUE_FC_HIDDEN_CHANNELS)), // 1*8*8 = 64 input features
    value_fc2_(torch::nn::LinearOptions(VALUE_FC_HIDDEN_CHANNELS, 1)) // 1 output value
{
    // Register input layer modules
    register_module("conv1", conv1_);
    register_module("bn1", bn1_);

    // Create and register residual blocks
    for (int i = 0; i < NUM_RES_BLOCKS; ++i) {
        resblocks_->push_back(ResBlock(NUM_CHANNELS));
    }
    register_module("resblocks", resblocks_);

    // Register policy head modules
    register_module("policy_conv", policy_conv_);
    register_module("policy_bn", policy_bn_);
    register_module("policy_fc", policy_fc_);

    // Register value head modules
    register_module("value_conv", value_conv_);
    register_module("value_bn", value_bn_);
    register_module("value_fc1", value_fc1_);
    register_module("value_fc2", value_fc2_);
}


std::pair<torch::Tensor, torch::Tensor> ChaturajiNNImpl::forward(torch::Tensor x) {
    // Input conv -> BN -> ReLU
    x = torch::relu(bn1_(conv1_(x)));

    // Residual blocks
    x = resblocks_->forward(x);

    // --- Policy Head ---
    torch::Tensor p = policy_conv_(x);
    p = policy_bn_(p);
    p = torch::relu(p);
    p = p.view({p.size(0), -1}); // Flatten - This automatically adapts to the new channel size from policy_conv_
    p = policy_fc_(p); // Output shape [Batch, 4096] (Logits)

    // --- Value Head ---
    torch::Tensor v = value_conv_(x);
    v = value_bn_(v);
    v = torch::relu(v);
    v = v.view({v.size(0), -1}); // Flatten - This automatically adapts to the new channel size from value_conv_
    v = value_fc1_(v);
    v = torch::relu(v);
    v = value_fc2_(v); // Output shape [Batch, 1]
    v = torch::tanh(v); // Apply tanh activation

    return {p, v};
}

// --- NEW: Batched Evaluation Implementation ---
std::vector<EvaluationResult> ChaturajiNNImpl::evaluate_batch(
    const std::vector<EvaluationRequest>& requests,
    torch::Device device)
{
    if (requests.empty()) {
        return {};
    }

    // 1. Collect state tensors and check dimensions
    std::vector<torch::Tensor> state_tensors;
    state_tensors.reserve(requests.size());
    for (const auto& req : requests) {
        // Ensure tensor has correct shape [C, H, W] before adding
        if (req.state_tensor.dim() != 3 || req.state_tensor.size(0) != 33 || req.state_tensor.size(1) != 8 || req.state_tensor.size(2) != 8) {
             throw std::runtime_error("Invalid tensor dimensions in EvaluationRequest. Expected [33, 8, 8], got " + std::string(req.state_tensor.sizes().vec().begin(), req.state_tensor.sizes().vec().end()));
        }
        // Assuming input tensors are on CPU, move them to the target device
        state_tensors.push_back(req.state_tensor.to(device));
    }

    // 2. Stack tensors into a batch
    torch::Tensor batch_tensor = torch::stack(state_tensors, 0); // Stacks along new dim 0 -> [B, C, H, W]

    // 3. Perform batched NN inference using the existing forward method
    torch::Tensor policy_logits_batch, value_batch;
    {
        torch::NoGradGuard no_grad;
        this->eval(); // Ensure evaluation mode
        std::tie(policy_logits_batch, value_batch) = this->forward(batch_tensor);
    }

    // 4. Move results to CPU for easier handling and packaging
    policy_logits_batch = policy_logits_batch.to(torch::kCPU); // Shape [B, 4096]
    value_batch = value_batch.to(torch::kCPU);           // Shape [B, 1]

    // Ensure batch sizes match
    if (policy_logits_batch.size(0) != static_cast<int64_t>(requests.size()) || value_batch.size(0) != static_cast<int64_t>(requests.size())) {
        throw std::runtime_error("Mismatch between request batch size and NN output batch size.");
    }

    // 5. Create EvaluationResult objects
    std::vector<EvaluationResult> results;
    results.reserve(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        EvaluationResult res;
        res.request_id = requests[i].request_id;
        res.policy_logits = policy_logits_batch[i]; // Extract the i-th policy tensor [4096]
        res.value = value_batch[i].item<float>();   // Extract the i-th value scalar
        results.push_back(res);
    }

    return results;
}
// --- End NEW ---

} // namespace chaturaji_cpp