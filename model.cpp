#include "model.h"
#include <vector>
#include <stdexcept> // For runtime_error

namespace chaturaji_cpp {

// --- Configuration Constants ---
// Network Body
const int NUM_RES_BLOCKS = 4;
const int NUM_CHANNELS = 64;

// Policy Head
const int POLICY_HEAD_CONV_CHANNELS = 8;

// Value Head
const int VALUE_HEAD_CONV_CHANNELS = 12;
const int VALUE_FC_HIDDEN_CHANNELS = 256;
const int NUM_VALUE_OUTPUTS = 4; // MODIFIED: Output 4 values for the 4 players
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
    value_conv_(torch::nn::Conv2dOptions(NUM_CHANNELS, VALUE_HEAD_CONV_CHANNELS, /*kernel_size=*/1)), 
    value_bn_(VALUE_HEAD_CONV_CHANNELS),
    value_fc1_(torch::nn::LinearOptions(VALUE_HEAD_CONV_CHANNELS * 8 * 8, VALUE_FC_HIDDEN_CHANNELS)), 
    value_fc2_(torch::nn::LinearOptions(VALUE_FC_HIDDEN_CHANNELS, NUM_VALUE_OUTPUTS)) // MODIFIED: Output 4 values
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
    p = p.view({p.size(0), -1}); 
    p = policy_fc_(p); // Output shape [Batch, 4096] (Logits)

    // --- Value Head ---
    torch::Tensor v = value_conv_(x);
    v = value_bn_(v);
    v = torch::relu(v);
    v = v.view({v.size(0), -1}); 
    v = value_fc1_(v);
    v = torch::relu(v);
    v = value_fc2_(v); // MODIFIED: Output shape [Batch, 4]
    v = torch::tanh(v); // Apply tanh activation (range -1 to 1 for each of the 4 values)

    return {p, v};
}

// --- Batched Evaluation Implementation ---
std::vector<EvaluationResult> ChaturajiNNImpl::evaluate_batch(
    const std::vector<EvaluationRequest>& requests,
    torch::Device device)
{
    if (requests.empty()) {
        return {};
    }

    // 1. Convert std::vector<float> (from requests) into a single Batch Tensor
    // We assume input is [33, 8, 8] which is 2112 floats.
    int64_t batch_size = static_cast<int64_t>(requests.size());
    int64_t feature_size = 33 * 8 * 8;

    // Create a large tensor to hold the entire batch of floats
    torch::Tensor batch_tensor_cpu = torch::empty({batch_size, 33, 8, 8}, torch::kFloat);
    float* batch_ptr = batch_tensor_cpu.data_ptr<float>();

    // Copy data from each request's vector into the batch tensor's memory
    for (int64_t i = 0; i < batch_size; ++i) {
        const auto& floats = requests[i].state_floats;
        if (floats.size() != feature_size) {
            throw std::runtime_error("Invalid state_floats size. Expected 2112.");
        }
        // Copy memory: Destination, Source, Size
        std::memcpy(batch_ptr + (i * feature_size), floats.data(), feature_size * sizeof(float));
    }

    // 2. Move to device (GPU)
    torch::Tensor batch_tensor_device = batch_tensor_cpu.to(device);

    // 3. Perform Inference
    torch::Tensor policy_logits_batch, value_batch;
    {
        torch::NoGradGuard no_grad;
        this->eval();
        std::tie(policy_logits_batch, value_batch) = this->forward(batch_tensor_device);
    }

    // 4. Move results back to CPU
    policy_logits_batch = policy_logits_batch.to(torch::kCPU); // [B, 4096]
    value_batch = value_batch.to(torch::kCPU);                 // [B, 4]

    // 5. Pack results into EvaluationResult (std::array)
    std::vector<EvaluationResult> results;
    results.reserve(requests.size());

    // Get raw pointers for fast access
    auto policy_accessor = policy_logits_batch.accessor<float, 2>();
    auto value_accessor = value_batch.accessor<float, 2>();

    for (size_t i = 0; i < requests.size(); ++i) {
        EvaluationResult res;
        res.request_id = requests[i].request_id;

        // Manually copy Policy (Tensor -> std::array)
        for (int k = 0; k < 4096; ++k) {
            res.policy_logits[k] = policy_accessor[i][k];
        }

        // Manually copy Value (Tensor -> std::array)
        for (int k = 0; k < 4; ++k) {
            res.value[k] = value_accessor[i][k];
        }

        results.push_back(res);
    }

    return results;
}

} // namespace chaturaji_cpp