#include "model.h"
#include <vector>
#include <stdexcept> // For runtime_error

namespace chaturaji_cpp {

// --- Configuration Constants ---
// Network Body
const int NUM_RES_BLOCKS = 10;
const int NUM_CHANNELS = 128;

// Policy Head
const int POLICY_HEAD_CONV_CHANNELS = 16;

// Value Head
const int VALUE_HEAD_CONV_CHANNELS = 16;
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

    // 1. Collect state tensors (all assumed to be on CPU initially)
    std::vector<torch::Tensor> state_tensors_cpu;
    state_tensors_cpu.reserve(requests.size());
    for (const auto& req : requests) {
        if (req.state_tensor.dim() != 3 || req.state_tensor.size(0) != 33 || req.state_tensor.size(1) != 8 || req.state_tensor.size(2) != 8) {
             throw std::runtime_error("Invalid tensor dimensions in EvaluationRequest. Expected [33, 8, 8], got " + std::string(req.state_tensor.sizes().vec().begin(), req.state_tensor.sizes().vec().end()));
        }
        state_tensors_cpu.push_back(req.state_tensor); // Keep on CPU
    }

    // 2. Stack tensors into a batch on CPU
    torch::Tensor batch_tensor_cpu = torch::stack(state_tensors_cpu, 0);

    // 3. Move the entire batch tensor to the target device once
    torch::Tensor batch_tensor_device = batch_tensor_cpu.to(device);

    // 4. Perform batched NN inference
    torch::Tensor policy_logits_batch, value_batch;
    {
        torch::NoGradGuard no_grad;
        this->eval();
        std::tie(policy_logits_batch, value_batch) = this->forward(batch_tensor_device); // Use the device tensor
    }

    // 5. Move results to CPU
    policy_logits_batch = policy_logits_batch.to(torch::kCPU); // Shape [B, 4096]
    value_batch = value_batch.to(torch::kCPU);           // Shape [B, 4]

    if (policy_logits_batch.size(0) != static_cast<int64_t>(requests.size()) || value_batch.size(0) != static_cast<int64_t>(requests.size())) {
        throw std::runtime_error("Mismatch between request batch size and NN output batch size.");
    }
    if (value_batch.size(1) != NUM_VALUE_OUTPUTS) { // Check second dimension of value_batch
        throw std::runtime_error("NN value output has incorrect number of player values. Expected " +
                                 std::to_string(NUM_VALUE_OUTPUTS) + ", got " + std::to_string(value_batch.size(1)));
    }


    // 5. Create EvaluationResult objects
    std::vector<EvaluationResult> results;
    results.reserve(requests.size());
    auto value_accessor = value_batch.accessor<float, 2>(); // Access as [Batch, NumPlayers]

    for (size_t i = 0; i < requests.size(); ++i) {
        EvaluationResult res;
        res.request_id = requests[i].request_id;
        res.policy_logits = policy_logits_batch[i]; 
        
        // MODIFIED: Populate the std::array<float, 4>
        for (int j = 0; j < NUM_VALUE_OUTPUTS; ++j) {
            res.value[j] = value_accessor[i][j];
        }
        results.push_back(res);
    }

    return results;
}

} // namespace chaturaji_cpp