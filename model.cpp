#include "model.h"

namespace chaturaji_cpp {

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
    // Input layer (40 input channels, 128 output channels)
    conv1_(torch::nn::Conv2dOptions(40, 128, /*kernel_size=*/3).padding(1)),
    bn1_(128),

    // Policy head
    policy_conv_(torch::nn::Conv2dOptions(128, 2, /*kernel_size=*/1)), // 128 in, 2 out
    policy_bn_(2),
    policy_fc_(torch::nn::LinearOptions(2 * 8 * 8, 4096)), // 2*8*8 = 128 input features

    // Value head
    value_conv_(torch::nn::Conv2dOptions(128, 1, /*kernel_size=*/1)), // 128 in, 1 out
    value_bn_(1),
    value_fc1_(torch::nn::LinearOptions(1 * 8 * 8, 128)), // 1*8*8 = 64 input features
    value_fc2_(torch::nn::LinearOptions(128, 1)) // 1 output value
{
    // Register input layer modules
    register_module("conv1", conv1_);
    register_module("bn1", bn1_);

    // Create and register residual blocks
    int num_res_blocks = 3; // Matches Python code
    for (int i = 0; i < num_res_blocks; ++i) {
        resblocks_->push_back(ResBlock(128));
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
    p = p.view({p.size(0), -1}); // Flatten [Batch, 2, 8, 8] -> [Batch, 128]
    p = policy_fc_(p); // Output shape [Batch, 4096] (Logits)

    // --- Value Head ---
    torch::Tensor v = value_conv_(x);
    v = value_bn_(v);
    v = torch::relu(v);
    v = v.view({v.size(0), -1}); // Flatten [Batch, 1, 8, 8] -> [Batch, 64]
    v = value_fc1_(v);
    v = torch::relu(v);
    v = value_fc2_(v); // Output shape [Batch, 1]
    v = torch::tanh(v); // Apply tanh activation

    return {p, v};
}

} // namespace chaturaji_cpp