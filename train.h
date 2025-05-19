#pragma once

#include <string>
#include <vector>
#include <array> // For std::array
#include <torch/torch.h>

#include "model.h"      
#include "self_play.h"  // For ReplayBuffer and GameDataStep
#include "types.h"      
#include "utils.h"      

namespace chaturaji_cpp {

class ChaturajiDataset : public torch::data::datasets::Dataset<ChaturajiDataset> {
public:
    explicit ChaturajiDataset(const std::vector<GameDataStep>& data); // GameDataStep now has std::array<double,4>

    // State tensor -> .data [C,H,W]
    // Target policy_tensor[4096] + value_tensor[4] -> .target [4100]
    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;
    torch::optional<size_t> size() const override;

private:
    torch::Tensor policy_to_tensor(const std::map<Move, double>& policy_map) const;

    std::vector<torch::Tensor> states_;   
    std::vector<torch::Tensor> policies_; 
    std::vector<torch::Tensor> values_;   // Each tensor will be of shape [4]
};


void train(
    int num_iterations = 65536,
    int num_games_per_iteration = 128, 
    double target_sampling_rate = 1.5,
    int training_batch_size = 1024, 
    int num_workers = 12,            
    int nn_batch_size = 1024,       
    int worker_batch_size = 48,
    double learning_rate = 0.001,
    double weight_decay = 1e-4,
    int simulations_per_move = 128,   
    const std::string& model_save_dir = "/content/drive/MyDrive/models", 
    const std::string& initial_model_path = "" 
);

} // namespace chaturaji_cpp