#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>

#include "model.h"      // Needs network definition
#include "self_play.h"  // Needs ReplayBuffer type definition
#include "types.h"      // Basic types
#include "utils.h"      // move_to_policy_index

namespace chaturaji_cpp {

// --- Training Dataset ---
class ChaturajiDataset : public torch::data::datasets::Dataset<ChaturajiDataset> {
public:
    // Constructor takes data from the replay buffer
    explicit ChaturajiDataset(const std::vector<GameDataStep>& data);

    // State tensor goes into .data
    // Concatenated policy + value tensor goes into .target
    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;

    // Returns the size of the dataset
    torch::optional<size_t> size() const override;

private:
    // Convert policy map to target tensor
    torch::Tensor policy_to_tensor(const std::map<Move, double>& policy_map) const;

    // Store pre-processed tensors for efficiency
    std::vector<torch::Tensor> states_;   // Board state tensors [C, H, W]
    std::vector<torch::Tensor> policies_; // Target policy tensors [4096]
    std::vector<torch::Tensor> values_;   // Target value tensors [1]
};


// --- Training Function ---
void train(
    int num_iterations = 65536,
    int num_games_per_iteration = 128, 
    int num_steps_per_iteration = 13,
    int training_batch_size = 1024, 
    int num_workers = 12,            
    int nn_batch_size = 1024,       
    int worker_batch_size = 48,
    double learning_rate = 0.001,
    double weight_decay = 1e-4,
    int simulations_per_move = 128,   
    const std::string& model_save_dir = "/content/drive/MyDrive/models", 
    const std::string& initial_model_path = "" // Path to load initial model from
);


} // namespace chaturaji_cpp