#pragma once

#include <string>
#include <vector>
#include <array>

#include "model.h"      
#include "self_play.h"  
#include "types.h"      
#include "utils.h"      

namespace chaturaji_cpp {

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
    int max_buffer_size = 200000,  
    int temp_decay_move = 20,
    double dirichlet_alpha = 0.4,
    double dirichlet_epsilon = 0.25,
    const std::string& model_save_dir = "/content/drive/MyDrive/models", 
    const std::string& initial_model_path = ""
);

} // namespace chaturaji_cpp