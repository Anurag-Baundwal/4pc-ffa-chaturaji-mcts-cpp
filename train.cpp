#include "train.h"
#include "self_play.h"
#include "model.h"
#include "utils.h" 

#include <torch/torch.h>
#include <torch/data/dataloader.h> 
#include <torch/data/datasets/base.h> 
#include <torch/optim/adam.h>     

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <fstream> 
#include <algorithm> // for std::copy with std::array
#include <cmath>     // For std::round

namespace fs = std::filesystem;

namespace chaturaji_cpp {

inline torch::Tensor policy_to_tensor_static_train(const std::map<Move, double>& policy_map) { // Renamed for local use
  torch::Tensor policy_tensor = torch::zeros({4096}, torch::kFloat32);
  auto policy_accessor = policy_tensor.accessor<float, 1>();
  for (const auto& pair : policy_map) {
      int index = move_to_policy_index(pair.first);
      if (index >= 0 && index < 4096) {
          policy_accessor[index] = static_cast<float>(pair.second);
      }
  }
  return policy_tensor;
}

ChaturajiDataset::ChaturajiDataset(const std::vector<GameDataStep>& data) {
    states_.reserve(data.size());
    policies_.reserve(data.size());
    values_.reserve(data.size());
    torch::Device cpu_device = torch::kCPU;

    for (const auto& step : data) {
        torch::Tensor state_t = board_to_tensor(std::get<0>(step), cpu_device).squeeze(0); 
        torch::Tensor policy_t = policy_to_tensor(std::get<1>(step)); 
        
        // MODIFIED: Convert std::array<double, 4> to torch::Tensor of shape [4]
        const std::array<double, 4>& player_rewards_array = std::get<3>(step);
        // Create a temporary C-style array to initialize tensor from data
        float rewards_float_carray[4];
        for(int i=0; i<4; ++i) rewards_float_carray[i] = static_cast<float>(player_rewards_array[i]);
        torch::Tensor value_t = torch::tensor(at::ArrayRef<float>(rewards_float_carray, 4), torch::kFloat32); // Shape [4]


        states_.push_back(state_t);
        policies_.push_back(policy_t);
        values_.push_back(value_t); 
    }
}

torch::data::Example<torch::Tensor, torch::Tensor> ChaturajiDataset::get(size_t index) {
     torch::Tensor state_tensor = states_[index]; 
     torch::Tensor policy_tensor = policies_[index]; 
     torch::Tensor value_tensor = values_[index];   // MODIFIED: Shape [4]

     // Concatenate policy [4096] and value [4] -> Target shape [4100]
     torch::Tensor target_tensor = torch::cat({policy_tensor, value_tensor}, /*dim=*/0);

     return {state_tensor, target_tensor};
}

torch::optional<size_t> ChaturajiDataset::size() const {
    return states_.size();
}

torch::Tensor ChaturajiDataset::policy_to_tensor(const std::map<Move, double>& policy_map) const {
     return policy_to_tensor_static_train(policy_map); // Use local static helper
}

void train(
  int num_iterations,
  int num_games_per_iteration,
  double target_sampling_rate_param,
  int training_batch_size,
  int num_workers,        
  int nn_batch_size,      
  int worker_batch_size, 
  double learning_rate,
  double weight_decay,
  int simulations_per_move,
  const std::string& model_save_dir_base,
  const std::string& initial_model_path)
{
  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Using device: " << device << std::endl;
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss_ts;
  ss_ts << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
  std::string timestamp = ss_ts.str();
  fs::path model_dir = fs::path(model_save_dir_base) / ("run_" + timestamp);
  std::ofstream log_file; 
  fs::path log_file_path = model_dir / "detailed_training_log.txt"; 

  try { fs::create_directories(model_dir); std::cout << "Model directory created: " << model_dir << std::endl; }
  catch (const std::exception& e) { std::cerr << "Error creating model directory: " << e.what() << std::endl; return; }

  log_file.open(log_file_path.string(), std::ios_base::app); 
  if (!log_file.is_open()) { std::cerr << "Warning: Could not open log file: " << log_file_path << std::endl; }
  else { 
      log_file << "Log file opened successfully: " << log_file_path << std::endl; 
      log_file.flush(); // ADDED FLUSH
  }

  if (log_file.is_open()) { 
      log_file << "Using device: " << device << std::endl; 
      log_file.flush(); // ADDED FLUSH
  }
  if (log_file.is_open()) { 
      log_file << "Model directory created: " << model_dir << std::endl; 
      log_file.flush(); // ADDED FLUSH
  }

  ChaturajiNN network;
  network->to(device); 
  if (!initial_model_path.empty() && fs::exists(initial_model_path)) {
      try { torch::load(network, initial_model_path, device); 
            std::cout << "Loaded initial model from: " << initial_model_path << std::endl; 
            if (log_file.is_open()) { 
                log_file << "Loaded initial model from: " << initial_model_path << std::endl; 
                log_file.flush(); // ADDED FLUSH
            }
      }
      catch (const c10::Error& e) { 
          std::cerr << "Error loading initial model: " << e.what() << ". Starting from scratch." << std::endl; 
          if (log_file.is_open()) { 
              log_file << "Error loading initial model: " << e.what() << ". Starting from scratch." << std::endl; 
              log_file.flush(); // ADDED FLUSH
          }
      }
  } else { 
      std::cout << "No initial model provided or found. Starting from scratch." << std::endl; 
      if (log_file.is_open()) { 
          log_file << "No initial model provided or found. Starting from scratch." << std::endl; 
          log_file.flush(); // ADDED FLUSH
      }
  }
  torch::optim::Adam optimizer(network->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));

  SelfPlay self_play_generator(
    network, device, num_workers, simulations_per_move, 1250000, 
    nn_batch_size, worker_batch_size, 2.5, 8, 0.45, 0.25
    );

  std::mt19937 buffer_rng(std::random_device{}());

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
      std::cout << "\n--- ITERATION " << (iteration + 1) << " ---" << std::endl;

      // 1. DATA GENERATION (C++)
      SelfPlay self_play_generator(network, device, num_workers, simulations_per_move, ...);
      self_play_generator.generate_data(num_games_per_iteration);

      // 2. MODEL TRAINING (PYTHON)
      std::cout << "Invoking Python training..." << std::endl;
      int ret = std::system("python train.py"); 
      if (ret != 0) {
          std::cerr << "Python training failed!" << std::endl;
          return;
      }

      // 3. RELOAD MODEL (C++ / Libtorch)
      // Load the model that Python just exported via JIT trace
      try {
          torch::load(network, "model.pt", device);
          std::cout << "Reloaded updated model from Python." << std::endl;
      } catch (const std::exception& e) {
          std::cerr << "Failed to reload model: " << e.what() << std::endl;
      }
    }
  } 

  std::cout << "\nTraining finished." << std::endl;
  if (log_file.is_open()) {
      log_file << "\nTraining finished." << std::endl;
      log_file.flush(); // ADDED FLUSH (before closing)
      log_file.close(); // Close also flushes
      std::cout << "Detailed log saved to: " << log_file_path << std::endl; 
  } else {
      std::cout << "Detailed log was not saved as the file could not be opened: " << log_file_path << std::endl;
  }
}

} // namespace chaturaji_cpp