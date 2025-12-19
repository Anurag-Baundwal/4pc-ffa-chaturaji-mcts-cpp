#include "train.h"
#include "self_play.h"
#include "model.h"
#include "utils.h" 

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <fstream> 
#include <cstdlib> // For std::system

namespace fs = std::filesystem;

namespace chaturaji_cpp {

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
  
  // Setup directories
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss_ts;
  ss_ts << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
  std::string timestamp = ss_ts.str();
  fs::path model_dir = fs::path(model_save_dir_base) / ("run_" + timestamp);
  fs::path training_data_dir = "training_data";

  try { 
      fs::create_directories(model_dir); 
      fs::create_directories(training_data_dir);
      std::cout << "Model directory: " << model_dir << std::endl; 
  }
  catch (const std::exception& e) { std::cerr << "Error creating directories: " << e.what() << std::endl; return; }

  // Load or Initialize Network
  ChaturajiNN network;
  network->to(device); 
  
  if (!initial_model_path.empty() && fs::exists(initial_model_path)) {
      try { 
          torch::load(network, initial_model_path, device); 
          std::cout << "Loaded initial model from: " << initial_model_path << std::endl; 
      }
      catch (const c10::Error& e) { 
          std::cerr << "Error loading initial model: " << e.what() << ". Starting from scratch." << std::endl; 
      }
  } else { 
      std::cout << "No initial model provided. Starting from scratch." << std::endl; 
  }

  // Initialize SelfPlay engine
  SelfPlay self_play_generator(
    network, 
    device, 
    num_workers, 
    simulations_per_move, 
    1250000, // max_buffer (unused now but kept for constructor)
    nn_batch_size, 
    worker_batch_size, 
    2.5, // c_puct
    8,   // temp decay
    0.45, // alpha
    0.25  // epsilon
  );

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
      std::cout << "\n========== ITERATION " << (iteration + 1) << " / " << num_iterations << " ==========" << std::endl;

      // 1. DATA GENERATION (C++)
      std::cout << "[C++] Generating " << num_games_per_iteration << " games..." << std::endl;
      auto start_gen = std::chrono::high_resolution_clock::now();
      
      size_t points = self_play_generator.generate_data(num_games_per_iteration);
      
      auto end_gen = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_gen = end_gen - start_gen;
      std::cout << "[C++] Generated " << points << " positions in " << duration_gen.count() << "s." << std::endl;

      // 2. MODEL TRAINING (PYTHON)
      std::cout << "[Python] Starting training process..." << std::endl;
      // Note: Ensure train.py is in the current working directory or in PATH
      int ret = std::system("python train.py"); 
      
      if (ret != 0) {
          std::cerr << "!!! Python training script failed with exit code " << ret << " !!!" << std::endl;
          // Optionally break here, or continue with old model
          // return; 
      }

      // 3. RELOAD MODEL (C++)
      // Python script exports JIT model to "model.pt"
      if (fs::exists("model.pt")) {
          try {
              torch::load(network, "model.pt", device);
              network->to(device); // Ensure it's on correct device
              network->eval();
              std::cout << "[C++] Reloaded updated model (model.pt)." << std::endl;
              
              // Archive the model for this iteration
              fs::path iter_model_path = model_dir / ("iter_" + std::to_string(iteration + 1) + ".pt");
              fs::copy("model.pt", iter_model_path, fs::copy_options::overwrite_existing);
          } catch (const std::exception& e) {
              std::cerr << "[C++] Failed to reload model: " << e.what() << std::endl;
          }
      } else {
          std::cerr << "[C++] Warning: model.pt not found. Using previous weights." << std::endl;
      }
  } 

  std::cout << "\nTraining finished." << std::endl;
}

} // namespace chaturaji_cpp