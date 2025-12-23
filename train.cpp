#include "train.h"
#include "self_play.h"
#include "model.h"
#include "utils.h" 

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <fstream> 
#include <cstdlib> 
#include <memory>
#include <sstream> // <--- ADDED THIS (Fixes C2079)
#include <ctime>   // <--- ADDED THIS (For std::localtime)

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
  std::unique_ptr<Model> network = nullptr;
  
  if (!initial_model_path.empty() && fs::exists(initial_model_path)) {
      try { 
          network = std::make_unique<Model>(initial_model_path);
          std::cout << "Loaded initial model from: " << initial_model_path << std::endl; 
      }
      catch (const std::exception& e) { 
          std::cerr << "Error loading initial model: " << e.what() << ". Training will fail if self-play starts without model." << std::endl;
          return;
      }
  } else { 
      std::cout << "No initial model provided. Triggering Python script to generate random initialization..." << std::endl;
      // Note: Make sure model.py is updated with export_random before running this
      int ret = std::system("python model.py export_random initial_random.onnx"); 
      if (ret == 0 && fs::exists("initial_random.onnx")) {
           network = std::make_unique<Model>("initial_random.onnx");
           std::cout << "Loaded random initial model." << std::endl;
      } else {
           std::cerr << "Failed to generate random model via Python. Aborting." << std::endl;
           return;
      }
  }

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
      std::cout << "\n========== ITERATION " << (iteration + 1) << " / " << num_iterations << " ==========" << std::endl;

      {
        // Scope the SelfPlay instance so it releases the Model pointer before we reload
        SelfPlay self_play_generator(
            network.get(), 
            num_workers, 
            simulations_per_move, 
            1250000, 
            nn_batch_size, 
            worker_batch_size, 
            2.5, 
            8,   
            0.45, 
            0.25  
        );

        // 1. DATA GENERATION (C++)
        std::cout << "[C++] Generating " << num_games_per_iteration << " games..." << std::endl;
        auto start_gen = std::chrono::high_resolution_clock::now();
        
        size_t points = self_play_generator.generate_data(num_games_per_iteration);
        
        auto end_gen = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_gen = end_gen - start_gen;
        std::cout << "[C++] Generated " << points << " positions in " << duration_gen.count() << "s." << std::endl;
      } // self_play_generator destroyed here

      // 2. MODEL TRAINING (PYTHON)
      std::cout << "[Python] Starting training process..." << std::endl;
      int ret = std::system("python train.py"); 
      
      if (ret != 0) {
          std::cerr << "!!! Python training script failed with exit code " << ret << " !!!" << std::endl;
      }

      // 3. RELOAD MODEL (C++)
      if (fs::exists("model.onnx")) {
          try {
              // Destroy old model and load new one
              network = std::make_unique<Model>("model.onnx");
              std::cout << "[C++] Reloaded updated model (model.onnx)." << std::endl;
              
              fs::path iter_model_path = model_dir / ("iter_" + std::to_string(iteration + 1) + ".onnx");
              fs::copy("model.onnx", iter_model_path, fs::copy_options::overwrite_existing);
          } catch (const std::exception& e) {
              std::cerr << "[C++] Failed to reload model: " << e.what() << std::endl;
          }
      } else {
          std::cerr << "[C++] Warning: model.onnx not found. Using previous weights." << std::endl;
      }
  } 

  std::cout << "\nTraining finished." << std::endl;
}

} // namespace chaturaji_cpp