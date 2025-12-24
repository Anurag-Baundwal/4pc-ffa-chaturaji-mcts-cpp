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
#include <sstream> 
#include <ctime>   

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
  int max_buffer_size,
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
            max_buffer_size,
            nn_batch_size, 
            worker_batch_size, 
            2.5, 
            4,   
            0.45, 
            0.25  
        );

        // 1. DATA GENERATION (C++)
        std::cout << "[C++] Generating " << num_games_per_iteration << " games..." << std::endl;
        auto start_gen = std::chrono::high_resolution_clock::now();
        
        // Generate data and capture exact number of samples created
        size_t points_generated = self_play_generator.generate_data(num_games_per_iteration);
        
        auto end_gen = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_gen = end_gen - start_gen;
        std::cout << "[C++] Generated " << points_generated << " positions in " << duration_gen.count() << "s." << std::endl;

        // 2. MODEL TRAINING (PYTHON)
        std::cout << "[Python] Starting training process..." << std::endl;
        
        // Construct the command to call the updated train.py with specific arguments
        std::string cmd = "python train.py";
        cmd += " --new-samples " + std::to_string(points_generated);
        cmd += " --sampling-rate " + std::to_string(target_sampling_rate_param);
        cmd += " --batch-size " + std::to_string(training_batch_size);
        cmd += " --lr " + std::to_string(learning_rate);
        cmd += " --wd " + std::to_string(weight_decay);
        cmd += " --max-buffer-size " + std::to_string(max_buffer_size);
        cmd += " --data-dir training_data";
        if (!initial_model_path.empty()) {
            cmd += " --load-weights " + initial_model_path;
        }

        int ret = std::system(cmd.c_str()); 
        
        if (ret != 0) {
            std::cerr << "!!! Python training script failed with exit code " << ret << " !!!" << std::endl;
        }

      } // self_play_generator destroyed here

      // 3. RELOAD MODEL (C++)
      if (fs::exists("model.onnx")) {
          try {
              network = std::make_unique<Model>("model.onnx");
              
              // 1. Archive the .onnx
              fs::path iter_model_path = model_dir / ("iter_" + std::to_string(iteration + 1) + ".onnx");
              fs::copy("model.onnx", iter_model_path, fs::copy_options::overwrite_existing);

              // 2. Archive the .pth and optimizer.pth so we can resume perfectly later
              if (fs::exists("model.pth")) {
                  fs::path iter_pth_path = model_dir / ("iter_" + std::to_string(iteration + 1) + ".pth");
                  fs::copy("model.pth", iter_pth_path, fs::copy_options::overwrite_existing);
              }
              if (fs::exists("optimizer.pth")) {
                  fs::path iter_opt_path = model_dir / ("iter_" + std::to_string(iteration + 1) + ".optimizer.pth");
                  fs::copy("optimizer.pth", iter_opt_path, fs::copy_options::overwrite_existing);
              }
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