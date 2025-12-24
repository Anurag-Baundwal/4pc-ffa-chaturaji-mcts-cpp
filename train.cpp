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
#include <memory>
#include <sstream> 

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
  // 1. PRE-CHECK LOAD MODEL
  if (!initial_model_path.empty() && !fs::exists(initial_model_path)) {
      std::cerr << "\n[C++] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
      std::cerr << "[C++] FATAL ERROR: Specified load-model path does not exist: " << initial_model_path << std::endl;
      std::cerr << "[C++] If you want to start a new training run, do not use --load-model." << std::endl;
      std::cerr << "[C++] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" << std::endl;
      return;
  }

  // 2. SETUP RUN DIRECTORY
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss_ts;
  ss_ts << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
  fs::path model_dir = fs::path(model_save_dir_base) / ("run_" + ss_ts.str());
  fs::path training_data_dir = "training_data";

  try { 
      fs::create_directories(model_dir); 
      fs::create_directories(training_data_dir);
      std::cout << "[C++] Output folder: " << model_dir << std::endl; 
  } catch (...) { return; }

  // 3. INITIALIZE ENGINE MODEL
  std::unique_ptr<Model> network = nullptr;
  std::string current_weights_path = ""; // Initialize as empty

  if (!initial_model_path.empty()) {
      // User provided a model to resume
      current_weights_path = initial_model_path; 
      network = std::make_unique<Model>(initial_model_path);
  } else { 
      std::cout << "[C++] No model provided. Initializing random weights..." << std::endl;
      // Tell Python to save the random init inside our specific folder
      std::string random_onnx_path = (model_dir / "iter_0.onnx").string();
      std::string init_cmd = "python model.py export_random " + random_onnx_path;
      
      if (std::system(init_cmd.c_str()) == 0) {
           // We load the random model into C++ so Self-Play can run
           network = std::make_unique<Model>(random_onnx_path);
           
           // NOTE: DO NOT set current_weights_path here.
           // If we leave it empty, the first call to 'python train.py' will NOT 
           // receive --load-weights, so Python will generate its own random .pth 
           // and start fresh, which is exactly what we want.
      } else { return; }
  }

  // 4. MAIN LOOP
  for (int iteration = 0; iteration < num_iterations; ++iteration) {
      std::cout << "\n========== ITERATION " << (iteration + 1) << " / " << num_iterations << " ==========" << std::endl;

      size_t points_generated = 0;
      {
          // Restored explicit hyperparameters
          SelfPlay self_play_generator(
              network.get(), 
              num_workers, 
              simulations_per_move, 
              max_buffer_size, 
              nn_batch_size, 
              worker_batch_size,
              2.5,  // c_puct
              4,    // temp_decay
              0.30, // alpha
              0.25  // epsilon
          );
          
          // Phase 1: Self-Play with Timing Info
          std::cout << "[C++] Generating " << num_games_per_iteration << " games..." << std::endl;
          auto start_gen = std::chrono::high_resolution_clock::now();
          
          points_generated = self_play_generator.generate_data(num_games_per_iteration);

          auto end_gen = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> duration_gen = end_gen - start_gen;
          
          std::cout << "[C++] Generated " << points_generated << " positions in " 
                    << std::fixed << std::setprecision(2) << duration_gen.count() << "s ";
          if (duration_gen.count() > 0) {
              std::cout << "(" << static_cast<int>(points_generated / duration_gen.count()) << " pos/s)";
          }
          std::cout << std::endl;

          // Phase 2: Python Training
          std::stringstream cmd;
          cmd << "python train.py"
              << " --save-dir \"" << model_dir.string() << "\""
              << " --new-samples " << points_generated
              << " --sampling-rate " << target_sampling_rate_param
              << " --batch-size " << training_batch_size
              << " --lr " << learning_rate
              << " --wd " << weight_decay
              << " --data-dir " << training_data_dir.string()
              << " --max-buffer-size " << max_buffer_size; // Restored max-buffer-size
          
          if (!current_weights_path.empty()) {
              cmd << " --load-weights \"" << current_weights_path << "\"";
          }

          std::cout << "[Python] Starting training process..." << std::endl;
          int ret = std::system(cmd.str().c_str()); 
          if (ret != 0) {
              std::cerr << "[C++] Python training crashed. Terminating C++ loop." << std::endl;
              return;
          }
      }

      // Phase 3: Archive Iteration & Reload
      // Python saved to 'latest.onnx'. We rename to 'iter_N.onnx' for history.
      try {
          fs::path latest_onnx = model_dir / "latest.onnx";
          fs::path latest_pth  = model_dir / "latest.pth";
          fs::path latest_opt  = model_dir / "latest.optimizer.pth";

          std::string suffix = "iter_" + std::to_string(iteration + 1);
          fs::path arch_onnx = model_dir / (suffix + ".onnx");
          fs::path arch_pth  = model_dir / (suffix + ".pth");
          fs::path arch_opt  = model_dir / (suffix + ".optimizer.pth");

          // Keep archived copies
          fs::copy(latest_onnx, arch_onnx, fs::copy_options::overwrite_existing);
          fs::copy(latest_pth,  arch_pth,  fs::copy_options::overwrite_existing);
          if (fs::exists(latest_opt)) fs::copy(latest_opt, arch_opt, fs::copy_options::overwrite_existing);

          // Update engine with the absolute path of the new iteration's ONNX
          current_weights_path = arch_onnx.string();
          network = std::make_unique<Model>(current_weights_path);
          
          std::cout << "[C++] Finished iteration " << (iteration + 1) << ". Weights: " << arch_onnx.filename() << std::endl;

      } catch (const std::exception& e) {
          std::cerr << "[C++] Error during file archiving: " << e.what() << std::endl;
          return;
      }
  }
}

} // namespace chaturaji_cpp