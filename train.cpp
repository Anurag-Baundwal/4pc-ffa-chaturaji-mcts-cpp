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
#include <regex>
#include <cstdlib> // For std::system

namespace fs = std::filesystem;

namespace chaturaji_cpp {

// Helper to extract iteration number from filename "iter_1234.onnx"
int extract_iteration_from_path(const std::string& path) {
    std::regex re("iter_(\\d+)\\.onnx");
    std::smatch match;
    std::filesystem::path p(path);
    std::string filename = p.filename().string();
    if (std::regex_search(filename, match, re)) {
        return std::stoi(match[1]);
    }
    return 0; 
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
  int max_buffer_size,
  int temp_decay_move,
  double dirichlet_alpha,
  double dirichlet_epsilon,
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

  // 3. INITIALIZE RUN STATS & ENGINE MODEL
  std::unique_ptr<Model> network = nullptr;
  std::string current_weights_path = ""; 
  RunStats stats;

  if (!initial_model_path.empty()) {
      // User provided a model to resume
      current_weights_path = initial_model_path; 
      network = std::make_unique<Model>(initial_model_path);
      
      // Attempt to load stats from the directory of the loaded model
      fs::path loaded_path(initial_model_path);
      fs::path parent_dir = loaded_path.parent_path();
      
      fs::path stats_file = parent_dir / "run_info.txt";
      if (fs::exists(stats_file)) {
          stats = RunStats::load(stats_file.string());
          std::cout << "[C++] Loaded run statistics from: " << stats_file.string() << std::endl;
          std::cout << "[C++] Resuming from Global Iteration: " << stats.global_iteration << std::endl;
      } else {
          // Fallback to filename guessing
          stats.global_iteration = extract_iteration_from_path(initial_model_path);
          std::cout << "[C++] Warning: run_info.txt not found. Guessed iteration " << stats.global_iteration << " from filename." << std::endl;
      }
      
  } else { 
      std::cout << "[C++] No model provided. Initializing random weights..." << std::endl;
      // Tell Python to save the random init inside our specific folder
      std::string random_onnx_path = (model_dir / "iter_0.onnx").string();
      std::string init_cmd = "python model.py export_random " + random_onnx_path;
      
      if (std::system(init_cmd.c_str()) == 0) {
           // We load the random model into C++ so Self-Play can run
           network = std::make_unique<Model>(random_onnx_path);
           stats.global_iteration = 0;
           stats.total_samples_generated = 0;
      } else { 
          std::cerr << "[C++] Error: Failed to generate random model." << std::endl;
          return; 
      }
  }

  // 4. MAIN LOOP
  // 'num_iterations' is interpreted as the Target Global Iteration
  if (stats.global_iteration >= num_iterations) {
      std::cout << "[C++] Target global iteration " << num_iterations << 
                 " already reached (Current: " << stats.global_iteration << "). Stopping." << std::endl;
      return;
  }

  std::cout << "[C++] Starting training session. Target Global Iteration: " << num_iterations << std::endl;

  while (stats.global_iteration < num_iterations) {
      stats.global_iteration++;
      stats.session_iterations++;

      std::cout << "\n========== ITERATION " << stats.global_iteration 
                << " (Target: " << num_iterations << ") ==========" << std::endl;

      size_t points_generated = 0;
      double duration_sec = 0.0;

      // Phase 1: Self-Play (Scoped to ensure threads join before Python starts)
      {
          SelfPlay self_play_generator(
              network.get(), 
              num_workers, 
              simulations_per_move, 
              max_buffer_size, 
              nn_batch_size, 
              worker_batch_size,
              2.5,  // c_puct
              temp_decay_move,
              dirichlet_alpha,
              dirichlet_epsilon
          );
          
          // Phase 1: Self-Play
          std::cout << "[C++] Generating " << num_games_per_iteration << " games..." << std::endl;
          auto start_gen = std::chrono::high_resolution_clock::now();
          
          points_generated = self_play_generator.generate_data(num_games_per_iteration);

          auto end_gen = std::chrono::high_resolution_clock::now();
          duration_sec = std::chrono::duration<double>(end_gen - start_gen).count();
      } 

      // Update Stats
      stats.total_samples_generated += points_generated;
      stats.session_samples += points_generated;

      // We must release the file handle to 'latest.onnx'
      // otherwise Python cannot overwrite 'latest.onnx' on Windows.
      network.reset();

      // Logging number of positions generated and speed (sims/s)
      std::cout << "[C++] Generated " << points_generated << " positions in " 
                << std::fixed << std::setprecision(2) << duration_sec << "s ";
      
      if (duration_sec > 0) {
          double total_sims = static_cast<double>(points_generated) * simulations_per_move;
          double sims_per_sec = total_sims / duration_sec;
          std::cout << "(" << std::fixed << std::setprecision(2) << sims_per_sec << " sims/s)";
      }
      std::cout << std::endl;

      // Phase 2: Python Training
      std::stringstream cmd;
      // Using python -u for unbuffered output to see progress
      cmd << "python -u train.py"
          << " --save-dir \"" << model_dir.string() << "\""
          << " --new-samples " << points_generated
          << " --sampling-rate " << target_sampling_rate_param
          << " --batch-size " << training_batch_size
          << " --lr " << learning_rate
          << " --wd " << weight_decay
          << " --data-dir " << training_data_dir.string()
          << " --max-buffer-size " << max_buffer_size;
      
      if (!current_weights_path.empty()) {
          cmd << " --load-weights \"" << current_weights_path << "\"";
      }

      std::cout << "[Python] Starting training process..." << std::endl;
      if (std::system(cmd.str().c_str()) != 0) {
          std::cerr << "[C++] Python training crashed. Terminating." << std::endl;
          return;
      }

      // Phase 3: Archive & Reload
      try {
          fs::path latest_onnx = model_dir / "latest.onnx";
          fs::path latest_pth  = model_dir / "latest.pth";
          fs::path latest_opt  = model_dir / "latest.optimizer.pth";

          // 1. Save RunStats
          fs::path info_file = model_dir / "run_info.txt";
          stats.save(info_file.string());

          // 2. Periodic Archiving       
          const int archive_interval = 25; 
          if (stats.global_iteration % archive_interval == 0) {
              std::string suffix = "iter_" + std::to_string(stats.global_iteration);
              fs::path arch_onnx = model_dir / (suffix + ".onnx");
              fs::path arch_pth  = model_dir / (suffix + ".pth");
              fs::path arch_opt  = model_dir / (suffix + ".optimizer.pth");

              fs::copy(latest_onnx, arch_onnx, fs::copy_options::overwrite_existing);
              fs::copy(latest_pth,  arch_pth,  fs::copy_options::overwrite_existing);
              if (fs::exists(latest_opt)) {
                  fs::copy(latest_opt, arch_opt, fs::copy_options::overwrite_existing);
              }
              std::cout << "[C++] Archived checkpoint: " << suffix << std::endl;
          }

          // 3. Reload engine
          // Re-initialize the C++ model with the new weights for the next Self-Play phase.
          current_weights_path = latest_onnx.string();
          network = std::make_unique<Model>(current_weights_path);
          
          std::cout << "[C++] Finished iteration " << stats.global_iteration << ". Weights: " << latest_onnx.filename().string() << std::endl;

      } catch (const std::exception& e) {
          std::cerr << "[C++] Error during file archiving: " << e.what() << std::endl;
          return;
      }
  }
}

} // namespace chaturaji_cpp