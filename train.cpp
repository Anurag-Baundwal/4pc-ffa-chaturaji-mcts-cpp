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
  else { log_file << "Log file opened successfully: " << log_file_path << std::endl; }

  if (log_file.is_open()) { log_file << "Using device: " << device << std::endl; }
  if (log_file.is_open()) { log_file << "Model directory created: " << model_dir << std::endl; }

  ChaturajiNN network;
  network->to(device); 
  if (!initial_model_path.empty() && fs::exists(initial_model_path)) {
      try { torch::load(network, initial_model_path, device); 
            std::cout << "Loaded initial model from: " << initial_model_path << std::endl; 
            if (log_file.is_open()) { log_file << "Loaded initial model from: " << initial_model_path << std::endl; }
      }
      catch (const c10::Error& e) { 
          std::cerr << "Error loading initial model: " << e.what() << ". Starting from scratch." << std::endl; 
          if (log_file.is_open()) { log_file << "Error loading initial model: " << e.what() << ". Starting from scratch." << std::endl; }
      }
  } else { 
      std::cout << "No initial model provided or found. Starting from scratch." << std::endl; 
      if (log_file.is_open()) { log_file << "No initial model provided or found. Starting from scratch." << std::endl; }
  }
  torch::optim::Adam optimizer(network->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));

  SelfPlay self_play_generator(
    network, device, num_workers, simulations_per_move, 1250000, 
    nn_batch_size, worker_batch_size, 2.5, 25, 0.45, 0.25
    );
  
  // --- START OF MODIFICATION: Pre-fill Replay Buffer ---
  const size_t TARGET_INITIAL_BUFFER_SIZE = 10000;
  std::cout << "\n--- Pre-filling Replay Buffer to " << TARGET_INITIAL_BUFFER_SIZE << " positions ---" << std::endl;
  if (log_file.is_open()) { log_file << "\n--- Pre-filling Replay Buffer to " << TARGET_INITIAL_BUFFER_SIZE << " positions ---" << std::endl; }

  while (self_play_generator.get_buffer().size() < TARGET_INITIAL_BUFFER_SIZE) {
      if (num_games_per_iteration <= 0) {
          std::cerr << "Error: num_games_per_iteration (" << num_games_per_iteration 
                    << ") is not positive. Cannot pre-fill buffer. Aborting pre-fill." << std::endl;
          if (log_file.is_open()) { 
              log_file << "Error: num_games_per_iteration (" << num_games_per_iteration 
                       << ") is not positive. Cannot pre-fill buffer. Aborting pre-fill." << std::endl; 
          }
          break; 
      }
      std::cout << "Current buffer size: " << self_play_generator.get_buffer().size() 
                << ". Generating " << num_games_per_iteration << " more games for pre-fill..." << std::endl;
      if (log_file.is_open()) {
          log_file << "Current buffer size: " << self_play_generator.get_buffer().size() 
                   << ". Generating " << num_games_per_iteration << " more games for pre-fill..." << std::endl;
      }

      auto start_prefill_batch = std::chrono::high_resolution_clock::now();
      size_t num_generated_prefill = self_play_generator.generate_data(num_games_per_iteration);
      auto end_prefill_batch = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> prefill_batch_duration = end_prefill_batch - start_prefill_batch;

      std::cout << "Pre-fill batch generated " << num_generated_prefill << " data points in " 
                << prefill_batch_duration.count() << " seconds. Buffer size: " << self_play_generator.get_buffer().size() << std::endl;
      if (log_file.is_open()) {
          log_file << "Pre-fill batch generated " << num_generated_prefill << " data points in " 
                   << prefill_batch_duration.count() << " seconds. Buffer size: " << self_play_generator.get_buffer().size() << std::endl;
      }
      
      // Safety break if max buffer size is somehow smaller than target (though current config is fine)
      // This check is a bit indirect as max_buffer_size_ is private to SelfPlay.
      // We rely on the fact that self_play_generator.get_buffer().size() won't grow if it hits its internal max.
      if (self_play_generator.get_buffer().size() < TARGET_INITIAL_BUFFER_SIZE &&
          self_play_generator.get_buffer().size() >= 1250000) { // Using the known max_buffer_size from SelfPlay constructor
            std::cout << "Warning: Replay buffer reached its maximum capacity (" 
                      << self_play_generator.get_buffer().size() 
                      << ") but is still less than the target pre-fill size ("
                      << TARGET_INITIAL_BUFFER_SIZE << "). Stopping pre-fill." << std::endl;
            if (log_file.is_open()) {
                log_file << "Warning: Replay buffer reached its maximum capacity (" 
                         << self_play_generator.get_buffer().size() 
                         << ") but is still less than the target pre-fill size ("
                         << TARGET_INITIAL_BUFFER_SIZE << "). Stopping pre-fill." << std::endl;
            }
            break; 
      }
  }
  std::cout << "--- Replay Buffer pre-filled. Final size: " << self_play_generator.get_buffer().size() << " ---" << std::endl;
  if (log_file.is_open()) { log_file << "--- Replay Buffer pre-filled. Final size: " << self_play_generator.get_buffer().size() << " ---" << std::endl; }
  // --- END OF MODIFICATION: Pre-fill Replay Buffer ---

  std::mt19937 buffer_rng(std::random_device{}());

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
      bool perform_training_this_iteration = true; 
      std::cout << "\n---------- ITERATION " << (iteration + 1) << "/" << num_iterations << " ----------" << std::endl;
      if (log_file.is_open()) { log_file << "\n---------- ITERATION " << (iteration + 1) << "/" << num_iterations << " ----------" << std::endl; }
      
      std::cout << "Generating " << num_games_per_iteration
                << " self-play games (Workers: " << num_workers
                << ", NN Batch: " << nn_batch_size
                << ", Worker Batch: " << worker_batch_size << ")..." << std::endl;
      if (log_file.is_open()) { 
          log_file << "Generating " << num_games_per_iteration
                   << " self-play games (Workers: " << num_workers
                   << ", NN Batch: " << nn_batch_size
                   << ", Worker Batch: " << worker_batch_size << ")..." << std::endl;
      }
      auto start_selfplay = std::chrono::high_resolution_clock::now();
      size_t num_generated_data_points = self_play_generator.generate_data(num_games_per_iteration); 
      auto end_selfplay = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> selfplay_duration = end_selfplay - start_selfplay;
      std::cout << "Self-play generation finished in " << selfplay_duration.count() << " seconds. Generated " << num_generated_data_points << " data points." << std::endl;
      if (log_file.is_open()) { log_file << "Self-play generation finished in " << selfplay_duration.count() << " seconds. Generated " << num_generated_data_points << " data points." << std::endl; }
      
      const ReplayBuffer& replay_buffer = self_play_generator.get_buffer();
      size_t current_buffer_size = replay_buffer.size();
      if (log_file.is_open()) { log_file << "Current replay buffer size: " << current_buffer_size << std::endl; }

      int dynamic_steps_per_iteration = 0; 
      if (num_generated_data_points > 0 && training_batch_size > 0) {
          double calculated_steps = static_cast<double>(num_generated_data_points) * target_sampling_rate_param / static_cast<double>(training_batch_size);
          dynamic_steps_per_iteration = static_cast<int>(std::round(calculated_steps));
          if (dynamic_steps_per_iteration == 0) {
              dynamic_steps_per_iteration = 1; 
          }
      }

      if (dynamic_steps_per_iteration == 0 || current_buffer_size < static_cast<size_t>(training_batch_size)) {
            std::cout << "  Skipping training this iteration. Dynamic steps: " << dynamic_steps_per_iteration
                      << ", Num generated data points: " << num_generated_data_points
                      << ", Buffer size: " << current_buffer_size << ", Training Batch size: " << training_batch_size 
                      << ", Target Sampling Rate: " << target_sampling_rate_param << std::endl;
            if (log_file.is_open()) {
                log_file << "  Skipping training this iteration. Dynamic steps: " << dynamic_steps_per_iteration
                         << ", Num generated data points: " << num_generated_data_points
                         << ", Buffer size: " << current_buffer_size << ", Training Batch size: " << training_batch_size 
                         << ", Target Sampling Rate: " << target_sampling_rate_param << std::endl;
            }
            perform_training_this_iteration = false;
      }

      if (perform_training_this_iteration) {
          std::cout << "Starting training for " << dynamic_steps_per_iteration << " steps (Batch Size: " << training_batch_size 
                    << ", Num data points generated this iteration: " << num_generated_data_points 
                    << ", Target sampling rate: " << target_sampling_rate_param << ")" << std::endl;
          if (log_file.is_open()) {
              log_file << "Starting training for " << dynamic_steps_per_iteration << " steps (Batch Size: " << training_batch_size 
                       << ", Num data points generated this iteration: " << num_generated_data_points 
                       << ", Target sampling rate: " << target_sampling_rate_param << ")" << std::endl;
          }
          
          std::vector<GameDataStep> training_data_vector(replay_buffer.begin(), replay_buffer.end());
          if (training_data_vector.empty() && dynamic_steps_per_iteration > 0) {
              std::cerr << "Error: Scheduled training steps (" << dynamic_steps_per_iteration 
                        << ") but training_data_vector is empty. Buffer size reported as: " << current_buffer_size 
                        << ". Skipping training." << std::endl;
              if (log_file.is_open()) {
                  log_file << "Error: Scheduled training steps (" << dynamic_steps_per_iteration 
                           << ") but training_data_vector is empty. Buffer size reported as: " << current_buffer_size 
                           << ". Skipping training." << std::endl;
              }
              perform_training_this_iteration = false; 
          }

          if (perform_training_this_iteration) { 
            network->train(); 
            double total_loss = 0.0;
            double total_policy_loss = 0.0;
            double total_value_loss = 0.0;
            int steps_performed_this_iter = 0;
            auto train_start_time = std::chrono::high_resolution_clock::now();

            for (int step = 0; step < dynamic_steps_per_iteration; ++step) {
                std::vector<torch::Tensor> state_batch_vec;
                std::vector<torch::Tensor> target_batch_vec; 
                state_batch_vec.reserve(training_batch_size);
                target_batch_vec.reserve(training_batch_size);
                std::uniform_int_distribution<size_t> dist(0, training_data_vector.size() - 1);

                for (int i = 0; i < training_batch_size; ++i) {
                    size_t sample_idx = dist(buffer_rng);
                    const GameDataStep& data_step = training_data_vector[sample_idx];
                    torch::Tensor state_t = board_to_tensor(std::get<0>(data_step), torch::kCPU).squeeze(0);
                    torch::Tensor policy_t = policy_to_tensor_static_train(std::get<1>(data_step));
                    
                    // MODIFIED: Convert std::array<double, 4> to torch::Tensor for target
                    const std::array<double, 4>& player_rewards_array_target = std::get<3>(data_step);
                    float rewards_float_carray_target[4];
                    for(int k=0; k<4; ++k) rewards_float_carray_target[k] = static_cast<float>(player_rewards_array_target[k]);
                    torch::Tensor value_t = torch::tensor(at::ArrayRef<float>(rewards_float_carray_target, 4), torch::kFloat32); // Shape [4]

                    torch::Tensor target_t = torch::cat({policy_t, value_t}, /*dim=*/0); // Shape [4100]

                    state_batch_vec.push_back(state_t);
                    target_batch_vec.push_back(target_t);
                }

                auto states = torch::stack(state_batch_vec, 0).to(device);
                auto targets = torch::stack(target_batch_vec, 0).to(device);
                
                // MODIFIED: Split targets correctly
                auto policy_target = targets.slice(/*dim=*/1, /*start=*/0, /*end=*/4096);    // Shape [B, 4096]
                auto value_target = targets.slice(/*dim=*/1, /*start=*/4096, /*end=*/4100); // Shape [B, 4]

                optimizer.zero_grad();
                torch::Tensor policy_pred, value_pred; // value_pred will be [B, 4]
                std::tie(policy_pred, value_pred) = network->forward(states); 

                auto policy_log_softmax = torch::log_softmax(policy_pred, /*dim=*/1);
                auto policy_loss = -torch::sum(policy_target * policy_log_softmax, /*dim=*/1).mean();
                // value_loss will compare [B, 4] with [B, 4]
                auto value_loss = torch::mse_loss(value_pred, value_target); 
                auto loss = policy_loss + value_loss;

                loss.backward();
                optimizer.step();

                if ((step + 1) % 1 == 0) { 
                    std::cout << "  Iter " << (iteration + 1)
                              << ", Step " << (step + 1) << "/" << dynamic_steps_per_iteration
                              << ": Loss: " << loss.item<double>()
                              << " (Policy: " << policy_loss.item<double>()
                              << ", Value: " << value_loss.item<double>() << ")"
                              << std::endl;
                    if (log_file.is_open()) {
                        log_file << "  Iter " << (iteration + 1)
                                 << ", Step " << (step + 1) << "/" << dynamic_steps_per_iteration
                                 << ": Loss: " << loss.item<double>()
                                 << " (Policy: " << policy_loss.item<double>()
                                 << ", Value: " << value_loss.item<double>() << ")"
                                 << std::endl;
                    }
                }
                total_loss += loss.item<double>();
                total_policy_loss += policy_loss.item<double>();
                total_value_loss += value_loss.item<double>();
                steps_performed_this_iter++;
            } 

            auto train_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> train_duration = train_end_time - train_start_time;

            if (steps_performed_this_iter > 0) {
                 std::cout << "  Training finished " << steps_performed_this_iter << " steps in " << train_duration.count() << "s."
                           << " Avg Loss: " << (total_loss / steps_performed_this_iter)
                           << " (Policy: " << (total_policy_loss / steps_performed_this_iter)
                           << ", Value: " << (total_value_loss / steps_performed_this_iter) << ")"
                           << std::endl;
                 if (log_file.is_open()) {
                     log_file << "  Training finished " << steps_performed_this_iter << " steps in " << train_duration.count() << "s."
                              << " Avg Loss: " << (total_loss / steps_performed_this_iter)
                              << " (Policy: " << (total_policy_loss / steps_performed_this_iter)
                              << ", Value: " << (total_value_loss / steps_performed_this_iter) << ")"
                              << std::endl;
                 }
            } else {
                 std::cout << "  No training steps performed this iteration (after checks)." << std::endl;
                 if (log_file.is_open()) { log_file << "  No training steps performed this iteration (after checks)." << std::endl; }
            }
            network->eval(); 
          }
        } // End if(perform_training_this_iteration) (outer one)

       if ((iteration + 1) % 1 == 0) { 
           fs::path save_path = model_dir / ("chaturaji_iter_" + std::to_string(iteration + 1) + ".pt");
           try {
               torch::save(network, save_path.string());
               std::cout << "Model saved after iteration " << (iteration + 1) << " to: " << save_path << std::endl;
               if (log_file.is_open()) { log_file << "Model saved after iteration " << (iteration + 1) << " to: " << save_path << std::endl; }
           } catch (const c10::Error& e) {
                std::cerr << "Error saving model: " << e.what() << std::endl;
                if (log_file.is_open()) { log_file << "Error saving model: " << e.what() << std::endl; }
           }
       }
  } 

  std::cout << "\nTraining finished." << std::endl;
  if (log_file.is_open()) {
      log_file << "\nTraining finished." << std::endl;
      log_file.close();
      std::cout << "Detailed log saved to: " << log_file_path << std::endl; 
  } else {
      std::cout << "Detailed log was not saved as the file could not be opened: " << log_file_path << std::endl;
  }
}

} // namespace chaturaji_cpp