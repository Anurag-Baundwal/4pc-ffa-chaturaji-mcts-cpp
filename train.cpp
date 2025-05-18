#include "train.h"
#include "self_play.h"
#include "model.h"
#include "utils.h" // For board_to_tensor

#include <torch/torch.h>
#include <torch/data/dataloader.h> // For DataLoader
#include <torch/data/datasets/base.h> // For Dataset
#include <torch/optim/adam.h>     // For Adam optimizer

// --- REMOVED AMP Header ---
// #if AT_CUDA_ENABLED
// #include <torch/cuda_amp.h> // This header does not exist in Libtorch distribution
// #define USE_AMP_IF_AVAILABLE true
// #else // AT_CUDA_ENABLED
// #define USE_AMP_IF_AVAILABLE false
// #endif // AT_CUDA_ENABLED
// --- END REMOVAL ---

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <fstream> // Added for std::ofstream

namespace fs = std::filesystem;

namespace chaturaji_cpp {

// Forward declaration for the helper function
inline torch::Tensor policy_to_tensor_static(const std::map<Move, double>& policy_map);

// --- ChaturajiDataset Implementation ---

ChaturajiDataset::ChaturajiDataset(const std::vector<GameDataStep>& data) {
    states_.reserve(data.size());
    policies_.reserve(data.size());
    values_.reserve(data.size());
    torch::Device cpu_device = torch::kCPU;
    for (const auto& step : data) {
        torch::Tensor state_t = board_to_tensor(std::get<0>(step), cpu_device).squeeze(0); // [C,H,W]
        torch::Tensor policy_t = policy_to_tensor(std::get<1>(step)); // [4096]
        // --- Store value as a tensor with one element ---
        torch::Tensor value_t = torch::tensor({std::get<3>(step)}, torch::kFloat32); // Shape [1]

        states_.push_back(state_t);
        policies_.push_back(policy_t);
        values_.push_back(value_t); // Store the [1] shaped tensor
    }
}

// --- FIX: Implement get() to return Example<Tensor, Tensor> ---
torch::data::Example<torch::Tensor, torch::Tensor> ChaturajiDataset::get(size_t index) {
     // Data is the state tensor
     torch::Tensor state_tensor = states_[index]; // Shape [C, H, W]

     // Target is the concatenation of policy and value
     torch::Tensor policy_tensor = policies_[index]; // Shape [4096]
     torch::Tensor value_tensor = values_[index];   // Shape [1]

     // Concatenate along a new dimension (dim 0) -> Shape [4097]
     torch::Tensor target_tensor = torch::cat({policy_tensor, value_tensor}, /*dim=*/0);

     return {state_tensor, target_tensor};
}
// --- END FIX ---

torch::optional<size_t> ChaturajiDataset::size() const {
    return states_.size();
}

torch::Tensor ChaturajiDataset::policy_to_tensor(const std::map<Move, double>& policy_map) const {
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


// --- Training Function Implementation ---
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
    // Timestamp, directories, model loading, optimizer, self-play gen setup...
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss_ts;
  ss_ts << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
  std::string timestamp = ss_ts.str();
  fs::path model_dir = fs::path(model_save_dir_base) / ("run_" + timestamp);

  std::ofstream log_file; // Declare log_file here
  fs::path log_file_path = model_dir / "detailed_training_log.txt"; // Save in the run-specific directory

  try { 
      fs::create_directories(model_dir); 
      std::cout << "Model directory created: " << model_dir << std::endl; 
  }
  catch (const std::exception& e) { 
      std::cerr << "Error creating model directory: " << e.what() << std::endl; 
      // Log file cannot be opened yet if directory creation fails
      return; 
  }

  log_file.open(log_file_path.string(), std::ios_base::app); // Open in append mode

  if (!log_file.is_open()) {
      std::cerr << "Warning: Could not open log file: " << log_file_path << std::endl;
  } else {
      log_file << "Log file opened successfully: " << log_file_path << std::endl;
  }

  if (log_file.is_open()) { log_file << "Using device: " << device << std::endl; }
  if (log_file.is_open()) { log_file << "Model directory created: " << model_dir << std::endl; }


  ChaturajiNN network;
  network->to(device); // Move network to device *before* passing to SelfPlay/Evaluator
  if (!initial_model_path.empty() && fs::exists(initial_model_path)) {
      try { 
          torch::load(network, initial_model_path, device); 
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


  // --- Instantiate SelfPlay with worker count and evaluator batch size ---
  SelfPlay self_play_generator(
    network, device,
    num_workers,            // Pass worker count
    simulations_per_move,
    1250000,                 // Default buffer size
    nn_batch_size,          // Pass NN evaluator batch size
    worker_batch_size,      // Pass worker batch size
    2.5,                    // Default c_puct
    25,                      // Default temp decay move
    0.45,                   // Default dirichlet alpha
    0.25                    // Default dirichlet epsilon
    );
  
  // Mersenne Twister for random sampling indices from buffer
  std::mt19937 buffer_rng(std::random_device{}());

  // --- Training Loop ---
  for (int iteration = 0; iteration < num_iterations; ++iteration) {
      bool perform_training_this_iteration = true; // Flag to control training execution
      std::cout << "\n---------- ITERATION " << (iteration + 1) << "/" << num_iterations << " ----------" << std::endl;
      if (log_file.is_open()) { log_file << "\n---------- ITERATION " << (iteration + 1) << "/" << num_iterations << " ----------" << std::endl; }
      
      // --- Self-Play Phase ---
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
      // Generate data using workers and append to buffer
      // But also capture the number of generated steps
      size_t num_generated_data_points = self_play_generator.generate_data(num_games_per_iteration); 
      auto end_selfplay = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> selfplay_duration = end_selfplay - start_selfplay;
      std::cout << "Self-play generation finished in " << selfplay_duration.count() << " seconds. Generated " << num_generated_data_points << " data points." << std::endl;
      if (log_file.is_open()) { log_file << "Self-play generation finished in " << selfplay_duration.count() << " seconds. Generated " << num_generated_data_points << " data points." << std::endl; }
      
      const ReplayBuffer& replay_buffer = self_play_generator.get_buffer();
      size_t current_buffer_size = replay_buffer.size();
      if (log_file.is_open()) { log_file << "Current replay buffer size: " << current_buffer_size << std::endl; }

      // --- Calculate dynamic steps_per_iteration ---
      int dynamic_steps_per_iteration = 0; // Default to 0 steps
      if (num_generated_data_points > 0 && training_batch_size > 0) {
          // Use the passed-in target_sampling_rate_param
          double calculated_steps = static_cast<double>(num_generated_data_points) * target_sampling_rate_param / static_cast<double>(training_batch_size);
          dynamic_steps_per_iteration = static_cast<int>(std::round(calculated_steps));
          
          // If rounding resulted in 0 steps, but some data was generated, ensure at least 1 step
          // to utilize the new data, provided the total buffer can form a batch.
          if (dynamic_steps_per_iteration == 0) {
              dynamic_steps_per_iteration = 1; 
          }
      }
      // If no new data was generated (num_generated_data_points == 0), dynamic_steps_per_iteration remains 0.


      // --- Condition for skipping training ---
      // Skip if:
      // 1. No training steps are to be performed (e.g., no new data and dynamic_steps_per_iteration is 0).
      // 2. OR, the entire replay buffer is too small to form even one training batch.
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
      
      // Create a vector from the replay buffer for efficient random access during sampling.
      // This copy happens once per training phase.
      std::vector<GameDataStep> training_data_vector(replay_buffer.begin(), replay_buffer.end());
            // Safety check: if training_data_vector is empty but steps were scheduled.
      // This should ideally be caught by the current_buffer_size check above.
      if (training_data_vector.empty() && dynamic_steps_per_iteration > 0) {
          std::cerr << "Error: Scheduled training steps (" << dynamic_steps_per_iteration 
                    << ") but training_data_vector is empty. Buffer size reported as: " << current_buffer_size 
                    << ". Skipping training." << std::endl;
          if (log_file.is_open()) {
              log_file << "Error: Scheduled training steps (" << dynamic_steps_per_iteration 
                       << ") but training_data_vector is empty. Buffer size reported as: " << current_buffer_size 
                       << ". Skipping training." << std::endl;
          }
          perform_training_this_iteration = false; // Update flag if this specific error occurs
      }
      if (perform_training_this_iteration) { // Re-check flag after the potential error above
        // --- Training Phase ---
        // std::cout << "Starting training for " << dynamic_steps_per_iteration << " steps (Batch Size: " << training_batch_size << ")..." << std::endl;
        network->train(); // Set network to training mode

        double total_loss = 0.0;
        double total_policy_loss = 0.0;
        double total_value_loss = 0.0;
        int steps_performed_this_iter = 0;
        auto train_start_time = std::chrono::high_resolution_clock::now();
        // Use dynamic_steps_per_iteration in the loop
        for (int step = 0; step < dynamic_steps_per_iteration; ++step) {
            // --- Manual Batch Sampling ---
            std::vector<torch::Tensor> state_batch_vec;
            std::vector<torch::Tensor> target_batch_vec; // Combined policy+value
            state_batch_vec.reserve(training_batch_size);
            target_batch_vec.reserve(training_batch_size);

            // Create distribution for sampling indices
            std::uniform_int_distribution<size_t> dist(0, training_data_vector.size() - 1);

            for (int i = 0; i < training_batch_size; ++i) {
                // Sample a random index
                size_t sample_idx = dist(buffer_rng);

                // Access the data point from the vector (Efficient O(1) access)
                const GameDataStep& data_step = training_data_vector[sample_idx];

                // Convert board, policy, value to tensors (on CPU initially is fine)
                // Ensure board_to_tensor returns [C,H,W] directly or squeeze batch dim
                torch::Tensor state_t = board_to_tensor(std::get<0>(data_step), torch::kCPU).squeeze(0);
                torch::Tensor policy_t = policy_to_tensor_static(std::get<1>(data_step));
                torch::Tensor value_t = torch::tensor({std::get<3>(data_step)}, torch::kFloat32); // Shape [1]
                torch::Tensor target_t = torch::cat({policy_t, value_t}, /*dim=*/0); // Shape [4097]

                state_batch_vec.push_back(state_t);
                target_batch_vec.push_back(target_t);
            }

            // Stack the collected tensors into batches and move to device
            auto states = torch::stack(state_batch_vec, 0).to(device);
            auto targets = torch::stack(target_batch_vec, 0).to(device);
            // --- End Manual Batch Sampling ---

            // --- Training Step ---
            auto policy_target = targets.slice(/*dim=*/1, /*start=*/0, /*end=*/4096);
            auto value_target = targets.slice(/*dim=*/1, /*start=*/4096, /*end=*/4097);

            optimizer.zero_grad();

            torch::Tensor policy_pred, value_pred;
            std::tie(policy_pred, value_pred) = network->forward(states); // Standard forward pass

            auto policy_log_softmax = torch::log_softmax(policy_pred, /*dim=*/1);
            auto policy_loss = -torch::sum(policy_target * policy_log_softmax, /*dim=*/1).mean();
            auto value_loss = torch::mse_loss(value_pred, value_target);
            auto loss = policy_loss + value_loss;

            loss.backward();
            optimizer.step();

            // --- PRINT LOSS FOR CURRENT STEP ---
            if ((step + 1) % 1 == 0) { // Print every step, or change 1 to N to print every N steps
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
            // --- END PRINT LOSS FOR CURRENT STEP ---

            total_loss += loss.item<double>();
            total_policy_loss += policy_loss.item<double>();
            total_value_loss += value_loss.item<double>();
            steps_performed_this_iter++;
            // --- End Training Step ---

        } // End step loop

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

        network->eval(); // Set back to eval mode
      }
    }
      // --- Save Model Checkpoint ---
       if ((iteration + 1) % 1 == 0) { // Save every iteration for now
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

  } // End iteration loop

  std::cout << "\nTraining finished." << std::endl;
  if (log_file.is_open()) {
      log_file << "\nTraining finished." << std::endl;
      log_file.close();
      std::cout << "Detailed log saved to: " << log_file_path << std::endl; // This cout is fine as it's a final status to the user
  } else {
      std::cout << "Detailed log was not saved as the file could not be opened: " << log_file_path << std::endl;
  }
}

// Helper function
inline torch::Tensor policy_to_tensor_static(const std::map<Move, double>& policy_map) {
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

} // namespace chaturaji_cpp