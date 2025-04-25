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

namespace fs = std::filesystem;

namespace chaturaji_cpp {


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
     torch::Tensor state_tensor = states_[index];

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
// (Setup remains the same)
void train(
    int num_iterations,
    int num_games_per_iteration,
    int num_epochs_per_iteration,
    int batch_size,
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
    try { fs::create_directories(model_dir); std::cout << "Model directory created: " << model_dir << std::endl; }
    catch (const std::exception& e) { std::cerr << "Error creating model directory: " << e.what() << std::endl; return; }
    ChaturajiNN network;
    network->to(device);
    if (!initial_model_path.empty() && fs::exists(initial_model_path)) {
        try { torch::load(network, initial_model_path, device); std::cout << "Loaded initial model from: " << initial_model_path << std::endl; }
        catch (const c10::Error& e) { std::cerr << "Error loading initial model: " << e.what() << ". Starting from scratch." << std::endl; }
    } else { std::cout << "No initial model provided or found. Starting from scratch." << std::endl; }
    torch::optim::Adam optimizer(network->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
    SelfPlay self_play_generator(network, device, simulations_per_move);

    // --- Training Loop ---
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        // Self-play generation...
        std::cout << "\n---------- ITERATION " << (iteration + 1) << "/" << num_iterations << " ----------" << std::endl;
        std::cout << "Generating " << num_games_per_iteration << " self-play games..." << std::endl;
        auto start_selfplay = std::chrono::high_resolution_clock::now();
        self_play_generator.clear_buffer();
        for (int g = 0; g < num_games_per_iteration; ++g) {
            std::cout << " Starting game " << (g + 1) << "/" << num_games_per_iteration << "..." << std::endl;
            self_play_generator.generate_game();
        }
        auto end_selfplay = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> selfplay_duration = end_selfplay - start_selfplay;
        std::cout << "Self-play generation finished in " << selfplay_duration.count() << " seconds." << std::endl;
        const ReplayBuffer& replay_buffer = self_play_generator.get_buffer();
        if (replay_buffer.empty()) { std::cerr << "Warning: Replay buffer is empty after self-play generation. Skipping training for this iteration." << std::endl; continue; }
        std::cout << "Generated " << replay_buffer.size() << " data points." << std::endl;
        std::vector<GameDataStep> training_data(replay_buffer.begin(), replay_buffer.end());


        // --- Prepare DataLoader ---
        // Dataset now returns Example<Tensor, Tensor>, Stack works as expected.
        auto dataset = ChaturajiDataset(training_data)
                           .map(torch::data::transforms::Stack<>());

        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2)
        );


        // --- Training Epochs ---
        std::cout << "Starting training for " << num_epochs_per_iteration << " epochs..." << std::endl;
        network->train();

        // // --- Conditional AMP setup (Not avaialble in Libtorch) ---
        // #if AT_CUDA_ENABLED
        // bool use_amp_runtime = device.is_cuda() && USE_AMP_IF_AVAILABLE;
        // torch::amp::GradScaler scaler(use_amp_runtime);
        // #else // AT_CUDA_ENABLED
        // const bool use_amp_runtime = false;
        // #endif // AT_CUDA_ENABLED

        for (int epoch = 0; epoch < num_epochs_per_iteration; ++epoch) {
            double total_loss = 0.0;
            double total_policy_loss = 0.0;
            double total_value_loss = 0.0;
            int batch_count = 0;
            auto epoch_start_time = std::chrono::high_resolution_clock::now();


            // --- DataLoader yields Example<Tensor, Tensor> ---
            for (auto& batch : *data_loader) {
                // --- FIX: Access data/target members and split target ---
                auto states = batch.data.to(device);        // Shape [B, C, H, W]
                auto targets = batch.target.to(device);      // Shape [B, 4097]

                // Split the combined target tensor
                // Policy target: first 4096 elements
                auto policy_target = targets.slice(/*dim=*/1, /*start=*/0, /*end=*/4096); // Shape [B, 4096]
                // Value target: last element, ensure correct shape [B, 1] for mse_loss
                auto value_target = targets.slice(/*dim=*/1, /*start=*/4096, /*end=*/4097); // Shape [B, 1]
                // Alternative for value: auto value_target = targets.select(/*dim=*/1, /*index=*/4096).unsqueeze(1);

                optimizer.zero_grad();

                // --- Forward pass (NO AMP CONTEXT NEEDED) ---
                torch::Tensor policy_pred, value_pred;
                std::tie(policy_pred, value_pred) = network->forward(states);

                // --- Calculate losses ---
                auto policy_log_softmax = torch::log_softmax(policy_pred, /*dim=*/1);
                auto policy_loss = -torch::sum(policy_target * policy_log_softmax, /*dim=*/1).mean(); // Use policy_target
                auto value_loss = torch::mse_loss(value_pred, value_target); // Use value_target

                auto loss = policy_loss + value_loss;


                // --- Backward pass (STANDARD, NO AMP/SCALER) ---
                loss.backward();
                optimizer.step();
                // --- END STANDARD BACKWARD ---

                total_loss += loss.item<double>();
                total_policy_loss += policy_loss.item<double>();
                total_value_loss += value_loss.item<double>();
                batch_count++;

            } // End batch loop

            // Epoch summary print...
            auto epoch_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;
            if (batch_count > 0) {
                 std::cout << "  Epoch " << (epoch + 1) << " finished in " << epoch_duration.count() << "s."
                           << " Avg Loss: " << (total_loss / batch_count)
                           << " (Policy: " << (total_policy_loss / batch_count)
                           << ", Value: " << (total_value_loss / batch_count) << ")"
                           << std::endl;
             } else {
                  std::cout << "  Epoch " << (epoch + 1) << " had no batches." << std::endl;
             }

        } // End epoch loop


        // --- Save Model Checkpoint (remains the same) ---
        if ((iteration + 1) % 1 == 0) {
             fs::path save_path = model_dir / ("chaturaji_iter_" + std::to_string(iteration + 1) + ".pt");
             try {
                 torch::save(network, save_path.string());
                 std::cout << "Model saved after iteration " << (iteration + 1) << " to: " << save_path << std::endl;
             } catch (const c10::Error& e) {
                  std::cerr << "Error saving model: " << e.what() << std::endl;
             }
        }

    } // End iteration loop

    std::cout << "\nTraining finished." << std::endl;
}


} // namespace chaturaji_cpp