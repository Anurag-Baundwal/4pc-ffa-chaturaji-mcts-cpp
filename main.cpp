#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <torch/torch.h>
#include <chrono> // For timing
#include <algorithm> // For std::find 
#include <memory> // For std::shared_ptr


#include "board.h"
#include "types.h"
#include "utils.h"
#include "model.h"
#include "search.h" // Includes sync MCTS function now
#include "train.h" // Include train function header
#include "strength_test.h" // Include strength test function header
#include "mcts_node.h" // For MCTSNode type

namespace fs = std::filesystem;

// Basic command-line argument parsing helper (replace with a proper library if needed)
std::string get_cmd_option(char** begin, char** end, const std::string& option) {
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return "";
}

bool cmd_option_exists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}


int main(int argc, char* argv[]) {
    // --- Mode Selection ---
    if (cmd_option_exists(argv, argv + argc, "--train")) {
        // --- Training Mode ---
        std::cout << "--- Starting Training Mode ---" << std::endl;

        // Get training parameters from command line or use defaults
        int iterations = 65536; 
        int games_per_iter = 128;
        double target_sampling_rate = 1.5; // ADD: Default
        int training_batch_size = 1024;
        int sims_per_move = 128;
        int num_workers = 12;   
        int nn_batch_size = 1024;
        int worker_batch_size = 48;
        std::string save_dir = "/content/drive/MyDrive/models";
        std::string load_path = "";

        std::string temp_str;
        temp_str = get_cmd_option(argv, argv+argc, "--iterations");
        if (!temp_str.empty()) iterations = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--games-per-iter");
        if (!temp_str.empty()) games_per_iter = std::stoi(temp_str);

        temp_str = get_cmd_option(argv, argv+argc, "--target-sampling-rate");
        if (!temp_str.empty()) target_sampling_rate = std::stod(temp_str); // Use stod for double
        temp_str = get_cmd_option(argv, argv+argc, "--train-batch"); // Argument for training data loader
        if (!temp_str.empty()) training_batch_size = std::stoi(temp_str);
         temp_str = get_cmd_option(argv, argv+argc, "--sims");
        if (!temp_str.empty()) sims_per_move = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--workers"); 
        if (!temp_str.empty()) num_workers = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--nn-batch");
        if (!temp_str.empty()) nn_batch_size = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--worker-batch");
        if (!temp_str.empty()) worker_batch_size = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--save-dir");
        if (!temp_str.empty()) save_dir = temp_str;
         temp_str = get_cmd_option(argv, argv+argc, "--load-model");
        if (!temp_str.empty()) load_path = temp_str;

        std::cout << "Parameters:" << std::endl;
        std::cout << "  Iterations:           " << iterations << std::endl;
        std::cout << "  Games/Iter:           " << games_per_iter << std::endl;
        std::cout << "  Target Sampling Rate: " << std::fixed << std::setprecision(2) << target_sampling_rate << std::endl;
        std::cout << "  Training Batch:       " << training_batch_size << std::endl;
        std::cout << "  Sims/Move:            " << sims_per_move << std::endl;
        std::cout << "  Workers:              " << num_workers << std::endl; // <-- Print new param
        std::cout << "  NN Eval Batch:        " << nn_batch_size << std::endl; // <-- Print new param
        std::cout << "  Worker MCTS Batch:    " << worker_batch_size << std::endl; // <-- Print new param
        std::cout << "  Save Dir:             " << save_dir << std::endl;
        std::cout << "  Load Model:           " << (load_path.empty() ? "None" : load_path) << std::endl;

        try {
            chaturaji_cpp::train(
                iterations, games_per_iter, target_sampling_rate,
                training_batch_size, // Training dataloader batch size
                num_workers,         // Number of self-play workers
                nn_batch_size,       // Evaluator NN batch size
                worker_batch_size,   // Pass worker batch size
                0.001, 1e-4,         // Default LR and weight decay
                sims_per_move,
                save_dir, load_path
            );
        } catch (const std::exception& e) {
            std::cerr << "Training failed with exception: " << e.what() << std::endl;
            return 1;
        }

    } else if (cmd_option_exists(argv, argv + argc, "--strength-test")) {
        // --- Strength Test Mode ---
        std::cout << "--- Entering Strength Test Mode ---" << std::endl;

        std::string new_model_path = "";
        std::string old_model_path = "";
        int games = 100;
        int sims = 250;
        int mcts_batch = 64;

        std::string temp_str;
        new_model_path = get_cmd_option(argv, argv + argc, "--new-model");
        old_model_path = get_cmd_option(argv, argv + argc, "--old-model");
        temp_str = get_cmd_option(argv, argv + argc, "--games");
        if (!temp_str.empty()) games = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv + argc, "--sims");
        if (!temp_str.empty()) sims = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv + argc, "--mcts-batch");
        if (!temp_str.empty()) mcts_batch = std::stoi(temp_str);

        if (new_model_path.empty()) { // Only --new-model is strictly required
            std::cerr << "Error: --new-model path must be provided for strength test." << std::endl;
            return 1;
        }
        // old_model_path can be empty; run_strength_test will handle it by creating a random model.

        try {
            chaturaji_cpp::run_strength_test(
                new_model_path, old_model_path, games, sims, mcts_batch
            );
        } catch (const std::exception& e) {
            std::cerr << "Strength test failed with exception: " << e.what() << std::endl;
            return 1;
        }

    } else {
        // --- Inference/Analysis Mode (Uses Synchronous MCTS) ---
        std::cout << "--- Starting Inference Mode ---" << std::endl;

        std::string model_path = "model.pt"; // Default model path
        int simulations = 1000; // Default simulations
        int mcts_sync_batch_size = 16; // Batch size for synchronous MCTS NN calls

        std::string temp_str;
         temp_str = get_cmd_option(argv, argv+argc, "--model");
        if (!temp_str.empty()) model_path = temp_str;
        temp_str = get_cmd_option(argv, argv+argc, "--sims");
        if (!temp_str.empty()) simulations = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--mcts-batch"); // Argument for sync MCTS batching
        if (!temp_str.empty()) mcts_sync_batch_size = std::stoi(temp_str);

         std::cout << "Parameters:" << std::endl;
         std::cout << "  Model Path:        " << model_path << std::endl;
         std::cout << "  Simulations:       " << simulations << std::endl;
         std::cout << "  MCTS Sync Batch:   " << mcts_sync_batch_size << std::endl;

         torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
         std::cout << "Using device: " << device << std::endl;

         // Load the model
         chaturaji_cpp::ChaturajiNN network;
         if (!fs::exists(model_path)) {
             std::cerr << "Error: Model file not found at " << model_path << std::endl;
             return 1;
         }
         try {
             torch::load(network, model_path, device);
             network->to(device);
             network->eval(); // Set to evaluation mode
             std::cout << "Model loaded successfully from " << model_path << std::endl;
         } catch (const c10::Error& e) {
             std::cerr << "Error loading model: " << e.what() << std::endl;
             return 1;
         }

        // Create initial board
        chaturaji_cpp::Board board;
        double total_execution_time = 0.0;
        int num_searches = 0;
        std::shared_ptr<chaturaji_cpp::MCTSNode> mcts_root_node_main = nullptr; // For tree reuse

        // Game loop (example: play N moves)
        int max_moves_to_play = 100;
        for (int i = 0; i < max_moves_to_play; ++i) {
            std::cout << "\nMove " << i + 1 << std::endl;
            std::cout << "Board state: " << std::endl;

            board.print_board();

            if (board.is_game_over()) {
                 std::cout << "Game Over!" << std::endl;
                 if(board.get_termination_reason()) {
                     std::cout << "Reason: " << *board.get_termination_reason() << std::endl;
                 }
                 mcts_root_node_main = nullptr; // Reset tree
                 break;
            }

             std::cout << "Searching for best move (Sync MCTS: " << simulations << " sims, batch " << mcts_sync_batch_size << ")..." << std::endl;
             auto start_time = std::chrono::high_resolution_clock::now();

             // Get best move using SYNCHRONOUS MCTS
             std::optional<chaturaji_cpp::Move> best_move_opt = chaturaji_cpp::get_best_move_mcts_sync( // Use sync function
              board, network, simulations, device,
              mcts_root_node_main, // Pass shared_ptr for tree reuse
              1.0, // Default c_puct
              mcts_sync_batch_size // Pass sync MCTS batch size
              );

             auto end_time = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double> execution_time = end_time - start_time;
             total_execution_time += execution_time.count();
             num_searches++;

             std::cout << "Search completed in " << execution_time.count() << " seconds." << std::endl;
             if (num_searches > 0) {
                 std::cout << "Average search time: " << (total_execution_time / num_searches) << " seconds." << std::endl;
             }


             if (best_move_opt) {
                 chaturaji_cpp::Move best_move = *best_move_opt;
                 std::cout << "Best move found: " << chaturaji_cpp::get_uci_string(best_move)
                           << " (SAN: " << chaturaji_cpp::get_san_string(best_move, board) << ")" << std::endl;

                 // Make the move on the board
                 board.make_move(best_move);
             } else {
                 std::cout << "No valid moves found for player " << static_cast<int>(board.get_current_player()) << "." << std::endl;
                 mcts_root_node_main = nullptr; // Reset tree
                 if (!board.is_game_over()) {
                      std::cout << "Player " << static_cast<int>(board.get_current_player()) << " resigns." << std::endl;
                      board.resign();
                 }
             }
             if (board.is_game_over()){
                 mcts_root_node_main = nullptr; // Reset tree if game ended after move
             }
        }

        // Print final state
        std::cout << "\n--- Final Board State ---" << std::endl;
        board.print_board();
        auto final_scores = board.get_game_result();
        std::cout << "Final Scores:" << std::endl;
        for(const auto& pair : final_scores) {
            std::cout << "  Player " << static_cast<int>(pair.first) << ": " << pair.second << std::endl;
        }
        auto winner = board.get_winner();
        if(winner) {
             std::cout << "Winner: Player " << static_cast<int>(*winner) << std::endl;
        } else if (board.is_game_over()) {
            std::cout << "Game is a draw or ended inconclusively." << std::endl;
        }

    } // End mode selection

    return 0;
}