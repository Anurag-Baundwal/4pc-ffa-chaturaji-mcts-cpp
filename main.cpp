#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <chrono> 
#include <algorithm> 
#include <memory> 
#include <iomanip>

#include "board.h"
#include "types.h"
#include "utils.h"
#include "model.h" // Now includes the ONNX Model class
#include "search.h" 
#include "train.h" 
#include "strength_test.h" 
#include "mcts_node.h" 

namespace fs = std::filesystem;

// Helper to parse args
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

        // --- Default Training Parameters ---
        // These serve as the primary defaults when running via the command line.
        // Because they are passed as explicit arguments to chaturaji_cpp::train(),
        // they effectively take precedence over the default values defined in train.h.
        int iterations = 65536; 
        int games_per_iter = 128;
        double target_sampling_rate = 1.5; 
        int training_batch_size = 1024;
        int sims_per_move = 128;
        int num_workers = 12;   
        int nn_batch_size = 1024;
        int worker_batch_size = 48;
        double learning_rate = 0.001;
        double weight_decay = 0.01;
        int max_buffer_size = 200000;
        int temp_decay_move = 20;
        double d_alpha = 0.4;
        double d_epsilon = 0.25;
        std::string save_dir = "models";
        std::string load_path = "";

        std::string temp_str;
        
        temp_str = get_cmd_option(argv, argv+argc, "--iterations");
        if (!temp_str.empty()) iterations = std::stoi(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--games-per-iter");
        if (!temp_str.empty()) games_per_iter = std::stoi(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--target-sampling-rate");
        if (!temp_str.empty()) target_sampling_rate = std::stod(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--train-batch"); 
        if (!temp_str.empty()) training_batch_size = std::stoi(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--sims");
        if (!temp_str.empty()) sims_per_move = std::stoi(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--workers"); 
        if (!temp_str.empty()) num_workers = std::stoi(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--nn-batch");
        if (!temp_str.empty()) nn_batch_size = std::stoi(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--worker-batch");
        if (!temp_str.empty()) worker_batch_size = std::stoi(temp_str);

        temp_str = get_cmd_option(argv, argv+argc, "--lr");
        if (!temp_str.empty()) learning_rate = std::stod(temp_str);

        temp_str = get_cmd_option(argv, argv+argc, "--wd");
        if (!temp_str.empty()) weight_decay = std::stod(temp_str);

        temp_str = get_cmd_option(argv, argv + argc, "--max-buffer-size");
        if (!temp_str.empty()) max_buffer_size = std::stoi(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--temp-decay-move");
        if (!temp_str.empty()) temp_decay_move = std::stoi(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--dirichlet-alpha");
        if (!temp_str.empty()) d_alpha = std::stod(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--dirichlet-epsilon");
        if (!temp_str.empty()) d_epsilon = std::stod(temp_str);
        
        temp_str = get_cmd_option(argv, argv+argc, "--save-dir");
        if (!temp_str.empty()) save_dir = temp_str;
        
        temp_str = get_cmd_option(argv, argv+argc, "--load-model");
        if (!temp_str.empty()) load_path = temp_str;

        try {
            chaturaji_cpp::train(
                iterations,
                games_per_iter,
                target_sampling_rate,
                training_batch_size,
                num_workers,
                nn_batch_size,
                worker_batch_size,
                learning_rate,
                weight_decay,
                sims_per_move,
                max_buffer_size,
                temp_decay_move,
                d_alpha,
                d_epsilon,
                save_dir,
                load_path
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

        if (new_model_path.empty()) { 
            std::cerr << "Error: --new-model path must be provided for strength test." << std::endl;
            return 1;
        }

        try {
            chaturaji_cpp::run_strength_test(
                new_model_path, old_model_path, games, sims, mcts_batch
            );
        } catch (const std::exception& e) {
            std::cerr << "Strength test failed with exception: " << e.what() << std::endl;
            return 1;
        }

    } else {
        // --- Inference/Analysis Mode ---
        std::cout << "--- Starting Inference Mode ---" << std::endl;

        std::string model_path = "model.onnx"; // Default to ONNX
        int simulations = 1000; 
        int mcts_sync_batch_size = 16; 

        std::string temp_str;
        temp_str = get_cmd_option(argv, argv+argc, "--model");
        if (!temp_str.empty()) model_path = temp_str;
        temp_str = get_cmd_option(argv, argv+argc, "--sims");
        if (!temp_str.empty()) simulations = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--mcts-batch"); 
        if (!temp_str.empty()) mcts_sync_batch_size = std::stoi(temp_str);

        std::cout << "Parameters:" << std::endl;
        std::cout << "  Model Path:        " << model_path << std::endl;
        std::cout << "  Simulations:       " << simulations << std::endl;
        std::cout << "  MCTS Sync Batch:   " << mcts_sync_batch_size << std::endl;

        // Load the ONNX model
        if (!fs::exists(model_path)) {
            std::cerr << "Error: Model file not found at " << model_path << std::endl;
            return 1;
        }
        
        std::unique_ptr<chaturaji_cpp::Model> network;
        try {
            network = std::make_unique<chaturaji_cpp::Model>(model_path);
            std::cout << "ONNX Model loaded successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return 1;
        }

        chaturaji_cpp::Board board;
        double total_execution_time = 0.0;
        int num_searches = 0;
        std::shared_ptr<chaturaji_cpp::MCTSNode> mcts_root_node_main = nullptr; 

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
                 mcts_root_node_main = nullptr; 
                 break;
            }

             std::cout << "Searching for best move (Sims: " << simulations << ")..." << std::endl;
             auto start_time = std::chrono::high_resolution_clock::now();

             // get_best_move_mcts_sync no longer takes torch::Device
             std::optional<chaturaji_cpp::Move> best_move_opt = chaturaji_cpp::get_best_move_mcts_sync( 
                  board, network.get(), simulations,
                  mcts_root_node_main, 
                  2.5, 
                  mcts_sync_batch_size 
              );

             auto end_time = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double> execution_time = end_time - start_time;
             total_execution_time += execution_time.count();
             num_searches++;

             std::cout << "Search completed in " << execution_time.count() << " seconds." << std::endl;

             if (best_move_opt) {
                 chaturaji_cpp::Move best_move = *best_move_opt;
                 std::cout << "Best move found: " << chaturaji_cpp::get_uci_string(best_move)
                           << " (SAN: " << chaturaji_cpp::get_san_string(best_move, board) << ")" << std::endl;
                 board.make_move(best_move);
             } else {
                 std::cout << "No valid moves found. Resigning." << std::endl;
                 mcts_root_node_main = nullptr; 
                 board.resign();
             }
        }
    } 

    return 0;
}