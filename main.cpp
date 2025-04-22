#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <torch/torch.h>
#include <chrono> // For timing


#include "board.h"
#include "types.h"
#include "utils.h"
#include "model.h"
#include "search.h"
#include "train.h" // Include train function header

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
        int iterations = 50;
        int games_per_iter = 50;
        int epochs_per_iter = 25;
        int batch_size = 4096;
        int sims_per_move = 50;
        std::string save_dir = "/content/drive/MyDrive/models"; // Default from Colab
        std::string load_path = "";

        std::string temp_str;
        temp_str = get_cmd_option(argv, argv+argc, "--iterations");
        if (!temp_str.empty()) iterations = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--games");
        if (!temp_str.empty()) games_per_iter = std::stoi(temp_str);
         temp_str = get_cmd_option(argv, argv+argc, "--epochs");
        if (!temp_str.empty()) epochs_per_iter = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--batch");
        if (!temp_str.empty()) batch_size = std::stoi(temp_str);
         temp_str = get_cmd_option(argv, argv+argc, "--sims");
        if (!temp_str.empty()) sims_per_move = std::stoi(temp_str);
        temp_str = get_cmd_option(argv, argv+argc, "--save-dir");
        if (!temp_str.empty()) save_dir = temp_str;
         temp_str = get_cmd_option(argv, argv+argc, "--load-model");
        if (!temp_str.empty()) load_path = temp_str;


        std::cout << "Parameters:" << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Games/Iter: " << games_per_iter << std::endl;
        std::cout << "  Epochs/Iter: " << epochs_per_iter << std::endl;
        std::cout << "  Batch Size: " << batch_size << std::endl;
        std::cout << "  Sims/Move: " << sims_per_move << std::endl;
        std::cout << "  Save Dir: " << save_dir << std::endl;
        std::cout << "  Load Model: " << (load_path.empty() ? "None" : load_path) << std::endl;

        try {
            chaturaji_cpp::train(
                iterations, games_per_iter, epochs_per_iter, batch_size,
                0.001, 1e-4, // Default LR and weight decay
                sims_per_move, save_dir, load_path
            );
        } catch (const std::exception& e) {
            std::cerr << "Training failed with exception: " << e.what() << std::endl;
            return 1;
        }

    } else {
        // --- Inference/Analysis Mode (like chaturaji_engine.py) ---
        std::cout << "--- Starting Inference Mode ---" << std::endl;

        std::string model_path = "model.pt"; // Default model path
        int simulations = 1000; // Default simulations

        std::string temp_str;
         temp_str = get_cmd_option(argv, argv+argc, "--model");
        if (!temp_str.empty()) model_path = temp_str;
        temp_str = get_cmd_option(argv, argv+argc, "--sims");
        if (!temp_str.empty()) simulations = std::stoi(temp_str);

         std::cout << "Parameters:" << std::endl;
         std::cout << "  Model Path: " << model_path << std::endl;
         std::cout << "  Simulations: " << simulations << std::endl;

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
                 break;
            }

             std::cout << "Searching for best move (" << simulations << " sims)..." << std::endl;
             auto start_time = std::chrono::high_resolution_clock::now();

             // Get best move using MCTS
             std::optional<chaturaji_cpp::Move> best_move_opt = chaturaji_cpp::get_best_move_mcts(
                 board, network, simulations, device
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
                 // In a real game, this might trigger resignation or indicate stalemate/checkmate
                 if (!board.is_game_over()) {
                      std::cout << "Player " << static_cast<int>(board.get_current_player()) << " resigns." << std::endl;
                      board.resign();
                 }
             }
        }

        // Print final state
        std::cout << "\n--- Final Board State ---" << std::endl;
        board.print_board();
        auto final_scores = board.get_game_result(); // Get scores after potential draw bonus
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