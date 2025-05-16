// strength_test.cpp
#include "strength_test.h"
#include "board.h"
#include "model.h"
#include "search.h" // For get_best_move_mcts_sync, get_reward_map (implicitly needed for ranking)
#include "types.h"
#include "utils.h"     // For move string conversion
#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::sort, std::find_if
#include <iomanip>   // For std::fixed, std::setprecision
#include <map>       // For std::map

namespace fs = std::filesystem;

namespace chaturaji_cpp {

namespace { // Anonymous namespace for helper functions

// Helper to convert player enum to string
std::string player_to_string(Player p) {
    switch (p) {
        case Player::RED: return "RED";
        case Player::BLUE: return "BLUE";
        case Player::YELLOW: return "YELLOW";
        case Player::GREEN: return "GREEN";
        default: return "UNKNOWN";
    }
}

} // end anonymous namespace


void run_strength_test(
    const std::string& new_model_path,
    const std::string& old_model_path, // Can be empty
    int num_games,
    int simulations_per_move,
    int mcts_batch_size
) {
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << device << std::endl;
    std::cout << "--- Starting Strength Test Mode ---" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  New Model Path:    " << new_model_path << std::endl;
    if (old_model_path.empty()) {
        std::cout << "  Old Model Path:    Not specified (using random)" << std::endl;
    } else {
        std::cout << "  Old Model Path:    " << old_model_path << std::endl;
    }
    std::cout << "  Games to Play:     " << num_games << std::endl;
    std::cout << "  Simulations/Move:  " << simulations_per_move << std::endl;
    std::cout << "  MCTS Batch Size:   " << mcts_batch_size << std::endl;
    std::cout << "  New Model Player:  Cycling (RED->BLUE->YELLOW->GREEN->...)" << std::endl;

    // --- Load Models ---
    ChaturajiNN new_network;
    ChaturajiNN old_network;

    // Load New Model
    if (!fs::exists(new_model_path)) {
        std::cerr << "Error: New model file not found at " << new_model_path << std::endl;
        return;
    }
    try {
        torch::load(new_network, new_model_path, device);
        new_network->to(device);
        new_network->eval(); // Set to evaluation mode
        std::cout << "New model loaded successfully." << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading new model: " << e.what() << std::endl;
        return;
    }

    // Load Old Model or Initialize Random
    if (old_model_path.empty()) {
        std::cout << "Old model path not specified. Using a randomly initialized model for the old model." << std::endl;
        old_network = ChaturajiNN(); // Default constructor creates a new, untrained network
        old_network->to(device);
        old_network->eval();
        std::cout << "Randomly initialized old model created successfully." << std::endl;
    } else {
        if (!fs::exists(old_model_path)) {
            std::cerr << "Error: Old model file not found at " << old_model_path << std::endl;
            return;
        }
        try {
            torch::load(old_network, old_model_path, device);
            old_network->to(device);
            old_network->eval(); // Set to evaluation mode
            std::cout << "Old model loaded successfully from " << old_model_path << "." << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading old model from path: " << e.what() << std::endl;
            return;
        }
    }

    // --- Simulation Loop ---
    double total_game_time = 0.0;
    std::map<int, int> new_model_rank_counts; // Key: rank (1-4), Value: count
    for (int i = 1; i <= 4; ++i) {
        new_model_rank_counts[i] = 0;
    }

    for (int game_idx = 0; game_idx < num_games; ++game_idx) {
        auto game_start_time = std::chrono::high_resolution_clock::now();
        Board board; // Start a fresh game

        // Determine which player gets the new model for this game
        Player new_model_player = static_cast<Player>(game_idx % 4);
        // std::cout << "Game " << game_idx + 1 << ": New model plays as " << player_to_string(new_model_player) << std::endl; // Optional detailed log
        
        int move_count = 0;
        while (!board.is_game_over()) {
            Player current_player = board.get_current_player();
            ChaturajiNN& current_network = (current_player == new_model_player) ? new_network : old_network;

            // Get best move using SYNCHRONOUS MCTS
            std::optional<Move> best_move_opt = get_best_move_mcts_sync(
                board, current_network, simulations_per_move, device,
                1.0, // Default c_puct
                mcts_batch_size
            );

            if (best_move_opt) {
                board.make_move(*best_move_opt);
            } else {
                // No valid moves found, the player might be stalemated or checkmated implicitly
                 // (though Chaturaji doesn't have checkmate in the same way).
                 // Or MCTS failed. Assume player must pass/resign if possible.
                if (!board.is_game_over() && board.get_active_players().count(current_player)) {
                    // std::cout << "Info: No move found for player " << player_to_string(current_player) << ". Resigning." << std::endl;
                    board.resign(); // Player resigns if no move is found by MCTS
                }
                // If already game over or player inactive, the loop condition will handle it.
            }
            move_count++;
            // Optional: Print board state every move for debugging
            // if (game_idx < 2) { // Only for first few games
            //     std::cout << "Move " << move_count << " (" << player_to_string(current_player) << "): "
            //               << (best_move_opt ? get_uci_string(*best_move_opt) : "resign") << std::endl;
            //     board.print_board();
            // }
        } // End single game loop

        auto game_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> game_duration = game_end_time - game_start_time;
        total_game_time += game_duration.count();

        // --- Analyze Result for Ranking ---
        Board::PlayerPointMap final_scores = board.get_game_result();
        std::vector<std::pair<Player, int>> sorted_scores_vec;
        // Ensure all players are in the vector for consistent sorting and ranking
        for (int p_idx = 0; p_idx < 4; ++p_idx) {
            Player p = static_cast<Player>(p_idx);
            int score = 0; // Default score
            auto it = final_scores.find(p);
            if (it != final_scores.end()) {
                score = it->second;
            }
            sorted_scores_vec.push_back({p, score});
        }

        std::sort(sorted_scores_vec.begin(), sorted_scores_vec.end(), [](const auto& a, const auto& b) {
            return a.second > b.second; // Sort by score descending
        });

        int current_new_model_rank = -1;
        for (size_t rank_idx = 0; rank_idx < sorted_scores_vec.size(); ++rank_idx) {
            if (sorted_scores_vec[rank_idx].first == new_model_player) {
                current_new_model_rank = rank_idx + 1;
                break;
            }
        }

        if (current_new_model_rank != -1 && current_new_model_rank >= 1 && current_new_model_rank <= 4) {
            new_model_rank_counts[current_new_model_rank]++;
        } else {
            std::cerr << "Warning: Could not determine rank for new model in game " << game_idx + 1 << std::endl;
        }
        
        // --- Print Progress ---
        // Update to use new_model_rank_counts[1] for "wins"
        if ((game_idx + 1) % 1 == 0 || (game_idx + 1) == num_games) {
            double avg_game_time = total_game_time / (game_idx + 1);
             std::cout << "Progress: Game " << std::setw(3) << (game_idx + 1) << "/" << num_games << " completed."
                       << " New Model (" << player_to_string(new_model_player) << ") got rank: " << current_new_model_rank << "."
                       << " Last game duration: " << std::fixed << std::setprecision(2) << game_duration.count() << "s."
                       << " New Model First Place Finishes: " << new_model_rank_counts[1] << "." // Use map for 1st place count
                       << " Avg game time: " << std::fixed << std::setprecision(2) << avg_game_time << "s." << std::endl;
        }

    } // End all games loop

    // --- Final Report ---
    std::cout << "\n--- Strength Test Finished ---" << std::endl;
    double first_place_percentage = (num_games > 0 && new_model_rank_counts.count(1)) ? (static_cast<double>(new_model_rank_counts.at(1)) / num_games * 100.0) : 0.0;
    std::cout << "New Model First Places: " << (new_model_rank_counts.count(1) ? new_model_rank_counts.at(1) : 0) << "/" << num_games
              << " (" << std::fixed << std::setprecision(2) << first_place_percentage << "%)" << std::endl;

    std::cout << "New Model Standings Distribution:" << std::endl;
    for (int rank = 1; rank <= 4; ++rank) {
        int count = new_model_rank_counts.count(rank) ? new_model_rank_counts.at(rank) : 0;
        double rank_percentage = (num_games > 0) ? (static_cast<double>(count) / num_games * 100.0) : 0.0;
        std::cout << "  Rank " << rank << ": " << std::setw(4) << count << " times ("
                  << std::fixed << std::setprecision(2) << rank_percentage << "%)" << std::endl;
    }

    if (num_games > 0) {
        std::cout << "Average Game Time: " << std::fixed << std::setprecision(2) << (total_game_time / num_games) << " seconds." << std::endl;
    }
}

} // namespace chaturaji_cpp