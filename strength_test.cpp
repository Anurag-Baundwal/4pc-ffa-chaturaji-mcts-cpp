#include "strength_test.h"
#include "board.h"
#include "model.h"
#include "search.h" 
#include "types.h"
#include "utils.h"     
#include "mcts_node.h" 
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <numeric>   
#include <algorithm> 
#include <iomanip>   
#include <map>       
#include <memory>    

namespace fs = std::filesystem;

namespace chaturaji_cpp {

namespace { 
std::string player_to_string(Player p) {
    switch (p) {
        case Player::RED: return "RED";
        case Player::BLUE: return "BLUE";
        case Player::YELLOW: return "YELLOW";
        case Player::GREEN: return "GREEN";
        default: return "UNKNOWN";
    }
}
} 

void run_strength_test(
    const std::string& new_model_path,
    const std::string& old_model_path, 
    int num_games,
    int simulations_per_move,
    int mcts_batch_size
) {
    std::cout << "--- Starting Strength Test Mode (ONNX) ---" << std::endl;
    std::cout << "  New Model Path:    " << new_model_path << std::endl;
    std::cout << "  Old Model Path:    " << old_model_path << std::endl;

    if (!fs::exists(new_model_path)) {
        std::cerr << "Error: New model file not found at " << new_model_path << std::endl;
        return;
    }
    if (!old_model_path.empty() && !fs::exists(old_model_path)) {
        std::cerr << "Error: Old model file not found at " << old_model_path << std::endl;
        return;
    }

    std::unique_ptr<Model> new_network;
    std::unique_ptr<Model> old_network;

    try {
        new_network = std::make_unique<Model>(new_model_path);
        std::cout << "New model loaded successfully." << std::endl;
        
        if (!old_model_path.empty()) {
            old_network = std::make_unique<Model>(old_model_path);
            std::cout << "Old model loaded successfully." << std::endl;
        } else {
             // In inference-only setups, "random model" handling is trickier if Model class strictly requires a file.
             // For now, we assume explicit paths are provided, or handle failure.
             std::cerr << "Error: Old model path is empty. Random initialization for ONNX Model not supported in this test harness yet." << std::endl;
             return;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception loading models: " << e.what() << std::endl;
        return;
    }

    double total_game_time = 0.0;
    std::map<int, int> new_model_rank_counts; 
    for (int i = 1; i <= 4; ++i) {
        new_model_rank_counts[i] = 0;
    }

    for (int game_idx = 0; game_idx < num_games; ++game_idx) {
        auto game_start_time = std::chrono::high_resolution_clock::now();
        Board board; 
        std::shared_ptr<MCTSNode> mcts_root_node_strength_test = nullptr; 

        Player new_model_player = static_cast<Player>(game_idx % 4);
        
        while (!board.is_game_over()) {
            Player current_player = board.get_current_player();
            Model* current_network_ptr = (current_player == new_model_player) ? new_network.get() : old_network.get();

            std::optional<Move> best_move_opt = get_best_move_mcts_sync(
                board, current_network_ptr, simulations_per_move, 
                mcts_root_node_strength_test, 
                1.0, 
                mcts_batch_size
            );

            if (best_move_opt) {
                board.make_move(*best_move_opt);
            } else {
                mcts_root_node_strength_test = nullptr; 
                if (!board.is_game_over() && board.get_active_players().count(current_player)) {
                    board.resign(); 
                }
            }
            if (board.is_game_over()) {
                mcts_root_node_strength_test = nullptr; 
            }
        } 

        auto game_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> game_duration = game_end_time - game_start_time;
        total_game_time += game_duration.count();

        Board::PlayerPointMap final_scores = board.get_game_result();
        std::vector<std::pair<Player, int>> sorted_scores_vec;
        for (int p_idx = 0; p_idx < 4; ++p_idx) {
            Player p = static_cast<Player>(p_idx);
            int score = 0; 
            auto it = final_scores.find(p);
            if (it != final_scores.end()) {
                score = it->second;
            }
            sorted_scores_vec.push_back({p, score});
        }

        std::sort(sorted_scores_vec.begin(), sorted_scores_vec.end(), [](const auto& a, const auto& b) {
            return a.second > b.second; 
        });

        int current_new_model_rank = -1;
        for (size_t rank_idx = 0; rank_idx < sorted_scores_vec.size(); ++rank_idx) {
            if (sorted_scores_vec[rank_idx].first == new_model_player) {
                current_new_model_rank = rank_idx + 1;
                break;
            }
        }

        if (current_new_model_rank != -1) {
            new_model_rank_counts[current_new_model_rank]++;
        }
        
        if ((game_idx + 1) % 1 == 0 || (game_idx + 1) == num_games) {
            double avg_game_time = total_game_time / (game_idx + 1);
             std::cout << "Progress: Game " << std::setw(3) << (game_idx + 1) << "/" << num_games << " completed."
                       << " New Model (" << player_to_string(new_model_player) << ") got rank: " << current_new_model_rank << "."
                       << " Last duration: " << std::fixed << std::setprecision(2) << game_duration.count() << "s."
                       << " Avg time: " << std::fixed << std::setprecision(2) << avg_game_time << "s." << std::endl;
        }
    } 

    std::cout << "\n--- Strength Test Finished ---" << std::endl;
    double first_place_percentage = (num_games > 0) ? (static_cast<double>(new_model_rank_counts[1]) / num_games * 100.0) : 0.0;
    std::cout << "New Model First Places: " << new_model_rank_counts[1] << "/" << num_games
              << " (" << std::fixed << std::setprecision(2) << first_place_percentage << "%)" << std::endl;
}

} // namespace chaturaji_cpp