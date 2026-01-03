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
#include <random> // Required for random opening

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
    std::cout << "--- Starting Strength Test---" << std::endl;
    std::cout << "  New Model Path:    " << new_model_path << std::endl;

    // --- 1. Model Loading / Random Initialization Logic ---
    if (!fs::exists(new_model_path)) {
        std::cerr << "Error: New model file not found at " << new_model_path << std::endl;
        return;
    }

    std::unique_ptr<Model> new_network;
    std::unique_ptr<Model> old_network;

    try {
        new_network = std::make_unique<Model>(new_model_path);
        
        if (!old_model_path.empty()) {
            if (!fs::exists(old_model_path)) {
                std::cerr << "Error: Old model file not found at " << old_model_path << std::endl;
                return;
            }
            old_network = std::make_unique<Model>(old_model_path);
            std::cout << "  Old Model Path:    " << old_model_path << std::endl;
        } else {
            // Restore "Random Model" initialization similar to train.cpp
            std::cout << "  Old Model Path:    [NONE] - Initializing random weights..." << std::endl;
            std::string random_onnx = "temp_strength_random.onnx";
            std::string init_cmd = "python model.py export_random " + random_onnx;
            if (std::system(init_cmd.c_str()) == 0) {
                old_network = std::make_unique<Model>(random_onnx);
            } else {
                std::cerr << "Error: Failed to generate random baseline model via Python." << std::endl;
                return;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception loading models: " << e.what() << std::endl;
        return;
    }

    // --- 2. Test Execution ---
    std::mt19937 rng(std::random_device{}());
    double total_game_time = 0.0;
    std::map<int, int> new_model_rank_counts = {{1,0}, {2,0}, {3,0}, {4,0}};

    // Initialize tt
    auto tt = std::make_unique<TranspositionTable>(1024);

    for (int game_idx = 0; game_idx < num_games; ++game_idx) {
        auto game_start_time = std::chrono::high_resolution_clock::now();
        Board board; 

        // --- 4-PLY RANDOM OPENING (One move per player) ---
        for (int p = 0; p < 4; ++p) {
            if (board.is_game_over()) break;
            std::vector<Move> moves = board.get_pseudo_legal_moves(board.get_current_player());
            if (moves.empty()) break;
            
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            board.make_move(moves[dist(rng)]);
        }

        Player new_model_player = static_cast<Player>(game_idx % 4);

        int move_count = 0;
        
        while (!board.is_game_over()) {
            Player current_player = board.get_current_player();
            Model* current_network_ptr = (current_player == new_model_player) ? new_network.get() : old_network.get();

            // Note: Tree reuse is diasbled in strength tests for fair testing

            // Update Age and set root
            uint32_t current_move_age = static_cast<uint32_t>(move_count++);
            std::shared_ptr<MCTSNode> mcts_root_node_strength_test = nullptr; 

            std::optional<Move> best_move_opt = get_best_move_mcts_sync(
                board, current_network_ptr, simulations_per_move, 
                mcts_root_node_strength_test, 
                2.5, 
                mcts_batch_size,
                tt.get(),
                static_cast<uint32_t>(move_count)
            );

            if (best_move_opt) {
                board.make_move(*best_move_opt);
            } else {
                if (!board.is_game_over() && board.get_active_players().count(current_player)) {
                    board.resign(); 
                }
            }
        } 

        auto game_end_time = std::chrono::high_resolution_clock::now();
        total_game_time += std::chrono::duration<double>(game_end_time - game_start_time).count();

        // Calculate Ranks
        Board::PlayerPointMap final_scores = board.get_game_result();
        std::vector<std::pair<Player, int>> results;
        for (int i = 0; i < 4; ++i) {
            Player p = static_cast<Player>(i);
            results.push_back({p, final_scores[p]});
        }
        std::sort(results.begin(), results.end(), [](auto& a, auto& b){ return a.second > b.second; });

        for (size_t rank = 0; rank < results.size(); ++rank) {
            if (results[rank].first == new_model_player) {
                new_model_rank_counts[rank + 1]++;
                break;
            }
        }
        
        if ((game_idx + 1) % 1 == 0 || (game_idx + 1) == num_games) {
            // Calculate how many digits wide the game number needs to be
            int width = std::to_string(num_games).length();

            std::cout << "Game " 
                      << std::right << std::setw(width) << (game_idx + 1) 
                      << "/" 
                      << std::left << std::setw(width) << num_games 
                      << " | NewModel (" << std::left << std::setw(6) << player_to_string(new_model_player) << ")"
                      << " | Rank: " << (std::find_if(results.begin(), results.end(), [&](auto& pair){return pair.first == new_model_player;}) - results.begin() + 1)
                      << " | Avg: " << std::fixed << std::setprecision(2) << (total_game_time / (game_idx + 1)) << "s" << std::endl;
        }
    } 

    std::cout << "\n--- Strength Test Results (After " << num_games << " games) ---" << std::endl;
    for (int i = 1; i <= 4; ++i) {
        double pct = (static_cast<double>(new_model_rank_counts[i]) / num_games) * 100.0;
        std::cout << "  Rank " << i << ": " << std::setw(3) << new_model_rank_counts[i] << " (" << pct << "%)" << std::endl;
    }
}

} // namespace chaturaji_cpp