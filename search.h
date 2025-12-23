#pragma once

#include <vector>
#include <map>
#include <optional>
#include <array>
#include <memory>

#include "board.h"     
#include "mcts_node.h" 
#include "model.h"     
#include "types.h"     
#include "utils.h"     

namespace chaturaji_cpp {

class Board;
class MCTSNode;
class Model; // Forward declaration of the new Model class
struct Move;

struct SimulationState {
  MCTSNode* current_node = nullptr; 
  std::vector<MCTSNode*> path;      
  std::optional<RequestId> pending_request_id = std::nullopt;
};

/**
 * @brief Backpropagates a vector of player-specific values up the MCTS path.
 * @param path The path from root to leaf (inclusive, leaf is at path.back()).
 * @param leaf_values_for_players The array of 4 values of the leaf state,
 *                                  for each of the 4 players (RED, BLUE, YELLOW, GREEN).
 */
void backpropagate_mcts_value(const std::vector<MCTSNode*>& path, const std::array<double, 4>& leaf_values_for_players);


std::map<Move, double> process_policy(const std::array<float, 4096>& policy_logits, const Board& board);

/**
 * @brief Runs MCTS to find the best move using the ONNX Model for inference.
 * @param network Pointer to the ONNX Model instance.
 * @param device Removed (handled internally by ONNX Runtime options if needed).
 */
std::optional<Move> get_best_move_mcts_sync( 
    const Board& board,
    Model* network, 
    int simulations,
    std::shared_ptr<MCTSNode>& current_mcts_root_shptr, 
    double c_puct = 2.5,
    int mcts_batch_size = 16 
);

std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores);

std::array<double, 4> convert_reward_map_to_array(const std::map<Player, double>& reward_map, double default_value = 0.0);

} // namespace chaturaji_cpp