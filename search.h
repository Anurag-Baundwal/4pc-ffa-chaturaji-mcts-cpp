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
class Model; 
struct Move;

struct SimulationState {
  MCTSNode* current_node = nullptr; 
  std::vector<MCTSNode*> path;      
  std::optional<RequestId> pending_request_id = std::nullopt;
};

/**
 * @brief Applies softmax to the 4 rank logits for each of the 4 players.
 * @param logits 16 raw values from the NN.
 */
void apply_value_softmax(std::array<float, 16>& logits);

/**
 * @brief Backpropagates a 16-element rank probability distribution up the tree.
 * @param path The path from root to leaf.
 * @param leaf_values_for_players Array of 16 values (4 players * 4 rank probs).
 */
void backpropagate_mcts_value(const std::vector<MCTSNode*>& path, const std::array<double, 16>& leaf_values_for_players);

std::map<Move, double> process_policy(const std::array<float, NN_POLICY_SIZE>& policy_logits, const Board& board);

/**
 * @brief Runs MCTS to find the best move using the ONNX Model for inference.
 * @param network Pointer to the ONNX Model instance.
 * @param simulations Number of simulations to run.
 * @param current_mcts_root_shptr Shared pointer to the root node (for tree reuse).
 * @param c_puct Exploration constant.
 * @param mcts_batch_size Batch size for NN inference.
 * @param verbose If true, prints detailed statistics (evaluations, move candidates) to stdout.
  * @param pessimism_factor Multiplier for negative rewards. E.g., 5.0 makes losses 5x more painful.
 */
std::optional<Move> get_best_move_mcts_sync( 
    const Board& board,
    Model* network, 
    int simulations,
    std::shared_ptr<MCTSNode>& current_mcts_root_shptr, 
    double c_puct = 2.5,
    int mcts_batch_size = 16,
    bool verbose = false,
    double pessimism_factor = 1.0 // Default 1.0 = Risk Neutral
);

std::array<double, 4> convert_reward_map_to_array(const std::map<Player, double>& reward_map, double default_value = 0.0);

} // namespace chaturaji_cpp