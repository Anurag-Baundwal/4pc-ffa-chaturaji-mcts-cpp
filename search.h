// search.h
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

void backpropagate_mcts_value(const std::vector<MCTSNode*>& path, const std::array<double, 4>& leaf_values_for_players);
std::map<Move, double> process_policy(const std::array<float, NN_POLICY_SIZE>& policy_logits, const Board& board);

/**
 * @brief Runs MCTS to find the best move using the ONNX Model for inference.
 * @param spite_weight Weight for utility mixing (Root Spite). Default 0.0.
 */
std::optional<Move> get_best_move_mcts_sync( 
    const Board& board,
    Model* network,
    int simulations,
    std::shared_ptr<MCTSNode>& current_mcts_root_shptr, 
    double c_puct = 2.5,
    int mcts_batch_size = 16,
    bool verbose = false,
    double spite_weight = 0.0 
);

std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores);
std::array<double, 4> convert_reward_map_to_array(const std::map<Player, double>& reward_map, double default_value = 0.0);

} // namespace chaturaji_cpp