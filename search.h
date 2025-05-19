#pragma once

#include <vector>
#include <map>
#include <optional>
#include <array> // For std::array
#include <torch/torch.h> 

#include "board.h"     
#include "mcts_node.h" 
#include "model.h"     
#include "types.h"     
#include "utils.h"     

namespace chaturaji_cpp {

class Board;
class MCTSNode;
class ChaturajiNN;
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
void backpropagate_mcts_value(const std::vector<MCTSNode*>& path, const std::array<double, 4>& leaf_values_for_players); // MODIFIED


std::map<Move, double> process_policy(const torch::Tensor& policy_logits, const Board& board);

std::optional<Move> get_best_move_mcts_sync( 
    const Board& board,
    ChaturajiNN& network, 
    int simulations,
    torch::Device device,
    double c_puct = 2.5,
    int mcts_batch_size = 16 
);

std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores);

/**
 * @brief Converts a map of player rewards to a fixed-size array, ordered by Player enum.
 * @param reward_map Map from Player enum to their reward.
 * @param default_value Value to use for players not present in the map.
 * @return std::array<double, 4> with rewards for RED, BLUE, YELLOW, GREEN.
 */
std::array<double, 4> convert_reward_map_to_array(const std::map<Player, double>& reward_map, double default_value = 0.0);


} // namespace chaturaji_cpp