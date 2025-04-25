#pragma once

#include <vector>
#include <map>
#include <optional>
#include <torch/torch.h> // For NN interaction

#include "board.h"     // Needs board state
#include "mcts_node.h" // Needs MCTSNode
#include "model.h"     // Needs NN model type
#include "types.h"     // Needs Move type
#include "utils.h"     // Needs move_to_policy_index, policy_index_to_move

namespace chaturaji_cpp {

// Forward declaration if needed, though includes should suffice
class Board;
class MCTSNode;
class ChaturajiNN;
struct Move;

// --- Helper Structures/Functions for Batched MCTS ---

/**
 * @brief Structure to hold the state of one simulation path during MCTS.
 */
struct SimulationState {
  MCTSNode* current_node = nullptr; // The current leaf node reached
  std::vector<MCTSNode*> path;      // Path from root to current_node (for backpropagation)
};

/**
* @brief Backpropagates a value iteratively up a stored path.
*
* @param path The path from the root to the leaf (inclusive).
* @param value The value to backpropagate (from the root's perspective).
*/
void backpropagate_path(const std::vector<MCTSNode*>& path, double value);

/**
* @brief Performs a batch evaluation of leaf nodes using the neural network.
*
* @param pending_eval Vector of simulation states waiting for evaluation.
* @param network The neural network module.
* @param device The torch device.
*/
void evaluate_and_expand_batch(
  std::vector<SimulationState>& pending_eval,
  ChaturajiNN& network,
  torch::Device device);

/**
* @brief Runs the core MCTS simulation loop with batching.
* Modifies the tree under the provided root node.
*
* @param root The root node of the MCTS search.
* @param network The neural network.
* @param simulations The total number of simulations to run.
* @param device The torch device.
* @param c_puct The exploration constant.
* @param batch_size The desired batch size for NN evaluations.
*/
void run_mcts_simulations_batch(
  MCTSNode& root,
  ChaturajiNN& network,
  int simulations,
  torch::Device device,
  double c_puct,
  int batch_size); // Added batch_size parameter

/**
 * @brief Processes the raw policy logits from the neural network.
 * Masks illegal moves and applies softmax to get probabilities.
 *
 * @param policy_logits Raw output tensor from the policy head [1, 4096].
 * @param board The current board state to determine legal moves.
 * @return A map from legal Move objects to their calculated probabilities.
 */
std::map<Move, double> process_policy(const torch::Tensor& policy_logits, const Board& board);

/**
 * @brief Runs the Monte Carlo Tree Search algorithm to find the best move.
 *
 * @param board The current root board state.
 * @param network The neural network used for evaluation and policy guidance.
 * @param simulations The number of MCTS simulations to perform.
 * @param device The torch device (CPU/CUDA) the network expects tensors on.
 * @param c_puct The exploration constant for PUCT calculation.
 * @return The best Move found, or std::nullopt if no legal moves exist.
 */
std::optional<Move> get_best_move_mcts(
    const Board& board,
    ChaturajiNN& network, // Pass by non-const reference as network state isn't changed, but inference isn't const
    int simulations,
    torch::Device device,
    double c_puct = 1.0,
    int mcts_batch_size = 8 // <-- NEW parameter with default
);

/**
 * @brief Calculates the reward map based on final player rankings.
 * Matches the reward structure used in Python self-play (+2, +0.5, -0.5, -2.0).
 *
 * @param final_scores A map of Player to their final score.
 * @return A map of Player to their assigned rank-based reward.
 */
std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores);


} // namespace chaturaji_cpp