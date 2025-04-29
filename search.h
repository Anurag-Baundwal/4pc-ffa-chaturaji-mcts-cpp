#pragma once

#include <vector>
#include <map>
#include <optional>
#include <torch/torch.h> // For NN interaction

#include "board.h"     // Needs board state
#include "mcts_node.h" // Needs MCTSNode
#include "model.h"     // Needs NN model type
#include "types.h"     // Needs Move type, EvaluationResult
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
  // RequestId associated with this path if it's pending evaluation
  std::optional<RequestId> pending_request_id = std::nullopt;
};

/**
* @brief Backpropagates a value iteratively up a stored path using the node's update_stats method.
*
* @param path The path from the root to the leaf (inclusive).
* @param value The value to backpropagate (from the root's perspective).
*/
void backpropagate_path(const std::vector<MCTSNode*>& path, double value);

// --- REMOVED evaluate_and_expand_batch DECLARATION ---
// This function's role is replaced by the async evaluator and result processing.
// void evaluate_and_expand_batch(...);


/**
* @brief Runs the core MCTS simulation loop.
*        VERSION FOR ASYNCHRONOUS SELF-PLAY (Conceptual Phase 1):
*        - Uses virtual loss during selection.
*        - Identifies leaf nodes needing evaluation.
*        - Includes placeholders for creating/queuing EvaluationRequests.
*        - Includes placeholders for processing received EvaluationResults.
*        NOTE: This function signature might change significantly when threading/queues are added.
*              The current implementation only modifies the *logic* conceptually.
*
* @param root The root node of the MCTS search.
* @param network The neural network (used conceptually, direct calls removed).
* @param simulations The total number of simulations to run.
* @param device The torch device (used conceptually).
* @param c_puct The exploration constant.
* @param mcts_batch_size The target batch size for the NN evaluator (used conceptually).
*/

/**
 * @brief Processes the raw policy logits from the neural network.
 * Masks illegal moves and applies softmax to get probabilities.
 *
 * @param policy_logits Raw output tensor from the policy head
 * @param board The current board state to determine legal moves.
 * @return A map from legal Move objects to their calculated probabilities.
 */
std::map<Move, double> process_policy(const torch::Tensor& policy_logits, const Board& board);

/**
 * @brief Runs the Monte Carlo Tree Search algorithm using the *synchronous* internal batching.
 *        Suitable for inference mode (e.g., playing a single game). Does *not* use virtual loss.
 *
 * @param board The current root board state.
 * @param network The neural network used for evaluation and policy guidance.
 * @param simulations The number of MCTS simulations to perform.
 * @param device The torch device (CPU/CUDA) the network expects tensors on.
 * @param c_puct The exploration constant for PUCT calculation.
 * @param mcts_batch_size Batch size for internal NN evaluation during the search.
 * @return The best Move found, or std::nullopt if no legal moves exist.
 */
std::optional<Move> get_best_move_mcts_sync( // Renamed for clarity
    const Board& board,
    ChaturajiNN& network, // Pass by non-const reference as network state isn't changed, but inference isn't const
    int simulations,
    torch::Device device,
    double c_puct = 1.0,
    int mcts_batch_size = 16 // Parameter for internal batching
);


/**
 * @brief Calculates the reward map based on final player rankings.
 *
 * @param final_scores A map of Player to their final score.
 * @return A map of Player to their assigned rank-based reward.
 */
std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores);


} // namespace chaturaji_cpp