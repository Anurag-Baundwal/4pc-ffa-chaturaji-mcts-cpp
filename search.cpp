#include "search.h"
#include "mcts_node.h" // Ensure MCTSNode definition is available
#include "utils.h"     // For move indexing and tensor conversion
#include <vector>
#include <algorithm> // For std::sort, std::max_element
#include <map>
#include <limits>

namespace chaturaji_cpp {


std::map<Move, double> process_policy(const torch::Tensor& policy_logits, const Board& board) {
    std::map<Move, double> policy_probs;
    std::vector<Move> legal_moves = board.get_pseudo_legal_moves(board.get_current_player());

    if (legal_moves.empty()) {
        return policy_probs; // No legal moves, return empty map
    }

    // Ensure logits are on CPU for easier indexing if they aren't already
    torch::Tensor logits_cpu = policy_logits.to(torch::kCPU).squeeze(); // Remove batch dim, move to CPU

    // Create a mask for legal moves
    // Use negative infinity for masking before softmax for numerical stability
    torch::Tensor masked_logits = torch::full_like(logits_cpu, -std::numeric_limits<float>::infinity());

    auto logits_accessor = logits_cpu.accessor<float, 1>();
    auto masked_logits_accessor = masked_logits.accessor<float, 1>();

    std::vector<int> legal_indices;
    legal_indices.reserve(legal_moves.size());
    for (const auto& move : legal_moves) {
        int index = move_to_policy_index(move);
        if (index >= 0 && index < logits_cpu.size(0)) {
            // Copy the original logit value for legal moves
            masked_logits_accessor[index] = logits_accessor[index];
            legal_indices.push_back(index); // Store index for later probability mapping
        } else {
             // Log error or throw: index out of bounds
             // std::cerr << "Error: move_to_policy_index returned out-of-bounds index: " << index << std::endl;
        }
    }

    // Apply softmax to the masked logits
    torch::Tensor probs_tensor = torch::softmax(masked_logits, /*dim=*/0);
    auto probs_accessor = probs_tensor.accessor<float, 1>();

    // Create the result map
    for (size_t i = 0; i < legal_moves.size(); ++i) {
        int index = legal_indices[i]; // Get the index corresponding to the i-th legal move
         if (index >= 0 && index < probs_tensor.size(0)) {
            policy_probs[legal_moves[i]] = static_cast<double>(probs_accessor[index]);
         }
    }

    return policy_probs;
}


std::optional<Move> get_best_move_mcts(
    const Board& board,
    ChaturajiNN& network,
    int simulations,
    torch::Device device,
    double c_puct)
{
    // Ensure network is in evaluation mode
    network->eval();

    // Create root node with a copy of the board
    MCTSNode root(board); // Root has no parent or move leading to it

    for (int i = 0; i < simulations; ++i) {
        MCTSNode* node = &root;
        std::vector<MCTSNode*> search_path; // Store nodes visited during selection
        search_path.push_back(node);

        // 1. Selection: Traverse the tree using PUCT until a leaf node is reached
        while (!node->is_leaf()) {
            node = node->select_child(c_puct);
            // Handle case where select_child returns nullptr (shouldn't happen if !is_leaf)
            if (!node) {
                 // This indicates an issue, maybe log an error or break
                 // std::cerr << "Error: select_child returned nullptr from non-leaf node." << std::endl;
                 continue;
            }
            search_path.push_back(node);
        }

        // 2. Expansion & Evaluation:
        double value = 0.0; // Value from the perspective of the player at the *root* node

        if (node->get_board().is_game_over()) {
            // Terminal node reached
            std::map<Player, int> final_scores = node->get_board().get_game_result();
            std::map<Player, double> reward_map = get_reward_map(final_scores); // Use helper

            // The value backpropagated should be from the root player's perspective
            value = reward_map.count(root.get_board().get_current_player()) ?
                    reward_map.at(root.get_board().get_current_player()) : -2.0; // Default to lowest reward if player somehow missing
        } else {
            // Non-terminal leaf node: Evaluate using the network and expand
            torch::Tensor state_tensor = board_to_tensor(node->get_board(), device);

            // Perform inference (no gradients needed here)
            torch::Tensor policy_logits, value_tensor;
            { // Scope for no_grad
                torch::NoGradGuard no_grad;
                std::tie(policy_logits, value_tensor) = network->forward(state_tensor);
            }


            // Process policy output
            std::map<Move, double> policy_probs = process_policy(policy_logits, node->get_board());

            // Expand the node
            if (!policy_probs.empty()) {
                 node->expand(policy_probs);
            }
             // else: No legal moves from this state, should be caught by is_game_over earlier?
             // If expand is called with empty policy_probs, it does nothing.

            // Get the value from the network output (shape [1, 1])
            // This value is typically tanh scaled (-1 to 1), representing expected outcome
            // for the player whose turn it is *at this node*.
            // The Python MCTS backpropagated this value directly, implying it was treated
            // as relative to the root player. Let's stick to that for now.
             value = value_tensor.item<double>(); // Get scalar value
        }

        // 3. Backpropagation: Update nodes along the search path
        // Iterate in reverse order? The Python update handles recursion. Our iterative search_path needs manual reverse.
        // The `update` method recursively calls parent->update, so just updating the leaf is sufficient.
        node->update(value);

    } // End of simulations loop

    // Choose the best move based on visit counts
    const auto& children = root.get_children();
    if (children.empty()) {
        return std::nullopt; // No legal moves from the root
    }

    auto best_child_it = std::max_element(children.begin(), children.end(),
        [](const std::unique_ptr<MCTSNode>& a, const std::unique_ptr<MCTSNode>& b) {
            return a->get_visit_count() < b->get_visit_count();
        });

    // Ensure a best child was found (should always happen if children is not empty)
    if (best_child_it != children.end()) {
        // Return the move that leads to the best child
        return (*best_child_it)->get_move(); // Move is stored in the child node
    } else {
        return std::nullopt; // Should be unreachable if children is not empty
    }
}


std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores) {
    std::vector<std::pair<Player, int>> sorted_scores(final_scores.begin(), final_scores.end());

    // Sort players by score descending
    std::sort(sorted_scores.begin(), sorted_scores.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second; // Higher score first
              });

    std::map<Player, double> reward_map;
    double rewards[] = {+1, +0, -0.25, -1}; // Rank 1 to 4 rewards

    // Assign rewards based on rank
    for (size_t i = 0; i < sorted_scores.size() && i < 4; ++i) {
        reward_map[sorted_scores[i].first] = rewards[i];
    }

     // Ensure all original players have an entry, default to lowest reward if somehow missing
     for (const auto& pair : final_scores) {
         if (reward_map.find(pair.first) == reward_map.end()) {
             reward_map[pair.first] = -2.0;
         }
     }

    return reward_map;
}


} // namespace chaturaji_cpp