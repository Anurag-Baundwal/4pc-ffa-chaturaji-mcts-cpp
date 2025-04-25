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

// --- NEW: Iterative Backpropagation ---
void backpropagate_path(const std::vector<MCTSNode*>& path, double value) {
  // Iterate path in reverse (from leaf up to, but not including, root's parent which is null)
  // Note: path includes the root itself. We update the root node's stats as well.
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    MCTSNode* node = *it;
    // Call the public update method on each node in the path.
    // The update method itself *no longer* recurses.
    node->update(value); // <-- Use the public method
}
}


// --- NEW: Batch Evaluation and Expansion ---
void evaluate_and_expand_batch(
  std::vector<SimulationState>& pending_eval,
  ChaturajiNN& network,
  torch::Device device)
{
  if (pending_eval.empty()) {
      return;
  }

  int batch_size = pending_eval.size();
  std::vector<torch::Tensor> state_tensors;
  state_tensors.reserve(batch_size);

  // 1. Collect state tensors (without batch dim)
  for (const auto& sim_state : pending_eval) {
      state_tensors.push_back(get_board_tensor_no_batch(sim_state.current_node->get_board(), device));
  }

  // 2. Stack tensors into a batch
  torch::Tensor batch_tensor = torch::stack(state_tensors, 0).to(device); // Stack along dim 0

  // 3. Perform batched NN inference
  torch::Tensor policy_logits_batch, value_batch;
  {
      torch::NoGradGuard no_grad;
      network->eval(); // Ensure eval mode
      std::tie(policy_logits_batch, value_batch) = network->forward(batch_tensor);
  }

  // Ensure outputs are on CPU for easier processing
  policy_logits_batch = policy_logits_batch.to(torch::kCPU);
  value_batch = value_batch.to(torch::kCPU);

  // 4. Process results for each simulation in the batch
  for (int i = 0; i < batch_size; ++i) {
      SimulationState& sim_state = pending_eval[i];
      MCTSNode* leaf_node = sim_state.current_node;
      const std::vector<MCTSNode*>& path = sim_state.path;

      // a. Process policy for this specific leaf
      //    Need policy_logits_batch[i] which is shape [4096]
      //    process_policy expects shape [1, 4096] or [4096], let's give it [4096]
      torch::Tensor policy_logits_single = policy_logits_batch[i];
      std::map<Move, double> policy_probs = process_policy(policy_logits_single, leaf_node->get_board());

      // b. Expand the leaf node
      if (!policy_probs.empty() && !leaf_node->get_board().is_game_over()) {
           leaf_node->expand(policy_probs);
      }

      // c. Get the value for this leaf
      //    value_batch[i] is shape [1]
      double value = value_batch[i].item<double>(); // Value is from root's perspective

      // d. Backpropagate the value up this leaf's path
      backpropagate_path(path, value);
  }

  // Clear the processed batch
  pending_eval.clear();
}


// --- NEW: Core Batched MCTS Simulation Loop ---
void run_mcts_simulations_batch(
  MCTSNode& root,
  ChaturajiNN& network,
  int simulations,
  torch::Device device,
  double c_puct,
  int batch_size)
{
  std::vector<SimulationState> pending_evaluation;
  pending_evaluation.reserve(batch_size);

  Player root_player = root.get_board().get_current_player(); // Needed for terminal reward perspective

  for (int i = 0; i < simulations; ++i) {
      SimulationState current_sim;
      current_sim.current_node = &root;
      current_sim.path.push_back(current_sim.current_node);

      // 1. Selection: Traverse the tree using PUCT until a leaf node is reached
      while (!current_sim.current_node->is_leaf()) {
          current_sim.current_node = current_sim.current_node->select_child(c_puct);
          if (!current_sim.current_node) {
               // Should not happen if !is_leaf, indicates potential issue
               std::cerr << "Warning: MCTS select_child returned nullptr from non-leaf." << std::endl;
               // How to recover? Maybe just break this simulation?
               goto next_simulation; // Use goto carefully, or refactor loop control
          }
          current_sim.path.push_back(current_sim.current_node);
      }

      // 2. Check if leaf is terminal or needs evaluation
      if (current_sim.current_node->get_board().is_game_over()) {
          // Terminal node: Calculate reward and backpropagate immediately
          std::map<Player, int> final_scores = current_sim.current_node->get_board().get_game_result();
          std::map<Player, double> reward_map = get_reward_map(final_scores);
          double value = reward_map.count(root_player) ? reward_map.at(root_player) : -2.0;
          backpropagate_path(current_sim.path, value);
      } else {
          // Non-terminal leaf: Add to pending batch
          pending_evaluation.push_back(std::move(current_sim)); // Move the state

          // 3. Evaluate batch if full
          if (pending_evaluation.size() >= batch_size) {
              evaluate_and_expand_batch(pending_evaluation, network, device);
          }
      }

      next_simulation:; // Label for goto target

  } // End of simulations loop

  // 4. Evaluate any remaining pending nodes
  evaluate_and_expand_batch(pending_evaluation, network, device);
}

std::optional<Move> get_best_move_mcts(
    const Board& board,
    ChaturajiNN& network,
    int simulations,
    torch::Device device,
    double c_puct,
    int mcts_batch_size) // Added batch_size parameter
{
    if (board.is_game_over()) {
      return std::nullopt;
    }

    // Ensure network is in evaluation mode
    network->eval();

    // Create root node with a copy of the board
    MCTSNode root(board); // Root has no parent or move leading to it

    // Run the batched simulations
    run_mcts_simulations_batch(root, network, simulations, device, c_puct, mcts_batch_size);
    
    // Choose the best move based on visit counts
    const auto& children = root.get_children();
    if (children.empty()) {
      // This might happen if root is terminal or expansion failed for some reason
      std::cerr << "Warning: MCTS root has no children after simulations." << std::endl;
      // Maybe return the first legal move if any? Or just nullopt.
      auto legal_moves = board.get_pseudo_legal_moves(board.get_current_player());
      if (!legal_moves.empty()) {
         // Maybe return a random legal move as fallback?
         // For now, stick to returning nullopt if MCTS failed to produce children/visits
          return std::nullopt;
      }
      return std::nullopt;
    }

    auto best_child_it = std::max_element(children.begin(), children.end(),
        [](const std::unique_ptr<MCTSNode>& a, const std::unique_ptr<MCTSNode>& b) {
            return a->get_visit_count() < b->get_visit_count();
        });

        if (best_child_it != children.end() && (*best_child_it)->get_visit_count() > 0) {
          // Return the move associated with the best child
          return (*best_child_it)->get_move();
      } else {
          // If no child was visited (e.g., only 1 simulation requested and it hit terminal?)
          // Or if max_element somehow fails (unlikely unless children empty)
          std::cerr << "Warning: Could not determine best move from MCTS visits (all 0?)." << std::endl;
           // Fallback: Return the first child's move if available? Or just nullopt.
           if (!children.empty()) return children[0]->get_move();
          return std::nullopt;
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