#include "search.h"
#include "mcts_node.h" // Ensure MCTSNode definition is available
#include "utils.h"     // For move indexing and tensor conversion
#include <vector>
#include <algorithm> // For std::sort, std::max_element
#include <map>
#include <limits>
#include <iostream> // For warnings/debug

namespace chaturaji_cpp {


std::map<Move, double> process_policy(const torch::Tensor& policy_logits, const Board& board) {
    std::map<Move, double> policy_probs;
    std::vector<Move> legal_moves = board.get_pseudo_legal_moves(board.get_current_player());

    if (legal_moves.empty()) {
        return policy_probs; // No legal moves, return empty map
    }

    // Ensure logits are on CPU (already specified in header doc) and correct shape
    if (policy_logits.device().type() != torch::kCPU) {
         std::cerr << "Warning: process_policy received logits not on CPU." << std::endl;
         // Potentially move to CPU here, or rely on caller convention
    }
     if (policy_logits.dim() != 1 || policy_logits.size(0) != 4096) {
          throw std::runtime_error("process_policy expects logits shape [4096], got shape " + std::to_string(policy_logits.dim())); // Better error needed for sizes
     }


    // Use negative infinity for masking before softmax for numerical stability
    torch::Tensor masked_logits = torch::full_like(policy_logits, -std::numeric_limits<float>::infinity());

    auto logits_accessor = policy_logits.accessor<float, 1>();
    auto masked_logits_accessor = masked_logits.accessor<float, 1>();

    std::vector<int> legal_indices;
    legal_indices.reserve(legal_moves.size());
    for (const auto& move : legal_moves) {
        int index = move_to_policy_index(move);
        if (index >= 0 && index < 4096) { // Check against policy size
            // Copy the original logit value for legal moves
            masked_logits_accessor[index] = logits_accessor[index];
            legal_indices.push_back(index); // Store index for later probability mapping
        } else {
             std::cerr << "Error: move_to_policy_index returned out-of-bounds index: " << index << std::endl;
             // Optionally throw or continue cautiously
        }
    }

    // Apply softmax to the masked logits
    torch::Tensor probs_tensor = torch::softmax(masked_logits, /*dim=*/0);
    auto probs_accessor = probs_tensor.accessor<float, 1>();

    // Create the result map
    for (size_t i = 0; i < legal_moves.size(); ++i) {
        int index = legal_indices[i]; // Get the index corresponding to the i-th legal move
         if (index >= 0 && index < 4096) { // Check again for safety
            policy_probs[legal_moves[i]] = static_cast<double>(probs_accessor[index]);
         }
    }

    return policy_probs;
}

// --- Iterative Backpropagation (Unchanged) ---
void backpropagate_path(const std::vector<MCTSNode*>& path, double value) {
  // Iterate path in reverse (from leaf up to, but not including, root's parent which is null)
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    MCTSNode* node = *it;
    // Call the public update_stats method on each node in the path.
    node->update_stats(value); // <-- Use the renamed public method
  }
}

// --- Synchronous MCTS Implementation (for inference/analysis) ---
// Uses internal batching via evaluate_and_expand_batch_sync.
// Does NOT use virtual loss.

// --- Private Helper for Synchronous MCTS: Batch Evaluation ---
void evaluate_and_expand_batch_sync(
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

  // 1. Collect state tensors
  for (const auto& sim_state : pending_eval) {
       state_tensors.push_back(get_board_tensor_no_batch(sim_state.current_node->get_board(), device));
  }

  // 2. Stack tensors into a batch
  torch::Tensor batch_tensor = torch::stack(state_tensors, 0);

  // 3. Perform batched NN inference
  torch::Tensor policy_logits_batch, value_batch;
  {
      torch::NoGradGuard no_grad;
      network->eval();
      std::tie(policy_logits_batch, value_batch) = network->forward(batch_tensor);
  }

  // Ensure outputs are on CPU for easier processing
  policy_logits_batch = policy_logits_batch.to(torch::kCPU);
  value_batch = value_batch.to(torch::kCPU);

  // 4. Process results for each simulation in the batch
  for (int i = 0; i < batch_size; ++i) {
      // --- Get references to the simulation state ---
      // Make sure we are using references if we intend modifications, though here we mainly read path/node
      const SimulationState& sim_state = pending_eval[i]; // Use const ref is sufficient here
      MCTSNode* leaf_node = sim_state.current_node;
      const std::vector<MCTSNode*>& path = sim_state.path; // Const ref to path

      if (!leaf_node) { // Safety check
          std::cerr << "Error: Nullptr leaf_node found in pending evaluation batch." << std::endl;
          continue;
      }

      // a. Process policy for this specific leaf
      torch::Tensor policy_logits_single = policy_logits_batch[i];
      std::map<Move, double> policy_probs = process_policy(policy_logits_single, leaf_node->get_board());

      // b. Expand the leaf node ONLY IF it's still a leaf and not terminal
      if (leaf_node->is_leaf() && !leaf_node->get_board().is_game_over()) {
           // Check policy is not empty before expanding
           if (!policy_probs.empty()) {
                leaf_node->expand(policy_probs);
           } else {
                // Log if policy is unexpectedly empty for non-terminal leaf
                std::cerr << "Warning (Sync MCTS): Empty policy from NN for non-terminal leaf node during batch processing." << std::endl;
           }
      } // else: Node is no longer a leaf (already expanded by another entry in this batch) or is terminal. Do not expand again.

      // c. Get the value for this leaf
      double value = value_batch[i].item<double>(); // Value is from root's perspective

      // d. Backpropagate the value up this leaf's path REGARDLESS of expansion success
      backpropagate_path(path, value);
  }

  // Clear the processed batch
  pending_eval.clear();
}


// --- Core Synchronous MCTS Simulation Loop ---
void run_mcts_simulations_sync( // Renamed from run_mcts_simulations_batch
  MCTSNode& root,
  ChaturajiNN& network,
  int simulations,
  torch::Device device,
  double c_puct,
  int batch_size) // Internal batch size for sync eval
{
  std::vector<SimulationState> pending_evaluation;
  pending_evaluation.reserve(batch_size);

  Player root_player = root.get_board().get_current_player();

  for (int i = 0; i < simulations; ++i) {
      SimulationState current_sim;
      current_sim.current_node = &root;
      current_sim.path.push_back(current_sim.current_node);

      // 1. Selection: Traverse the tree using PUCT.
      //    *Crucially, does NOT use virtual loss*. It calls the standard select_child.
      while (!current_sim.current_node->is_leaf()) {
          // Select child *without* virtual loss consideration (standard PUCT)
          // We achieve this because pending_visits_ will always be 0 in this sync loop.
           MCTSNode* next_node = current_sim.current_node->select_child(c_puct);

          // Simple check to prevent infinite loops if selection fails unexpectedly
          if (next_node == nullptr || next_node == current_sim.current_node) {
                 std::cerr << "Warning: MCTS sync select_child failed or didn't advance." << std::endl;
                 // If it's null, the node was likely terminal but considered non-leaf? Error state.
                 // If it's the same node, could be an issue with PUCT calculation or tree state.
                 continue;
          }
          current_sim.current_node = next_node;
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
          // Non-terminal leaf: Add to pending batch for synchronous evaluation
          pending_evaluation.push_back(std::move(current_sim)); // Move the state

          // 3. Evaluate batch if full
          if (pending_evaluation.size() >= static_cast<size_t>(batch_size)) {
              evaluate_and_expand_batch_sync(pending_evaluation, network, device);
          }
      }

  } // End of simulations loop

  // 4. Evaluate any remaining pending nodes
  evaluate_and_expand_batch_sync(pending_evaluation, network, device);
}

// --- Interface function for synchronous MCTS (inference mode) ---
std::optional<Move> get_best_move_mcts_sync( // Renamed from get_best_move_mcts
    const Board& board,
    ChaturajiNN& network,
    int simulations,
    torch::Device device,
    double c_puct,
    int mcts_batch_size) // Renamed param for clarity
{
    if (board.is_game_over()) {
      return std::nullopt;
    }

    // Ensure network is in evaluation mode
    network->eval();

    // Create root node with a copy of the board
    MCTSNode root(board);

    // Run the SYNCHRONOUS simulations with internal batching
    run_mcts_simulations_sync(root, network, simulations, device, c_puct, mcts_batch_size); // Call the renamed sync version

    // Choose the best move based on visit counts (same logic as before)
    const auto& children = root.get_children();
    if (children.empty()) {
      std::cerr << "Warning: MCTS root (sync) has no children after simulations." << std::endl;
      auto legal_moves = board.get_pseudo_legal_moves(board.get_current_player());
       if (!legal_moves.empty()) { return legal_moves[0]; } // Fallback?
      return std::nullopt;
    }

    auto best_child_it = std::max_element(children.begin(), children.end(),
        [](const std::unique_ptr<MCTSNode>& a, const std::unique_ptr<MCTSNode>& b) {
            return a->get_visit_count() < b->get_visit_count();
        });

    if (best_child_it != children.end() && (*best_child_it)->get_visit_count() > 0) {
        return (*best_child_it)->get_move();
    } else {
        std::cerr << "Warning: Could not determine best move from MCTS sync visits (all 0?)." << std::endl;
        if (!children.empty()) return children[0]->get_move(); // Fallback?
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
    // Standard rewards: +1 (1st), 0 (2nd), -0.25 (3rd), -1 (4th) - adjust if needed
    // If fewer than 4 players, ranks still apply (e.g., 1st=+1, 2nd=0, 3rd=-0.25)
    double rewards[] = {+1.0, 0.0, -0.25, -1.0};

    // Assign rewards based on rank
    for (size_t i = 0; i < sorted_scores.size(); ++i) {
         // Ensure we don't go out of bounds for the rewards array
         if (i < sizeof(rewards)/sizeof(rewards[0])) {
             reward_map[sorted_scores[i].first] = rewards[i];
         } else {
              // Should not happen with 4 players max, but good safety check
             reward_map[sorted_scores[i].first] = rewards[sizeof(rewards)/sizeof(rewards[0])-1]; // Assign lowest reward
         }
    }

     // Ensure all original players have an entry, default to lowest reward if somehow missing
     // (This handles cases where a player might have score 0 and not be in sorted_scores if map iteration order isn't guaranteed)
     for(int p_idx = 0; p_idx < 4; ++p_idx) {
         Player p = static_cast<Player>(p_idx);
         if (final_scores.count(p) && reward_map.find(p) == reward_map.end()) {
              // Find rank based on score compared to others
              int score_p = final_scores.at(p);
              size_t rank = 0;
              for(const auto& sp : sorted_scores) {
                   if (sp.second > score_p) {
                        rank++;
                   } else if (sp.second == score_p) {
                        // Tie-breaking? For now, assign based on first occurrence in sort
                        break;
                   }
              }
              if (rank < sizeof(rewards)/sizeof(rewards[0])) {
                  reward_map[p] = rewards[rank];
              } else {
                   reward_map[p] = rewards[sizeof(rewards)/sizeof(rewards[0])-1];
              }
         } else if (!final_scores.count(p)) {
            // Player wasn't even in the final scores? Assign lowest reward.
             reward_map[p] = rewards[sizeof(rewards)/sizeof(rewards[0])-1];
         }
     }


    return reward_map;
}


} // namespace chaturaji_cpp