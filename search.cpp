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
         // Simple workaround for shape check: Print sizes. A more robust solution would format the sizes vector.
        std::string shape_str = "[";
        for(int64_t s : policy_logits.sizes()) { shape_str += std::to_string(s) + ","; }
        if (shape_str.length() > 1) shape_str.pop_back(); // Remove trailing comma
        shape_str += "]";
        throw std::runtime_error("process_policy expects logits shape [4096], got shape " + shape_str);
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

// --- Backpropagation Function ---
void backpropagate_mcts_value(const std::vector<MCTSNode*>& path, double leaf_value_for_leaf_player) {
    double current_value_for_node_player = leaf_value_for_leaf_player;
    // Iterate path in reverse (from leaf up to the root)
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        MCTSNode* node = *it;
        // MCTSNode::update_stats (visit_count_++, total_value_ += value)
        // receives a value that is now correctly signed for 'node's player.
        node->update_stats(current_value_for_node_player);

        // For the parent of 'node', the value of this outcome is the negative
        // of what it was for 'node'.
        current_value_for_node_player *= -1.0;
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
       // Generate tensor on CPU for potentially easier transfer if evaluator is on GPU
       state_tensors.push_back(get_board_tensor_no_batch(sim_state.current_node->get_board(), torch::kCPU));
  }

  // 2. Stack tensors into a batch and move to target device
  torch::Tensor batch_tensor = torch::stack(state_tensors, 0).to(device);

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
      double value_for_leaf_player = value_batch[i].item<double>(); // Value is from leaf's current player's perspective

      // d. Backpropagate the value up this leaf's path REGARDLESS of expansion success
      backpropagate_mcts_value(path, value_for_leaf_player);
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
  // --- Handle edge case: 0 simulations ---
  // If 0 simulations, we still might need the root policy for the fallback in get_best_move_mcts_sync.
  // We trigger *one* evaluation of the root if it's not terminal and not yet expanded.
  if (simulations == 0 && root.is_leaf() && !root.get_board().is_game_over()) {
      std::vector<SimulationState> initial_eval;
      SimulationState root_state;
      root_state.current_node = &root;
      root_state.path.push_back(&root);
      initial_eval.push_back(std::move(root_state));
      std::cout << "Info (Sync MCTS): simulations=0, evaluating root node directly for policy." << std::endl;
      evaluate_and_expand_batch_sync(initial_eval, network, device);
      return; // No further simulations needed
  }
  // --- End 0 simulations handling ---


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
                 std::cerr << "Warning: MCTS sync select_child failed or didn't advance."
                           << " Parent visits: " << current_sim.current_node->get_visit_count()
                           << ", Children: " << current_sim.current_node->get_children().size()
                           << ", IsGameOver: " << current_sim.current_node->get_board().is_game_over() << std::endl;
                 // If selection failed, we cannot proceed down this path. Backpropagate terminal value if possible.
                 if (current_sim.current_node->get_board().is_game_over()){
                    MCTSNode* terminal_leaf = current_sim.current_node; // This node is the "leaf" of this failed path
                    Player player_at_this_terminal_node = terminal_leaf->get_board().get_current_player();
                    std::map<Player, int> final_scores = terminal_leaf->get_board().get_game_result();
                    std::map<Player, double> reward_map = get_reward_map(final_scores);
                    double value_for_this_node = reward_map.count(player_at_this_terminal_node) ? reward_map.at(player_at_this_terminal_node) : -1.0; // Use appropriate default
                    backpropagate_mcts_value(current_sim.path, value_for_this_node);
                 } else {
                     // Non-terminal node where selection failed? This is odd.
                     // Backpropagate a neutral value (0?) or error value? Let's use 0 for now.
                     backpropagate_mcts_value(current_sim.path, 0.0);
                 }

                 goto next_simulation; // Use goto to jump to the start of the next simulation cleanly
          }
          current_sim.current_node = next_node;
          current_sim.path.push_back(current_sim.current_node);
      } // End selection loop


      // 2. Check if leaf is terminal or needs evaluation
      if (current_sim.current_node->get_board().is_game_over()) {
          MCTSNode* terminal_leaf = current_sim.current_node;
          Player player_at_terminal_leaf = terminal_leaf->get_board().get_current_player();
          // Note: If game ends, current_player might be the *next* player if the game ended due to
          // the previous player's move resulting in elimination/draw.
          // The reward should ideally be for the player whose action *led to* or *was about to be made from* this state.
          // For simplicity, if get_current_player() on a terminal board returns the player who would have
          // hypothetically moved next, using their reward is a common approach.

          std::map<Player, int> final_scores = terminal_leaf->get_board().get_game_result();
          std::map<Player, double> reward_map = get_reward_map(final_scores); // reward_map has {Player: rank_reward}

          double value_for_leaf_player_at_terminal = 0.0;
          if (reward_map.count(player_at_terminal_leaf)) {
              value_for_leaf_player_at_terminal = reward_map.at(player_at_terminal_leaf);
          } else {
               std::cerr << "Warning (Sync MCTS): Player " << static_cast<int>(player_at_terminal_leaf)
                         << " not found in reward_map for terminal node. Using 0.0 for backprop." << std::endl;
          }
          backpropagate_mcts_value(current_sim.path, value_for_leaf_player_at_terminal);
      } else {
          // Non-terminal leaf: Add to pending batch for synchronous evaluation
          pending_evaluation.push_back(std::move(current_sim)); // Move the state

          // 3. Evaluate batch if full
          if (pending_evaluation.size() >= static_cast<size_t>(batch_size)) {
              evaluate_and_expand_batch_sync(pending_evaluation, network, device);
          }
      }

      next_simulation:; // Label for the goto statement

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
    // This will now handle simulations=0 correctly by evaluating the root.
    run_mcts_simulations_sync(root, network, simulations, device, c_puct, mcts_batch_size); // Call the renamed sync version

    // --- Move Selection Logic ---
    const auto& children = root.get_children();

    // Case 1: No children generated (root is terminal, or simulations=0 and no legal moves)
    if (children.empty()) {
        auto legal_moves = board.get_pseudo_legal_moves(board.get_current_player());
        if (legal_moves.empty()) {
            std::cerr << "Warning (get_best_move): Root has no children and no legal moves. Returning nullopt." << std::endl;
            return std::nullopt; // No moves possible
        } else {
            // This should only happen if simulations=0 and the root eval failed or yielded empty policy
            std::cerr << "Warning (get_best_move): Root has no children despite legal moves existing (Sims=" << simulations << "). Returning first legal move as fallback." << std::endl;
            return legal_moves[0];
        }
    }

    // Case 2: Children exist, find best by visit count first
    auto best_child_by_visit_it = std::max_element(children.begin(), children.end(),
        [](const std::unique_ptr<MCTSNode>& a, const std::unique_ptr<MCTSNode>& b) {
            return a->get_visit_count() < b->get_visit_count();
        });

    // If the best child has visits > 0, use standard MCTS result
    if (best_child_by_visit_it != children.end() && (*best_child_by_visit_it)->get_visit_count() > 0) {
         // Standard MCTS result based on visits
        // Ensure the move is valid before returning
        if((*best_child_by_visit_it)->get_move()) {
             return (*best_child_by_visit_it)->get_move();
        } else {
            std::cerr << "Error (get_best_move): Best child by visit has no associated move." << std::endl;
            // Fallback logic in case of error
            if (!children.empty() && children[0]->get_move()) return children[0]->get_move();
            return std::nullopt;
        }
    } else {
        // Fallback: All children have 0 visits. Use prior probability.
        // This happens when simulations < mcts_batch_size or simulations=0.
        // The root *should* have been expanded by run_mcts_simulations_sync if sims=0
        // or by the final evaluate_and_expand_batch_sync call otherwise.
        std::cerr << "Warning (get_best_move): All child nodes have zero visits (Sims=" << simulations << "). Using prior probabilities from policy." << std::endl;

         auto best_child_by_prior_it = std::max_element(children.begin(), children.end(),
             [](const std::unique_ptr<MCTSNode>& a, const std::unique_ptr<MCTSNode>& b) {
                 return a->get_prior() < b->get_prior(); // Compare priors
             });

         if (best_child_by_prior_it != children.end()) {
              // Check if the move is valid before returning
              if ((*best_child_by_prior_it)->get_move()) {
                   return (*best_child_by_prior_it)->get_move();
              } else {
                   std::cerr << "Error (get_best_move): Best child by prior has no associated move." << std::endl;
                   // Fallback
                   if (!children.empty() && children[0]->get_move()) return children[0]->get_move();
                   return std::nullopt;
              }
         } else {
             // This case (children exist but max_element fails) shouldn't normally happen
             std::cerr << "Error (get_best_move): Could not determine best move from priors even though children exist." << std::endl;
             // Fallback
              if (!children.empty() && children[0]->get_move()) return children[0]->get_move();
             return std::nullopt;
         }
    }
}


std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores) {
    std::vector<std::pair<Player, int>> sorted_scores;
    // Ensure all 4 players are considered, even if score is 0 or negative implicitly
    for (int p_idx = 0; p_idx < 4; ++p_idx) {
        Player p = static_cast<Player>(p_idx);
        int score = final_scores.count(p) ? final_scores.at(p) : 0; // Default score 0 if not in map
        sorted_scores.emplace_back(p, score);
    }


    // Sort players by score descending
    std::sort(sorted_scores.begin(), sorted_scores.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second; // Higher score first
              });

    std::map<Player, double> reward_map;
    double rewards[] = {+1.0, +0.25, -0.25, -1.0};

    // Assign rewards based on rank
    for (size_t i = 0; i < sorted_scores.size(); ++i) {
         // Use index i directly as rank (0-based index matches reward index)
         reward_map[sorted_scores[i].first] = rewards[i];
    }

    return reward_map;
}


} // namespace chaturaji_cpp