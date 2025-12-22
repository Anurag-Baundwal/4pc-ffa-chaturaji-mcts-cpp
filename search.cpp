#include "search.h"
#include "mcts_node.h" 
#include "utils.h"     
#include <vector>
#include <algorithm> 
#include <map>
#include <limits>
#include <iostream> 
#include <memory> // For std::shared_ptr, std::make_shared, std::move

namespace chaturaji_cpp {

// Helper function definition
std::array<double, 4> convert_reward_map_to_array(const std::map<Player, double>& reward_map, double default_value) {
    std::array<double, 4> player_rewards;
    for (int i = 0; i < 4; ++i) {
        Player p = static_cast<Player>(i);
        auto it = reward_map.find(p);
        if (it != reward_map.end()) {
            player_rewards[i] = it->second;
        } else {
            player_rewards[i] = default_value;
        }
    }
    return player_rewards;
}


std::map<Move, double> process_policy(const torch::Tensor& policy_logits, const Board& board) {
    std::map<Move, double> policy_probs;
    std::vector<Move> legal_moves = board.get_pseudo_legal_moves(board.get_current_player());

    if (legal_moves.empty()) {
        return policy_probs; 
    }
    if (policy_logits.device().type() != torch::kCPU) {
         std::cerr << "Warning: process_policy received logits not on CPU." << std::endl;
    }
     if (policy_logits.dim() != 1 || policy_logits.size(0) != 4096) {
        std::string shape_str = "[";
        for(int64_t s : policy_logits.sizes()) { shape_str += std::to_string(s) + ","; }
        if (shape_str.length() > 1) shape_str.pop_back(); 
        shape_str += "]";
        throw std::runtime_error("process_policy expects logits shape [4096], got shape " + shape_str);
     }

    torch::Tensor masked_logits = torch::full_like(policy_logits, -std::numeric_limits<float>::infinity());
    auto logits_accessor = policy_logits.accessor<float, 1>();
    auto masked_logits_accessor = masked_logits.accessor<float, 1>();
    std::vector<int> legal_indices;
    legal_indices.reserve(legal_moves.size());

    for (const auto& move : legal_moves) {
        int index = move_to_policy_index(move);
        if (index >= 0 && index < 4096) { 
            masked_logits_accessor[index] = logits_accessor[index];
            legal_indices.push_back(index); 
        } else {
             std::cerr << "Error: move_to_policy_index returned out-of-bounds index: " << index << std::endl;
        }
    }

    torch::Tensor probs_tensor = torch::softmax(masked_logits, /*dim=*/0);
    auto probs_accessor = probs_tensor.accessor<float, 1>();

    for (size_t i = 0; i < legal_moves.size(); ++i) {
        int index = legal_indices[i]; 
         if (index >= 0 && index < 4096) { 
            policy_probs[legal_moves[i]] = static_cast<double>(probs_accessor[index]);
         }
    }
    return policy_probs;
}

void backpropagate_mcts_value(const std::vector<MCTSNode*>& path, const std::array<double, 4>& leaf_values_for_players) { // MODIFIED
    // The leaf_values_for_players are the direct outcomes/predictions for each of the 4 players.
    // This same vector of values is propagated up the tree.
    // No sign flipping is needed because MCTSNode::update_stats now stores an array of 4 values,
    // and UCT selection uses the component relevant to the decision-making player.
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        MCTSNode* node = *it;
        node->update_stats(leaf_values_for_players); // Pass the full 4-value array
        // REMOVED: current_value_for_node_player *= -1.0;
    }
}

void evaluate_and_expand_batch_sync(
  std::vector<SimulationState>& pending_eval,
  ChaturajiNN& network,
  torch::Device device)
{
  if (pending_eval.empty()) return;

  int64_t batch_size = pending_eval.size();
  int64_t feature_size = 33 * 8 * 8;

  torch::Tensor batch_tensor = torch::empty({batch_size, 33, 8, 8}, torch::kFloat);
  float* batch_ptr = batch_tensor.data_ptr<float>();

  for (int64_t i = 0; i < batch_size; ++i) {
      std::vector<float> floats = board_to_floats(pending_eval[i].current_node->get_board());
      std::memcpy(batch_ptr + (i * feature_size), floats.data(), feature_size * sizeof(float));
  }
  
  batch_tensor = batch_tensor.to(device);

  torch::Tensor policy_logits_batch, value_pred_batch;
  {
      torch::NoGradGuard no_grad;
      network->eval();
      std::tie(policy_logits_batch, value_pred_batch) = network->forward(batch_tensor);
  }

  policy_logits_batch = policy_logits_batch.to(torch::kCPU);
  value_pred_batch = value_pred_batch.to(torch::kCPU); // value_pred_batch is [B, 4]

  auto value_accessor = value_pred_batch.accessor<float, 2>(); // Access as [Batch, NumPlayers]

  for (int i = 0; i < batch_size; ++i) {
      const SimulationState& sim_state = pending_eval[i]; 
      MCTSNode* leaf_node = sim_state.current_node;
      const std::vector<MCTSNode*>& path = sim_state.path; 

      if (!leaf_node) { 
          std::cerr << "Error: Nullptr leaf_node found in pending evaluation batch." << std::endl;
          continue;
      }

      torch::Tensor policy_logits_single = policy_logits_batch[i];
      std::map<Move, double> policy_probs = process_policy(policy_logits_single, leaf_node->get_board());

      if (leaf_node->is_leaf() && !leaf_node->get_board().is_game_over()) {
           if (!policy_probs.empty()) {
                leaf_node->expand(policy_probs);
           } else {
                std::cerr << "Warning (Sync MCTS): Empty policy from NN for non-terminal leaf node during batch processing." << std::endl;
           }
      } 
      
      // MODIFIED: Extract the 4 player values
      std::array<double, 4> player_values_from_nn;
      for(int p_idx = 0; p_idx < 4; ++p_idx) {
          player_values_from_nn[p_idx] = static_cast<double>(value_accessor[i][p_idx]);
      }
      backpropagate_mcts_value(path, player_values_from_nn); // Pass the array of 4 values
  }
  pending_eval.clear();
}

void run_mcts_simulations_sync( 
  MCTSNode& root,
  ChaturajiNN& network,
  int simulations,
  torch::Device device,
  double c_puct,
  int batch_size) 
{
  if (simulations == 0 && root.is_leaf() && !root.get_board().is_game_over()) {
      std::vector<SimulationState> initial_eval;
      SimulationState root_state;
      root_state.current_node = &root;
      root_state.path.push_back(&root);
      initial_eval.push_back(std::move(root_state));
      // std::cout << "Info (Sync MCTS): simulations=0, evaluating root node directly for policy." << std::endl;
      evaluate_and_expand_batch_sync(initial_eval, network, device);
      return; 
  }

  std::vector<SimulationState> pending_evaluation;
  pending_evaluation.reserve(batch_size);
  // Player root_player = root.get_board().get_current_player(); // Not strictly needed here anymore for value perspective

  for (int i = 0; i < simulations; ++i) {
      SimulationState current_sim;
      current_sim.current_node = &root;
      current_sim.path.push_back(current_sim.current_node);

      while (!current_sim.current_node->is_leaf()) {
           MCTSNode* next_node = current_sim.current_node->select_child(c_puct);
          if (next_node == nullptr || next_node == current_sim.current_node) {
                 std::cerr << "Warning: MCTS sync select_child failed or didn't advance."
                           << " Parent visits: " << current_sim.current_node->get_visit_count()
                           << ", Children: " << current_sim.current_node->get_children().size()
                           << ", IsGameOver: " << current_sim.current_node->get_board().is_game_over() << std::endl;
                 if (current_sim.current_node->get_board().is_game_over()){
                    MCTSNode* terminal_leaf = current_sim.current_node; 
                    std::map<Player, int> final_scores_map = terminal_leaf->get_board().get_game_result();
                    std::map<Player, double> reward_map = get_reward_map(final_scores_map);
                    std::array<double, 4> terminal_player_values = convert_reward_map_to_array(reward_map); // Convert to array
                    backpropagate_mcts_value(current_sim.path, terminal_player_values);
                 } else {
                     std::array<double, 4> neutral_values = {0.0, 0.0, 0.0, 0.0};
                     backpropagate_mcts_value(current_sim.path, neutral_values);
                 }
                 goto next_simulation_sync; 
          }
          current_sim.current_node = next_node;
          current_sim.path.push_back(current_sim.current_node);
      } 

      if (current_sim.current_node->get_board().is_game_over()) {
          MCTSNode* terminal_leaf = current_sim.current_node;
          std::map<Player, int> final_scores_map = terminal_leaf->get_board().get_game_result();
          std::map<Player, double> reward_map = get_reward_map(final_scores_map);
          std::array<double, 4> terminal_player_values = convert_reward_map_to_array(reward_map); // Convert to array
          backpropagate_mcts_value(current_sim.path, terminal_player_values);
      } else {
          pending_evaluation.push_back(std::move(current_sim)); 
          if (pending_evaluation.size() >= static_cast<size_t>(batch_size)) {
              evaluate_and_expand_batch_sync(pending_evaluation, network, device);
          }
      }
      next_simulation_sync:; 
  } 
  evaluate_and_expand_batch_sync(pending_evaluation, network, device);
}


std::optional<Move> get_best_move_mcts_sync( 
    const Board& board,
    ChaturajiNN& network,
    int simulations,
    torch::Device device,
    std::shared_ptr<MCTSNode>& current_mcts_root_shptr, // MODIFIED for tree reuse
    double c_puct,
    int mcts_batch_size) 
{
    if (board.is_game_over()) {
      current_mcts_root_shptr = nullptr; // Clear tree if game is over
      return std::nullopt;
    }

    // Tree Reuse Logic: Check if the passed-in root matches the current board state
    if (current_mcts_root_shptr && current_mcts_root_shptr->get_board().get_position_key() == board.get_position_key()) {
        // std::cout << "Info (get_best_move): Reusing existing MCTS root node." << std::endl;
    } else {
        // std::cout << "Info (get_best_move): Creating new MCTS root node for board key " << board.get_position_key() << std::endl;
        current_mcts_root_shptr = std::make_shared<MCTSNode>(board);
    }
    
    network->eval(); // Ensure network is in eval mode (might be redundant if always called before)
    run_mcts_simulations_sync(*current_mcts_root_shptr, network, simulations, device, c_puct, mcts_batch_size); 

    const auto& children_const_ref = current_mcts_root_shptr->get_children(); // Const reference for selection
    if (children_const_ref.empty()) {
        auto legal_moves = board.get_pseudo_legal_moves(board.get_current_player());
        if (legal_moves.empty()) {
            std::cerr << "Warning (get_best_move): Root has no children and no legal moves. Returning nullopt." << std::endl;
            current_mcts_root_shptr = nullptr; // No future tree
            return std::nullopt; 
        } else {
            std::cerr << "Warning (get_best_move): Root has no children despite legal moves existing (Sims=" << simulations << "). Returning first legal move as fallback." << std::endl;
            current_mcts_root_shptr = nullptr; // No tree to reuse based on this fallback
            return legal_moves[0];
        }
    }

    // Select best child by visit count
    auto best_child_by_visit_it = std::max_element(children_const_ref.begin(), children_const_ref.end(),
        [](const std::unique_ptr<MCTSNode>& a, const std::unique_ptr<MCTSNode>& b) {
            return a->get_visit_count() < b->get_visit_count();
        });

    MCTSNode* chosen_child_raw_ptr = nullptr;
    if (best_child_by_visit_it != children_const_ref.end() && (*best_child_by_visit_it)->get_visit_count() > 0) {
        chosen_child_raw_ptr = (*best_child_by_visit_it).get();
    } else {
        // Fallback: if all children have zero visits, pick by prior
        std::cerr << "Warning (get_best_move): All child nodes have zero visits (Sims=" << simulations << "). Using prior probabilities from policy." << std::endl;
        auto best_child_by_prior_it = std::max_element(children_const_ref.begin(), children_const_ref.end(),
            [](const std::unique_ptr<MCTSNode>& a, const std::unique_ptr<MCTSNode>& b) {
                return a->get_prior() < b->get_prior(); 
            });
        if (best_child_by_prior_it != children_const_ref.end()) {
            chosen_child_raw_ptr = (*best_child_by_prior_it).get();
        }
    }

    if (chosen_child_raw_ptr && chosen_child_raw_ptr->get_move()) {
        std::optional<Move> chosen_move = chosen_child_raw_ptr->get_move();
        
        // Transfer ownership of the chosen child to become the new root
        auto& old_root_children_vec_for_reuse = current_mcts_root_shptr->get_children_for_reuse(); // Non-const access
        std::unique_ptr<MCTSNode> new_root_candidate_uptr;

        for (auto it = old_root_children_vec_for_reuse.begin(); it != old_root_children_vec_for_reuse.end(); ++it) {
            if (it->get() == chosen_child_raw_ptr) {
                new_root_candidate_uptr = std::move(*it); // Steal the unique_ptr
                old_root_children_vec_for_reuse.erase(it); // Remove the now-empty slot
                break;
            }
        }

        if (new_root_candidate_uptr) {
            new_root_candidate_uptr->set_parent(nullptr); // Detach from old parent
            current_mcts_root_shptr = std::move(new_root_candidate_uptr); // Update the shared_ptr to point to the new root
        } else {
            std::cerr << "Error (get_best_move): Failed to extract chosen child unique_ptr for tree reuse. Resetting tree." << std::endl;
            current_mcts_root_shptr = nullptr; // Fallback: reset tree
        }
        return chosen_move;

    } else {
        std::cerr << "Error (get_best_move): Could not determine a valid best child. Resetting tree." << std::endl;
        current_mcts_root_shptr = nullptr; // Cannot reuse, reset for next turn.
        // Fallback to first legal move if any children existed, or nullopt
        if (!children_const_ref.empty() && children_const_ref[0]->get_move()) {
            return children_const_ref[0]->get_move();
        }
        return std::nullopt;
    }
}


std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores) {
    std::vector<std::pair<Player, int>> sorted_scores;
    for (int p_idx = 0; p_idx < 4; ++p_idx) {
        Player p = static_cast<Player>(p_idx);
        int score = final_scores.count(p) ? final_scores.at(p) : 0; 
        sorted_scores.emplace_back(p, score);
    }

    std::sort(sorted_scores.begin(), sorted_scores.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second; 
              });

    std::map<Player, double> reward_map;
    // Standard rank-based rewards
    double rewards[] = {+1.0, +0.25, -0.25, -1.0}; // Rank 1, 2, 3, 4

    // Handle ties by averaging rewards for tied ranks
    size_t i = 0;
    while (i < sorted_scores.size()) {
        size_t j = i;
        // Find all players tied at current rank
        while (j < sorted_scores.size() && sorted_scores[j].second == sorted_scores[i].second) {
            j++;
        }
        // Players from index i to j-1 are tied.
        // Calculate average reward for these ranks.
        double sum_rewards_for_tied_ranks = 0.0;
        for (size_t k = i; k < j; ++k) {
            sum_rewards_for_tied_ranks += rewards[k]; // Use rank index k for reward array
        }
        double avg_reward = sum_rewards_for_tied_ranks / (j - i);

        // Assign this average reward to all tied players
        for (size_t k = i; k < j; ++k) {
            reward_map[sorted_scores[k].first] = avg_reward;
        }
        i = j; // Move to the next distinct rank
    }
    return reward_map;
}

} // namespace chaturaji_cpp