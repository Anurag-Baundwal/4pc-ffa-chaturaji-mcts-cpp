#include "search.h"
#include "mcts_node.h" 
#include "utils.h"     
#include <vector>
#include <algorithm> 
#include <map>
#include <limits>
#include <iostream> 
#include <memory> 
#include <cmath>

namespace chaturaji_cpp {

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

std::map<Move, double> process_policy(const std::array<float, NN_POLICY_SIZE>& policy_logits, const Board& board) {
    std::map<Move, double> policy_probs;
    std::vector<Move> legal_moves = board.get_pseudo_legal_moves(board.get_current_player());

    if (legal_moves.empty()) {
        return policy_probs;
    }

    // 1. Gather logits for legal moves only
    std::vector<float> legal_logits;
    legal_logits.reserve(legal_moves.size());
    std::vector<Move> valid_moves;
    valid_moves.reserve(legal_moves.size());

    float max_logit = -std::numeric_limits<float>::infinity();

    for (const auto& move : legal_moves) {
        int index = move_to_policy_index(move);
        if (index >= 0 && index < NN_POLICY_SIZE) {
            float logit = policy_logits[index];
            legal_logits.push_back(logit);
            valid_moves.push_back(move);
            if (logit > max_logit) {
                max_logit = logit;
            }
        }
    }

    if (legal_logits.empty()) return policy_probs;

    // 2. Compute Softmax manually
    float sum_exp = 0.0f;
    for (float& val : legal_logits) {
        val = std::exp(val - max_logit); // Subtract max for stability
        sum_exp += val;
    }

    // 3. Normalize and populate map
    for (size_t i = 0; i < valid_moves.size(); ++i) {
        if (sum_exp > 0.0f) {
            policy_probs[valid_moves[i]] = static_cast<double>(legal_logits[i] / sum_exp);
        } else {
            policy_probs[valid_moves[i]] = 1.0 / valid_moves.size();
        }
    }

    return policy_probs;
}

void backpropagate_mcts_value(const std::vector<MCTSNode*>& path, const std::array<double, 4>& leaf_values_for_players) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        MCTSNode* node = *it;
        node->update_stats(leaf_values_for_players);
    }
}

void evaluate_and_expand_batch_sync(
  std::vector<SimulationState>& pending_eval,
  Model* network)
{
  if (pending_eval.empty()) return;

  // 1. Prepare Requests
  std::vector<EvaluationRequest> requests;
  requests.reserve(pending_eval.size());

  for (size_t i = 0; i < pending_eval.size(); ++i) {
      EvaluationRequest req;
      req.request_id = static_cast<RequestId>(i); // Use index as ID for sync correlation
      req.state_floats = board_to_floats(pending_eval[i].current_node->get_board());
      requests.push_back(std::move(req));
  }

  // 2. Run Inference (Synchronous)
  std::vector<EvaluationResult> results = network->evaluate_batch(requests);

  // 3. Process Results
  for (const auto& result : results) {
      // Because we used index as request_id and it's synchronous, we can map back directly.
      // However, robustness check is good.
      size_t idx = static_cast<size_t>(result.request_id);
      if (idx >= pending_eval.size()) continue;

      const SimulationState& sim_state = pending_eval[idx];
      MCTSNode* leaf_node = sim_state.current_node;
      const std::vector<MCTSNode*>& path = sim_state.path;

      if (!leaf_node) continue;

      std::map<Move, double> policy_probs = process_policy(result.policy_logits, leaf_node->get_board());

      if (leaf_node->is_leaf() && !leaf_node->get_board().is_game_over()) {
           if (!policy_probs.empty()) {
                leaf_node->expand(policy_probs);
           }
      } 
      
      // --- Un-rotate the values ---
      // The NN returns values [Relative0, Relative1, Relative2, Relative3]
      // where 0 is "Current Player". We need to map this back to [Red, Blue, Yellow, Green]
      std::array<double, 4> player_values_absolute;
      Player cp = leaf_node->get_board().get_current_player();
      int cp_idx = static_cast<int>(cp);

      for(int rel_i = 0; rel_i < 4; ++rel_i) {
          int abs_p_idx = (cp_idx + rel_i) % 4;
          player_values_absolute[abs_p_idx] = static_cast<double>(result.value[rel_i]);
      }

      backpropagate_mcts_value(path, player_values_absolute);
  }
  pending_eval.clear();
}

void run_mcts_simulations_sync( 
  MCTSNode& root,
  Model* network,
  int simulations,
  double c_puct,
  int batch_size) 
{
  if (simulations == 0 && root.is_leaf() && !root.get_board().is_game_over()) {
      std::vector<SimulationState> initial_eval;
      SimulationState root_state;
      root_state.current_node = &root;
      root_state.path.push_back(&root);
      initial_eval.push_back(std::move(root_state));
      evaluate_and_expand_batch_sync(initial_eval, network);
      return; 
  }

  std::vector<SimulationState> pending_evaluation;
  pending_evaluation.reserve(batch_size);

  for (int i = 0; i < simulations; ++i) {
      SimulationState current_sim;
      current_sim.current_node = &root;
      current_sim.path.push_back(current_sim.current_node);

      while (!current_sim.current_node->is_leaf()) {
           MCTSNode* next_node = current_sim.current_node->select_child(c_puct);
          if (next_node == nullptr || next_node == current_sim.current_node) {
                 if (current_sim.current_node->get_board().is_game_over()){
                    MCTSNode* terminal_leaf = current_sim.current_node; 
                    std::map<Player, int> final_scores_map = terminal_leaf->get_board().get_game_result();
                    std::map<Player, double> reward_map = get_reward_map(final_scores_map);
                    std::array<double, 4> terminal_player_values = convert_reward_map_to_array(reward_map);
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
          std::array<double, 4> terminal_player_values = convert_reward_map_to_array(reward_map);
          backpropagate_mcts_value(current_sim.path, terminal_player_values);
      } else {
          pending_evaluation.push_back(std::move(current_sim)); 
          if (pending_evaluation.size() >= static_cast<size_t>(batch_size)) {
              evaluate_and_expand_batch_sync(pending_evaluation, network);
          }
      }
      next_simulation_sync:; 
  } 
  evaluate_and_expand_batch_sync(pending_evaluation, network);
}


std::optional<Move> get_best_move_mcts_sync( 
    const Board& board,
    Model* network,
    int simulations,
    std::shared_ptr<MCTSNode>& current_mcts_root_shptr,
    double c_puct,
    int mcts_batch_size) 
{
    if (board.is_game_over()) {
      current_mcts_root_shptr = nullptr;
      return std::nullopt;
    }

    if (current_mcts_root_shptr && current_mcts_root_shptr->get_board().get_position_key() == board.get_position_key()) {
        // Reuse
    } else {
        current_mcts_root_shptr = std::make_shared<MCTSNode>(board);
    }
    
    // ONNX models don't need explicit .eval() mode setting like Libtorch
    run_mcts_simulations_sync(*current_mcts_root_shptr, network, simulations, c_puct, mcts_batch_size); 

    const auto& children_const_ref = current_mcts_root_shptr->get_children();
    if (children_const_ref.empty()) {
        auto legal_moves = board.get_pseudo_legal_moves(board.get_current_player());
        if (legal_moves.empty()) {
            current_mcts_root_shptr = nullptr;
            return std::nullopt; 
        } else {
            std::cerr << "Warning (get_best_move): Root has no children. Returning first legal move." << std::endl;
            current_mcts_root_shptr = nullptr;
            return legal_moves[0];
        }
    }

    auto best_child_by_visit_it = std::max_element(children_const_ref.begin(), children_const_ref.end(),
        [](const std::unique_ptr<MCTSNode>& a, const std::unique_ptr<MCTSNode>& b) {
            return a->get_visit_count() < b->get_visit_count();
        });

    MCTSNode* chosen_child_raw_ptr = nullptr;
    if (best_child_by_visit_it != children_const_ref.end() && (*best_child_by_visit_it)->get_visit_count() > 0) {
        chosen_child_raw_ptr = (*best_child_by_visit_it).get();
    } else {
        std::cerr << "Warning (get_best_move): All child nodes have zero visits. Using prior." << std::endl;
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
        
        auto& old_root_children_vec_for_reuse = current_mcts_root_shptr->get_children_for_reuse();
        std::unique_ptr<MCTSNode> new_root_candidate_uptr;

        for (auto it = old_root_children_vec_for_reuse.begin(); it != old_root_children_vec_for_reuse.end(); ++it) {
            if (it->get() == chosen_child_raw_ptr) {
                new_root_candidate_uptr = std::move(*it);
                old_root_children_vec_for_reuse.erase(it);
                break;
            }
        }

        if (new_root_candidate_uptr) {
            new_root_candidate_uptr->set_parent(nullptr);
            current_mcts_root_shptr = std::move(new_root_candidate_uptr);
        } else {
            current_mcts_root_shptr = nullptr;
        }
        return chosen_move;

    } else {
        current_mcts_root_shptr = nullptr;
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
    double rewards[] = {+1.0, +0.25, -0.25, -1.0}; // Rank 1, 2, 3, 4

    size_t i = 0;
    while (i < sorted_scores.size()) {
        size_t j = i;
        while (j < sorted_scores.size() && sorted_scores[j].second == sorted_scores[i].second) {
            j++;
        }
        double sum_rewards_for_tied_ranks = 0.0;
        for (size_t k = i; k < j; ++k) {
            sum_rewards_for_tied_ranks += rewards[k];
        }
        double avg_reward = sum_rewards_for_tied_ranks / (j - i);

        for (size_t k = i; k < j; ++k) {
            reward_map[sorted_scores[k].first] = avg_reward;
        }
        i = j;
    }
    return reward_map;
}

} // namespace chaturaji_cpp