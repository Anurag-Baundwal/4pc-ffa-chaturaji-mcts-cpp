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
#include <iomanip>

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
    MoveList legal_moves;
    board.get_pseudo_legal_moves(board.get_current_player(), legal_moves);

    if (legal_moves.empty()) {
        return policy_probs;
    }

    // 1. Gather logits for legal moves only (Stack allocated)
    StackFloatVector legal_logits;
    MoveList valid_moves;

    float max_logit = -std::numeric_limits<float>::infinity();

    for (const auto& move : legal_moves) {
        int index = move_to_policy_index(move, board.get_current_player());
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

    // 2. Compute Softmax manually with temperature scaling
    float sum_exp = 0.0f;
    const float policy_temperature = 1.36f; 
    for (size_t i = 0; i < legal_logits.size(); ++i) {
        legal_logits[i] = std::exp((legal_logits[i] - max_logit) / policy_temperature);
        sum_exp += legal_logits[i];
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
      req.request_id = static_cast<RequestId>(i);
      board_to_floats_into(pending_eval[i].current_node->get_board(), req.state_floats);
      requests.push_back(std::move(req));
  }

  // 2. Run Inference (Synchronous)
  std::vector<EvaluationResult> results = network->evaluate_batch(requests);

  // 3. Process Results
  for (const auto& result : results) {
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
                    auto final_scores_map = terminal_leaf->get_board().get_game_result();
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
          auto final_scores_map = terminal_leaf->get_board().get_game_result();
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
    int mcts_batch_size,
    bool verbose) 
{
    if (board.is_game_over()) {
      current_mcts_root_shptr = nullptr;
      return std::nullopt;
    }

    if (current_mcts_root_shptr && current_mcts_root_shptr->get_board().get_position_key() == board.get_position_key()) {
        // Reuse existing tree if hash matches
    } else {
        current_mcts_root_shptr = std::make_shared<MCTSNode>(board);
    }
    
    // Run MCTS
    run_mcts_simulations_sync(*current_mcts_root_shptr, network, simulations, c_puct, mcts_batch_size); 

    // --- VERBOSE OUTPUT START ---
    if (verbose) {
        std::cout << "--- Search Statistics ---" << std::endl;
        int root_visits = current_mcts_root_shptr->get_visit_count();
        auto root_values = current_mcts_root_shptr->get_total_player_values();

        // Print expected rewards (Average Value)
        std::cout << "Root Expected Rewards (+1.0 to -1.0):" << std::endl;
        const char* player_colors[] = {"RED", "BLUE", "YELLOW", "GREEN"};
        const char* color_codes[] = {"\033[31m", "\033[34m", "\033[33m", "\033[32m"}; // ANSI codes
        const char* reset_code = "\033[0m";

        for (int i = 0; i < 4; ++i) {
            double avg_val = root_values[i] / std::max(1, root_visits);
            std::cout << "  " << color_codes[i] << std::left << std::setw(7) << player_colors[i] 
                      << reset_code << ": " << std::showpos << std::fixed << std::setprecision(4) 
                      << avg_val << std::noshowpos;
            if (i < 3) std::cout << "  ";
        }
        std::cout << std::endl << std::endl;

        // Gather children for sorting (Linked List Iteration)
        std::vector<MCTSNode*> children_ptrs;
        for (MCTSNode* c = current_mcts_root_shptr->get_first_child(); c; c = c->get_next_sibling()) {
            if(c->get_visit_count() > 0) children_ptrs.push_back(c);
        }

        // Sort by visits (descending)
        std::sort(children_ptrs.begin(), children_ptrs.end(), [](MCTSNode* a, MCTSNode* b){
            return a->get_visit_count() > b->get_visit_count();
        });

        std::cout << "Top Candidate Moves:" << std::endl;
        std::cout << "  # | Move     | Visits | Prior | CurPlayer Val | Full Values [R, B, Y, G]" << std::endl;
        std::cout << "-------------------------------------------------------------------------------" << std::endl;

        int count = 0;
        Player cp = board.get_current_player();

        for (auto* child : children_ptrs) {
            if(count++ >= 8) break; // Show top 8 moves

            if(!child->get_move()) continue;
            Move m = *child->get_move();
            
            int visits = child->get_visit_count();
            double prior = child->get_prior();
            auto c_values = child->get_total_player_values();
            double cp_val = c_values[static_cast<int>(cp)] / std::max(1, visits);

            std::cout << std::right << std::setw(3) << count << " | "
                      << std::left << std::setw(8) << get_uci_string(m)
                      << " | " << std::right << std::setw(6) << visits
                      << " | " << std::fixed << std::setprecision(3) << prior
                      << " | " << std::showpos << std::setw(12) << cp_val << std::noshowpos
                      << " | [";
            
            for(int i=0; i<4; ++i) {
                double v = c_values[i] / std::max(1, visits);
                std::cout << std::fixed << std::setprecision(3) << std::showpos << v << std::noshowpos;
                if(i<3) std::cout << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "-------------------------------------------------------------------------------" << std::endl;
    }
    // --- VERBOSE OUTPUT END ---

    if (!current_mcts_root_shptr->get_first_child()) {
        MoveList legal_moves;
        board.get_pseudo_legal_moves(board.get_current_player(), legal_moves);
        if (legal_moves.empty()) {
            current_mcts_root_shptr = nullptr;
            return std::nullopt; 
        } else {
            std::cerr << "Warning (get_best_move): Root has no children after search. Returning first legal move." << std::endl;
            current_mcts_root_shptr = nullptr;
            return legal_moves[0];
        }
    }

    // Select Best Child (Robust: Visits > Prior)
    // Iterate linked list manually to find max
    MCTSNode* best_child_by_visit = nullptr;
    int max_visits = -1;

    for (MCTSNode* c = current_mcts_root_shptr->get_first_child(); c; c = c->get_next_sibling()) {
        if (c->get_visit_count() > max_visits) {
            max_visits = c->get_visit_count();
            best_child_by_visit = c;
        }
    }

    MCTSNode* chosen_child_raw_ptr = nullptr;
    if (best_child_by_visit && best_child_by_visit->get_visit_count() > 0) {
        chosen_child_raw_ptr = best_child_by_visit;
    } else {
        if (verbose) std::cerr << "Warning: All child nodes have zero visits. Using prior." << std::endl;
        // Fallback to prior
        MCTSNode* best_child_by_prior = nullptr;
        double max_prior = -1.0;
        for (MCTSNode* c = current_mcts_root_shptr->get_first_child(); c; c = c->get_next_sibling()) {
            if (c->get_prior() > max_prior) {
                max_prior = c->get_prior();
                best_child_by_prior = c;
            }
        }
        chosen_child_raw_ptr = best_child_by_prior;
    }

    // Move the tree root (Tree Reuse with Linked List)
    if (chosen_child_raw_ptr && chosen_child_raw_ptr->get_move()) {
        std::optional<Move> chosen_move = chosen_child_raw_ptr->get_move();
        
        // Unlink chosen child from the list so it doesn't get deleted by old root
        MCTSNode* prev = nullptr;
        for (MCTSNode* c = current_mcts_root_shptr->get_first_child(); c; c = c->get_next_sibling()) {
            if (c == chosen_child_raw_ptr) {
                if (prev) prev->set_next_sibling(c->get_next_sibling());
                else current_mcts_root_shptr->set_first_child(c->get_next_sibling());

                c->set_next_sibling(nullptr); // Detach
                c->set_parent(nullptr);
                break;
            }
            prev = c;
        }

        // Reset shared_ptr to new root. Old root dies and deletes all other children.
        current_mcts_root_shptr.reset(chosen_child_raw_ptr);
        return chosen_move;

    } else {
        current_mcts_root_shptr = nullptr;
        // Return first child move as fallback if it exists
        MCTSNode* first = current_mcts_root_shptr ? current_mcts_root_shptr->get_first_child() : nullptr;
        if (first && first->get_move()) {
            return first->get_move();
        }
        return std::nullopt;
    }
}


std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores) {
    std::vector<std::pair<Player, int>> sorted_scores;
    for (int p_idx = 0; p_idx < 4; ++p_idx) {
        Player p = static_cast<Player>(p_idx);
        auto it = final_scores.find(p);
        int score = (it != final_scores.end()) ? it->second : 0;
        sorted_scores.emplace_back(p, score);
    }

    std::sort(sorted_scores.begin(), sorted_scores.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second; 
              });

    std::map<Player, double> reward_map;
    double rewards[] = {+1.0, +0.333, -0.333, -1.0}; // Rank 1, 2, 3, 4

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