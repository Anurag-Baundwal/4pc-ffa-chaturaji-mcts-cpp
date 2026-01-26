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

    // 1. Gather logits for legal moves only
    std::vector<float> legal_logits;
    legal_logits.reserve(legal_moves.size());
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
    for (float& val : legal_logits) {
        val = std::exp((val - max_logit) / policy_temperature); 
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

// --- Helper for Pessimism ---
void apply_pessimism(std::array<double, 4>& values, double factor) {
    if (factor == 1.0) return; // Optimization for default case
    for (int i = 0; i < 4; ++i) {
        if (values[i] < 0.0) {
            values[i] *= factor;
        }
    }
}

void evaluate_and_expand_batch_sync(
  std::vector<SimulationState>& pending_eval,
  Model* network,
  double pessimism_factor)
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
      
      // --- Un-rotate the values ---
      // The NN returns values [Relative0, Relative1, Relative2, Relative3]
      // where 0 is "Current Player". Map this back to [Red, Blue, Yellow, Green]
      std::array<double, 4> player_values_absolute;
      Player cp = leaf_node->get_board().get_current_player();
      int cp_idx = static_cast<int>(cp);

      for(int rel_i = 0; rel_i < 4; ++rel_i) {
          int abs_p_idx = (cp_idx + rel_i) % 4;
          player_values_absolute[abs_p_idx] = static_cast<double>(result.value[rel_i]);
      }

      // --- APPLY PESSIMISM TO NN OUTPUT ---
      apply_pessimism(player_values_absolute, pessimism_factor);

      backpropagate_mcts_value(path, player_values_absolute);
  }
  pending_eval.clear();
}

void run_mcts_simulations_sync( 
  MCTSNode& root,
  Model* network,
  int simulations,
  double c_puct,
  int batch_size,
  double pessimism_factor)
{
  if (simulations == 0 && root.is_leaf() && !root.get_board().is_game_over()) {
      std::vector<SimulationState> initial_eval;
      SimulationState root_state;
      root_state.current_node = &root;
      root_state.path.push_back(&root);
      initial_eval.push_back(std::move(root_state));
      evaluate_and_expand_batch_sync(initial_eval, network, pessimism_factor);
      return; 
  }

  std::vector<SimulationState> pending_evaluation;
  pending_evaluation.reserve(batch_size);

  for (int i = 0; i < simulations; ++i) {
      SimulationState current_sim;
      current_sim.current_node = &root;
      current_sim.path.push_back(current_sim.current_node);

      // Traversal using select_child
      while (!current_sim.current_node->is_leaf()) {
           MCTSNode* next_node = current_sim.current_node->select_child(c_puct);
          if (next_node == nullptr || next_node == current_sim.current_node) {
                 if (current_sim.current_node->get_board().is_game_over()){
                    MCTSNode* terminal_leaf = current_sim.current_node; 
                    std::array<int, 4> final_scores = terminal_leaf->get_board().get_game_result();
                    std::array<double, 4> terminal_player_values = get_reward_map_array(final_scores);
                    
                    // --- APPLY PESSIMISM TO TERMINAL STATE ---
                    apply_pessimism(terminal_player_values, pessimism_factor);

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
          std::array<int, 4> final_scores = terminal_leaf->get_board().get_game_result();         
          std::array<double, 4> terminal_player_values = get_reward_map_array(final_scores);
          
          // --- APPLY PESSIMISM TO TERMINAL STATE ---
          apply_pessimism(terminal_player_values, pessimism_factor);
          
          backpropagate_mcts_value(current_sim.path, terminal_player_values);
      } else {
          pending_evaluation.push_back(std::move(current_sim)); 
          if (pending_evaluation.size() >= static_cast<size_t>(batch_size)) {
              evaluate_and_expand_batch_sync(pending_evaluation, network, pessimism_factor);
          }
      }
      next_simulation_sync:; 
  } 
  evaluate_and_expand_batch_sync(pending_evaluation, network, pessimism_factor);
}


std::optional<Move> get_best_move_mcts_sync( 
    const Board& board,
    Model* network,
    int simulations,
    std::shared_ptr<MCTSNode>& current_mcts_root_shptr,
    double c_puct,
    int mcts_batch_size,
    bool verbose,
    double pessimism_factor) 
{
    if (board.is_game_over()) {
      current_mcts_root_shptr = nullptr;
      return std::nullopt;
    }

    if (current_mcts_root_shptr && current_mcts_root_shptr->get_board().get_position_key() == board.get_position_key()) {
        // Reuse existing tree
    } else {
        current_mcts_root_shptr = std::make_shared<MCTSNode>(board);
    }
    
    // Run MCTS
    run_mcts_simulations_sync(*current_mcts_root_shptr, network, simulations, c_puct, mcts_batch_size, pessimism_factor); 

    // --- VERBOSE OUTPUT ---
    if (verbose) {
        std::cout << "--- Search Statistics (Pessimism: " << pessimism_factor << "x) ---" << std::endl;
        int root_visits = current_mcts_root_shptr->get_visit_count();
        auto root_values = current_mcts_root_shptr->get_total_player_values();

        std::cout << "Root Expected Rewards (Scaled):" << std::endl;
        const char* player_colors[] = {"RED", "BLUE", "YELLOW", "GREEN"};
        const char* color_codes[] = {"\033[31m", "\033[34m", "\033[33m", "\033[32m"}; 
        const char* reset_code = "\033[0m";

        for (int i = 0; i < 4; ++i) {
            double avg_val = root_values[i] / std::max(1, root_visits);
            std::cout << "  " << color_codes[i] << std::left << std::setw(7) << player_colors[i] 
                      << reset_code << ": " << std::showpos << std::fixed << std::setprecision(4) 
                      << avg_val << std::noshowpos;
            if (i < 3) std::cout << "  ";
        }
        std::cout << std::endl << std::endl;

        // Collect children pointers
        std::vector<MCTSNode*> children_ptrs;
        MCTSNode* curr_child = current_mcts_root_shptr->get_first_child();
        while (curr_child) {
            if (curr_child->get_visit_count() > 0) {
                children_ptrs.push_back(curr_child);
            }
            curr_child = curr_child->get_next_sibling();
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
            if(count++ >= 8) break; 

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

    // --- BEST CHILD SELECTION ---
    if (current_mcts_root_shptr->is_leaf()) {
        MoveList legal_moves;
        board.get_pseudo_legal_moves(board.get_current_player(), legal_moves);
        if (legal_moves.empty()) {
            current_mcts_root_shptr = nullptr;
            return std::nullopt; 
        } else {
            if(verbose) std::cerr << "Warning: Root has no children. Returning first legal move." << std::endl;
            current_mcts_root_shptr = nullptr;
            return legal_moves[0];
        }
    }

    MCTSNode* best_child = nullptr;
    int max_visits = -1;
    double best_prior = -1.0;

    MCTSNode* curr = current_mcts_root_shptr->get_first_child();
    
    // Robust selection: Max visits, tie-break on Prior
    while (curr) {
        int v = curr->get_visit_count();
        if (v > max_visits) {
            max_visits = v;
            best_child = curr;
            best_prior = curr->get_prior();
        } else if (v == max_visits) {
             if (curr->get_prior() > best_prior) {
                 best_prior = curr->get_prior();
                 best_child = curr;
             }
        }
        curr = curr->get_next_sibling();
    }

    // --- TREE REUSE ---
    if (best_child && best_child->get_move()) {
        std::optional<Move> chosen_move = best_child->get_move();
        
        // Safety: detach_child_and_clear_others already sets parent to nullptr
        MCTSNode* new_root_raw = current_mcts_root_shptr->detach_child_and_clear_others(best_child);
        current_mcts_root_shptr.reset(new_root_raw);
        return chosen_move;
    } else {
        // MCTS failed to find a move (likely 0 simulations or empty tree)
        MoveList legal_moves;
        board.get_pseudo_legal_moves(board.get_current_player(), legal_moves);

        std::optional<Move> fallback = std::nullopt;
        if (!legal_moves.empty()) {
            // Pick the first move that is NOT a resignation
            for (const auto& m : legal_moves) {
                if (!m.is_resignation()) {
                    fallback = m;
                    break;
                }
            }
            // If only resignation is possible, then take it
            if (!fallback) fallback = legal_moves[0];
        }

        // Clean up the tree entirely since the search was inconclusive
        current_mcts_root_shptr = nullptr; 
        return fallback;
    }
}


std::array<double, 4> get_reward_map_array(const std::array<int, 4>& final_points) {
    // 1. Create pairs of (PlayerIndex, Score)
    struct PScore { int p_idx; int score; };
    std::array<PScore, 4> sorted_scores;
    
    for(int i=0; i<4; ++i) {
        sorted_scores[i] = {i, final_points[i]};
    }

    // 2. Sort by score descending
    std::sort(sorted_scores.begin(), sorted_scores.end(), 
              [](const PScore& a, const PScore& b) {
                  return a.score > b.score;
              });

    std::array<double, 4> result_rewards; 
    
    // Base rewards: 1st=+1, 2nd=+0.33, 3rd=-0.33, 4th=-1
    // Note: Pessimism is applied inside run_mcts_simulations_sync, not here
    const double rank_rewards[] = {+1.0, +0.333, -0.333, -1.0};

    size_t i = 0;
    while (i < 4) {
        size_t j = i;
        // Find end of tie group
        while (j < 4 && sorted_scores[j].score == sorted_scores[i].score) {
            j++;
        }
        
        // Calculate average reward for this group
        double sum_rewards = 0.0;
        for (size_t k = i; k < j; ++k) {
            sum_rewards += rank_rewards[k];
        }
        double avg_reward = sum_rewards / (j - i);

        // Assign to players
        for (size_t k = i; k < j; ++k) {
            int p_idx = sorted_scores[k].p_idx;
            result_rewards[p_idx] = avg_reward;
        }
        i = j;
    }

    return result_rewards;
}

} // namespace chaturaji_cpp