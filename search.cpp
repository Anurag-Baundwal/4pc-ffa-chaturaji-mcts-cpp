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

void apply_value_softmax(std::array<float, 16>& logits) {
    for (int p = 0; p < 4; ++p) {
        // Find max for numerical stability
        float max_l = -std::numeric_limits<float>::infinity();
        for (int r = 0; r < 4; ++r) {
            max_l = std::max(max_l, logits[p * 4 + r]);
        }
        
        float sum = 0.0f;
        for (int r = 0; r < 4; ++r) {
            logits[p * 4 + r] = std::exp(logits[p * 4 + r] - max_l);
            sum += logits[p * 4 + r];
        }
        
        for (int r = 0; r < 4; ++r) {
            logits[p * 4 + r] /= (sum + 1e-9f);
        }
    }
}

/**
 * @brief Backpropagates a vector of player-specific values up the MCTS path.
 * @param path The path from root to leaf (inclusive, leaf is at path.back()).
 * @param leaf_values_for_players The array of 16 values of the leaf state,
 *                                  for each of the 4 players (RED, BLUE, YELLOW, GREEN) 
 *                                  and each of the 4 ranks.
 */
void backpropagate_mcts_value(const std::vector<MCTSNode*>& path, const std::array<double, 16>& leaf_values_for_players) {
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

      req.input_planes = TensorPool::acquire_planes();
      req.input_scalars = TensorPool::acquire_scalars();

      board_to_tensors(pending_eval[i].current_node->get_board(), req.input_planes->data(), req.input_scalars->data());
      requests.push_back(std::move(req));
  }

  // 2. Run Inference (Synchronous)
  std::vector<EvaluationResult> results = network->evaluate_batch(requests);

  // 3. Process Results
  for (auto& result : results) {
      size_t idx = static_cast<size_t>(result.request_id);
      if (idx >= pending_eval.size()) continue;

      const SimulationState& sim_state = pending_eval[idx];
      MCTSNode* leaf_node = sim_state.current_node;
      const std::vector<MCTSNode*>& path = sim_state.path;

      if (!leaf_node) continue;

      std::map<Move, double> policy_probs = process_policy(*(result.policy_logits), leaf_node->get_board());

      if (leaf_node->is_leaf() && !leaf_node->get_board().is_game_over()) {
           if (!policy_probs.empty()) {
                leaf_node->expand(policy_probs);
           }
      } 
      
      // 1. Get the raw values from the result
      std::array<float, 16> leaf_logits = *result.value;
      
      // 2. APPLY SOFTMAX to convert logits to probabilities
      apply_value_softmax(leaf_logits);

      // 3. Un-rotate the probabilities (Relative -> Absolute)
      std::array<double, 16> player_values_absolute;
      player_values_absolute.fill(0.0);
      Player cp = leaf_node->get_board().get_current_player();
      int cp_idx = static_cast<int>(cp);

      for(int rel_p = 0; rel_p < 4; ++rel_p) {
          int abs_p = (cp_idx + rel_p) % 4;
          for(int rank = 0; rank < 4; ++rank) {
              player_values_absolute[abs_p * 4 + rank] = static_cast<double>(leaf_logits[rel_p * 4 + rank]);
          }
      }

      backpropagate_mcts_value(path, player_values_absolute);

      // Return Output memory to the pool
      TensorPool::release_policy(std::move(result.policy_logits));
      TensorPool::release_value(std::move(result.value));
  }

  // 4. Return Input memory to the pool
  for (auto& req : requests) {
      TensorPool::release_planes(std::move(req.input_planes));
      TensorPool::release_scalars(std::move(req.input_scalars));
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
      evaluate_and_expand_batch_sync(initial_eval, network);
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
           MCTSNode* next_node = current_sim.current_node->select_child(c_puct, pessimism_factor);
          if (next_node == nullptr || next_node == current_sim.current_node) {
            if (current_sim.current_node->get_board().is_game_over()){
                auto scores = current_sim.current_node->get_board().get_game_result();
                backpropagate_mcts_value(current_sim.path, get_rank_probabilities_target(scores));
            } else {
                std::array<double, 16> neutral;
                neutral.fill(0.25); // 25% chance for each rank
                backpropagate_mcts_value(current_sim.path, neutral);
            }
                 goto next_simulation_sync; 
          }
          current_sim.current_node = next_node;
          current_sim.path.push_back(current_sim.current_node);
      } 

      if (current_sim.current_node->get_board().is_game_over()) {
          MCTSNode* terminal_leaf = current_sim.current_node;
          std::array<int, 4> final_scores = terminal_leaf->get_board().get_game_result();         
          std::array<double, 16> terminal_values = get_rank_probabilities_target(final_scores);
          backpropagate_mcts_value(current_sim.path, terminal_values);
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

        std::cout << "Root Outcome Probabilities & Expected Rewards:" << std::endl;
        const char* player_colors[] = {"RED", "BLUE", "YELLOW", "GREEN"};
        const char* color_codes[] = {"\033[31m", "\033[34m", "\033[33m", "\033[32m"}; 
        const char* reset_code = "\033[0m";

        for (int i = 0; i < 4; ++i) {
            // Scalar EV for overview
            double ev = get_expected_value(root_values, i, root_visits, pessimism_factor);
            
            // Calculate individual rank percentages (1st, 2nd, 3rd, 4th)
            double p1 = (root_values[i * 4 + 0] / std::max(1, root_visits)) * 100.0;
            double p2 = (root_values[i * 4 + 1] / std::max(1, root_visits)) * 100.0;
            double p3 = (root_values[i * 4 + 2] / std::max(1, root_visits)) * 100.0;
            double p4 = (root_values[i * 4 + 3] / std::max(1, root_visits)) * 100.0;

            std::cout << "  " << color_codes[i] << std::left << std::setw(7) << player_colors[i] << reset_code 
                      << ": EV: " << std::showpos << std::fixed << std::setprecision(3) << ev << std::noshowpos
                      << " | 1st: " << std::setw(5) << std::fixed << std::setprecision(1) << p1 << "%"
                      << " 2nd: " << std::setw(5) << p2 << "%"
                      << " 3rd: " << std::setw(5) << p3 << "%"
                      << " 4th: " << std::setw(5) << p4 << "%" << std::endl;
        }
        std::cout << std::endl;

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

        std::cout << "Top Candidate Moves (Expected Utility):" << std::endl;
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
            
            // Expected value for current player specifically
            double cp_val = get_expected_value(c_values, static_cast<int>(cp), visits, pessimism_factor);

            std::cout << std::right << std::setw(3) << count << " | "
                      << std::left << std::setw(8) << get_uci_string(m) 
                      << " | " << std::right << std::setw(6) << visits
                      << " | " << std::fixed << std::setprecision(3) << prior
                      << " | " << std::showpos << std::setw(12) << cp_val << std::noshowpos
                      << " | [";
            
            for(int i=0; i<4; ++i) {
                // Calculate EV for each player for the candidate list
                double v = get_expected_value(c_values, i, visits, pessimism_factor);
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

} // namespace chaturaji_cpp