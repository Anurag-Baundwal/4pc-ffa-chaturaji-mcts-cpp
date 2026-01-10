// search.cpp
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
        player_rewards[i] = (it != reward_map.end()) ? it->second : default_value;
    }
    return player_rewards;
}

std::map<Move, double> process_policy(const std::array<float, NN_POLICY_SIZE>& policy_logits, const Board& board) {
    std::map<Move, double> policy_probs;
    std::vector<Move> legal_moves = board.get_pseudo_legal_moves(board.get_current_player());
    if (legal_moves.empty()) return policy_probs;

    std::vector<float> legal_logits;
    std::vector<Move> valid_moves;
    float max_logit = -std::numeric_limits<float>::infinity();

    for (const auto& move : legal_moves) {
        int index = move_to_policy_index(move, board.get_current_player());
        if (index >= 0 && index < NN_POLICY_SIZE) {
            float logit = policy_logits[index];
            legal_logits.push_back(logit);
            valid_moves.push_back(move);
            if (logit > max_logit) max_logit = logit;
        }
    }

    if (legal_logits.empty()) return policy_probs;

    float sum_exp = 0.0f;
    const float policy_temperature = 1.36f; 
    for (float& val : legal_logits) {
        val = std::exp((val - max_logit) / policy_temperature);
        sum_exp += val;
    }

    for (size_t i = 0; i < valid_moves.size(); ++i) {
        policy_probs[valid_moves[i]] = (sum_exp > 0.0f) ? static_cast<double>(legal_logits[i] / sum_exp) : (1.0 / valid_moves.size());
    }
    return policy_probs;
}

void backpropagate_mcts_value(const std::vector<MCTSNode*>& path, const std::array<double, 4>& leaf_values_for_players) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) (*it)->update_stats(leaf_values_for_players);
}

void evaluate_and_expand_batch_sync(std::vector<SimulationState>& pending_eval, Model* network) {
  if (pending_eval.empty()) return;
  std::vector<EvaluationRequest> requests;
  for (size_t i = 0; i < pending_eval.size(); ++i) {
      EvaluationRequest req;
      req.request_id = static_cast<RequestId>(i);
      req.state_floats = board_to_floats(pending_eval[i].current_node->get_board());
      requests.push_back(std::move(req));
  }
  std::vector<EvaluationResult> results = network->evaluate_batch(requests);
  for (const auto& result : results) {
      size_t idx = static_cast<size_t>(result.request_id);
      MCTSNode* leaf_node = pending_eval[idx].current_node;
      std::map<Move, double> policy_probs = process_policy(result.policy_logits, leaf_node->get_board());
      if (leaf_node->is_leaf() && !leaf_node->get_board().is_game_over() && !policy_probs.empty()) leaf_node->expand(policy_probs);
      std::array<double, 4> player_values_absolute;
      int cp_idx = static_cast<int>(leaf_node->get_board().get_current_player());
      for(int rel_i = 0; rel_i < 4; ++rel_i) player_values_absolute[(cp_idx + rel_i) % 4] = static_cast<double>(result.value[rel_i]);
      backpropagate_mcts_value(pending_eval[idx].path, player_values_absolute);
  }
  pending_eval.clear();
}

void run_mcts_simulations_sync(MCTSNode& root, Model* network, int simulations, double c_puct, int batch_size, Player root_player, double spite_weight) {
  if (simulations == 0 && root.is_leaf() && !root.get_board().is_game_over()) {
      std::vector<SimulationState> initial_eval;
      SimulationState root_state; root_state.current_node = &root; root_state.path.push_back(&root);
      initial_eval.push_back(std::move(root_state));
      evaluate_and_expand_batch_sync(initial_eval, network);
      return; 
  }

  std::vector<SimulationState> pending_evaluation;
  for (int i = 0; i < simulations; ++i) {
      SimulationState current_sim;
      current_sim.current_node = &root;
      current_sim.path.push_back(current_sim.current_node);

      while (!current_sim.current_node->is_leaf()) {
           // Pass the root player and spite weight to selection
           MCTSNode* next_node = current_sim.current_node->select_child(c_puct, root_player, spite_weight);
           if (!next_node || next_node == current_sim.current_node) {
                 if (current_sim.current_node->get_board().is_game_over()){
                    std::array<double, 4> term_vals = convert_reward_map_to_array(get_reward_map(current_sim.current_node->get_board().get_game_result()));
                    backpropagate_mcts_value(current_sim.path, term_vals);
                 } else backpropagate_mcts_value(current_sim.path, {0,0,0,0});
                 goto next_simulation_sync; 
          }
          current_sim.current_node = next_node;
          current_sim.path.push_back(current_sim.current_node);
      } 

      if (current_sim.current_node->get_board().is_game_over()) {
          std::array<double, 4> term_vals = convert_reward_map_to_array(get_reward_map(current_sim.current_node->get_board().get_game_result()));
          backpropagate_mcts_value(current_sim.path, term_vals);
      } else {
          pending_evaluation.push_back(std::move(current_sim)); 
          if (pending_evaluation.size() >= static_cast<size_t>(batch_size)) evaluate_and_expand_batch_sync(pending_evaluation, network);
      }
      next_simulation_sync:; 
  } 
  evaluate_and_expand_batch_sync(pending_evaluation, network);
}

std::optional<Move> get_best_move_mcts_sync(const Board& board, Model* network, int simulations, std::shared_ptr<MCTSNode>& current_mcts_root_shptr, double c_puct, int mcts_batch_size, bool verbose, double spite_weight) {
    if (board.is_game_over()) return std::nullopt;

    if (!current_mcts_root_shptr || current_mcts_root_shptr->get_board().get_position_key() != board.get_position_key()) {
        current_mcts_root_shptr = std::make_shared<MCTSNode>(board);
    }
    
    // --- DYNAMIC SPITE CALCULATION ---
    Player root_player = board.get_current_player();
    int num_eliminated = 4 - static_cast<int>(board.get_active_players().size());
    double effective_spite = spite_weight;
    for (int i = 0; i < num_eliminated; ++i) effective_spite *= 0.66;
    
    if (verbose && spite_weight > 0.0) {
        std::cout << "info string Active Players: " << board.get_active_players().size() 
                  << " | Effective Spite: " << std::fixed << std::setprecision(3) << effective_spite << std::endl;
    }

    run_mcts_simulations_sync(*current_mcts_root_shptr, network, simulations, c_puct, mcts_batch_size, root_player, effective_spite); 

    if (verbose) {
        std::cout << "--- Search Statistics ---" << std::endl;
        int root_visits = current_mcts_root_shptr->get_visit_count();
        auto root_values = current_mcts_root_shptr->get_total_player_values();
        std::cout << "Root Expected Rewards (+1.0 to -1.0):" << std::endl;
        const char* player_colors[] = {"RED", "BLUE", "YELLOW", "GREEN"};
        const char* color_codes[] = {"\033[31m", "\033[34m", "\033[33m", "\033[32m"}; 
        for (int i = 0; i < 4; ++i) {
            std::cout << "  " << color_codes[i] << std::left << std::setw(7) << player_colors[i] 
                      << "\033[0m: " << std::showpos << std::fixed << std::setprecision(4) 
                      << (root_values[i] / std::max(1, root_visits)) << "  ";
        }
        std::cout << std::endl;
        std::vector<MCTSNode*> children_ptrs;
        for(const auto& c : current_mcts_root_shptr->get_children()) if(c->get_visit_count() > 0) children_ptrs.push_back(c.get());
        std::sort(children_ptrs.begin(), children_ptrs.end(), [](MCTSNode* a, MCTSNode* b){ return a->get_visit_count() > b->get_visit_count(); });
        std::cout << "Top Candidate Moves:" << std::endl;
        int count = 0;
        Player cp = board.get_current_player();
        for (auto* child : children_ptrs) {
            if(count++ >= 8) break;
            int visits = child->get_visit_count();
            auto c_values = child->get_total_player_values();
            std::cout << std::right << std::setw(3) << count << " | " << std::left << std::setw(8) << get_uci_string(*child->get_move())
                      << " | V: " << std::right << std::setw(6) << visits << " | P: " << std::fixed << std::setprecision(3) << child->get_prior()
                      << " | Us: " << std::showpos << std::setw(6) << (c_values[static_cast<int>(cp)] / std::max(1, visits)) << " | [";
            for(int i=0; i<4; ++i) std::cout << std::fixed << std::setprecision(2) << (c_values[i]/std::max(1, visits)) << (i<3?" ":"");
            std::cout << "]" << std::endl;
        }
    }

    const auto& children = current_mcts_root_shptr->get_children();
    if (children.empty()) return std::nullopt;

    auto best_child_it = std::max_element(children.begin(), children.end(), [](const auto& a, const auto& b) { return a->get_visit_count() < b->get_visit_count(); });
    MCTSNode* chosen = best_child_it->get();

    if (chosen && chosen->get_move()) {
        std::optional<Move> chosen_move = chosen->get_move();
        auto& old_vec = current_mcts_root_shptr->get_children_for_reuse();
        std::unique_ptr<MCTSNode> new_root;
        for (auto it = old_vec.begin(); it != old_vec.end(); ++it) {
            if (it->get() == chosen) { new_root = std::move(*it); old_vec.erase(it); break; }
        }
        if (new_root) { new_root->set_parent(nullptr); current_mcts_root_shptr = std::move(new_root); }
        else current_mcts_root_shptr = nullptr;
        return chosen_move;
    }
    return std::nullopt;
}

std::map<Player, double> get_reward_map(const std::map<Player, int>& final_scores) {
    std::vector<std::pair<Player, int>> sorted_scores;
    for (int p_idx = 0; p_idx < 4; ++p_idx) sorted_scores.emplace_back(static_cast<Player>(p_idx), final_scores.count(static_cast<Player>(p_idx)) ? final_scores.at(static_cast<Player>(p_idx)) : 0);
    std::sort(sorted_scores.begin(), sorted_scores.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::map<Player, double> reward_map;
    double rewards[] = {+1.0, +0.333, -0.333, -1.0};
    size_t i = 0;
    while (i < sorted_scores.size()) {
        size_t j = i;
        while (j < sorted_scores.size() && sorted_scores[j].second == sorted_scores[i].second) j++;
        double sum = 0.0;
        for (size_t k = i; k < j; ++k) sum += rewards[k];
        for (size_t k = i; k < j; ++k) reward_map[sorted_scores[k].first] = sum / (j - i);
        i = j;
    }
    return reward_map;
}

} // namespace chaturaji_cpp