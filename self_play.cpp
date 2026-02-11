#include "self_play.h"
#include "data_writer.h"
#include <iostream> 
#include <numeric>  
#include <cmath>    
#include <algorithm>
#include <stdexcept>
#include <vector>   
#include <future>   
#include <mutex>    
#include <memory>

namespace chaturaji_cpp {

/**
 * Internal struct to manage the state of a single game session.
 * Used to allow workers to cycle between multiple games while waiting for the GPU.
 */
struct GameSession {
    int session_id;
    Board board;
    std::unique_ptr<MCTSNode> mcts_root;
    std::vector<std::tuple<Board, std::map<Move, double>, Player>> history_for_rewards;
    int move_count = 0;

    bool is_active = false;
    bool waiting_for_inference = false;
    bool root_noise_applicable = true;
    int target_visit_count = 0; 

    std::vector<SimulationState> pending_batch;
    std::vector<std::future<EvaluationResult>> pending_futures;

    GameSession(int id) : session_id(id) {
        pending_batch.reserve(128);
        pending_futures.reserve(128);
    }
    
    void reset_for_new_game() {
        board = Board();
        mcts_root = nullptr;
        history_for_rewards.clear();
        move_count = 0;
        waiting_for_inference = false;
        root_noise_applicable = true;
        pending_batch.clear();
        pending_futures.clear();
        is_active = true;
        target_visit_count = 0; 
    }
};

SelfPlay::SelfPlay(
    Model* network,
    int num_workers,
    int games_per_worker,
    int simulations_per_move,
    int nn_batch_size,
    int worker_batch_size,
    double c_puct,
    int temperature_decay_move,
    double dirichlet_alpha,
    double dirichlet_epsilon
) :
    network_handle_(network), 
    num_workers_(num_workers),
    games_per_worker_(games_per_worker),
    simulations_per_move_(simulations_per_move),
    worker_batch_size_(worker_batch_size), 
    mcts_c_puct_(c_puct),
    temperature_decay_move_(temperature_decay_move),
    dirichlet_alpha_(dirichlet_alpha),
    dirichlet_epsilon_(dirichlet_epsilon),
    rng_(std::random_device{}()) 
{
    if (!network) { 
        throw std::runtime_error("SelfPlay received a null network pointer.");
    }
    evaluator_ = std::make_unique<Evaluator>(network, nn_batch_size);
    evaluator_->start();
}

SelfPlay::~SelfPlay() {
    if (evaluator_) {
        evaluator_->stop(); 
    }
}

void SelfPlay::submit_inference_batch(
    std::vector<SimulationState>& batch,
    std::vector<std::future<EvaluationResult>>& out_futures
) {
    out_futures.clear(); 
    
    for (size_t i = 0; i < batch.size(); ++i) {
        MCTSNode* leaf_node = batch[i].current_node;
        if (!leaf_node) {
            throw std::runtime_error("SelfPlay: Critical Error - Attempted to submit a null leaf node for inference.");
        }
        EvaluationRequest req;
        req.request_id = static_cast<RequestId>(i);
        req.input_planes = TensorPool::acquire_planes();
        req.input_scalars = TensorPool::acquire_scalars();
        board_to_tensors(leaf_node->get_board(), req.input_planes->data(), req.input_scalars->data());
        out_futures.emplace_back(evaluator_->submit_request(std::move(req)));
    }
}

void SelfPlay::process_inference_results(
    std::vector<SimulationState>& batch,
    std::vector<std::future<EvaluationResult>>& futures,
    bool& root_noise_applicable 
) {
    for (size_t i = 0; i < batch.size(); ++i) {
        MCTSNode* leaf_node = batch[i].current_node;
        const std::vector<MCTSNode*>& path = batch[i].path;

        if (!leaf_node) continue; 

        try {
            EvaluationResult result = futures[i].get(); 
            leaf_node->decrement_pending_visits();

            std::map<Move, double> policy_probs = process_policy(*result.policy_logits, leaf_node->get_board());
            bool is_root_node_eval = (leaf_node == path[0]); 

            if (!policy_probs.empty()) {
                // Apply noise to root if this is the first expansion of a new move/game
                if (is_root_node_eval && root_noise_applicable) {
                    policy_probs = add_dirichlet_noise(policy_probs, dirichlet_alpha_, dirichlet_epsilon_);
                    root_noise_applicable = false; 
                }
                if (leaf_node->is_leaf() && !leaf_node->get_board().is_game_over()) {
                     leaf_node->expand(policy_probs);
                }
            }
            
            // Map relative NN values back to absolute player indices
            std::array<double, 4> player_values_absolute;
            Player cp = leaf_node->get_board().get_current_player();
            int cp_idx = static_cast<int>(cp);

            for(int rel_i = 0; rel_i < 4; ++rel_i) {
                int abs_p_idx = (cp_idx + rel_i) % 4;
                player_values_absolute[abs_p_idx] = static_cast<double>((*result.value)[rel_i]);
            }

            backpropagate_mcts_value(path, player_values_absolute); 

            // Return the result memory to the pool
            TensorPool::release_policy(std::move(result.policy_logits));
            TensorPool::release_value(std::move(result.value));

        } catch (const std::exception& e) {
            std::cerr << "Exception processing batch result: " << e.what() << std::endl;
            if (leaf_node) leaf_node->decrement_pending_visits();
        }
    } 
    batch.clear();
    futures.clear();
}

std::map<Move, double> SelfPlay::add_dirichlet_noise(
  const std::map<Move, double>& policy_probs,
  double alpha,
  double epsilon)
{
  if (policy_probs.empty() || alpha <= 0.0 || epsilon <= 0.0) {
      return policy_probs;
  }
  size_t num_actions = policy_probs.size();
  std::vector<double> noise_samples(num_actions);
  std::gamma_distribution<double> gamma_dist(alpha, 1.0); 
  double noise_sum = 0.0;
  for (size_t i = 0; i < num_actions; ++i) {
      noise_samples[i] = gamma_dist(rng_);
      noise_sum += noise_samples[i];
  }
  if (noise_sum > 1e-9) {
      for (size_t i = 0; i < num_actions; ++i) { noise_samples[i] /= noise_sum; }
  } else {
      double uniform_noise = 1.0 / static_cast<double>(num_actions);
       for (size_t i = 0; i < num_actions; ++i) {
          noise_samples[i] = uniform_noise;
      }
  }
  std::map<Move, double> noisy_policy;
  size_t noise_idx = 0;
  for (const auto& pair : policy_probs) {
      const Move& move = pair.first;
      double original_prob = pair.second;
      double noise_val = noise_samples[noise_idx++];
      noisy_policy[move] = (1.0 - epsilon) * original_prob + epsilon * noise_val;
  }
  return noisy_policy;
}

size_t SelfPlay::generate_data(int num_games) {
    worker_threads_.clear(); 
    std::atomic<int> games_started_counter(0);
    std::atomic<int> games_completed_counter(0);
    std::vector<std::vector<GameDataStep>> local_buffers(num_workers_);

    std::string filename = "training_data/gen_" + std::to_string(std::time(nullptr)) + ".bin";
    auto writer = std::make_unique<DataWriter>(filename);

    for (int i = 0; i < num_workers_; ++i) {
        worker_threads_.emplace_back(
            &SelfPlay::run_game_simulation, 
            this, 
            i, 
            std::ref(games_started_counter),
            std::ref(games_completed_counter), 
            num_games, 
            std::ref(local_buffers[i])       
        );
    }

    for (auto& thread : worker_threads_) {
        if (thread.joinable()) thread.join();
    }

    size_t total_points = 0;
    for (auto& local_buf : local_buffers) {
        total_points += local_buf.size();
        writer->write_batch(local_buf);
        local_buf.clear();
    }

    return total_points;
}

void SelfPlay::run_game_simulation(
  int worker_id,
  std::atomic<int>& games_started_counter,
  std::atomic<int>& games_completed_counter,
  int target_games,
  std::vector<GameDataStep>& local_buffer
) {
  std::mt19937 thread_rng(std::random_device{}() + worker_id);
  
  std::vector<GameSession> sessions;
  sessions.reserve(games_per_worker_);
  for(int i=0; i<games_per_worker_; ++i) {
      sessions.emplace_back(i);
  }

  while (true) {
      if (games_completed_counter.load() >= target_games) {
          bool any_active = false;
          for(const auto& s : sessions) if(s.is_active) any_active = true;
          if (!any_active) break;
      }

      int active_session_count = 0;
      bool any_action_taken = false;

      for (auto& session : sessions) {
          if (!session.is_active) {
              int game_idx = games_started_counter.fetch_add(1);
              if (game_idx < target_games) {
                  session.reset_for_new_game();
                  session.mcts_root = std::make_unique<MCTSNode>(session.board);
                  // Flag noise for application after the first expansion
                  session.root_noise_applicable = true; 
                  // Set initial target. Since new root has 0 visits:
                  session.target_visit_count = simulations_per_move_;
              } else {
                  continue; 
              }
          }
          
          active_session_count++;

          if (session.waiting_for_inference) {
              bool ready = false;
              if (!session.pending_futures.empty()) {
                  auto status = session.pending_futures[0].wait_for(std::chrono::seconds(0));
                  if (status == std::future_status::ready) {
                      ready = true;
                  }
              }

              if (ready) {
                  process_inference_results(session.pending_batch, session.pending_futures, session.root_noise_applicable);
                  session.waiting_for_inference = false;
                  any_action_taken = true;
              } else {
                  continue; 
              }
          }

          // Inner MCTS loop: generate simulations until batch is full or move is decided
          while (session.is_active && !session.waiting_for_inference) {
              int current_visits = session.mcts_root->get_visit_count(); 

              if (current_visits >= session.target_visit_count) {
                   MCTSNode& root = *session.mcts_root;
                   double current_temperature = (session.move_count < temperature_decay_move_) ? 1.0 : 0.0;
                   bool can_resign = (session.board.get_full_move_number() > 40);
                   
                   std::map<Move, double> final_policy = get_action_probs(root, current_temperature, can_resign);
                   if (final_policy.empty()) {
                       games_started_counter.fetch_sub(1);
                       session.is_active = false;
                       break;
                   }

                   session.history_for_rewards.emplace_back(session.board, final_policy, session.board.get_current_player());

                   Move chosen_move = choose_move(root, current_temperature, can_resign);
                   session.board.make_move(chosen_move);
                   session.move_count++;

                   // Tree Reuse
                   MCTSNode* chosen_child = nullptr;
                   MCTSNode* curr = root.get_first_child();
                   while (curr) {
                       if (curr->get_move() && curr->get_move().value() == chosen_move) {
                           chosen_child = curr;
                           break;
                       }
                       curr = curr->get_next_sibling();
                   }

                   if (chosen_child) {
                       MCTSNode* new_root = session.mcts_root->detach_child_and_clear_others(chosen_child);
                       session.mcts_root.reset(new_root);
                   } else {
                       session.mcts_root = std::make_unique<MCTSNode>(session.board);
                   }
                   
                   // Update target for the next move
                   // Target = (Visits carried over from reuse) + (New simulations to run)
                   session.target_visit_count = session.mcts_root->get_visit_count() + simulations_per_move_;
                   session.pending_batch.clear();

                   if (session.board.is_game_over()) {
                       process_game_result(session.history_for_rewards, session.board, local_buffer);
                       int completed_count = games_completed_counter.fetch_add(1) + 1;
                       std::cout << "Worker " << worker_id << " finished game " << completed_count 
                                 << "/" << target_games << " (" << session.move_count << " moves)." << std::endl; 
                       session.is_active = false;
                   } else {
                       if (!session.mcts_root->is_leaf()) {
                           session.mcts_root->inject_noise(dirichlet_alpha_, dirichlet_epsilon_, thread_rng);
                           session.root_noise_applicable = false; 
                       } else {
                           session.root_noise_applicable = true; 
                       }
                   }

                   any_action_taken = true;
                   break; // Move to next session
              }

              SimulationState current_sim_path;
              current_sim_path.current_node = session.mcts_root.get();
              current_sim_path.path.push_back(current_sim_path.current_node);
              
              bool selection_failed = false;
              while (!current_sim_path.current_node->is_leaf()) {
                  MCTSNode* next_node = current_sim_path.current_node->select_child(mcts_c_puct_);
                  if (!next_node || next_node == current_sim_path.current_node) {
                      selection_failed = true;
                      break;
                  }
                  current_sim_path.current_node = next_node;
                  current_sim_path.path.push_back(current_sim_path.current_node);
              }

              if (selection_failed) {
                  // Fallback: If MCTS selection gets stuck, force evaluation of root to progress
                  if (current_sim_path.current_node == session.mcts_root.get() && session.mcts_root->is_leaf()) {
                      // Handled by batching below
                  } else {
                      break; 
                  }
              }

              MCTSNode* leaf = current_sim_path.current_node;
              if (leaf->get_board().is_game_over()) {
                  PlayerPointMap scores = leaf->get_board().get_game_result();
                  backpropagate_mcts_value(current_sim_path.path, get_reward_map_array(scores));
              } else {
                  leaf->increment_pending_visits();
                  session.pending_batch.push_back(std::move(current_sim_path));

                  // Submit if batch is full OR if we have enough simulations 
                  // in flight to finish the search for this move.
                  size_t in_flight = session.pending_batch.size();
                  if (in_flight >= (size_t)worker_batch_size_ || (current_visits + in_flight) >= (size_t)session.target_visit_count) {
                      submit_inference_batch(session.pending_batch, session.pending_futures);
                      session.waiting_for_inference = true;
                      any_action_taken = true;
                      break;  
                  }
              }
          }
      }

      if (active_session_count > 0 && !any_action_taken) {
          for (auto& s : sessions) {
              if (s.is_active && s.waiting_for_inference) {
                  s.pending_futures[0].wait();
                  break; 
              }
          }
      }
      
      if (active_session_count == 0 && games_started_counter.load() >= target_games) {
          break;
      }
  } 
}

std::map<Move, double> SelfPlay::get_action_probs(const MCTSNode& root, double temperature, bool allow_resignation) const {
    std::map<Move, double> probs;
    if (root.is_leaf()) { return probs; }

    std::vector<double> visit_counts;
    std::vector<Move> moves;
    
    MCTSNode* curr = root.get_first_child();
    while (curr) {
        if (curr->get_move()) {
            Move m = *curr->get_move();
            if (!allow_resignation && m.is_resignation()) {
                curr = curr->get_next_sibling();
                continue;
            }
            visit_counts.push_back(static_cast<double>(curr->get_visit_count()));
            moves.push_back(m);
        }
        curr = curr->get_next_sibling();
    }
    
    if (moves.empty()) return probs;

    if (temperature == 0.0) {
        auto max_it = std::max_element(visit_counts.begin(), visit_counts.end());
        size_t max_index = std::distance(visit_counts.begin(), max_it);
        for (size_t i = 0; i < moves.size(); ++i) {
            probs[moves[i]] = (i == max_index) ? 1.0 : 0.0;
        }
    } else {
        double total_visits_pow = 0.0;
        std::vector<double> powered_visits;
        powered_visits.reserve(visit_counts.size());
        double inv_temp = 1.0 / temperature;
        for (double count : visit_counts) {
            double powered_count = std::pow(count, inv_temp);
            powered_visits.push_back(powered_count);
            total_visits_pow += powered_count;
        }
        if (total_visits_pow > 1e-9) { 
            for (size_t i = 0; i < moves.size(); ++i) {
                probs[moves[i]] = powered_visits[i] / total_visits_pow;
            }
        } else {
             double uniform_prob = moves.empty() ? 0.0 : (1.0 / static_cast<double>(moves.size()));
             for (size_t i = 0; i < moves.size(); ++i) { probs[moves[i]] = uniform_prob; }
        }
    }
    return probs;
}

Move SelfPlay::choose_move(const MCTSNode& root, double temperature, bool allow_resignation) {
    std::map<Move, double> action_probs = get_action_probs(root, temperature, allow_resignation);
    
    if (action_probs.empty()) { 
        if (!allow_resignation) return choose_move(root, temperature, true);
        throw std::runtime_error("Cannot choose move: No legal actions found."); 
    }

    std::vector<Move> moves;
    std::vector<double> probabilities;
    moves.reserve(action_probs.size());
    probabilities.reserve(action_probs.size());
    for (const auto& pair : action_probs) {
        moves.push_back(pair.first);
        probabilities.push_back(pair.second);
    }
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    int chosen_index = dist(rng_);
    return moves[chosen_index];
}

void SelfPlay::process_game_result(
    std::vector<std::tuple<Board, std::map<Move, double>, Player>>& game_history_for_rewards, 
    const Board& final_board,
    std::vector<GameDataStep>& output_buffer 
) {
    std::array<int, 4> final_scores = final_board.get_game_result();
    std::array<double, 4> game_rewards_array = get_reward_map_array(final_scores);

    for (const auto& history_step : game_history_for_rewards) {
        const Board& board_state = std::get<0>(history_step);
        const std::map<Move, double>& policy = std::get<1>(history_step);
        output_buffer.emplace_back(board_state, policy, std::get<2>(history_step), game_rewards_array);
    }
}

} // namespace chaturaji_cpp