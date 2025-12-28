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

std::mutex buffer_mutex_self_play;

SelfPlay::SelfPlay(
    Model* network,
    int num_workers,
    int simulations_per_move,
    size_t max_buffer_size,
    int nn_batch_size,
    int worker_batch_size,
    double c_puct,
    int temperature_decay_move,
    double dirichlet_alpha,
    double dirichlet_epsilon
) :
    network_handle_(network), 
    num_workers_(num_workers),
    simulations_per_move_(simulations_per_move),
    max_buffer_size_(max_buffer_size),
    buffer_(),
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


const ReplayBuffer& SelfPlay::get_buffer() const {
    return buffer_;
}

void SelfPlay::clear_buffer() {
     std::lock_guard<std::mutex> lock(buffer_mutex_self_play); 
    buffer_.clear();
}

void SelfPlay::process_worker_batch(
  std::vector<SimulationState>& pending_batch,
  Player root_player, 
  bool& root_noise_applicable 
) {
  if (pending_batch.empty()) {
      return;
  }

  size_t batch_size = pending_batch.size();
  std::vector<std::future<EvaluationResult>> futures;
  futures.reserve(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
      MCTSNode* leaf_node = pending_batch[i].current_node;
      if (!leaf_node) { 
           std::cerr << "Error: Nullptr leaf_node found in pending worker batch." << std::endl;
           continue;
      }
      EvaluationRequest req;
      req.state_floats = board_to_floats(leaf_node->get_board());
      futures.push_back(evaluator_->submit_request(std::move(req)));
      pending_batch[i].pending_request_id = req.request_id; 
  }

  for (size_t i = 0; i < batch_size; ++i) {
      MCTSNode* leaf_node = pending_batch[i].current_node;
      const std::vector<MCTSNode*>& path = pending_batch[i].path;

      if (!leaf_node) continue; 

      try {
          EvaluationResult result = futures[i].get(); 
          leaf_node->decrement_pending_visits();

          // 1. Process Policy (Standard)
          std::map<Move, double> policy_probs = process_policy(result.policy_logits, leaf_node->get_board());
          bool is_root_node_eval = (leaf_node == path[0]); 

          if (!policy_probs.empty()) {
              if (is_root_node_eval && root_noise_applicable) {
                  policy_probs = add_dirichlet_noise(policy_probs, dirichlet_alpha_, dirichlet_epsilon_);
                  root_noise_applicable = false; 
              }
              if (leaf_node->is_leaf() && !leaf_node->get_board().is_game_over()) {
                   leaf_node->expand(policy_probs);
              }
          }
          
          // 2. Process Value - CRITICAL FIX HERE
          // The NN returns values relative to the current player of the *leaf node*.
          // We must rotate them back to absolute player indices for the MCTS tree.
          std::array<double, 4> player_values_absolute;
          
          
          Player cp = leaf_node->get_board().get_current_player();
          int cp_idx = static_cast<int>(cp);

          for(int rel_i = 0; rel_i < 4; ++rel_i) {
              // Map Relative Index (0=Current, 1=Next...) back to Absolute Index (Red/Blue...)
              int abs_p_idx = (cp_idx + rel_i) % 4;
              player_values_absolute[abs_p_idx] = static_cast<double>(result.value[rel_i]);
          }

          backpropagate_mcts_value(path, player_values_absolute); 

      } catch (const std::future_error& e) {
          std::cerr << "Future error processing worker batch item " << i << ": " << e.what() << std::endl;
          if (leaf_node) leaf_node->decrement_pending_visits();
      } catch (const std::exception& e) {
          std::cerr << "Exception processing worker batch item " << i << ": " << e.what() << std::endl;
           if (leaf_node) leaf_node->decrement_pending_visits();
      }
  } 
  pending_batch.clear();
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

  while (true) {
      int game_idx = games_started_counter.fetch_add(1);
      if (game_idx >= target_games) break;

      Board board; 
      std::unique_ptr<MCTSNode> mcts_root_uptr = nullptr;

      std::vector<std::tuple<Board, std::map<Move, double>, Player>> game_history_for_rewards;
      int move_count = 0;

      while (!board.is_game_over()) {
          // Check if we can reuse the existing tree (if it exists and matches current state)
          // Otherwise, create a fresh root.
          if (!mcts_root_uptr || mcts_root_uptr->get_board().get_position_key() != board.get_position_key()) {
              mcts_root_uptr = std::make_unique<MCTSNode>(board);
          }
          MCTSNode& current_root_ref = *mcts_root_uptr; 

          Player root_player = board.get_current_player(); 

          std::vector<SimulationState> pending_worker_batch;
          pending_worker_batch.reserve(worker_batch_size_);
          
          bool root_noise_applicable = true; 

          // If the tree is REUSED, the root is not a leaf, so process_worker_batch 
          // will never trigger the noise logic. We must inject it manually here.
          if (!current_root_ref.is_leaf()) {
              current_root_ref.inject_noise(dirichlet_alpha_, dirichlet_epsilon_, rng_);
              root_noise_applicable = false; // Prevent double application
          }


          for (int sim = 0; sim < simulations_per_move_; ++sim) {
              SimulationState current_mcts_path;
              current_mcts_path.current_node = &current_root_ref;
              current_mcts_path.path.push_back(current_mcts_path.current_node);
              bool selection_failed = false; 

              while (!current_mcts_path.current_node->is_leaf()) {
                  MCTSNode* next_node = current_mcts_path.current_node->select_child(mcts_c_puct_);
                  if (next_node == nullptr || next_node == current_mcts_path.current_node) {
                       selection_failed = true; 
                       break; 
                  }
                  current_mcts_path.current_node = next_node;
                  current_mcts_path.path.push_back(current_mcts_path.current_node);
              } 

              if (selection_failed) {
                  continue; 
              }

              MCTSNode* leaf_node = current_mcts_path.current_node;
              if (leaf_node->get_board().is_game_over()) {
                  std::map<Player, int> final_scores = leaf_node->get_board().get_game_result();
                  std::map<Player, double> reward_map_terminal = get_reward_map(final_scores);
                  std::array<double, 4> terminal_player_values = convert_reward_map_to_array(reward_map_terminal);
                  backpropagate_mcts_value(current_mcts_path.path, terminal_player_values);
              } else {
                  leaf_node->increment_pending_visits();
                  pending_worker_batch.push_back(std::move(current_mcts_path));
                  if (pending_worker_batch.size() >= static_cast<size_t>(worker_batch_size_)) {
                      process_worker_batch(pending_worker_batch, root_player, root_noise_applicable);
                  }
              }
          } 

          if (!pending_worker_batch.empty()) {
              process_worker_batch(pending_worker_batch, root_player, root_noise_applicable);
          }

          double current_temperature = (move_count < temperature_decay_move_) ? 1.0 : 0.0;
          std::map<Move, double> final_policy = get_action_probs(current_root_ref, current_temperature);

          if (final_policy.empty()) {
              mcts_root_uptr = nullptr;
              break; 
          }

          game_history_for_rewards.emplace_back(board, final_policy, root_player);
          Move chosen_move = choose_move(current_root_ref, current_temperature);
          board.make_move(chosen_move); 

          // --- Tree Reuse Logic ---
          MCTSNode* chosen_child_raw_ptr = nullptr;
          for (const auto& child_uptr_loop : current_root_ref.get_children()) {
              if (child_uptr_loop->get_move() && child_uptr_loop->get_move().value() == chosen_move) {
                  chosen_child_raw_ptr = child_uptr_loop.get();
                  break;
              }
          }

          if (chosen_child_raw_ptr) {
              auto& old_root_children_vec = current_root_ref.get_children_for_reuse();
              std::unique_ptr<MCTSNode> new_root_candidate_uptr;
              for (auto it = old_root_children_vec.begin(); it != old_root_children_vec.end(); ++it) {
                  if (it->get() == chosen_child_raw_ptr) {
                      new_root_candidate_uptr = std::move(*it); 
                      old_root_children_vec.erase(it);          
                      break;
                  }
              }
              if (new_root_candidate_uptr) {
                  new_root_candidate_uptr->set_parent(nullptr);
                  mcts_root_uptr = std::move(new_root_candidate_uptr);
              } else {
                  mcts_root_uptr = std::make_unique<MCTSNode>(board); 
              }
          } else {
              mcts_root_uptr = std::make_unique<MCTSNode>(board); 
          }
          move_count++;
      } 

      int completed_count = games_completed_counter.fetch_add(1) + 1;
      
      std::cout << "Worker " << worker_id << " finished game " << completed_count << "/" << target_games << " (" << move_count << " moves)." << std::endl;
      
      process_game_result(game_history_for_rewards, board, local_buffer);
  } 
}

std::map<Move, double> SelfPlay::get_action_probs(const MCTSNode& root, double temperature) const {
     std::map<Move, double> probs;
    const auto& children = root.get_children();
    if (children.empty()) { return probs; }

    std::vector<double> visit_counts;
    std::vector<Move> moves;
    visit_counts.reserve(children.size());
    moves.reserve(children.size());

    for (const auto& child : children) {
        visit_counts.push_back(static_cast<double>(child->get_visit_count()));
        if (child->get_move()) { moves.push_back(*child->get_move()); }
    }

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


Move SelfPlay::choose_move(const MCTSNode& root, double temperature) {
    std::map<Move, double> action_probs = get_action_probs(root, temperature);
    if (action_probs.empty()) { throw std::runtime_error("Cannot choose move: No legal actions found."); }

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
    std::map<Player, int> final_scores = final_board.get_game_result();
    std::map<Player, double> reward_map_for_game = get_reward_map(final_scores);
    std::array<double, 4> game_rewards_array = convert_reward_map_to_array(reward_map_for_game);

    for (const auto& history_step : game_history_for_rewards) {
        const Board& board_state = std::get<0>(history_step);
        const std::map<Move, double>& policy = std::get<1>(history_step);
        output_buffer.emplace_back(board_state, policy, std::get<2>(history_step), game_rewards_array);
    }
}

} // namespace chaturaji_cpp