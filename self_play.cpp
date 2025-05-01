#include "self_play.h"
#include <iostream> // For progress printing
#include <numeric>  // For std::accumulate
#include <cmath>    // For std::pow, std::exp, std::log
#include <algorithm>// For std::max_element
#include <stdexcept>// Include for runtime_error
#include <vector>   // For storing noise samples, threads, local buffers
#include <future>   // For std::future from evaluator
#include <mutex>    // For protecting main buffer access

namespace chaturaji_cpp {

// Mutex to protect the main replay buffer when combining results
std::mutex buffer_mutex;

SelfPlay::SelfPlay(
    ChaturajiNN network,
    torch::Device device,
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
    network_handle_(network), // Store handle
    device_(device),
    num_workers_(num_workers),
    simulations_per_move_(simulations_per_move),
    max_buffer_size_(max_buffer_size),
    buffer_(),
    worker_batch_size_(worker_batch_size), 
    mcts_c_puct_(c_puct),
    temperature_decay_move_(temperature_decay_move),
    dirichlet_alpha_(dirichlet_alpha),
    dirichlet_epsilon_(dirichlet_epsilon),
    rng_(std::random_device{}()) // Seed the random number generator
{
    // Ensure the passed network is valid
    if (!network) { // Check the original parameter
        throw std::runtime_error("SelfPlay received an invalid network module.");
    }
    // Create and start the evaluator
    evaluator_ = std::make_unique<Evaluator>(network, device, nn_batch_size);
    evaluator_->start();
}

// Destructor to stop evaluator thread properly
SelfPlay::~SelfPlay() {
    if (evaluator_) {
        evaluator_->stop(); // Stops the background thread and joins it
    }
    // Worker threads should already be joined by generate_data
}


const ReplayBuffer& SelfPlay::get_buffer() const {
    return buffer_;
}

void SelfPlay::clear_buffer() {
     std::lock_guard<std::mutex> lock(buffer_mutex); // Protect buffer access
    buffer_.clear();
}

// --- New Helper Function Implementation ---
void SelfPlay::process_worker_batch(
  std::vector<SimulationState>& pending_batch,
  Player root_player,
  bool& root_noise_applicable // Use reference to modify the flag
) {
  if (pending_batch.empty()) {
      return;
  }

  size_t batch_size = pending_batch.size();
  std::vector<std::future<EvaluationResult>> futures;
  futures.reserve(batch_size);

  // 1. Submit all requests without waiting
  for (size_t i = 0; i < batch_size; ++i) {
      MCTSNode* leaf_node = pending_batch[i].current_node;
      if (!leaf_node) { // Safety check
           std::cerr << "Error: Nullptr leaf_node found in pending worker batch." << std::endl;
           // Need a way to handle this - maybe skip submission and create a dummy future?
           // For now, let's assume valid nodes. A robust solution might need dummy results.
           continue;
      }
      EvaluationRequest req;
      // Generate tensor on CPU for potentially easier transfer if evaluator is on GPU
      req.state_tensor = get_board_tensor_no_batch(leaf_node->get_board(), torch::kCPU);
      // Submit and store the future
      futures.push_back(evaluator_->submit_request(std::move(req)));
      // Associate future with the request (implicitly done by index)
      pending_batch[i].pending_request_id = req.request_id; // Store ID if needed for debugging/logging
  }

  // 2. Wait for and process results
  for (size_t i = 0; i < batch_size; ++i) {
      MCTSNode* leaf_node = pending_batch[i].current_node;
      const std::vector<MCTSNode*>& path = pending_batch[i].path;

      if (!leaf_node) continue; // Skip if node was invalid earlier

      try {
          // Wait for the i-th future to complete
          EvaluationResult result = futures[i].get();

          // Decrement pending visits (remove virtual loss effect)
          leaf_node->decrement_pending_visits();

          // Process policy
          std::map<Move, double> policy_probs = process_policy(result.policy_logits, leaf_node->get_board());

          // Check if this evaluation was for the *original* root node search
          // AND if noise hasn't been applied yet for this root.
          bool is_root_node_eval = (leaf_node == path[0]); // path[0] is always the root for this search

          // Expand the node (potentially with noise for the root)
          if (!policy_probs.empty()) {
              // Only apply noise if it's the root's first evaluation pass AND flag is true
              if (is_root_node_eval && root_noise_applicable) {
                  policy_probs = add_dirichlet_noise(policy_probs, dirichlet_alpha_, dirichlet_epsilon_);
                  root_noise_applicable = false; // Noise applied, set flag for this move search
                  // std::cout << "Applied root noise" << std::endl; // Debug
              }
              // Expand only if it's still a leaf (another thread/batch might have expanded it)
              if (leaf_node->is_leaf() && !leaf_node->get_board().is_game_over()) {
                   leaf_node->expand(policy_probs);
              }
          } else if (!leaf_node->get_board().is_game_over()) {
               // Policy empty but not terminal? Log warning.
                std::cerr << "Warning (Worker Batch): Empty policy from NN for non-terminal leaf." << std::endl;
          }


          // Backpropagate value (always from root player's perspective)
          backpropagate_path(path, static_cast<double>(result.value));

      } catch (const std::future_error& e) {
          std::cerr << "Future error processing worker batch item " << i << ": " << e.what() << " Code: " << e.code() << std::endl;
          // Clean up pending visit if future failed
          if (leaf_node) {
              leaf_node->decrement_pending_visits();
          }
      } catch (const std::exception& e) {
          std::cerr << "Exception processing worker batch item " << i << ": " << e.what() << std::endl;
          // Clean up pending visit
           if (leaf_node) {
              leaf_node->decrement_pending_visits();
          }
      }
  } // End processing loop

  // 3. Clear the processed batch
  pending_batch.clear();
}

// --- Helper for adding Dirichlet Noise ---
std::map<Move, double> SelfPlay::add_dirichlet_noise(
  const std::map<Move, double>& policy_probs,
  double alpha,
  double epsilon)
{
  if (policy_probs.empty() || alpha <= 0.0 || epsilon <= 0.0) {
      // No noise needed if no moves, alpha invalid, or epsilon is zero
      return policy_probs;
  }

  size_t num_actions = policy_probs.size();
  std::vector<double> noise_samples(num_actions);
  std::gamma_distribution<double> gamma_dist(alpha, 1.0); // Shape alpha, scale 1.0

  // Generate samples from gamma distribution
  double noise_sum = 0.0;
  for (size_t i = 0; i < num_actions; ++i) {
      noise_samples[i] = gamma_dist(rng_);
      noise_sum += noise_samples[i];
  }
  // Normalize noise samples if sum is valid
  if (noise_sum > 1e-9) {
      for (size_t i = 0; i < num_actions; ++i) { noise_samples[i] /= noise_sum; }
  } else {
      // If sum is too small (e.g., alpha was tiny), distribute uniformly.
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


// --- Main Data Generation Function ---
void SelfPlay::generate_data(int num_games) {
    worker_threads_.clear(); // Clear any previous threads
    std::atomic<int> games_completed_counter(0);

    // Create thread-local storage for results
    std::vector<std::vector<GameDataStep>> local_buffers(num_workers_);

    std::cout << "Starting data generation with " << num_workers_ << " workers for " << num_games << " games..." << std::endl;

    // Launch worker threads
    for (int i = 0; i < num_workers_; ++i) {
        worker_threads_.emplace_back(
            &SelfPlay::run_game_simulation, // Member function pointer
            this,                            // 'this' pointer for the object
            i,                               // Worker ID
            std::ref(games_completed_counter),// Reference to atomic counter
            num_games,                       // Target games
            std::ref(local_buffers[i])       // Reference to worker's local buffer
        );
    }

    // Wait for all worker threads to complete
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    std::cout << "All workers finished. Combining results..." << std::endl;

    // Combine results from local buffers into the main buffer
    { // Lock scope for main buffer
        std::lock_guard<std::mutex> lock(buffer_mutex);
        size_t total_steps = 0;
        for (const auto& local_buf : local_buffers) {
            total_steps += local_buf.size();
            for (const auto& step : local_buf) {
                if (buffer_.size() >= max_buffer_size_) {
                    buffer_.pop_front(); // Maintain buffer size limit
                }
                buffer_.push_back(step); // Add data step
            }
        }
         std::cout << "Combined " << total_steps << " data steps. Final buffer size: " << buffer_.size() << std::endl;
    } // Mutex unlock

    // Note: Evaluator is stopped in the destructor ~SelfPlay
}


// --- Worker Thread Function (Revised - No Goto) ---
void SelfPlay::run_game_simulation(
  int worker_id,
  std::atomic<int>& games_completed_counter,
  int target_games,
  std::vector<GameDataStep>& local_buffer
) {
  std::mt19937 thread_rng(std::random_device{}() + worker_id);

  while (games_completed_counter < target_games) {
      Board board;
      std::vector<std::tuple<Board, std::map<Move, double>, Player>> game_data_temp;
      int move_count = 0;

      // --- Game Loop ---
      while (!board.is_game_over()) {
          if (games_completed_counter >= target_games) break;

          MCTSNode current_root(board); // Root for the current move search
          Player root_player = board.get_current_player(); // Player at the root

          // --- Batching Structures for this move search ---
          std::vector<SimulationState> pending_worker_batch;
          pending_worker_batch.reserve(worker_batch_size_);
          bool root_noise_applicable = true; // Flag to apply noise only once for this root

          // --- Run MCTS Simulations for the current move ---
          for (int sim = 0; sim < simulations_per_move_; ++sim) {
              SimulationState current_mcts_path;
              current_mcts_path.current_node = &current_root;
              current_mcts_path.path.push_back(current_mcts_path.current_node);

              bool selection_failed = false; // Flag to track selection issues

              // 1. Selection (with virtual loss)
              while (!current_mcts_path.current_node->is_leaf()) {
                  MCTSNode* next_node = current_mcts_path.current_node->select_child(mcts_c_puct_);

                  if (next_node == nullptr || next_node == current_mcts_path.current_node) {
                       std::cerr << "Worker " << worker_id << ": MCTS select_child failed or didn't advance. Node:"
                                 << (next_node ? " same" : " null") << ". Sims left: " << (simulations_per_move_ - sim - 1) << std::endl;
                       selection_failed = true; // Set the flag
                       break; // Break out of the inner selection loop
                  }
                  current_mcts_path.current_node = next_node;
                  current_mcts_path.path.push_back(current_mcts_path.current_node);
              } // End Selection while loop

              // --- Check if selection failed ---
              if (selection_failed) {
                  // Skip the rest of the logic for *this simulation* and move to the next one.
                  continue; // Go to the next iteration of the outer `for` loop (simulations)
              }

              // --- If selection succeeded, proceed with leaf handling ---
              MCTSNode* leaf_node = current_mcts_path.current_node;

              // 2. Check if leaf is terminal
              if (leaf_node->get_board().is_game_over()) {
                  std::map<Player, int> final_scores = leaf_node->get_board().get_game_result();
                  std::map<Player, double> reward_map = get_reward_map(final_scores);
                  double value = reward_map.count(root_player) ? reward_map.at(root_player) : -2.0;
                  backpropagate_path(current_mcts_path.path, value);
              } else {
                  // 3. Non-terminal leaf: Add to pending batch
                  leaf_node->increment_pending_visits();
                  pending_worker_batch.push_back(std::move(current_mcts_path));

                  // 4. Check if batch is full
                  if (pending_worker_batch.size() >= static_cast<size_t>(worker_batch_size_)) {
                      process_worker_batch(pending_worker_batch, root_player, root_noise_applicable);
                  }
              }
              // End leaf handling (no goto needed)

          } // End MCTS simulation loop for one move

          // --- Process any remaining nodes in the batch ---
          if (!pending_worker_batch.empty()) {
              process_worker_batch(pending_worker_batch, root_player, root_noise_applicable);
          }

          // --- Choose Move (based on completed search) ---
          double current_temperature = (move_count < temperature_decay_move_) ? 1.0 : 0.0;
          std::map<Move, double> final_policy = get_action_probs(current_root, current_temperature);

          if (final_policy.empty()) {
               if (!board.is_game_over()) {
                   std::cerr << "Worker " << worker_id << ": Warning - No moves generated from MCTS search, but game not over. Ending game." << std::endl;
               }
              break; // End this game simulation
          }

          // Store state and policy
          game_data_temp.emplace_back(board, final_policy, root_player);

          // Choose and make move
          Move chosen_move = choose_move(current_root, current_temperature);
          board.make_move(chosen_move);
          move_count++;

      } // End game loop (while !board.is_game_over())

      // --- Game Finished ---
      int completed_count = games_completed_counter.fetch_add(1) + 1;
      std::cout << "Worker " << worker_id << " finished game " << completed_count << "/" << target_games
                << " (" << move_count << " moves)." << std::endl;

      // Process results into worker's local buffer
      process_game_result(game_data_temp, board, local_buffer);

  } // End worker loop (while games_completed_counter < target_games)
  std::cout << "Worker " << worker_id << " exiting." << std::endl;
}
// --- Helper Functions

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
        else { throw std::runtime_error("MCTS child node without a move."); }
    }

    if (temperature == 0.0) {
        auto max_it = std::max_element(visit_counts.begin(), visit_counts.end());
        size_t max_index = std::distance(visit_counts.begin(), max_it);

        // Assign probability 1.0 to the best move, 0.0 to others
        for (size_t i = 0; i < moves.size(); ++i) {
            probs[moves[i]] = (i == max_index) ? 1.0 : 0.0;
        }
    } else {
        // Temperature-based sampling (using power method)
        double total_visits_pow = 0.0;
        std::vector<double> powered_visits;
        powered_visits.reserve(visit_counts.size());
        double inv_temp = 1.0 / temperature;
        for (double count : visit_counts) {
            double powered_count = std::pow(count, inv_temp);
            powered_visits.push_back(powered_count);
            total_visits_pow += powered_count;
        }

        // Normalize to get probabilities
        if (total_visits_pow > 1e-9) { // Avoid division by zero
            for (size_t i = 0; i < moves.size(); ++i) {
                probs[moves[i]] = powered_visits[i] / total_visits_pow;
            }
        } else {
             // If total is near zero (e.g., all counts were 0), distribute uniformly
             double uniform_prob = 1.0 / static_cast<double>(moves.size());
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
    // Use member RNG - potential contention if SelfPlay object is reused by multiple threads,
    // but in the current structure, each worker runs in its own thread context.
    // Consider thread_local RNG if needed later.
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    int chosen_index = dist(rng_);
    return moves[chosen_index];
}

// --- Modified to output to a provided buffer ---
void SelfPlay::process_game_result(
    std::vector<std::tuple<Board, std::map<Move, double>, Player>>& game_data_temp,
    const Board& final_board,
    std::vector<GameDataStep>& output_buffer // Changed from modifying member buffer
) {
    std::map<Player, int> final_scores = final_board.get_game_result();
    std::map<Player, double> reward_map = get_reward_map(final_scores);

    // Add processed data to the provided output buffer
    for (const auto& step_data : game_data_temp) {
        const Board& board_state = std::get<0>(step_data);
        const std::map<Move, double>& policy = std::get<1>(step_data);
        Player player_turn = std::get<2>(step_data);

        double reward = reward_map.count(player_turn) ? reward_map.at(player_turn) : -2.0;

        // Add the full tuple: Board state, Policy map, Player, Reward
        output_buffer.emplace_back(board_state, policy, player_turn, reward);
    }
}


} // namespace chaturaji_cpp