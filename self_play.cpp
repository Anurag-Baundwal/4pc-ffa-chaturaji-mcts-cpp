#include "self_play.h"
#include <iostream> // For progress printing
#include <numeric>  // For std::accumulate
#include <cmath>    // For std::pow, std::exp, std::log
#include <algorithm>// For std::max_element
#include <stdexcept>// Include for runtime_error
#include <vector>   // For storing noise samples

namespace chaturaji_cpp {

SelfPlay::SelfPlay(
    ChaturajiNN network,
    torch::Device device,
    int simulations_per_move,
    size_t buffer_size,
    double c_puct,
    int temperature_decay_move,
    int mcts_batch_size,
    double dirichlet_alpha,
    double dirichlet_epsilon
) :
    network_(network), // Copy/move the module handle
    device_(device),
    simulations_per_move_(simulations_per_move),
    buffer_(buffer_size), // Initialize deque with max size
    mcts_c_puct_(c_puct),
    temperature_decay_move_(temperature_decay_move),
    mcts_batch_size_(mcts_batch_size), // Store batch size
    dirichlet_alpha_(dirichlet_alpha),
    dirichlet_epsilon_(dirichlet_epsilon),
    rng_(std::random_device{}()) // Seed the random number generator
{
    // Ensure the passed network is valid
    if (!network_) {
        throw std::runtime_error("SelfPlay received an invalid network module.");
    }
     network_->eval(); // Set network to evaluation mode
}

const ReplayBuffer& SelfPlay::get_buffer() const {
    return buffer_;
}

void SelfPlay::clear_buffer() {
    buffer_.clear();
}

// --- NEW: Dirichlet Noise Helper ---
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
  if (noise_sum > 1e-9) { // Avoid division by zero
      for (size_t i = 0; i < num_actions; ++i) {
          noise_samples[i] /= noise_sum;
      }
  } else {
      // If sum is too small (e.g., alpha was tiny), maybe distribute uniformly?
      // Or just return original probs. Let's distribute uniformly for robustness.
      double uniform_noise = 1.0 / static_cast<double>(num_actions);
       for (size_t i = 0; i < num_actions; ++i) {
          noise_samples[i] = uniform_noise;
      }
  }

  // Combine original policy with noise
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

// --- Refactored generate_game ---
void SelfPlay::generate_game() {
    Board board;
    // Temporary storage for the current game's data before assigning final rewards
    std::vector<std::tuple<Board, std::map<Move, double>, Player>> game_data_temp;
    int move_count = 0;

    network_->eval(); // Ensure network is in eval mode

    while (!board.is_game_over()) {
        // Optional: Print progress
        // if (move_count % 5 == 0) { 
        //     std::cout << "Move " << move_count << " - Buffer size: " << buffer_.size() << std::endl;
        // }

        // Create MCTS root node for the current state
        MCTSNode root(board); // Create a copy of the board for the root

        // --- Step 1: Initial Network Evaluation for Root (if not terminal) ---
        std::map<Move, double> noisy_prior_probs; // Will hold priors (potentially noisy) for expansion

        if (!root.get_board().is_game_over()) { // Only evaluate if not terminal
              torch::Tensor root_tensor = get_board_tensor_no_batch(root.get_board(), device_);
              torch::Tensor policy_logits_root, value_root; // value_root not used here

            { // NoGradGuard scope
                torch::NoGradGuard no_grad;
                std::tie(policy_logits_root, value_root) = network_->forward(root_tensor.unsqueeze(0));
            }

            // --- Step 2: Process Policy and Add Dirichlet Noise ---
            std::map<Move, double> prior_probs = process_policy(policy_logits_root.squeeze(0), root.get_board());
            noisy_prior_probs = add_dirichlet_noise(prior_probs, dirichlet_alpha_, dirichlet_epsilon_);

            // --- Step 3: Expand Root Node using (potentially noisy) priors ---
            if (!noisy_prior_probs.empty()) {
                  root.expand(noisy_prior_probs);
            } else {
                  // No legal moves from root? Should mean game over.
                  if (!root.get_board().is_game_over()) {
                      std::cerr << "Warning: No prior probabilities generated for non-terminal root in self-play. Game might end unexpectedly." << std::endl;
                  }
                  // Expansion won't happen, MCTS loop below might do nothing.
            }
        }
        // --- End Root Preparation ---

        // --- Step 4: Run Batched MCTS Simulations ---
        // Note: If root was terminal or had no moves, MCTS won't run effectively, which is okay.
        // The MCTS simulation will use the priors stored in the root's children (which came from noisy_prior_probs).
        if (!root.get_children().empty()) { // Only run if root was expanded
            run_mcts_simulations_batch(
              root,
              network_,
              simulations_per_move_,
              device_,
              mcts_c_puct_,
              mcts_batch_size_ // Pass the stored batch size
            );
        }
        // --- End MCTS ---

        // Determine temperature
        double current_temperature = (move_count < temperature_decay_move_) ? 1.0 : 0.0;

        // Get action probabilities from MCTS visit counts
        std::map<Move, double> action_probs = get_action_probs(root, current_temperature);


        if (action_probs.empty()) {
          // MCTS failed to produce moves, likely game over or an issue.
          if (!board.is_game_over()) { // Check again, maybe MCTS detected it?
             std::cerr << "Warning: No action probabilities generated from MCTS root in self-play, but board not flagged as game over. Ending game early." << std::endl;
          }
          break; // Exit game generation loop
        }


        // Store state and policy *before* making the move
        // Important: Store a *copy* of the board state as it was *before* the move.
        game_data_temp.emplace_back(board, action_probs, board.get_current_player());

        // Choose and make the move
        Move chosen_move = choose_move(root, current_temperature);
        board.make_move(chosen_move); // Modify the main board state

        move_count++;

    } // End game loop

    // Game finished, process results and add to buffer
    process_game_result(game_data_temp, board);

     // Optional: Print final scores
     std::cout << "Game finished (" << move_count << " moves). Final Scores: ";
     const auto& final_points = board.get_player_points();
     for(const auto& pair : final_points) {
         switch(pair.first) {
            case Player::RED: std::cout << "R:" << pair.second; break;
            case Player::BLUE: std::cout << " B:" << pair.second; break;
            case Player::YELLOW: std::cout << " Y:" << pair.second; break;
            case Player::GREEN: std::cout << " G:" << pair.second; break;
         }
     }
     std::cout << " | Buffer size: " << buffer_.size() << std::endl;
}


std::map<Move, double> SelfPlay::get_action_probs(const MCTSNode& root, double temperature) const {
    std::map<Move, double> probs;
    const auto& children = root.get_children();

    if (children.empty()) {
        return probs;
    }

    std::vector<double> visit_counts;
    visit_counts.reserve(children.size());
    std::vector<Move> moves;
    moves.reserve(children.size());

    for (const auto& child : children) {
        visit_counts.push_back(static_cast<double>(child->get_visit_count()));
        if (child->get_move()) { // Should always have a move for a child
            moves.push_back(*child->get_move());
        } else {
            // This would be an error in MCTS logic
            throw std::runtime_error("MCTS child node without a move.");
        }
    }

    if (temperature == 0.0) {
        // Greedy selection: find the max visit count
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
            // Add epsilon? Python code didn't explicitly add epsilon here but did in PUCT calc.
            // Let's omit it for now, assuming visit counts > 0 if temp != 0.
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
              for (size_t i = 0; i < moves.size(); ++i) {
                probs[moves[i]] = uniform_prob;
            }
        }
    }
    return probs;
}


Move SelfPlay::choose_move(const MCTSNode& root, double temperature) {
    std::map<Move, double> action_probs = get_action_probs(root, temperature);

    if (action_probs.empty()) {
        // Should be handled before calling choose_move, but as a fallback...
        throw std::runtime_error("Cannot choose move: No legal actions found.");
    }

    // C++ equivalent of numpy.random.choice
    std::vector<Move> moves;
    std::vector<double> probabilities;
    moves.reserve(action_probs.size());
    probabilities.reserve(action_probs.size());

    for (const auto& pair : action_probs) {
        moves.push_back(pair.first);
        probabilities.push_back(pair.second);
    }

    // Create a distribution object
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

    // Sample an index using the generator
    int chosen_index = dist(rng_);

    return moves[chosen_index];
}


void SelfPlay::process_game_result(
    std::vector<std::tuple<Board, std::map<Move, double>, Player>>& game_data_temp,
    const Board& final_board)
{
    std::map<Player, int> final_scores = final_board.get_game_result(); // Use get_game_result which includes draw bonus
    std::map<Player, double> reward_map = get_reward_map(final_scores);

    // Add processed data to the main buffer
    for (const auto& step_data : game_data_temp) {
        const Board& board_state = std::get<0>(step_data);
        const std::map<Move, double>& policy = std::get<1>(step_data);
        Player player_turn = std::get<2>(step_data);

        // Find the reward for the player whose turn it was
        double reward = reward_map.count(player_turn) ? reward_map.at(player_turn) : -2.0; // Default lowest

        // Add to buffer, potentially removing oldest if full
        if (buffer_.size() == buffer_.max_size()) {
            buffer_.pop_front(); // Remove oldest element
        }
        // Add the full tuple: Board state, Policy map, Player, Reward
        buffer_.emplace_back(board_state, policy, player_turn, reward);
    }
}


} // namespace chaturaji_cpp