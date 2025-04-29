#pragma once

#include <vector>
#include <deque>
#include <map>
#include <tuple> // For storing game data
#include <memory> // For shared_ptr
#include <random> // For probabilistic move selection
#include <thread> // For std::thread
#include <atomic> // For atomic flags/counters
#include <vector> // For storing threads

#include "board.h"
#include "mcts_node.h"
#include "model.h"
#include "search.h" // For process_policy, get_reward_map
#include "types.h"
#include "utils.h" // For board_to_tensor
#include "evaluator.h" // Include the new Evaluator class


namespace chaturaji_cpp {

// Define the structure for storing one step of game data (Unchanged)
using GameDataStep = std::tuple<Board, std::map<Move, double>, Player, double>;
using ReplayBuffer = std::deque<GameDataStep>;


class SelfPlay {
public:
    /**
     * @param network Shared pointer to the neural network model.
     * @param device The device (CPU/CUDA) for NN inference.
     * @param num_workers Number of parallel game simulation threads.
     * @param simulations_per_move Number of MCTS simulations for each move decision.
     * @param buffer_size Maximum size of the replay buffer.
     * @param nn_batch_size The batch size used by the NN evaluator thread.
     * @param c_puct MCTS exploration constant.
     * @param temperature_decay_move Move number after which temperature becomes 0.
     * @param dirichlet_alpha The alpha parameter for Dirichlet noise.
     * @param dirichlet_epsilon The weight factor for Dirichlet noise.
     */
    SelfPlay(
        ChaturajiNN network, // Pass the model module directly
        torch::Device device,
        int num_workers = 4,
        int simulations_per_move = 100,
        size_t buffer_size = 250000,
        int nn_batch_size = 4096, // Batch size for NN evaluations
        double c_puct = 1.0,
        int temperature_decay_move = 5,
        double dirichlet_alpha = 0.3,
        double dirichlet_epsilon = 0.25
    );

    ~SelfPlay(); // Destructor to manage evaluator lifetime

    /**
     * @brief Generates a specified number of self-play games using worker threads and the evaluator.
     *        Adds the generated game data to the internal replay buffer.
     * @param num_games The total number of games to generate across all workers.
     */
    void generate_data(int num_games);

    /**
     * @brief Provides access to the internal replay buffer.
     * @return Const reference to the replay buffer deque.
     */
    const ReplayBuffer& get_buffer() const;

    /**
     * @brief Clears the internal replay buffer.
     */
    void clear_buffer();


private:
    // --- Renamed generate_game to run_game_simulation ---
    /**
     * @brief Runs a single game simulation loop, interacting with the evaluator.
     *        This function is executed by each worker thread.
     * @param worker_id Identifier for the worker thread (for logging).
     * @param games_completed_counter Atomic counter shared by workers.
     * @param target_games Total games to generate across all workers.
     * @param local_buffer Buffer local to this worker thread to store results.
     */
    void run_game_simulation(
        int worker_id,
        std::atomic<int>& games_completed_counter,
        int target_games,
        std::vector<GameDataStep>& local_buffer
    );


    // (get_action_probs, choose_move, process_game_result remain similar)
    std::map<Move, double> get_action_probs(const MCTSNode& root, double temperature) const;
    Move choose_move(const MCTSNode& root, double temperature);
    void process_game_result(
        std::vector<std::tuple<Board, std::map<Move, double>, Player>>& game_data_temp,
        const Board& final_board,
        std::vector<GameDataStep>& output_buffer // Store directly in output buffer
    );
     std::map<Move, double> add_dirichlet_noise(
      const std::map<Move, double>& policy_probs,
      double alpha,
      double epsilon
    );

    // Member Variables
    ChaturajiNN network_handle_; // Keep handle for potential future use? Or remove? Keep for now.
    torch::Device device_; // Keep track of intended device
    int num_workers_;
    int simulations_per_move_;
    ReplayBuffer buffer_; // Main shared buffer (filled after workers finish)
    double mcts_c_puct_;
    int temperature_decay_move_;
    double dirichlet_alpha_;
    double dirichlet_epsilon_;

    std::mt19937 rng_; // Mersenne Twister engine (maybe make thread-local later?)

    // --- Evaluator and Worker Management ---
    std::unique_ptr<Evaluator> evaluator_; // Use unique_ptr to manage lifetime
    std::vector<std::thread> worker_threads_;
};

} // namespace chaturaji_cpp