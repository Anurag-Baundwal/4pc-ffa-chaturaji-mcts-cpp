#pragma once

#include <vector>
#include <deque>
#include <map>
#include <tuple> // For storing game data
#include <memory> // For network pointer if needed
#include <random> // For probabilistic move selection

#include "board.h"
#include "mcts_node.h"
#include "model.h"
#include "search.h" // For process_policy, get_reward_map
#include "types.h"
#include "utils.h" // For board_to_tensor

namespace chaturaji_cpp {

// Define the structure for storing one step of game data
// Board state (copied), Policy map (Move -> prob), Player whose turn it was, Final reward for that player
using GameDataStep = std::tuple<Board, std::map<Move, double>, Player, double>;
using ReplayBuffer = std::deque<GameDataStep>;


class SelfPlay {
public:
    /**
     * @param network Shared pointer to the neural network model.
     * @param device The device (CPU/CUDA) for NN inference.
     * @param simulations_per_move Number of MCTS simulations for each move decision.
     * @param buffer_size Maximum size of the replay buffer.
     * @param c_puct MCTS exploration constant.
     * @param temperature_decay_move Move number after which temperature becomes 0 for greedy selection.
     */
    SelfPlay(
        ChaturajiNN network, // Pass the model module directly
        torch::Device device,
        int simulations_per_move = 100,
        size_t buffer_size = 10000,
        double c_puct = 1.0,
        int temperature_decay_move = 5
    );

    /**
     * @brief Generates one full game of self-play using MCTS.
     *        Adds the generated game data to the internal replay buffer.
     */
    void generate_game();

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
    ChaturajiNN network_; // Store the network module
    torch::Device device_;
    int simulations_per_move_;
    ReplayBuffer buffer_; // Stores (board, policy_map, player, reward) tuples
    double mcts_c_puct_;
    int temperature_decay_move_;

    // Random number generation for temperature-based move selection
    std::mt19937 rng_; // Mersenne Twister engine

    /**
     * @brief Calculates action probabilities based on MCTS visit counts and temperature.
     * @param root The root node of the MCTS search for the current position.
     * @param temperature Controls exploration (1.0 = proportional, 0.0 = greedy).
     * @return Map from Move to its selection probability.
     */
    std::map<Move, double> get_action_probs(const MCTSNode& root, double temperature) const;

    /**
     * @brief Chooses a move based on the calculated action probabilities.
     * @param root The root node of the MCTS search.
     * @param temperature The temperature for selection.
     * @return The chosen Move.
     */
    Move choose_move(const MCTSNode& root, double temperature);

    /**
     * @brief Processes the final game result and assigns rewards to buffer entries.
     * @param game_data_temp Temporary vector holding (Board, Policy, Player) for the finished game.
     * @param final_board The board state at the end of the game.
     */
    void process_game_result(
        std::vector<std::tuple<Board, std::map<Move, double>, Player>>& game_data_temp,
        const Board& final_board
    );
};

} // namespace chaturaji_cpp