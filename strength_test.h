// strength_test.h
#pragma once

#include <string>
#include <vector>

namespace chaturaji_cpp {

/**
 * @brief Runs a strength test by pitting two models against each other.
 *
 * The new model plays one color per game, cycling through RED, BLUE, YELLOW, GREEN.
 * The old model plays the other three colors.
 *
 * @param new_model_path Path to the "newer" model file.
 * @param old_model_path Path to the "older" model file.
 * @param num_games Total number of games to simulate.
 * @param simulations_per_move Number of MCTS simulations per move decision.
 * @param mcts_batch_size Batch size for synchronous MCTS network evaluations.
 */
void run_strength_test(
    const std::string& new_model_path,
    const std::string& old_model_path,
    int num_games,
    int simulations_per_move,
    int mcts_batch_size
);

} // namespace chaturaji_cpp