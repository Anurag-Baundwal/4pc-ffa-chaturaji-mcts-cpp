#pragma once

#include <string>
#include <vector>

namespace chaturaji_cpp {

/**
 * @brief Runs a strength test by pitting two models against each other.
 * @param new_model_path Path to the "newer" ONNX model file.
 * @param old_model_path Path to the "older" ONNX model file.
 */
void run_strength_test(
    const std::string& new_model_path,
    const std::string& old_model_path,
    int num_games,
    int simulations_per_move,
    int mcts_batch_size
);

} // namespace chaturaji_cpp