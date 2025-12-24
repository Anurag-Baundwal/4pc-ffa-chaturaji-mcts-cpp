#pragma once

#include <vector>
#include <string>
#include <cstdint>

#include "board.h" 
#include "types.h" 

namespace chaturaji_cpp {

// --- Tensor Conversion (Decoupled) ---

/**
 * @brief Converts the current board state into a flat vector of floats [C, H, W].
 *
 * Channels (NN_INPUT_CHANNELS total):
 *   0-19:  Piece planes (5 types * 4 players)
 *   20-23: Active player status
 *   24-27: Current player turn
 *   28-31: Player points
 *   32:    50-move counter
 *   33:    Incoming attacks
 *
 * @param board The board object to convert.
 * @return std::vector<float> Raw float data of size NN_INPUT_SIZE (NN_INPUT_CHANNELS * 8 * 8).
 */
std::vector<float> board_to_floats(const Board& board);

// --- Move Indexing ---

int move_to_policy_index(const Move& move);
Move policy_index_to_move(int index);

// --- Notation Utilities ---

std::string get_san_string(const Move& move, const Board& board);
std::string get_uci_string(const Move& move);

} // namespace chaturaji_cpp