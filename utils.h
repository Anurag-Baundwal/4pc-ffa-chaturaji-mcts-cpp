#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <fstream> // Added for RunStats
#include <iostream> // Added for RunStats

#include "board.h" 
#include "types.h" 

namespace chaturaji_cpp {

// --- Stats For The Current Training Run ---

struct RunStats {
    int global_iteration = 0;
    size_t total_samples_generated = 0;
    
    // Stats specific to the current active session (reset on program start)
    int session_iterations = 0;
    size_t session_samples = 0;

    void save(const std::string& filepath) const;
    static RunStats load(const std::string& filepath);
};

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

int move_to_policy_index(const Move& move, Player p);
Move policy_index_to_move(int index, Player p);

// --- Notation Utilities ---

std::string get_san_string(const Move& move, const Board& board);
std::string get_uci_string(const Move& move);

/**
 * @brief Parses a move string (e.g., "e2e4", "e2-e4", "a7a8r", "a7-a8=R") 
 * and returns the corresponding legal Move object.
 * Throws std::invalid_argument if the move is invalid or illegal in the current position.
 */
Move parse_string_to_move(const Board& board, const std::string& move_str);

} // namespace chaturaji_cpp