#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>

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

// --- Tensor Conversion ---

void board_to_tensors(const Board& board, std::vector<float>& out_planes, std::vector<float>& out_scalars);

/**
 * @brief Compress a board state, policy, and rewards into a PackedSample struct.
 * Used for saving training data efficiently.
 */
PackedSample create_packed_sample(
    const Board& board, 
    const std::map<Move, double>& policy, 
    const std::array<double, 4>& rewards
);

// --- Move Indexing ---

int move_to_policy_index(const Move& move, Player p);
Move policy_index_to_move(int index, Player p);

// --- Notation Utilities ---

std::string get_san_string(const Move& move, const Board& board);
std::string get_uci_string(const Move& move);
Move parse_string_to_move(const Board& board, const std::string& move_str);

} // namespace chaturaji_cpp