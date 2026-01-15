#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>

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

std::vector<float> board_to_floats(const Board& board);

// --- Move Indexing ---

int move_to_policy_index(const Move& move, Player p);
Move policy_index_to_move(int index, Player p);

// --- Notation Utilities ---

std::string get_san_string(const Move& move, const Board& board);
std::string get_uci_string(const Move& move);
Move parse_string_to_move(const Board& board, const std::string& move_str);

} // namespace chaturaji_cpp