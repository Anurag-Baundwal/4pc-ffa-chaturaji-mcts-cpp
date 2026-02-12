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

class TensorPool {
public:
    // Acquire re-usable memory (or allocate new if pool empty)
    static std::unique_ptr<PlanesArray> acquire_planes();
    static std::unique_ptr<ScalarsArray> acquire_scalars();
    static std::unique_ptr<PolicyArray> acquire_policy();
    static std::unique_ptr<ValueArray> acquire_value();

    // Return memory to the pool
    static void release_planes(std::unique_ptr<PlanesArray> ptr);
    static void release_scalars(std::unique_ptr<ScalarsArray> ptr);
    static void release_policy(std::unique_ptr<PolicyArray> ptr);
    static void release_value(std::unique_ptr<ValueArray> ptr);
};

// --- Tensor Conversion ---

void board_to_tensors(const Board& board, float* out_planes, float* out_scalars);

/**
 * @brief Compress a board state, policy, and rewards into a PackedSample struct.
 * Used for saving training data efficiently.
 */
PackedSample create_packed_sample(
    const Board& board, 
    const std::map<Move, double>& policy, 
    const std::array<double, 16>& rewards
);

// --- Move Indexing ---

int move_to_policy_index(const Move& move, Player p);
Move policy_index_to_move(int index, const Board& board);

// --- Rank Probability & Expected Value ---

/**
 * @brief Converts rank probabilities into a single scalar expected value (utility).
 * 
 * Maps probabilities of [1st, 2nd, 3rd, 4th] place to a score using the scale:
 * 1st: +1.0, 2nd: +0.333, 3rd: -0.333, 4th: -1.0.
 * 
 * @param total_vals The accumulated 16-value probability array from a node.
 * @param player_idx Absolute index of the player (0-3).
 * @param visits The visit count of the node to calculate the average.
 * @param pessimism_factor Multiplier for negative rewards (3rd and 4th place).
 * @return double The scalar expected utility for the current player.
 */
inline double get_expected_value(const std::array<double, 16>& total_vals, int player_idx, int visits, double pessimism_factor = 1.0) {
    if (visits <= 0) return 0.0;
    
    // Calculate average probabilities for this player
    double p1 = total_vals[player_idx * 4 + 0] / visits;
    double p2 = total_vals[player_idx * 4 + 1] / visits;
    double p3 = total_vals[player_idx * 4 + 2] / visits;
    double p4 = total_vals[player_idx * 4 + 3] / visits;

    // Base rewards
    double r1 = 1.0;
    double r2 = 0.333;
    double r3 = -0.333;
    double r4 = -1.0;

    // Apply pessimism to 'losing' ranks (3rd and 4th)
    if (pessimism_factor != 1.0) {
        r3 *= pessimism_factor;
        r4 *= pessimism_factor;
    }

    return (p1 * r1) + (p2 * r2) + (p3 * r3) + (p4 * r4);
}

/**
 * @brief Calculates the ground-truth rank probabilities for a finished game.
 * Handles ties by distributing probability mass equally across tied ranks.
 * @param final_points The absolute points of all 4 players.
 * @return std::array<double, 16> Target rank probabilities (One-hot or split).
 */
std::array<double, 16> get_rank_probabilities_target(const std::array<int, 4>& final_points);

// --- Notation Utilities ---

std::string get_san_string(const Move& move, const Board& board);
std::string get_uci_string(const Move& move);
Move parse_string_to_move(const Board& board, const std::string& move_str);

} // namespace chaturaji_cpp