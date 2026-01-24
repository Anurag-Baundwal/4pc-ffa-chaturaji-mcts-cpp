/**
 * @file pack_test_gen.cpp
 * @brief Verification Tool: Generates Ground Truth Data.
 * 
 * This program sets up a specific board state, creates a fake policy and reward,
 * and exports two versions of the data:
 * 1. Raw Floats (Ground Truth): The state unpacked into floats directly by C++.
 * 2. Packed Binary: The struct format used for training data storage.
 * 
 * These files are consumed by `pack_verify.py` to ensure the Python-side
 * unpacking logic matches the C++ ground truth bit-for-bit.
 */
 
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <filesystem>
#include <stdexcept>

#include "board.h"
#include "types.h"
#include "utils.h"
#include "magic_utils.h"

using namespace chaturaji_cpp;

void write_raw_floats(const std::string& filename, const std::vector<float>& data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Could not open " + filename + " for writing.");
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    out.close();
}

void write_packed_struct(const std::string& filename, const PackedSample& sample) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Could not open " + filename + " for writing.");
    out.write(reinterpret_cast<const char*>(&sample), sizeof(PackedSample));
    out.close();
}

int main() {
    try {
        std::cout << "--- Generating Packing Test Data ---" << std::endl;

        Board board;
        
        // --- 1. SETUP STATE WITH LEGAL MOVES ---
        // Red (a2-a3)
        board.make_move(parse_string_to_move(board, "a2-a3")); 
        // Blue (b7-c7)
        board.make_move(parse_string_to_move(board, "b7-c7"));

        std::cout << "Board state prepared. Current Player: " << (int)board.get_current_player() << std::endl;

        // --- 2. CREATE FAKE POLICY ---
        // Note: The player is now YELLOW (2) because Red and Blue moved.
        // Let's create moves for Yellow: e7-e6
        std::map<Move, double> policy;
        Move m1 = parse_string_to_move(board, "e7-e6");
        policy[m1] = 1.0;

        // --- 3. CREATE FAKE REWARDS ---
        std::array<double, 4> rewards = {0.5, -0.5, 1.0, 0.0};

        // --- 4. GROUND TRUTH (Old Way) ---
        std::vector<float> ground_truth_input(NN_INPUT_SIZE);
        board_to_floats_into(board, ground_truth_input);

        std::vector<float> ground_truth_policy(NN_POLICY_SIZE, 0.0f);
        Player p = board.get_current_player();
        int idx = move_to_policy_index(m1, p);
        ground_truth_policy[idx] = 1.0f;

        // --- 5. PACKED SAMPLE (New Way) ---
        PackedSample packed = create_packed_sample(board, policy, rewards);

        // --- 6. SAVE ---
        write_raw_floats("test_truth_input.bin", ground_truth_input);
        write_raw_floats("test_truth_policy.bin", ground_truth_policy);
        write_raw_floats("test_truth_value.bin", { (float)rewards[0], (float)rewards[1], (float)rewards[2], (float)rewards[3] });
        write_packed_struct("test_packed.bin", packed);

        std::cout << "Successfully wrote test files." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}