#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "types.h"
#include "utils.h"

namespace chaturaji_cpp {

class DataWriter {
public:
    DataWriter(const std::string& filename) {
        outfile_.open(filename, std::ios::binary | std::ios::app);
        if (!outfile_.is_open()) {
            std::cerr << "Error opening data file for writing: " << filename << std::endl;
        }
    }

    ~DataWriter() {
        if (outfile_.is_open()) {
            outfile_.close();
        }
    }

    void write_batch(const std::vector<GameDataStep>& data) {
        if (!outfile_.is_open()) return;

        for (const auto& step : data) {
            // 0. Extract shared data first
            const Board& board = std::get<0>(step);
            const auto& policy_map = std::get<1>(step);
            Player move_player = std::get<2>(step);
            const std::array<double, 4>& abs_rewards = std::get<3>(step);

            // 1. Board State
            std::vector<float> state_floats = board_to_floats(board);
            
            if (state_floats.size() != NN_INPUT_SIZE) {
                std::cerr << "Error: board_to_floats returned incorrect size: " << state_floats.size() 
                          << " expected " << NN_INPUT_SIZE << std::endl;
            }
            
            outfile_.write(reinterpret_cast<const char*>(state_floats.data()), sizeof(float) * state_floats.size());

            // 2. Policy Tensor (NN_POLICY_SIZE = 4096)
            std::vector<float> policy_dense(NN_POLICY_SIZE, 0.0f);
            for (const auto& kv : policy_map) {
                int idx = move_to_policy_index(kv.first, move_player); 
                if (idx >= 0 && idx < NN_POLICY_SIZE) {
                    policy_dense[idx] = static_cast<float>(kv.second);
                }
            }
            outfile_.write(reinterpret_cast<const char*>(policy_dense.data()), sizeof(float) * NN_POLICY_SIZE);

            // 3. Value (NN_VALUE_SIZE = 4)
            int cp_idx = static_cast<int>(move_player);

            float rel_rewards[4];
            for (int rel_i = 0; rel_i < 4; ++rel_i) {
                rel_rewards[rel_i] = static_cast<float>(abs_rewards[(cp_idx + rel_i) % 4]);
            }
            outfile_.write(reinterpret_cast<const char*>(rel_rewards), sizeof(float) * 4);
        }
        outfile_.flush();
    }

private:
    std::ofstream outfile_;
};

}