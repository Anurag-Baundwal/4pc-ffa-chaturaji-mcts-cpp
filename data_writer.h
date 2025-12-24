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
            // 1. Board State
            std::vector<float> state_floats = board_to_floats(std::get<0>(step));
            
            // Validate size (NN_INPUT_SIZE = 34 * 8 * 8)
            if (state_floats.size() != NN_INPUT_SIZE) {
                std::cerr << "Error: board_to_floats returned incorrect size: " << state_floats.size() 
                          << " expected " << NN_INPUT_SIZE << std::endl;
            }
            
            outfile_.write(reinterpret_cast<const char*>(state_floats.data()), sizeof(float) * state_floats.size());

            // 2. Policy Tensor (NN_POLICY_SIZE = 4096)
            std::vector<float> policy_dense(NN_POLICY_SIZE, 0.0f);
            const auto& policy_map = std::get<1>(step);
            for (const auto& kv : policy_map) {
                int idx = move_to_policy_index(kv.first);
                if (idx >= 0 && idx < NN_POLICY_SIZE) {
                    policy_dense[idx] = static_cast<float>(kv.second);
                }
            }
            outfile_.write(reinterpret_cast<const char*>(policy_dense.data()), sizeof(float) * NN_POLICY_SIZE);

            // 3. Value (NN_VALUE_SIZE = 4)
            const std::array<double, NN_VALUE_SIZE>& rewards = std::get<3>(step);
            float rewards_float[NN_VALUE_SIZE];
            for (int i = 0; i < NN_VALUE_SIZE; ++i) rewards_float[i] = static_cast<float>(rewards[i]);
            outfile_.write(reinterpret_cast<const char*>(rewards_float), sizeof(float) * NN_VALUE_SIZE);
        }
        outfile_.flush();
    }

private:
    std::ofstream outfile_;
};

}