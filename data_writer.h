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
            // 1. Board State (using std::vector<float>)
            std::vector<float> state_floats = board_to_floats(std::get<0>(step));
            
            // Validate size (34 channels * 8 * 8)
            if (state_floats.size() != 34 * 8 * 8) {
                std::cerr << "Error: board_to_floats returned incorrect size: " << state_floats.size() << std::endl;
            }
            
            outfile_.write(reinterpret_cast<const char*>(state_floats.data()), sizeof(float) * state_floats.size());

            // 2. Policy Tensor (4096)
            std::vector<float> policy_dense(4096, 0.0f);
            const auto& policy_map = std::get<1>(step);
            for (const auto& kv : policy_map) {
                int idx = move_to_policy_index(kv.first);
                if (idx >= 0 && idx < 4096) {
                    policy_dense[idx] = static_cast<float>(kv.second);
                }
            }
            outfile_.write(reinterpret_cast<const char*>(policy_dense.data()), sizeof(float) * 4096);

            // 3. Value (4)
            const std::array<double, 4>& rewards = std::get<3>(step);
            float rewards_float[4];
            for (int i = 0; i < 4; ++i) rewards_float[i] = static_cast<float>(rewards[i]);
            outfile_.write(reinterpret_cast<const char*>(rewards_float), sizeof(float) * 4);
        }
        outfile_.flush();
    }

private:
    std::ofstream outfile_;
};

}