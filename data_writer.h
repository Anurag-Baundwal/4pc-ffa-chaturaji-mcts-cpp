#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <torch/torch.h>
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
            // 1. Board State Tensor (33, 8, 8)
            // We assume board_to_tensor works and returns a CPU tensor [1, 33, 8, 8]
            // We need to construct it here because GameDataStep holds the Board object, not the tensor
            // Note: This relies on linking against board_to_tensor from utils
            torch::Tensor state = board_to_tensor(std::get<0>(step), torch::kCPU).squeeze(0); // [33, 8, 8]
            
            // Ensure contiguous memory for raw write
            state = state.contiguous();
            outfile_.write(reinterpret_cast<const char*>(state.data_ptr<float>()), sizeof(float) * 33 * 8 * 8);

            // 2. Policy Tensor (4096)
            // Convert map to dense vector
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