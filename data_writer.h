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
            const Board& board = std::get<0>(step);
            const auto& policy_map = std::get<1>(step);
            // Extract data
            const std::array<double, 16>& abs_rewards = std::get<3>(step); 

            // Pack it
            PackedSample sample = create_packed_sample(board, policy_map, abs_rewards);

            // Write raw struct bytes
            outfile_.write(reinterpret_cast<const char*>(&sample), sizeof(PackedSample));
        }
        outfile_.flush();
    }

private:
    std::ofstream outfile_;
};

}