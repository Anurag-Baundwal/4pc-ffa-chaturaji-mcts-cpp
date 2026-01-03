#pragma once

#include <vector>
#include <array>
#include <atomic>
#include <optional>
#include <cstdint>
#include <map>
#include "types.h"

namespace chaturaji_cpp {

// Configuration
constexpr int TT_MAX_MOVES = 32;       
constexpr float TT_MIN_PROB = 0.0001f; 
constexpr int TT_CLUSTER_SIZE = 4;     

#pragma pack(push, 1)
struct SparsePolicyEntry {
    uint16_t move_idx;
    float prob; 
};

struct alignas(64) TTEntry {
    ZobristKey key = 0;
    std::array<float, NN_VALUE_SIZE> value;
    uint32_t age = 0;
    uint16_t num_moves = 0;
    SparsePolicyEntry policy_sparse[TT_MAX_MOVES];

    bool is_empty() const { return key == 0; }
};
#pragma pack(pop)

// This struct is used to pass TT data back to search quickly
struct TTData {
    std::array<float, NN_VALUE_SIZE> value;
    uint16_t num_moves;
    const SparsePolicyEntry* policy_entries;
};

class TranspositionTable {
public:
    TranspositionTable(size_t size_in_mb = 1024);

    void store(ZobristKey key, 
               const std::array<float, NN_VALUE_SIZE>& value,
               const std::map<Move, double>& policy_probs,
               Player p,
               uint32_t age);

    std::optional<TTData> probe(ZobristKey key);

    void clear();

    // --- Statistics ---
    double get_hit_rate() const;
    uint64_t get_hits() const { return hits_.load(); }
    uint64_t get_misses() const { return misses_.load(); }
    void reset_stats() { hits_ = 0; misses_ = 0; }

private:
    struct Bucket {
        std::atomic_flag lock = ATOMIC_FLAG_INIT;
        TTEntry entries[TT_CLUSTER_SIZE];
        void acquire() { while (lock.test_and_set(std::memory_order_acquire)); }
        void release() { lock.clear(std::memory_order_release); }
    };

    std::vector<Bucket> table_;
    size_t num_buckets_;

    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
};

} // namespace chaturaji_cpp