#pragma once

#include <vector>
#include <array>
#include <atomic>
#include <optional>
#include <cstdint>
#include "types.h"

namespace chaturaji_cpp {

// Configuration
constexpr int TT_MAX_MOVES = 32;       // Top-32 moves cover almost all probability mass
constexpr float TT_MIN_LOGIT = -20.0f; // Softmax(exp(-20)) is effectively zero
constexpr int TT_CLUSTER_SIZE = 3;     // 3-way associativity (3 entries per bucket)

#pragma pack(push, 1)
struct SparsePolicyEntry {
    uint16_t move_idx;
    float logit;
};

// Align to 64 bytes to match CPU cache lines
struct alignas(64) TTEntry {
    ZobristKey key = 0;
    std::array<float, NN_VALUE_SIZE> value;
    uint32_t age = 0;
    uint16_t num_moves = 0;
    SparsePolicyEntry policy_sparse[TT_MAX_MOVES];

    // Helper to check if entry is empty
    bool is_empty() const { return key == 0; }
};
#pragma pack(pop)

class TranspositionTable {
public:
    /**
     * @param size_in_mb Total memory to allocate.
     */
    TranspositionTable(size_t size_in_mb = 1024);

    // Disable copying
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;

    /**
     * Stores a position. Uses stack-based Top-K sorting.
     */
    void store(ZobristKey key, 
               const std::array<float, NN_POLICY_SIZE>& full_policy_logits, 
               const std::array<float, NN_VALUE_SIZE>& value);

    /**
     * Probes for a position. Lock-protected per bucket.
     */
    std::optional<EvaluationResult> probe(ZobristKey key);

    void clear();
    void set_age(uint32_t age) { age_ = age; }

private:
    struct Bucket {
        std::atomic_flag lock = ATOMIC_FLAG_INIT;
        TTEntry entries[TT_CLUSTER_SIZE];

        // Lightweight spinlock
        void acquire() { while (lock.test_and_set(std::memory_order_acquire)); }
        void release() { lock.clear(std::memory_order_release); }
    };

    std::vector<Bucket> table_;
    size_t num_buckets_;
    uint32_t age_ = 0;
};

} // namespace chaturaji_cpp