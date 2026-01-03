#include "transposition_table.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace chaturaji_cpp {

TranspositionTable::TranspositionTable(size_t size_in_mb) {
    size_t bytes_per_bucket = sizeof(Bucket);
    num_buckets_ = (size_in_mb * 1024 * 1024) / bytes_per_bucket;
    if (num_buckets_ < 2) num_buckets_ = 2;
    table_ = std::vector<Bucket>(num_buckets_);
    
    std::cout << "[TT] Size: " << size_in_mb << " MB | " 
              << "Buckets: " << num_buckets_ << " | "
              << "Entries: " << num_buckets_ * TT_CLUSTER_SIZE << std::endl;
}

void TranspositionTable::store(ZobristKey key, 
                               const std::array<float, NN_POLICY_SIZE>& full_policy_logits, 
                               const std::array<float, NN_VALUE_SIZE>& value) {
    
    size_t bucket_idx = key % num_buckets_;
    Bucket& bucket = table_[bucket_idx];

    // --- Thread-local buffer & Thresholding ---
    struct RawEntry { float logit; int idx; };
    // Allocate once per thread to avoid stack overflow and heap overhead
    static thread_local std::vector<RawEntry> candidate_buffer(NN_POLICY_SIZE);
    
    // 1. Find Max Logit (Linear pass)
    float max_logit = -1e20f;
    for (float l : full_policy_logits) if (l > max_logit) max_logit = l;

    // 2. Thresholding: Only keep "significant" moves
    // Logits more than 15.0 below max contribute effectively 0 to probability
    float threshold = max_logit - 15.0f;
    int valid_count = 0;
    for (int i = 0; i < NN_POLICY_SIZE; ++i) {
        if (full_policy_logits[i] > threshold) {
            candidate_buffer[valid_count++] = { full_policy_logits[i], i };
        }
    }

    // 3. Partial Sort only the significant moves
    int count_to_store = std::min((int)TT_MAX_MOVES, valid_count);
    std::partial_sort(candidate_buffer.begin(), candidate_buffer.begin() + count_to_store, 
                      candidate_buffer.begin() + valid_count,
        [](const RawEntry& a, const RawEntry& b) { return a.logit > b.logit; });

    // 4. Access Bucket and Find Slot
    bucket.acquire();
    TTEntry* best_slot = nullptr;
    
    // Replacement Strategy:
    // 1. If key matches, overwrite.
    // 2. If an empty slot exists, take it.
    // 3. Otherwise, replace the "oldest" entry in the bucket.
    for (int i = 0; i < TT_CLUSTER_SIZE; ++i) {
        if (bucket.entries[i].key == key) {
            best_slot = &bucket.entries[i];
            break;
        }
        // Replacement: empty slot > oldest age
        if (!best_slot || bucket.entries[i].is_empty() || bucket.entries[i].age < best_slot->age) {
            best_slot = &bucket.entries[i];
        }
    }

    // 5. Write Data
    best_slot->key = key;
    best_slot->value = value;
    best_slot->age = age_; // Store current engine age
    best_slot->num_moves = static_cast<uint16_t>(count_to_store);

    for (int i = 0; i < count_to_store; ++i) {
        best_slot->policy_sparse[i].move_idx = static_cast<uint16_t>(candidate_buffer[i].idx);
        best_slot->policy_sparse[i].logit = candidate_buffer[i].logit;
    }
    bucket.release();
}

std::optional<EvaluationResult> TranspositionTable::probe(ZobristKey key) {
    size_t bucket_idx = key % num_buckets_;
    Bucket& bucket = table_[bucket_idx];

    bucket.acquire();
    
    TTEntry* match = nullptr;
    for (int i = 0; i < TT_CLUSTER_SIZE; ++i) {
        if (bucket.entries[i].key == key) {
            match = &bucket.entries[i];
            break;
        }
    }

    if (!match) {
        bucket.release();
        return std::nullopt;
    }

    // Found match - Decompress
    EvaluationResult res;
    // Note: request_id should be set by the caller (Evaluator)
    std::fill(res.policy_logits.begin(), res.policy_logits.end(), -100.0f); // Default to very low

    for (int i = 0; i < match->num_moves; ++i) {
        res.policy_logits[match->policy_sparse[i].move_idx] = match->policy_sparse[i].logit;
    }
    res.value = match->value;

    bucket.release();
    return res;
}

void TranspositionTable::clear() {
    for (auto& bucket : table_) {
        bucket.acquire();
        for (int i = 0; i < TT_CLUSTER_SIZE; ++i) {
            bucket.entries[i].key = 0;
        }
        bucket.release();
    }
}

} // namespace chaturaji_cpp