#include "transposition_table.h"
#include "utils.h"
#include <algorithm>
#include <iostream>

namespace chaturaji_cpp {

TranspositionTable::TranspositionTable(size_t size_in_mb) {
    size_t bytes_per_bucket = sizeof(Bucket);
    num_buckets_ = (size_in_mb * 1024 * 1024) / bytes_per_bucket;
    if (num_buckets_ < 2) num_buckets_ = 2;
    table_ = std::vector<Bucket>(num_buckets_);
    hits_ = 0;
    misses_ = 0;
    std::cout << "[TT] Optimized Probabilities Mode | Size: " << size_in_mb << " MB" << std::endl;
}

void TranspositionTable::store(ZobristKey key, 
                               const std::array<float, NN_VALUE_SIZE>& value,
                               const std::map<Move, double>& policy_probs,
                               Player p,
                               uint32_t age) {
    
    size_t bucket_idx = key % num_buckets_;
    Bucket& bucket = table_[bucket_idx];

    struct TempMove { uint16_t idx; float prob; };
    static thread_local std::vector<TempMove> top_moves;
    top_moves.clear();

    for (const auto& [move, prob] : policy_probs) {
        if (prob > TT_MIN_PROB) {
            top_moves.push_back({ (uint16_t)move_to_policy_index(move, p), (float)prob });
        }
    }

    // Sort the legal moves map (usually small)
    std::sort(top_moves.begin(), top_moves.end(), [](const auto& a, const auto& b) {
        return a.prob > b.prob;
    });

    int count = std::min((int)TT_MAX_MOVES, (int)top_moves.size());

    bucket.acquire();
    TTEntry* best_slot = &bucket.entries[0];
    for (int i = 0; i < TT_CLUSTER_SIZE; ++i) {
        if (bucket.entries[i].key == key) {
            best_slot = &bucket.entries[i];
            break;
        }
        if (bucket.entries[i].is_empty() || bucket.entries[i].age < best_slot->age) {
            best_slot = &bucket.entries[i];
        }
    }

    best_slot->key = key;
    best_slot->value = value;
    best_slot->age = age;
    best_slot->num_moves = (uint16_t)count;
    for (int i = 0; i < count; ++i) {
        best_slot->policy_sparse[i].move_idx = top_moves[i].idx;
        best_slot->policy_sparse[i].prob = top_moves[i].prob;
    }
    bucket.release();
}

std::optional<TTData> TranspositionTable::probe(ZobristKey key) {
    size_t bucket_idx = key % num_buckets_;
    Bucket& bucket = table_[bucket_idx];

    for (int i = 0; i < TT_CLUSTER_SIZE; ++i) {
        if (bucket.entries[i].key == key) {
            hits_++;
            return TTData{ bucket.entries[i].value, bucket.entries[i].num_moves, bucket.entries[i].policy_sparse };
        }
    }
    misses_++;
    return std::nullopt;
}

void TranspositionTable::clear() {
    for (auto& bucket : table_) {
        bucket.acquire();
        for (int i = 0; i < TT_CLUSTER_SIZE; ++i) bucket.entries[i].key = 0;
        bucket.release();
    }
}

double TranspositionTable::get_hit_rate() const {
    uint64_t h = hits_.load();
    uint64_t m = misses_.load();
    uint64_t total = h + m;
    if (total == 0) return 0.0;
    return static_cast<double>(h) / static_cast<double>(total);
}

} // namespace chaturaji_cpp