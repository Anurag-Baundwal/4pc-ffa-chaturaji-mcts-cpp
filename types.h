#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <stdint.h>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <memory>

namespace chaturaji_cpp {

// --- Board Dimensions & NN Configuration ---
constexpr int BOARD_DIM = 8;
constexpr int BOARD_AREA = 64; // 8 * 8

// --- Architectural Configuration ---
// Input is split into two distinct parts:
// 1. Spatial Planes: 28 planes total
//    - 20 (Pieces: 5 types * 4 players)
//    - 4 (X-Ray Attacks: 1 * 4 players)
//    - 4 (Standard Attacks: 1 * 4 players)
constexpr int NN_INPUT_PLANES = 28; 
constexpr int NN_INPUT_PLANES_SIZE = NN_INPUT_PLANES * BOARD_AREA;

// 2. Scalar Input: 18 global features total
//    4(Material) + 4(ActiveStatus) + 4(Points) + 1(50MoveClock) + 4(InCheck) + 1(OpponentCount)
constexpr int NN_INPUT_SCALARS = 18;

// --- Spatial Policy Output Configuration ---
// Output: Policy (Move probabilities) and Value (Win probabilities)
// We use a Spatial representation similar to AlphaZero: 64 Planes * 8x8 Board.
// Planes breakdown:
//  - 56 "Queen" planes (8 directions * 7 distances)
//  - 8 Knight planes
// Total spatial index [0-4095] = (PlaneIndex * 64) + FromSquareIndex (Relative)
//
// Additionally, we append one extra index for the Resignation option.
// Index 4096 = Resign
constexpr int NN_POLICY_PLANES = 64;
constexpr int NN_POLICY_INDEX_RESIGN = 4096;
constexpr int NN_POLICY_SIZE = (NN_POLICY_PLANES * BOARD_AREA) + 1; // 4097 total indices
constexpr int NN_VALUE_SIZE = 16;     // 4 players * 4 possible ranks

using Bitboard = uint64_t;

template <typename T, size_t Capacity>
class StaticVector {
public:
    StaticVector() : size_(0) {}

    // Copy constructor
    // Only copy the active elements, ignore the rest of the array.
    StaticVector(const StaticVector& other) : size_(other.size_) {
        std::copy(other.data_.begin(), other.data_.begin() + size_, data_.begin());
    }

    // Assignment operator
    StaticVector& operator=(const StaticVector& other) {
        if (this != &other) {
            size_ = other.size_;
            std::copy(other.data_.begin(), other.data_.begin() + size_, data_.begin());
        }
        return *this;
    }

    // Move ops
    StaticVector(StaticVector&& other) noexcept : size_(other.size_) {
        std::copy(other.data_.begin(), other.data_.begin() + size_, data_.begin());
        other.size_ = 0; 
    }

    StaticVector& operator=(StaticVector&& other) noexcept {
        if (this != &other) {
            size_ = other.size_;
            std::copy(other.data_.begin(), other.data_.begin() + size_, data_.begin());
            other.size_ = 0; 
        }
        return *this;
    }

    void push_back(const T& value) {
        assert(size_ < Capacity && "StaticVector capacity exceeded");
        data_[size_++] = value;
    }

    template <typename... Args>
    void emplace_back(Args&&... args) {
        assert(size_ < Capacity && "StaticVector capacity exceeded");
        data_[size_++] = T(std::forward<Args>(args)...);
    }

    void pop_back() {
        if (size_ > 0) size_--;
    }

    void clear() {
        size_ = 0;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    // Element access
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
    // Front/Back access
    T& back() { return data_[size_ - 1]; }
    const T& back() const { return data_[size_ - 1]; }

    // Iterators
    T* begin() { return data_.data(); }
    const T* begin() const { return data_.data(); }
    T* end() { return data_.data() + size_; }
    const T* end() const { return data_.data() + size_; }

private:
    std::array<T, Capacity> data_;
    size_t size_;
};

enum class Player {
    RED = 0,
    BLUE = 1,
    YELLOW = 2,
    GREEN = 3
};

enum class PieceType {
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    KING = 5,
};

enum class TerminationReason : uint8_t {
    ELIMINATION,
    FIFTY_MOVE_RULE,
    THREEFOLD_REPETITION,
    AUTOCLAIM
};

// Helper to convert termination reason enum to string for printing
inline std::string to_string(TerminationReason reason) {
    switch (reason) {
        case TerminationReason::ELIMINATION: return "elimination";
        case TerminationReason::FIFTY_MOVE_RULE: return "fifty_move_rule";
        case TerminationReason::THREEFOLD_REPETITION: return "threefold_repetition";
        case TerminationReason::AUTOCLAIM: return "autoclaim";
        default: return "unknown";
    }
}

// Operator to allow direct printing of TerminationReason to streams
inline std::ostream& operator<<(std::ostream& os, TerminationReason reason) {
    return os << to_string(reason);
}

using ZobristKey = uint64_t;

struct BoardLocation {
    int row = -1;
    int col = -1;

    // Default constructor
    BoardLocation() = default;

    // Parameterized constructor
    BoardLocation(int r, int c) : row(r), col(c) {}

    // Equality operator for comparisons (useful for maps/sets if needed)
    bool operator==(const BoardLocation& other) const {
        return row == other.row && col == other.col;
    }
    // Less than operator (needed for using BoardLocation as key in std::map/std::set)
     bool operator<(const BoardLocation& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

struct Move {
    BoardLocation from_loc;
    BoardLocation to_loc;
    std::optional<PieceType> promotion_piece_type;

    // Default constructor
    Move() = default;

    // Parameterized constructor
    Move(BoardLocation from, BoardLocation to, std::optional<PieceType> promotion = std::nullopt)
        : from_loc(from), to_loc(to), promotion_piece_type(promotion) {}

    // Equality operator
     bool operator==(const Move& other) const {
        return from_loc == other.from_loc &&
               to_loc == other.to_loc &&
               promotion_piece_type == other.promotion_piece_type;
    }

    bool operator<(const Move& other) const {
        if (from_loc < other.from_loc) return true;
        if (other.from_loc < from_loc) return false;
        // from_locs are equal, compare to_loc
        if (to_loc < other.to_loc) return true;
        if (other.to_loc < to_loc) return false;
        // to_locs are equal, compare promotion_piece_type
        // std::optional comparison: nullopt is less than any value
        return promotion_piece_type < other.promotion_piece_type;
    }

    // Static helper for Resignation Move
    static Move Resign() {
        return Move({-1, -1}, {-1, -1});
    }

    // Helper to check if this is a resignation move
    bool is_resignation() const {
        return from_loc.row == -1 && from_loc.col == -1;
    }
};

using MoveList = StaticVector<Move, 128>;

// --- Structures for Asynchronous Evaluation ---

using RequestId = uint64_t;
using PolicyArray = std::array<float, NN_POLICY_SIZE>;
using ValueArray = std::array<float, NN_VALUE_SIZE>;
using PlanesArray = std::array<float, NN_INPUT_PLANES_SIZE>;
using ScalarsArray = std::array<float, NN_INPUT_SCALARS>;

/**
 * @brief Data sent from an MCTS worker to the evaluator.
 */
struct EvaluationRequest {
    RequestId request_id;
    std::unique_ptr<PlanesArray> input_planes;
    std::unique_ptr<ScalarsArray> input_scalars;

    EvaluationRequest() : input_planes(nullptr), input_scalars(nullptr) {}

    EvaluationRequest(EvaluationRequest&&) = default;
    EvaluationRequest& operator=(EvaluationRequest&&) = default;
};

/**
 * @brief Data sent from the evaluator back to the MCTS worker.
 */
struct EvaluationResult {
    RequestId request_id;
    std::unique_ptr<PolicyArray> policy_logits;
    std::unique_ptr<ValueArray> value;

    EvaluationResult() : policy_logits(nullptr), value(nullptr) {}
    
    EvaluationResult(EvaluationResult&&) = default;
    EvaluationResult& operator=(EvaluationResult&&) = default;
};

// Maximum sparse policy entries to store (moves with very low prob are truncated)
constexpr int MAX_STORED_MOVES = 64; 

// Ensure byte alignment is 1 (Compact binary storage)
#pragma pack(push, 1)

/**
 * @brief A compact representation of a training sample.
 */
struct PackedSample {
    // --- Board State (Bitboards) ---
    uint64_t piece_bitboards[4][5];    // 20 Piece Bitboards (4 players * 5 types)
    uint64_t attack_bitboards[4];      // 4 Attack Bitboards
    uint64_t xray_attack_bitboards[4]; // 4 X-Ray Attack Bitboards

    // --- Hand-crafted Supplemental Features ---
    float material_score[4];

    // --- Game state scalars ---
    int32_t player_points[4];
    int32_t full_move_number;
    int32_t move_number_last_reset;
    uint8_t active_mask;
    uint8_t current_player;

    // Padding to align next section
    uint8_t _padding[2]; 

    // --- Sparse Policy ---
    int32_t num_policy_entries;
    uint16_t move_indices[MAX_STORED_MOVES]; // Policy indices (0-4095)
    float move_probs[MAX_STORED_MOVES];      // Probabilities associated with indices

    // --- Value ---
    float values[16]; // Final game result
};

#pragma pack(pop)

// --- Type Aliases for Board State ---
using PlayerPointMap = std::array<int, 4>; // Replaced std::map with std::array for POD layout
using ActivePlayerSet = uint8_t;           // Replaced std::set with bitmask

} // namespace chaturaji_cpp

// --- Hash Specializations ---
namespace std {
    template <>
    struct hash<chaturaji_cpp::BoardLocation> {
        size_t operator()(const chaturaji_cpp::BoardLocation& loc) const {
            // Simple hash combination
            size_t h1 = std::hash<int>{}(loc.row);
            size_t h2 = std::hash<int>{}(loc.col);
            // Combine hashes
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };

    template <>
    struct hash<chaturaji_cpp::Move> {
         size_t operator()(const chaturaji_cpp::Move& move) const {
            size_t h1 = std::hash<chaturaji_cpp::BoardLocation>{}(move.from_loc);
            size_t h2 = std::hash<chaturaji_cpp::BoardLocation>{}(move.to_loc);
            size_t h3 = 0;
            if (move.promotion_piece_type) {
                // Hash the underlying enum value if promotion exists
                 h3 = std::hash<int>{}(static_cast<int>(*move.promotion_piece_type));
            }
            // Combine hashes
            size_t seed = 0;
            seed ^= h1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
} // namespace std