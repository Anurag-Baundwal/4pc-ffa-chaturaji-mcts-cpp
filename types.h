#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <optional>

namespace chaturaji_cpp {

// --- Board Dimensions & NN Configuration ---
constexpr int BOARD_DIM = 8;
constexpr int BOARD_AREA = 64; // 8 * 8

// Input: 34 channels (Pieces, History, Meta) * 64 squares
constexpr int NN_INPUT_CHANNELS = 34; 
constexpr int NN_INPUT_SIZE = NN_INPUT_CHANNELS * BOARD_AREA; // 34 * 8 * 8 = 2176

// Output: Policy (Move probabilities) and Value (Win probabilities)
constexpr int NN_POLICY_SIZE = 4096; // 64 from_sq * 64 to_sq
constexpr int NN_VALUE_SIZE = 4;     // One value per player

using Bitboard = uint64_t;

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
};

// --- Structures for Asynchronous Evaluation ---

// Unique identifier for an evaluation request (can be for a batch)
using RequestId = uint64_t;

/**
 * @brief Data sent from an MCTS worker to the evaluator.
 */
struct EvaluationRequest {
    RequestId request_id;
    std::vector<float> state_floats; // Size: NN_INPUT_SIZE (34 * 8 * 8 = 2176)
};

/**
 * @brief Data sent from the evaluator back to the MCTS worker.
 */
struct EvaluationResult {
    RequestId request_id;
    std::array<float, NN_POLICY_SIZE> policy_logits; // Size: 4096
    std::array<float, NN_VALUE_SIZE> value;          // Size: 4
};

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