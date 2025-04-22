#pragma once // Use pragma once for include guard (common practice)

#include <optional> // For optional promotion piece type
#include <functional> // For std::hash needed later

namespace chaturaji_cpp {

// Equivalent to Python's Player Enum
enum class Player {
    RED = 0,
    BLUE = 1,
    YELLOW = 2,
    GREEN = 3
};

// Equivalent to Python's PieceType Enum
enum class PieceType {
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    QUEEN = 5,          // Keep for potential future use / evaluation function?
    KING = 6,
    ONE_POINT_QUEEN = 7,// Keep for potential future use / evaluation function?
    DEAD_KING = 9
};

// Equivalent to Python's BoardLocation class
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
        if (row != other.row) {
            return row < other.row;
        }
        return col < other.col;
    }
};

// Equivalent to Python's Move class
struct Move {
    BoardLocation from_loc;
    BoardLocation to_loc;
    std::optional<PieceType> promotion_piece_type; // Use optional for promotion

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

    // --- FIX: Add operator< for std::map ---
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
    // --- END FIX ---
};

} // namespace chaturaji_cpp

// Provide hash specializations for BoardLocation and Move if needed for unordered containers
namespace std {
    template <>
    struct hash<chaturaji_cpp::BoardLocation> {
        size_t operator()(const chaturaji_cpp::BoardLocation& loc) const {
            // Simple hash combination
            size_t h1 = std::hash<int>{}(loc.row);
            size_t h2 = std::hash<int>{}(loc.col);
            // Combine hashes (improved version to reduce collisions)
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