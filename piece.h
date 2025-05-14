#pragma once

#include "types.h" // Include the shared types

namespace chaturaji_cpp {

// Equivalent to Python's Piece class
class Piece {
public:
    Player player;
    PieceType piece_type;

    // Default constructor (maybe needed for optional/array initialization)
    Piece() : player(Player::RED), piece_type(PieceType::PAWN) {} // Default to a Red Pawn

    // Parameterized constructor
    Piece(Player p, PieceType pt) : player(p), piece_type(pt) {}

    // Copy constructor (default should be fine)
    Piece(const Piece& other) = default;

    // Copy assignment operator (default should be fine)
    Piece& operator=(const Piece& other) = default;

    // Move constructor (default should be fine)
    Piece(Piece&& other) = default;

    // Move assignment operator (default should be fine)
    Piece& operator=(Piece&& other) = default;

     // Equality operator (optional, but can be useful)
    bool operator==(const Piece& other) const {
        return player == other.player &&
               piece_type == other.piece_type;
    }
    
    // Inequality operator
    bool operator!=(const Piece& other) const {
      return !(*this == other); // Define in terms of operator==
    }
};

} // namespace chaturaji_cpp