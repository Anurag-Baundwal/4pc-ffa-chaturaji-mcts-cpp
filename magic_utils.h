#pragma once

#include <array>
#include <vector>
#include <string>
#include <cstdint> // For uint64_t

#include "types.h" // For Bitboard, BoardLocation

#ifdef _MSC_VER
#include <intrin.h> // For MSVC specific intrinsics
#endif

namespace chaturaji_cpp {
namespace magic_utils {

// --- Constants ---
constexpr int BOARD_SIZE = 8;
constexpr int NUM_SQUARES = 64;

// --- Bit Manipulation Helpers ---
inline void set_bit(Bitboard& bb, int sq_idx) { 
    bb |= (1ULL << sq_idx); 
}

inline void clear_bit(Bitboard& bb, int sq_idx) { 
    bb &= ~(1ULL << sq_idx); 
}

inline bool get_bit(Bitboard bb, int sq_idx) { 
    return (bb >> sq_idx) & 1ULL; 
}

inline int pop_lsb(Bitboard& bb) {
    if (bb == 0) return -1;
    #ifdef _MSC_VER
        unsigned long index;
        _BitScanForward64(&index, bb);
        bb &= bb - 1; // Clears the LSB
        return static_cast<int>(index);
    #else // Assuming GCC/Clang
        int index = __builtin_ctzll(bb); // Count trailing zeros (index of LSB)
        bb &= bb - 1; // Clears the LSB
        return index;
    #endif
}

inline int get_lsb_index(Bitboard bb) {
    if (bb == 0) return -1;
    #ifdef _MSC_VER
        unsigned long index;
        _BitScanForward64(&index, bb);
        return static_cast<int>(index);
    #else // Assuming GCC/Clang
        return __builtin_ctzll(bb);
    #endif
}

#if __cplusplus >= 202002L && defined(__cpp_lib_popcount)
// Use C++20 std::popcount if available
inline int pop_count(Bitboard bb) {
    return std::popcount(bb);
}
#elif defined(_MSC_VER)
// MSVC specific
inline int pop_count(Bitboard bb) {
    return static_cast<int>(__popcnt64(bb));
}
#elif defined(__GNUC__) || defined(__clang__)
// GCC/Clang specific
inline int pop_count(Bitboard bb) {
    return __builtin_popcountll(bb);
}
#else
// Fallback pop_count (less efficient)
inline int pop_count(Bitboard bb) {
    int count = 0;
    while (bb > 0) {
        bb &= (bb - 1);
        count++;
    }
    return count;
}
#endif

// --- Square Conversion Utilities ---
inline int to_sq_idx(int r, int c) {
    return r * BOARD_SIZE + c;
}

inline BoardLocation from_sq_idx(int sq_idx) {
    return {sq_idx / BOARD_SIZE, sq_idx % BOARD_SIZE};
}

// --- Mask Generation for Sliding Pieces ---
Bitboard generate_rook_mask(int sq);
Bitboard generate_bishop_mask(int sq);

// --- On-the-fly Attack Calculation for Sliding Pieces ---
Bitboard calculate_rook_attacks_on_the_fly(int sq, Bitboard occupied);
Bitboard calculate_bishop_attacks_on_the_fly(int sq, Bitboard occupied);

// --- Occupancy Subset Generation ---
Bitboard get_occupancy_subset(int index, int bits_in_mask, Bitboard mask);

// --- Pre-generated Magic Numbers and Shifts ---
extern const std::array<Bitboard, NUM_SQUARES> RookMagics;
extern const std::array<Bitboard, NUM_SQUARES> BishopMagics;
extern const std::array<int, NUM_SQUARES> RookShifts;
extern const std::array<int, NUM_SQUARES> BishopShifts;

} // namespace magic_utils
} // namespace chaturaji_cpp