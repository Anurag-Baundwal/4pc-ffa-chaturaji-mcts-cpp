#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm> // For std::min
#include <iomanip>   // For std::hex, std::setw, std::setfill
#include <stdexcept> // For std::runtime_error

// --- Include necessary definitions from your project ---
// Assuming these are in a shared header or you can copy relevant parts
#include "types.h" // For Bitboard, BoardLocation (if not directly defined here)
// --- END Includes ---


// --- Make sure these constants and functions are accessible ---
// --- You might need to make them static members of a helper class or put them in a namespace ---
// --- For simplicity, I'm defining them globally or assuming they are accessible ---

constexpr int BOARD_SIZE_MF = 8;
constexpr int NUM_SQUARES_MF = 64;
using Bitboard_MF = chaturaji_cpp::Bitboard; // Or uint64_t directly

// Forward declare if needed, or include definitions
namespace chaturaji_cpp {
    // Assuming these are static methods in your Board class or standalone functions
    // If they are in Board class, you'd call Board::to_sq_idx etc.
    int to_sq_idx(int r, int c) { return r * BOARD_SIZE_MF + c; }
    chaturaji_cpp::BoardLocation from_sq_idx(int sq_idx) { return {sq_idx / BOARD_SIZE_MF, sq_idx % BOARD_SIZE_MF}; }
    void set_bit(Bitboard_MF& bb, int sq_idx) { bb |= (1ULL << sq_idx); }
    // bool get_bit(Bitboard_MF bb, int sq_idx) { return (bb >> sq_idx) & 1ULL; } // Not directly used by finder, but by attack calcs
    int pop_lsb(Bitboard_MF& bb) {
        if (bb == 0) return -1;
        #ifdef _MSC_VER
            unsigned long index;
            _BitScanForward64(&index, bb);
            bb &= bb - 1;
            return static_cast<int>(index);
        #else
            int index = __builtin_ctzll(bb);
            bb &= bb - 1;
            return index;
        #endif
    }
} // namespace chaturaji_cpp


#if __cplusplus >= 202002L && defined(__cpp_lib_popcount)
inline int pop_count_mf(Bitboard_MF bb) { return std::popcount(bb); }
#elif defined(_MSC_VER)
inline int pop_count_mf(Bitboard_MF bb) { return static_cast<int>(__popcnt64(bb)); }
#elif defined(__GNUC__) || defined(__clang__)
inline int pop_count_mf(Bitboard_MF bb) { return __builtin_popcountll(bb); }
#else
inline int pop_count_mf(Bitboard_MF bb) {
    int count = 0;
    while (bb > 0) { bb &= (bb - 1); count++; }
    return count;
}
#endif
// --- End Prerequisite Definitions ---


// --- On-the-fly attack calculators (copied from your board.cpp for standalone use) ---
Bitboard_MF calculate_rook_attacks_on_the_fly_mf(int sq, Bitboard_MF occupied) {
    Bitboard_MF attacks = 0ULL;
    chaturaji_cpp::BoardLocation loc = chaturaji_cpp::from_sq_idx(sq);
    int r0 = loc.row;
    int c0 = loc.col;
    const int dr[] = {-1, 1, 0, 0};
    const int dc[] = {0, 0, 1, -1};

    for (int i = 0; i < 4; ++i) {
        for (int k = 1; k < BOARD_SIZE_MF; ++k) {
            int r = r0 + dr[i] * k;
            int c = c0 + dc[i] * k;
            if (r < 0 || r >= BOARD_SIZE_MF || c < 0 || c >= BOARD_SIZE_MF) break;
            int target_sq = chaturaji_cpp::to_sq_idx(r,c);
            chaturaji_cpp::set_bit(attacks, target_sq);
            if ((occupied >> target_sq) & 1ULL) break;
        }
    }
    return attacks;
}

Bitboard_MF calculate_bishop_attacks_on_the_fly_mf(int sq, Bitboard_MF occupied) {
    Bitboard_MF attacks = 0ULL;
    chaturaji_cpp::BoardLocation loc = chaturaji_cpp::from_sq_idx(sq);
    int r0 = loc.row;
    int c0 = loc.col;
    const int dr[] = {-1, 1, 1, -1};
    const int dc[] = {1, 1, -1, -1};

    for (int i = 0; i < 4; ++i) {
        for (int k = 1; k < BOARD_SIZE_MF; ++k) {
            int r = r0 + dr[i] * k;
            int c = c0 + dc[i] * k;
             if (r < 0 || r >= BOARD_SIZE_MF || c < 0 || c >= BOARD_SIZE_MF) break;
            int target_sq = chaturaji_cpp::to_sq_idx(r,c);
            chaturaji_cpp::set_bit(attacks, target_sq);
            if ((occupied >> target_sq) & 1ULL) break;
        }
    }
    return attacks;
}
// --- End On-the-fly attack calculators ---


// --- Blocker Mask Generation (copied & adapted from your board.cpp) ---
Bitboard_MF generate_rook_mask_mf(int sq) {
    Bitboard_MF mask = 0ULL;
    chaturaji_cpp::BoardLocation loc = chaturaji_cpp::from_sq_idx(sq);
    int r0 = loc.row; int c0 = loc.col;

    for (int r_idx = r0 + 1; r_idx < BOARD_SIZE_MF - 1; ++r_idx) chaturaji_cpp::set_bit(mask, chaturaji_cpp::to_sq_idx(r_idx, c0)); // South (inner)
    for (int r_idx = r0 - 1; r_idx > 0; --r_idx) chaturaji_cpp::set_bit(mask, chaturaji_cpp::to_sq_idx(r_idx, c0));             // North (inner)
    for (int c_idx = c0 + 1; c_idx < BOARD_SIZE_MF - 1; ++c_idx) chaturaji_cpp::set_bit(mask, chaturaji_cpp::to_sq_idx(r0, c_idx)); // East (inner)
    for (int c_idx = c0 - 1; c_idx > 0; --c_idx) chaturaji_cpp::set_bit(mask, chaturaji_cpp::to_sq_idx(r0, c_idx));             // West (inner)
    return mask;
}

Bitboard_MF generate_bishop_mask_mf(int sq) {
    Bitboard_MF mask = 0ULL;
    chaturaji_cpp::BoardLocation loc = chaturaji_cpp::from_sq_idx(sq);
    int r0 = loc.row; int c0 = loc.col;

    const int dr[] = {-1, 1, 1, -1}; const int dc[] = {1, 1, -1, -1};
    for (int dir_idx = 0; dir_idx < 4; ++dir_idx) {
        for (int k = 1; k < BOARD_SIZE_MF ; ++k) { // Iterate up to one square before edge
            int r = r0 + dr[dir_idx] * k;
            int c = c0 + dc[dir_idx] * k;
            // Only add to mask if the *next* square is on the board
            // (i.e., r,c is not an edge square for this ray segment)
            if (r > 0 && r < BOARD_SIZE_MF - 1 && c > 0 && c < BOARD_SIZE_MF - 1) {
                chaturaji_cpp::set_bit(mask, chaturaji_cpp::to_sq_idx(r, c));
            } else {
                break; // Reached edge or beyond
            }
        }
    }
    return mask;
}
// --- End Blocker Mask Generation ---

// Helper to get the i-th occupancy permutation for a given mask
Bitboard_MF get_occupancy_subset_mf(int subset_index, int bits_in_mask, Bitboard_MF mask) {
    Bitboard_MF occupancy = 0ULL;
    Bitboard_MF temp_mask = mask;
    for (int i = 0; i < bits_in_mask; ++i) {
        int lsb_sq = chaturaji_cpp::pop_lsb(temp_mask);
        if (lsb_sq == -1) break;
        if ((subset_index >> i) & 1) {
            chaturaji_cpp::set_bit(occupancy, lsb_sq);
        }
    }
    return occupancy;
}

// Random number generator for magic candidates
std::mt19937_64 rng_magic(std::random_device{}());

Bitboard_MF generate_random_magic_candidate() {
    // "Sparse" magics (fewer bits set) often work better/faster
    return rng_magic() & rng_magic() & rng_magic();
}

// Function to find a magic number for a given square and piece type
Bitboard_MF find_magic_for_square(int sq, bool is_rook,
                                 std::vector<Bitboard_MF>& occupancies, // Out param
                                 std::vector<Bitboard_MF>& attacks,     // Out param
                                 int& shift_bits)                       // Out param
{
    Bitboard_MF blocker_mask = is_rook ? generate_rook_mask_mf(sq) : generate_bishop_mask_mf(sq);
    int num_mask_bits = pop_count_mf(blocker_mask);
    shift_bits = 64 - num_mask_bits; // This is the common shift

    size_t num_occupancy_permutations = 1ULL << num_mask_bits;
    occupancies.assign(num_occupancy_permutations, 0ULL);
    attacks.assign(num_occupancy_permutations, 0ULL);

    // 1. Generate all occupancy permutations and their corresponding true attack sets
    for (size_t i = 0; i < num_occupancy_permutations; ++i) {
        occupancies[i] = get_occupancy_subset_mf(i, num_mask_bits, blocker_mask);
        if (is_rook) {
            attacks[i] = calculate_rook_attacks_on_the_fly_mf(sq, occupancies[i]);
        } else {
            attacks[i] = calculate_bishop_attacks_on_the_fly_mf(sq, occupancies[i]);
        }
    }

    // 2. Find a magic number
    // This temporary table is used to check for collisions for a *single* magic candidate
    std::vector<Bitboard_MF> temp_attack_table(num_occupancy_permutations, 0ULL);
    std::vector<bool>        temp_table_used(num_occupancy_permutations, false);

    for (int try_count = 0; try_count < 1000000; ++try_count) { // Try up to 1M candidates
        Bitboard_MF magic_candidate = generate_random_magic_candidate();
        
        // Optimization: if magic * mask has too few high bits, it's unlikely to work well
        if (pop_count_mf((blocker_mask * magic_candidate) & 0xFF00000000000000ULL) < 6) {
             continue;
        }

        std::fill(temp_table_used.begin(), temp_table_used.end(), false);
        bool possible_magic = true;

        for (size_t i = 0; i < num_occupancy_permutations; ++i) {
            unsigned int magic_index = (occupancies[i] * magic_candidate) >> shift_bits;

            if (!temp_table_used[magic_index]) {
                temp_attack_table[magic_index] = attacks[i];
                temp_table_used[magic_index] = true;
            } else if (temp_attack_table[magic_index] != attacks[i]) {
                possible_magic = false;
                break; // Collision with different attack set, this magic fails
            }
        }

        if (possible_magic) {
            return magic_candidate; // Found a working magic!
        }
    }

    std::cerr << "!!! FAILED to find magic for sq " << sq << (is_rook ? " (ROOK)" : " (BISHOP)")
              << " after many tries." << std::endl;
    return 0ULL; // Indicate failure
}


int main() {
    std::cout << "Generating Magic Bitboard Numbers...\n" << std::endl;

    std::array<Bitboard_MF, NUM_SQUARES_MF> rook_magics;
    std::array<int, NUM_SQUARES_MF> rook_shifts;
    std::array<Bitboard_MF, NUM_SQUARES_MF> bishop_magics;
    std::array<int, NUM_SQUARES_MF> bishop_shifts;

    // Temporary storage for each square's specific attack table entries (not the global table)
    std::vector<Bitboard_MF> occupancies_temp;
    std::vector<Bitboard_MF> attacks_temp;

    std::cout << "// --- ROOK MAGICS ---" << std::endl;
    std::cout << "const std::array<Bitboard, NUM_SQUARES_BB> RookMagics = {" << std::endl;
    for (int sq = 0; sq < NUM_SQUARES_MF; ++sq) {
        rook_magics[sq] = find_magic_for_square(sq, true, occupancies_temp, attacks_temp, rook_shifts[sq]);
        std::cout << "    0x" << std::hex << std::setw(16) << std::setfill('0') << rook_magics[sq] << "ULL,"
                  << " // sq " << std::dec << sq << " (shift " << rook_shifts[sq] << ")" << std::endl;
        if (rook_magics[sq] == 0ULL) {
             std::cerr << "CRITICAL: Could not find rook magic for square " << sq << std::endl;
             return 1;
        }
    }
    std::cout << "};" << std::endl << std::endl;

    // You would also print rook_shifts_ (though they are derivable if you store masks)
    // and populate the actual rook_attack_table_ in your Board class using these.

    std::cout << "// --- BISHOP MAGICS ---" << std::endl;
    std::cout << "const std::array<Bitboard, NUM_SQUARES_BB> BishopMagics = {" << std::endl;
    for (int sq = 0; sq < NUM_SQUARES_MF; ++sq) {
        bishop_magics[sq] = find_magic_for_square(sq, false, occupancies_temp, attacks_temp, bishop_shifts[sq]);
        std::cout << "    0x" << std::hex << std::setw(16) << std::setfill('0') << bishop_magics[sq] << "ULL,"
                  << " // sq " << std::dec << sq << " (shift " << bishop_shifts[sq] << ")" << std::endl;
         if (bishop_magics[sq] == 0ULL) {
             std::cerr << "CRITICAL: Could not find bishop magic for square " << sq << std::endl;
             return 1;
        }
    }
    std::cout << "};" << std::endl << std::endl;

    std::cout << "// --- ROOK SHIFTS (can be derived, but useful for verification) ---" << std::endl;
    std::cout << "const std::array<int, NUM_SQUARES_BB> RookShifts = {" << std::endl;
    for(int sq=0; sq<NUM_SQUARES_MF; ++sq) {
        std::cout << "    " << rook_shifts[sq] << ((sq % 8 == 7) ? ",\n" : ", ");
    }
    std::cout << "};" << std::endl << std::endl;

    std::cout << "// --- BISHOP SHIFTS (can be derived, but useful for verification) ---" << std::endl;
    std::cout << "const std::array<int, NUM_SQUARES_BB> BishopShifts = {" << std::endl;
    for(int sq=0; sq<NUM_SQUARES_MF; ++sq) {
        std::cout << "    " << bishop_shifts[sq] << ((sq % 8 == 7) ? ",\n" : ", ");
    }
    std::cout << "};" << std::endl << std::endl;


    std::cout << "\nMagic number generation complete." << std::endl;
    std::cout << "Remember to integrate these into your Board class and thoroughly test!" << std::endl;
    std::cout << "You will also need to generate the rook_masks_ and bishop_masks_ within your Board class "
              << "and use them with these magics and shifts to populate your global attack tables." << std::endl;

    return 0;
}