#include "magic_utils.h"
#include <stdexcept> // For std::out_of_range (though not directly used in these specific functions)

namespace chaturaji_cpp {
namespace magic_utils {

// --- Mask Generation Definitions ---
Bitboard generate_rook_mask(int sq) {
    Bitboard mask = 0ULL;
    BoardLocation loc = from_sq_idx(sq); // Uses magic_utils::from_sq_idx
    int r0 = loc.row; int c0 = loc.col;

    // Corrected Mask Generation (simpler approach):
    // Mask includes relevant squares for blocking, excluding edges unless piece is on edge.
    // These are squares whose occupancy can change the attack set from sq.
    // "Inner" squares means not on rank 1/8 or file A/H.
    for (int r_idx = r0 + 1; r_idx < BOARD_SIZE - 1; ++r_idx) set_bit(mask, to_sq_idx(r_idx, c0)); // South (inner)
    for (int r_idx = r0 - 1; r_idx > 0; --r_idx) set_bit(mask, to_sq_idx(r_idx, c0));             // North (inner)
    for (int c_idx = c0 + 1; c_idx < BOARD_SIZE - 1; ++c_idx) set_bit(mask, to_sq_idx(r0, c_idx)); // East (inner)
    for (int c_idx = c0 - 1; c_idx > 0; --c_idx) set_bit(mask, to_sq_idx(r0, c_idx));             // West (inner)
    return mask;
}

Bitboard generate_bishop_mask(int sq) {
    Bitboard mask = 0ULL;
    BoardLocation loc = from_sq_idx(sq); // Uses magic_utils::from_sq_idx
    int r0 = loc.row; int c0 = loc.col;
    
    const int bishop_dr[] = {-1, 1, 1, -1}; 
    const int bishop_dc[] = {1, 1, -1, -1};
    for (int dir_idx = 0; dir_idx < 4; ++dir_idx) {
        for (int k = 1; k < BOARD_SIZE; ++k) { 
            int r = r0 + bishop_dr[dir_idx] * k;
            int c = c0 + bishop_dc[dir_idx] * k;
            // Only add to mask if the square is an "inner" board square
            // (not on the 1st/8th rank or a/h file for this ray segment)
            if (r > 0 && r < BOARD_SIZE - 1 && c > 0 && c < BOARD_SIZE - 1) {
                set_bit(mask, to_sq_idx(r, c));
            } else {
                // Stop this ray if it hits an edge (or goes off-board for non-inner squares)
                // because squares on the edge itself don't block further for "inner" squares.
                break; 
            }
        }
    }
    return mask;
}

// --- On-the-fly Attack Calculation Definitions ---
Bitboard calculate_rook_attacks_on_the_fly(int sq, Bitboard occupied) {
    Bitboard attacks = 0ULL;
    BoardLocation loc = from_sq_idx(sq); // Uses magic_utils::from_sq_idx
    int r0 = loc.row;
    int c0 = loc.col;
    // N, S, E, W
    const int dr[] = {-1, 1, 0, 0};
    const int dc[] = {0, 0, 1, -1};

    for (int i = 0; i < 4; ++i) { // For each of the 4 rook directions
        for (int k = 1; k < BOARD_SIZE; ++k) {
            int r = r0 + dr[i] * k;
            int c = c0 + dc[i] * k;
            if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) break; // Off board
            int target_sq = to_sq_idx(r,c); // Uses magic_utils::to_sq_idx
            set_bit(attacks, target_sq);    // Uses magic_utils::set_bit
            if (get_bit(occupied, target_sq)) break; // Blocked by a piece in 'occupied'
        }
    }
    return attacks;
}

Bitboard calculate_bishop_attacks_on_the_fly(int sq, Bitboard occupied) {
    Bitboard attacks = 0ULL;
    BoardLocation loc = from_sq_idx(sq); // Uses magic_utils::from_sq_idx
    int r0 = loc.row;
    int c0 = loc.col;
    // NE, SE, SW, NW
    const int dr[] = {-1, 1, 1, -1};
    const int dc[] = {1, 1, -1, -1};

    for (int i = 0; i < 4; ++i) { // For each of the 4 bishop directions
        for (int k = 1; k < BOARD_SIZE; ++k) {
            int r = r0 + dr[i] * k;
            int c = c0 + dc[i] * k;
            if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) break; // Off board
            int target_sq = to_sq_idx(r,c); // Uses magic_utils::to_sq_idx
            set_bit(attacks, target_sq);    // Uses magic_utils::set_bit
            if (get_bit(occupied, target_sq)) break; // Blocked
        }
    }
    return attacks;
}

// --- Occupancy Subset Generation Definition ---
Bitboard get_occupancy_subset(int index, int bits_in_mask, Bitboard mask) {
    Bitboard occupancy = 0ULL;
    Bitboard temp_mask = mask; // Iterate over bits in the mask
    for (int i = 0; i < bits_in_mask; ++i) {
        int lsb_sq = pop_lsb(temp_mask); // Uses magic_utils::pop_lsb
        if (lsb_sq == -1) break; // Should not happen if bits_in_mask is correct
        if ((index >> i) & 1) {    // If the i-th bit of `index` is set
            set_bit(occupancy, lsb_sq); // Uses magic_utils::set_bit
        }
    }
    return occupancy;
}


// --- Magic Bitboard Constants (copied from board.cpp anonymous namespace) ---
// --- ROOK MAGICS ---
const std::array<Bitboard, NUM_SQUARES> RookMagics = {
    0x2280005882604000ULL, // sq 0 (shift 52)
    0x214000c010006002ULL, // sq 1 (shift 53)
    0x0100082000401100ULL, // sq 2 (shift 53)
    0x9100082100041001ULL, // sq 3 (shift 53)
    0x0280040028008012ULL, // sq 4 (shift 53)
    0xa880018002000400ULL, // sq 5 (shift 53)
    0x4580008009000200ULL, // sq 6 (shift 53)
    0x3080008000502100ULL, // sq 7 (shift 52)
    0x4180802040008000ULL, // sq 8 (shift 53)
    0x4400400020005001ULL, // sq 9 (shift 54)
    0x4280802000801000ULL, // sq 10 (shift 54)
    0x4f00808008001000ULL, // sq 11 (shift 54)
    0x0381806400480080ULL, // sq 12 (shift 54)
    0x0004804400800200ULL, // sq 13 (shift 54)
    0x0004004170020408ULL, // sq 14 (shift 54)
    0x0001000a00815500ULL, // sq 15 (shift 53)
    0x0000248000400099ULL, // sq 16 (shift 53)
    0x8040008020008040ULL, // sq 17 (shift 54)
    0x5021010020004812ULL, // sq 18 (shift 54)
    0x0888008010000880ULL, // sq 19 (shift 54)
    0x4018010008041100ULL, // sq 20 (shift 54)
    0x2422008002800400ULL, // sq 21 (shift 54)
    0x40c00c0091082a10ULL, // sq 22 (shift 54)
    0x0202020020840041ULL, // sq 23 (shift 53)
    0x0248800280244000ULL, // sq 24 (shift 53)
    0x0000208200410208ULL, // sq 25 (shift 54)
    0x020d014100102000ULL, // sq 26 (shift 54)
    0x0381002100081000ULL, // sq 27 (shift 54)
    0x8201040180080080ULL, // sq 28 (shift 54)
    0x0242001200103844ULL, // sq 29 (shift 54)
    0x2204100400024148ULL, // sq 30 (shift 54)
    0x0000012200004084ULL, // sq 31 (shift 53)
    0x0480204000800080ULL, // sq 32 (shift 53)
    0x0000802002804005ULL, // sq 33 (shift 54)
    0x0000200080801008ULL, // sq 34 (shift 54)
    0x0030080080801004ULL, // sq 35 (shift 54)
    0x2004000800800480ULL, // sq 36 (shift 54)
    0x5000040080800200ULL, // sq 37 (shift 54)
    0x00810004c1000200ULL, // sq 38 (shift 54)
    0x0200800040800100ULL, // sq 39 (shift 53)
    0x1540614000928000ULL, // sq 40 (shift 53)
    0x0c08200050084000ULL, // sq 41 (shift 54)
    0x0610002408002000ULL, // sq 42 (shift 54)
    0x9410c22201120008ULL, // sq 43 (shift 54)
    0x0003000800050012ULL, // sq 44 (shift 54)
    0x6000020004008080ULL, // sq 45 (shift 54)
    0x4010420108040010ULL, // sq 46 (shift 54)
    0x1000410080420004ULL, // sq 47 (shift 53)
    0x80008010a3400280ULL, // sq 48 (shift 53)
    0x0401004000208100ULL, // sq 49 (shift 54)
    0x0020001003802280ULL, // sq 50 (shift 54)
    0x4018100100082100ULL, // sq 51 (shift 54)
    0x4004008006080080ULL, // sq 52 (shift 54)
    0x4042000204008080ULL, // sq 53 (shift 54)
    0x0001000442002100ULL, // sq 54 (shift 54)
    0x0104800100005880ULL, // sq 55 (shift 53)
    0x004200810020104aULL, // sq 56 (shift 52)
    0x0040804000142105ULL, // sq 57 (shift 53)
    0x0280200008110041ULL, // sq 58 (shift 53)
    0x8010010008200411ULL, // sq 59 (shift 53)
    0x4012000c60100932ULL, // sq 60 (shift 53)
    0x10ca004804104102ULL, // sq 61 (shift 53)
    0x1820081002012084ULL, // sq 62 (shift 53)
    0x0680002400410082ULL, // sq 63 (shift 52)
};

// --- BISHOP MAGICS ---
const std::array<Bitboard, NUM_SQUARES> BishopMagics = {
    0x0410022084008204ULL, // sq 0 (shift 58)
    0x8004010812008404ULL, // sq 1 (shift 59)
    0x02441400a2000009ULL, // sq 2 (shift 59)
    0x1808204044001108ULL, // sq 3 (shift 59)
    0x8110882020018000ULL, // sq 4 (shift 59)
    0x0c20880540000000ULL, // sq 5 (shift 59)
    0x0011009004208080ULL, // sq 6 (shift 59)
    0x2003008090011000ULL, // sq 7 (shift 58)
    0x0300620210010502ULL, // sq 8 (shift 59)
    0x0000200401104518ULL, // sq 9 (shift 59)
    0x2000484800608000ULL, // sq 10 (shift 59)
    0x4002040410880005ULL, // sq 11 (shift 59)
    0x0a1002121010000eULL, // sq 12 (shift 59)
    0x0100371006100020ULL, // sq 13 (shift 59)
    0x2000020801480800ULL, // sq 14 (shift 59)
    0x0344008404220204ULL, // sq 15 (shift 59)
    0x4084401010420800ULL, // sq 16 (shift 59)
    0x001054e401025401ULL, // sq 17 (shift 59)
    0x183400aa10220201ULL, // sq 18 (shift 57)
    0x200c018804121400ULL, // sq 19 (shift 57)
    0x0284800400a04100ULL, // sq 20 (shift 57)
    0x0042006901008280ULL, // sq 21 (shift 57)
    0x0004000084218800ULL, // sq 22 (shift 59)
    0x8088400212020100ULL, // sq 23 (shift 59)
    0x0088090205a00830ULL, // sq 24 (shift 59)
    0x1001090420480104ULL, // sq 25 (shift 59)
    0x20010100408c0100ULL, // sq 26 (shift 57)
    0x0c04004144010102ULL, // sq 27 (shift 55)
    0x2002840000812000ULL, // sq 28 (shift 55)
    0x020092001308022bULL, // sq 29 (shift 57)
    0x0088011002008202ULL, // sq 30 (shift 59)
    0x1000808001040082ULL, // sq 31 (shift 59)
    0x30042020000b0208ULL, // sq 32 (shift 59)
    0x200108080a821000ULL, // sq 33 (shift 59)
    0x1014104407080810ULL, // sq 34 (shift 57)
    0x2009200800010104ULL, // sq 35 (shift 55)
    0x0044080201002008ULL, // sq 36 (shift 55)
    0x0050210a00004040ULL, // sq 37 (shift 57)
    0x2084011048060840ULL, // sq 38 (shift 59)
    0xae08010058002204ULL, // sq 39 (shift 59)
    0x8045243004004102ULL, // sq 40 (shift 59)
    0x02c409043000c220ULL, // sq 41 (shift 59)
    0x48c0211058041000ULL, // sq 42 (shift 57)
    0x0207046091004800ULL, // sq 43 (shift 57)
    0x8100310a02005420ULL, // sq 44 (shift 57)
    0x4001050307000200ULL, // sq 45 (shift 57)
    0x00a8081140440400ULL, // sq 46 (shift 59)
    0x000200aa12000480ULL, // sq 47 (shift 59)
    0x0044008884700400ULL, // sq 48 (shift 59)
    0x0c02220110184023ULL, // sq 49 (shift 59)
    0x0000194406210040ULL, // sq 50 (shift 59)
    0x4000c8828404420cULL, // sq 51 (shift 59)
    0x40c1120610440000ULL, // sq 52 (shift 59)
    0x0400102230044014ULL, // sq 53 (shift 59)
    0x0840100200a10400ULL, // sq 54 (shift 59)
    0x0004081208420500ULL, // sq 55 (shift 59)
    0x0402410090012104ULL, // sq 56 (shift 58)
    0x0801002201100810ULL, // sq 57 (shift 59)
    0x0000000842009000ULL, // sq 58 (shift 59)
    0x0051400044208810ULL, // sq 59 (shift 59)
    0x0004000020024416ULL, // sq 60 (shift 59)
    0x000c084011140522ULL, // sq 61 (shift 59)
    0x8400200202c80900ULL, // sq 62 (shift 59)
    0x00083801004a0204ULL, // sq 63 (shift 58)
};

// --- ROOK SHIFTS ---
const std::array<int, NUM_SQUARES> RookShifts = {
    52,     53,     53,     53,     53,     53,     53,     52,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    52,     53,     53,     53,     53,     53,     53,     52,
};

// --- BISHOP SHIFTS ---
const std::array<int, NUM_SQUARES> BishopShifts = {
    58,     59,     59,     59,     59,     59,     59,     58,
    59,     59,     59,     59,     59,     59,     59,     59,
    59,     59,     57,     57,     57,     57,     59,     59,
    59,     59,     57,     55,     55,     57,     59,     59,
    59,     59,     57,     55,     55,     57,     59,     59,
    59,     59,     57,     57,     57,     57,     59,     59,
    59,     59,     59,     59,     59,     59,     59,     59,
    58,     59,     59,     59,     59,     59,     59,     58,
};


} // namespace magic_utils
} // namespace chaturaji_cpp