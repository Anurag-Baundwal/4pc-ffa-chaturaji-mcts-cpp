// board.cpp
#include "board.h"
#include <algorithm> // For std::find, std::max_element, std::copy
#include <array>     // For Zobrist key storage and bitboard arrays
#include <cmath>     // For std::ceil, std::round (used in evaluate, get_game_result)
#include <cstdint>   // For ZobristKey (uint64_t), Bitboard (uint64_t)
#include <iostream>  // For print_board, print_bitboard
#include <limits>    // For numeric_limits (Zobrist key generation)
#include <numeric>   // For std::accumulate, std::popcount (C++20)
#include <random>    // For Zobrist key generation (std::mt19937_64)
#include <sstream>   // Potentially for string conversions (not directly used here)
#include <stdexcept> // For std::out_of_range, std::runtime_error
#include <utility>   // For std::move, std::pair
#include <vector>    // For move lists, undo stack, directional constants

#ifdef _MSC_VER
#include <intrin.h> // For MSVC specific intrinsics like _BitScanForward64, __popcnt64
#endif

namespace chaturaji_cpp {

// --- Bit Manipulation Helpers (including pop_count) ---
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


// Anonymous namespace for Zobrist, internal constants, and lookup table initializers
namespace { 
// --- Zobrist Hashing Constants ---
const int NUM_PIECE_TYPES_FOR_HASH = 5; // P, N, B, R, K (for Zobrist keys)
// --- Bitboard Related Constants ---
const int NUM_BB_PIECE_TYPES = 5;       // P, N, B, R, K (for bitboard array indexing)
const int NUM_PLAYERS_BB = 4;           // Number of players (for bitboard array indexing)

// --- DIRECTIONAL CONSTANTS (primarily for evaluate() ) ---
const std::vector<std::pair<int, int>> BISHOP_DIRS_EVAL = { {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
const std::vector<std::pair<int, int>> ROOK_DIRS_EVAL = { {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
const std::vector<std::pair<int, int>> KING_DIRS_EVAL = { {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
const std::vector<std::pair<int, int>> KNIGHT_MOVES_EVAL = { {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};

// --- Magic Bitboard Constants ---
// --- ROOK MAGICS ---
const std::array<Bitboard, NUM_SQUARES_BB> RookMagics = {
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
const std::array<Bitboard, NUM_SQUARES_BB> BishopMagics = {
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

// --- ROOK SHIFTS (can be derived, but useful for verification) ---
const std::array<int, NUM_SQUARES_BB> RookShifts = {
    52,     53,     53,     53,     53,     53,     53,     52,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    53,     54,     54,     54,     54,     54,     54,     53,
    52,     53,     53,     53,     53,     53,     53,     52,
};

// --- BISHOP SHIFTS (can be derived, but useful for verification) ---
const std::array<int, NUM_SQUARES_BB> BishopShifts = {
    58,     59,     59,     59,     59,     59,     59,     58,
    59,     59,     59,     59,     59,     59,     59,     59,
    59,     59,     57,     57,     57,     57,     59,     59,
    59,     59,     57,     55,     55,     57,     59,     59,
    59,     59,     57,     55,     55,     57,     59,     59,
    59,     59,     57,     57,     57,     57,     59,     59,
    59,     59,     59,     59,     59,     59,     59,     59,
    58,     59,     59,     59,     59,     59,     59,     58,
};

// Helper to map PieceType to bitboard array index (0-4)
int piece_type_to_bb_idx_internal(PieceType pt) {
    int val = static_cast<int>(pt) - 1; 
    if (val < 0 || val >= NUM_BB_PIECE_TYPES) {
        throw std::out_of_range("Invalid PieceType for bitboard index.");
    }
    return val;
}

// --- Zobrist Hashing Data Structure and Initialization (Unchanged) ---
struct ZobristData {
  std::array<std::array<std::array<ZobristKey, NUM_SQUARES_BB>, NUM_PLAYERS_BB>, NUM_PIECE_TYPES_FOR_HASH> piece_keys;
  std::array<ZobristKey, NUM_PLAYERS_BB> turn_keys;
  std::array<ZobristKey, NUM_PLAYERS_BB> active_player_status_keys;
  ZobristData() {
    std::mt19937_64 rng(0xBADFACE); 
    std::uniform_int_distribution<ZobristKey> dist(0, std::numeric_limits<ZobristKey>::max());
    for (int type_idx = 0; type_idx < NUM_PIECE_TYPES_FOR_HASH; ++type_idx) {
      for (int player_idx = 0; player_idx < NUM_PLAYERS_BB; ++player_idx) {
          for (int sq_idx = 0; sq_idx < NUM_SQUARES_BB; ++sq_idx) {
            piece_keys[type_idx][player_idx][sq_idx] = dist(rng);
          }
      }
    }
    for (int player_idx = 0; player_idx < NUM_PLAYERS_BB; ++player_idx) {
      turn_keys[player_idx] = dist(rng);
    }
    for (int player_idx = 0; player_idx < NUM_PLAYERS_BB; ++player_idx) {
        active_player_status_keys[player_idx] = dist(rng);
    }
  }
  ZobristKey get_piece_key(PieceType type, Player player, int square_index) const {
    if (square_index < 0 || square_index >= NUM_SQUARES_BB) {
      throw std::out_of_range("Square index out of range for Zobrist key lookup.");
    }
    int type_idx = static_cast<int>(type) - 1; 
    if (type_idx < 0 || type_idx >= NUM_PIECE_TYPES_FOR_HASH) {
      throw std::out_of_range("PieceType out of range for Zobrist key lookup.");
    }
    int player_idx = static_cast<int>(player); 
    if (player_idx < 0 || player_idx >= NUM_PLAYERS_BB) {
      throw std::out_of_range("Player out of range for Zobrist key lookup.");
    }
    return piece_keys[type_idx][player_idx][square_index]; 
    }
  ZobristKey get_turn_key(Player player) const {
    int player_idx = static_cast<int>(player);
    if (player_idx < 0 || player_idx >= NUM_PLAYERS_BB) {
      throw std::out_of_range("Player out of range for Zobrist key lookup.");
    }
    return turn_keys[player_idx];
  }
  ZobristKey get_active_player_status_key(Player player) const {
    int player_idx = static_cast<int>(player);
    if (player_idx < 0 || player_idx >= NUM_PLAYERS_BB) {
        throw std::out_of_range("Player out of range for Zobrist active status key lookup.");
    }
    return active_player_status_keys[player_idx];
  }
};
const ZobristData &get_zobrist_data() {
  static const ZobristData instance; 
  return instance;
}

// --- Bitboard constants for pawn move generation (file checks) ---
const Bitboard FILE_A_BB = 0x0101010101010101ULL; 
const Bitboard FILE_H_BB = 0x8080808080808080ULL; 
// --- Pawn Promotion Target Coordinates (Bitboard version) ---
const int PROMOTION_ROW_RED_BB = 0;    
const int PROMOTION_COL_BLUE_BB = 7;   
const int PROMOTION_ROW_YELLOW_BB = 7; 
const int PROMOTION_COL_GREEN_BB = 0;  

// --- Helper function for Magic Bitboard Initialization: Generate Occupancy Subsets ---
Bitboard get_occupancy_subset(int index, int bits_in_mask, Bitboard mask) {
    Bitboard occupancy = 0ULL;
    Bitboard temp_mask = mask; // Iterate over bits in the mask
    for (int i = 0; i < bits_in_mask; ++i) {
        int lsb_sq = Board::pop_lsb(temp_mask); // Get the square index of the i-th bit in the mask
        if (lsb_sq == -1) break; // Should not happen if bits_in_mask is correct
        if ((index >> i) & 1) {    // If the i-th bit of `index` is set
            Board::set_bit(occupancy, lsb_sq); // Set this bit in the occupancy
        }
    }
    return occupancy;
}

// --- Helper functions for Magic Bitboard Initialization: On-the-fly attack generation ---
Bitboard calculate_rook_attacks_on_the_fly(int sq, Bitboard occupied) {
    Bitboard attacks = 0ULL;
    BoardLocation loc = Board::from_sq_idx(sq);
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
            int target_sq = Board::to_sq_idx(r,c);
            Board::set_bit(attacks, target_sq);
            if (Board::get_bit(occupied, target_sq)) break; // Blocked by a piece in 'occupied'
        }
    }
    return attacks;
}

Bitboard calculate_bishop_attacks_on_the_fly(int sq, Bitboard occupied) {
    Bitboard attacks = 0ULL;
    BoardLocation loc = Board::from_sq_idx(sq);
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
            int target_sq = Board::to_sq_idx(r,c);
            Board::set_bit(attacks, target_sq);
            if (Board::get_bit(occupied, target_sq)) break; // Blocked
        }
    }
    return attacks;
}

} // end anonymous namespace


// --- Static Lookup Tables for Bitboard Move Generation ---
std::array<Bitboard, NUM_SQUARES_BB> Board::knight_attacks_; 
std::array<Bitboard, NUM_SQUARES_BB> Board::king_attacks_;   

std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> Board::pawn_attacks_red_;
std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> Board::pawn_attacks_blue_;
std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> Board::pawn_attacks_yellow_;
std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> Board::pawn_attacks_green_;

std::array<Bitboard, NUM_SQUARES_BB> Board::pawn_fwd_moves_red_;
std::array<Bitboard, NUM_SQUARES_BB> Board::pawn_fwd_moves_blue_;
std::array<Bitboard, NUM_SQUARES_BB> Board::pawn_fwd_moves_yellow_;
std::array<Bitboard, NUM_SQUARES_BB> Board::pawn_fwd_moves_green_;

// --- NEW: Magic Bitboard related static members ---
std::array<Bitboard, NUM_SQUARES_BB> Board::rook_masks_;
std::array<Bitboard, NUM_SQUARES_BB> Board::bishop_masks_;
std::array<int, NUM_SQUARES_BB> Board::rook_shift_bits_;
std::array<int, NUM_SQUARES_BB> Board::bishop_shift_bits_;
std::vector<Bitboard> Board::rook_attack_table_;    // Flat table for all rook attacks
std::vector<Bitboard> Board::bishop_attack_table_;  // Flat table for all bishop attacks
std::array<unsigned int, NUM_SQUARES_BB> Board::rook_attack_offsets_;   // Offsets into rook_attack_table_
std::array<unsigned int, NUM_SQUARES_BB> Board::bishop_attack_offsets_; // Offsets into bishop_attack_table_
// --- END NEW ---

Board::StaticInitializer Board::static_initializer_; 


// --- Bitboard Helper Functions (Public Static) ---
int Board::piece_type_to_bb_idx(PieceType pt) {
    return piece_type_to_bb_idx_internal(pt); 
}
bool Board::is_valid_sq_idx(int sq_idx) {
    return sq_idx >= 0 && sq_idx < NUM_SQUARES_BB;
}
int Board::to_sq_idx(int r, int c) {
    return r * BOARD_SIZE + c; 
}
BoardLocation Board::from_sq_idx(int sq_idx) {
    return {sq_idx / BOARD_SIZE, sq_idx % BOARD_SIZE};
}

// --- Lookup Table Initialization ---
void Board::initialize_lookup_tables() {
    // Knight and King attacks (unchanged)
    const int kn_moves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2},  {1, 2},  {2, -1},  {2, 1}};
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r, c);
            knight_attacks_[sq_idx] = 0ULL;
            for (auto& move : kn_moves) {
                int nr = r + move[0]; int nc = c + move[1];
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    set_bit(knight_attacks_[sq_idx], to_sq_idx(nr, nc));
                }
            }
        }
    }
    const int ki_moves[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1},   {1, -1}, {1, 0},  {1, 1}};
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r, c);
            king_attacks_[sq_idx] = 0ULL;
            for (auto& move : ki_moves) {
                int nr = r + move[0]; int nc = c + move[1];
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    set_bit(king_attacks_[sq_idx], to_sq_idx(nr, nc));
                }
            }
        }
    }

    // Pawn attacks and forward moves (unchanged)
    for (int r = 0; r < BOARD_SIZE; ++r) { // Red
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c); pawn_fwd_moves_red_[sq_idx] = 0ULL; pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx] = 0ULL; 
            if (r > 0) { 
                set_bit(pawn_fwd_moves_red_[sq_idx], to_sq_idx(r-1, c));
                if (c > 0) set_bit(pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx], to_sq_idx(r-1, c-1));
                if (c < BOARD_SIZE - 1) set_bit(pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx], to_sq_idx(r-1, c+1));
            }
        }
    } // Blue
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c); pawn_fwd_moves_blue_[sq_idx] = 0ULL; pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx] = 0ULL;
             if (c < BOARD_SIZE -1) { 
                set_bit(pawn_fwd_moves_blue_[sq_idx], to_sq_idx(r, c+1));
                if (r > 0) set_bit(pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx], to_sq_idx(r-1, c+1));
                if (r < BOARD_SIZE - 1) set_bit(pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx], to_sq_idx(r+1, c+1));
            }
        }
    } // Yellow
     for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c); pawn_fwd_moves_yellow_[sq_idx] = 0ULL; pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx] = 0ULL;
             if (r < BOARD_SIZE -1) { 
                set_bit(pawn_fwd_moves_yellow_[sq_idx], to_sq_idx(r+1, c));
                if (c > 0) set_bit(pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx], to_sq_idx(r+1, c-1));
                if (c < BOARD_SIZE - 1) set_bit(pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx], to_sq_idx(r+1, c+1));
            }
        }
    } // Green
     for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c); pawn_fwd_moves_green_[sq_idx] = 0ULL; pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx] = 0ULL;
             if (c > 0) { 
                set_bit(pawn_fwd_moves_green_[sq_idx], to_sq_idx(r, c-1));
                if (r > 0) set_bit(pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx], to_sq_idx(r-1, c-1));
                if (r < BOARD_SIZE - 1) set_bit(pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx], to_sq_idx(r+1, c-1));
            }
        }
    }
    
    // --- Magic Bitboard Initialization ---
    // 1. Generate Blocker Masks (rook_masks_ and bishop_masks_)
    //    The mask for a square `s` includes all squares on its rays,
    //    EXCLUDING the squares on the very edges of the board (rank 0/7, file 0/7),
    //    unless `s` itself is on such an edge and the ray segment is between `s` and the other edge.
    //    Essentially, it's the squares whose occupancy status can change the attack set from `s`.
    for (int sq = 0; sq < NUM_SQUARES_BB; ++sq) {
        rook_masks_[sq] = 0ULL;
        bishop_masks_[sq] = 0ULL;
        BoardLocation loc = from_sq_idx(sq);
        int r0 = loc.row; int c0 = loc.col;

        // Rook Masks
        const int rook_dr[] = {-1, 1, 0, 0}; const int rook_dc[] = {0, 0, 1, -1};
        for (int dir_idx = 0; dir_idx < 4; ++dir_idx) {
            for (int k = 1; k < BOARD_SIZE -1; ++k) { // Iterate up to one square before edge
                int r = r0 + rook_dr[dir_idx] * k;
                int c = c0 + rook_dc[dir_idx] * k;
                // Check if (r,c) is on the board AND not an edge square for THIS ray segment
                // i.e. if the next square (r + dr, c + dc) is still on the board
                if (r > 0 && r < BOARD_SIZE - 1 && c >= 0 && c < BOARD_SIZE && (rook_dr[dir_idx] !=0)) { // Horizontal inner squares for vertical rays
                     set_bit(rook_masks_[sq], to_sq_idx(r, c));
                } else if (r >= 0 && r < BOARD_SIZE && c > 0 && c < BOARD_SIZE-1 && (rook_dc[dir_idx] != 0)) { // Vertical inner squares for horizontal rays
                     set_bit(rook_masks_[sq], to_sq_idx(r, c));
                }
                 // A simpler approach (common): mask is relevant squares *excluding* sq itself and board edges
                 // For square (r0, c0):
                 // Horizontal: (r0, c) for c in 1..6. If c0 is 0 or 7, include c0.
                 // Vertical:   (r, c0) for r in 1..6. If r0 is 0 or 7, include r0.
                 // This needs careful handling of edge cases for squares on edges/corners.
                 // Let's use a standard approach: iterate ray, add if not edge
            }
        }
        // Corrected Mask Generation (simpler approach):
        rook_masks_[sq] = 0ULL; // Reset
        for (int r_idx = r0 + 1; r_idx < BOARD_SIZE - 1; ++r_idx) set_bit(rook_masks_[sq], to_sq_idx(r_idx, c0)); // South (inner)
        for (int r_idx = r0 - 1; r_idx > 0; --r_idx) set_bit(rook_masks_[sq], to_sq_idx(r_idx, c0));             // North (inner)
        for (int c_idx = c0 + 1; c_idx < BOARD_SIZE - 1; ++c_idx) set_bit(rook_masks_[sq], to_sq_idx(r0, c_idx)); // East (inner)
        for (int c_idx = c0 - 1; c_idx > 0; --c_idx) set_bit(rook_masks_[sq], to_sq_idx(r0, c_idx));             // West (inner)

        // Bishop Masks
        bishop_masks_[sq] = 0ULL; // Reset
        const int bishop_dr[] = {-1, 1, 1, -1}; const int bishop_dc[] = {1, 1, -1, -1};
        for (int dir_idx = 0; dir_idx < 4; ++dir_idx) {
            for (int k = 1; k < BOARD_SIZE; ++k) { // Iterate up to one square before edge
                int r = r0 + bishop_dr[dir_idx] * k;
                int c = c0 + bishop_dc[dir_idx] * k;
                if (r > 0 && r < BOARD_SIZE - 1 && c > 0 && c < BOARD_SIZE - 1) { // Check if it's an "inner" board square
                    set_bit(bishop_masks_[sq], to_sq_idx(r, c));
                } else {
                  break;
                }
            }
        }
    }
    
    // 2. Calculate shift bits and total table sizes
    unsigned int total_rook_table_entries = 0;
    unsigned int total_bishop_table_entries = 0;
    for (int sq = 0; sq < NUM_SQUARES_BB; ++sq) {
        // Copy the pre-generated shifts to the member variables
        rook_shift_bits_[sq] = RookShifts[sq];
        bishop_shift_bits_[sq] = BishopShifts[sq];
        
        rook_attack_offsets_[sq] = total_rook_table_entries;
        total_rook_table_entries += (1ULL << pop_count(rook_masks_[sq]));
        
        bishop_attack_offsets_[sq] = total_bishop_table_entries;
        total_bishop_table_entries += (1ULL << pop_count(bishop_masks_[sq]));
    }
    rook_attack_table_.resize(total_rook_table_entries);
    bishop_attack_table_.resize(total_bishop_table_entries);

    // 3. Populate Attack Tables
    for (int sq = 0; sq < NUM_SQUARES_BB; ++sq) {
        // Rooks
        Bitboard r_mask = rook_masks_[sq];
        int r_num_mask_bits = pop_count(r_mask);
        unsigned int r_num_entries_for_sq = (1ULL << r_num_mask_bits);
        for (unsigned int i = 0; i < r_num_entries_for_sq; ++i) {
            Bitboard occupancy = get_occupancy_subset(i, r_num_mask_bits, r_mask);
            Bitboard attacks = calculate_rook_attacks_on_the_fly(sq, occupancy);
            unsigned int magic_idx = (occupancy * RookMagics[sq]) >> rook_shift_bits_[sq];
            rook_attack_table_[rook_attack_offsets_[sq] + magic_idx] = attacks;
        }

        // Bishops
        Bitboard b_mask = bishop_masks_[sq];
        int b_num_mask_bits = pop_count(b_mask);
        unsigned int b_num_entries_for_sq = (1ULL << b_num_mask_bits);
        for (unsigned int i = 0; i < b_num_entries_for_sq; ++i) {
            Bitboard occupancy = get_occupancy_subset(i, b_num_mask_bits, b_mask);
            Bitboard attacks = calculate_bishop_attacks_on_the_fly(sq, occupancy);
            unsigned int magic_idx = (occupancy * BishopMagics[sq]) >> bishop_shift_bits_[sq];
            bishop_attack_table_[bishop_attack_offsets_[sq] + magic_idx] = attacks;
        }
    }
}

// --- Constructor --- (Zobrist part unchanged)
Board::Board()
    : current_player_(Player::RED), full_move_number_(1),
      move_number_of_last_reset_(0), termination_reason_(std::nullopt) {
  for (auto& player_bb_array : piece_bitboards_) player_bb_array.fill(0ULL);
  player_bitboards_.fill(0ULL);
  occupied_bitboard_ = 0ULL;
  for (int i = 0; i < 4; ++i) {
    Player p = static_cast<Player>(i);
    player_points_[p] = 0;
    active_players_.insert(p);
  }
  setup_initial_board(); 
  const auto& zobrist_data = get_zobrist_data();
  current_hash_ = 0; 
  for (int p_idx = 0; p_idx < NUM_PLAYERS_BB; ++p_idx) {
      Player player = static_cast<Player>(p_idx);
      for (int pt_bb_idx = 0; pt_bb_idx < NUM_BB_PIECE_TYPES; ++pt_bb_idx) {
          PieceType piece_type = static_cast<PieceType>(pt_bb_idx + 1); 
          Bitboard current_piece_bb = piece_bitboards_[p_idx][pt_bb_idx];
          Bitboard temp_bb = current_piece_bb;
          while(temp_bb) { 
              int sq_idx = pop_lsb(temp_bb); 
              current_hash_ ^= zobrist_data.get_piece_key(piece_type, player, sq_idx);
          }
      }
  }
  current_hash_ ^= zobrist_data.get_turn_key(current_player_);
  for (Player p : active_players_) { 
      current_hash_ ^= zobrist_data.get_active_player_status_key(p);
  }
  position_history_.push_back(current_hash_);
}

// --- Copy Constructor, Move Constructor, Assignments (Bitboard parts unchanged) ---
Board::Board(const Board &other)
    : active_players_(other.active_players_), player_points_(other.player_points_),
      current_player_(other.current_player_), position_history_(other.position_history_),
      full_move_number_(other.full_move_number_), move_number_of_last_reset_(other.move_number_of_last_reset_),
      termination_reason_(other.termination_reason_), current_hash_(other.current_hash_),
      undo_stack_(other.undo_stack_), piece_bitboards_(other.piece_bitboards_),
      player_bitboards_(other.player_bitboards_), occupied_bitboard_(other.occupied_bitboard_) {}

Board::Board(Board &&other) noexcept
    : active_players_(std::move(other.active_players_)), player_points_(std::move(other.player_points_)),
      current_player_(other.current_player_), position_history_(std::move(other.position_history_)),
      full_move_number_(other.full_move_number_), move_number_of_last_reset_(other.move_number_of_last_reset_),
      termination_reason_(std::move(other.termination_reason_)), current_hash_(other.current_hash_),
      undo_stack_(std::move(other.undo_stack_)), piece_bitboards_(std::move(other.piece_bitboards_)),
      player_bitboards_(std::move(other.player_bitboards_)), occupied_bitboard_(other.occupied_bitboard_) {
  other.full_move_number_ = 1; other.move_number_of_last_reset_ = 0; other.current_hash_ = 0; 
  other.occupied_bitboard_ = 0ULL; for(auto& arr : other.piece_bitboards_) arr.fill(0ULL); other.player_bitboards_.fill(0ULL);
}

Board &Board::operator=(const Board &other) {
  if (this != &other) {
    active_players_ = other.active_players_; player_points_ = other.player_points_; current_player_ = other.current_player_;
    position_history_ = other.position_history_; full_move_number_ = other.full_move_number_;
    move_number_of_last_reset_ = other.move_number_of_last_reset_; termination_reason_ = other.termination_reason_;
    current_hash_ = other.current_hash_; undo_stack_ = other.undo_stack_;
    piece_bitboards_ = other.piece_bitboards_; player_bitboards_ = other.player_bitboards_;
    occupied_bitboard_ = other.occupied_bitboard_;
  }
  return *this;
}

Board &Board::operator=(Board &&other) noexcept {
  if (this != &other) { 
    active_players_ = std::move(other.active_players_); player_points_ = std::move(other.player_points_);
    current_player_ = other.current_player_; position_history_ = std::move(other.position_history_);
    full_move_number_ = other.full_move_number_; move_number_of_last_reset_ = other.move_number_of_last_reset_;
    termination_reason_ = std::move(other.termination_reason_); current_hash_ = other.current_hash_;
    undo_stack_ = std::move(other.undo_stack_); piece_bitboards_ = std::move(other.piece_bitboards_);
    player_bitboards_ = std::move(other.player_bitboards_); occupied_bitboard_ = other.occupied_bitboard_;
    other.full_move_number_ = 1; other.move_number_of_last_reset_ = 0; other.current_hash_ = 0;
    other.occupied_bitboard_ = 0ULL; for(auto& arr : other.piece_bitboards_) arr.fill(0ULL); other.player_bitboards_.fill(0ULL);
  }
  return *this;
}

// --- MCTS Child Board Creation (Unchanged) ---
Board Board::create_mcts_child_board(const Board& parent_board, const Move& move) {
  Board child_board; 
  child_board.active_players_ = parent_board.active_players_;
  child_board.player_points_ = parent_board.player_points_;
  child_board.current_player_ = parent_board.current_player_; 
  child_board.full_move_number_ = parent_board.full_move_number_;
  child_board.move_number_of_last_reset_ = parent_board.move_number_of_last_reset_;
  child_board.current_hash_ = parent_board.current_hash_; 
  child_board.piece_bitboards_ = parent_board.piece_bitboards_;
  child_board.player_bitboards_ = parent_board.player_bitboards_;
  child_board.occupied_bitboard_ = parent_board.occupied_bitboard_;
  child_board.make_move_for_mcts(move);
  return child_board; 
}

// --- Get Piece At Square (Unchanged) ---
std::optional<Piece> Board::get_piece_at_sq(int sq_idx) const {
    if (!is_valid_sq_idx(sq_idx)) return std::nullopt; 
    if (!get_bit(occupied_bitboard_, sq_idx)) return std::nullopt; 
    for (int p_idx = 0; p_idx < NUM_PLAYERS_BB; ++p_idx) {
        if (get_bit(player_bitboards_[p_idx], sq_idx)) { 
            Player player = static_cast<Player>(p_idx);
            for (int pt_bb_idx = 0; pt_bb_idx < NUM_BB_PIECE_TYPES; ++pt_bb_idx) {
                if (get_bit(piece_bitboards_[p_idx][pt_bb_idx], sq_idx)) {
                    PieceType pt = static_cast<PieceType>(pt_bb_idx + 1); 
                    return Piece(player, pt); 
                }
            }
            throw std::runtime_error("Bitboard inconsistency in get_piece_at_sq: Player bit set, but no piece type bit.");
        }
    }
    return std::nullopt; 
}

// --- Initial Board Setup (Unchanged) ---
void Board::setup_initial_board() {
  for (auto& player_bbs : piece_bitboards_) player_bbs.fill(0ULL);
  player_bitboards_.fill(0ULL);
  occupied_bitboard_ = 0ULL;
  auto place_piece = [&](Player p, PieceType pt, int r, int c) {
      int sq_idx = to_sq_idx(r, c); int player_idx = static_cast<int>(p); int pt_bb_idx = piece_type_to_bb_idx(pt);
      set_bit(piece_bitboards_[player_idx][pt_bb_idx], sq_idx); set_bit(player_bitboards_[player_idx], sq_idx); set_bit(occupied_bitboard_, sq_idx);                      
  };
  place_piece(Player::RED, PieceType::ROOK, 7, 0); place_piece(Player::RED, PieceType::KNIGHT, 7, 1); place_piece(Player::RED, PieceType::BISHOP, 7, 2); place_piece(Player::RED, PieceType::KING, 7, 3);
  for (int col = 0; col < 4; ++col) place_piece(Player::RED, PieceType::PAWN, 6, col);
  place_piece(Player::BLUE, PieceType::ROOK, 0, 0); place_piece(Player::BLUE, PieceType::KNIGHT, 1, 0); place_piece(Player::BLUE, PieceType::BISHOP, 2, 0); place_piece(Player::BLUE, PieceType::KING, 3, 0);
  for (int row = 0; row < 4; ++row) place_piece(Player::BLUE, PieceType::PAWN, row, 1);
  place_piece(Player::YELLOW, PieceType::ROOK, 0, 7); place_piece(Player::YELLOW, PieceType::KNIGHT, 0, 6); place_piece(Player::YELLOW, PieceType::BISHOP, 0, 5); place_piece(Player::YELLOW, PieceType::KING, 0, 4);
  for (int col = 4; col < 8; ++col) place_piece(Player::YELLOW, PieceType::PAWN, 1, col);
  place_piece(Player::GREEN, PieceType::KING, 4, 7); place_piece(Player::GREEN, PieceType::BISHOP, 5, 7); place_piece(Player::GREEN, PieceType::KNIGHT, 6, 7); place_piece(Player::GREEN, PieceType::ROOK, 7, 7);
  for (int row = 4; row < 8; ++row) place_piece(Player::GREEN, PieceType::PAWN, row, 6);
}

// --- Square Validity Check (Unchanged) ---
bool Board::is_valid_square(int row, int col) const {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

// --- Pseudo-Legal Move Generation (Master Function) (Unchanged overall structure) ---
std::vector<Move> Board::get_pseudo_legal_moves(Player player) const {
  std::vector<Move> pseudo_legal_moves;
  pseudo_legal_moves.reserve(128); 
  if (!active_players_.count(player)) { return pseudo_legal_moves; }
  get_pawn_moves_bb(player, pseudo_legal_moves);
  get_knight_moves_bb(player, pseudo_legal_moves);
  get_bishop_moves_bb(player, pseudo_legal_moves); // Will use magic bitboards
  get_rook_moves_bb(player, pseudo_legal_moves);   // Will use magic bitboards
  get_king_moves_bb(player, pseudo_legal_moves);
  return pseudo_legal_moves;
}

// --- Bitboard-Based Move Generation Helpers ---

// Pawn moves (unchanged)
void Board::get_pawn_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard pawns = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::PAWN)];
    Bitboard my_pieces = player_bitboards_[p_idx];
    Bitboard opp_pieces = occupied_bitboard_ & ~my_pieces; 
    Bitboard empty_sqs = ~occupied_bitboard_;            
    const Bitboard* current_fwd_moves_table = nullptr; 
    const std::array<Bitboard, NUM_SQUARES_BB>* current_atk_table_for_player = nullptr;
    int promotion_target_coord = -1; bool check_row_for_promo = false; 
    switch (player) {
        case Player::RED:    current_fwd_moves_table = &pawn_fwd_moves_red_[0];    current_atk_table_for_player = &pawn_attacks_red_[p_idx];    promotion_target_coord = PROMOTION_ROW_RED_BB;   check_row_for_promo = true; break;
        case Player::BLUE:   current_fwd_moves_table = &pawn_fwd_moves_blue_[0];   current_atk_table_for_player = &pawn_attacks_blue_[p_idx];   promotion_target_coord = PROMOTION_COL_BLUE_BB;  check_row_for_promo = false; break;
        case Player::YELLOW: current_fwd_moves_table = &pawn_fwd_moves_yellow_[0]; current_atk_table_for_player = &pawn_attacks_yellow_[p_idx]; promotion_target_coord = PROMOTION_ROW_YELLOW_BB; check_row_for_promo = true; break;
        case Player::GREEN:  current_fwd_moves_table = &pawn_fwd_moves_green_[0];  current_atk_table_for_player = &pawn_attacks_green_[p_idx];  promotion_target_coord = PROMOTION_COL_GREEN_BB; check_row_for_promo = false; break;
    }
    if (!current_fwd_moves_table || !current_atk_table_for_player) return; 
    Bitboard temp_pawns = pawns; 
    while (temp_pawns) {
        int from_sq = pop_lsb(temp_pawns); 
        BoardLocation from_loc = from_sq_idx(from_sq);
        Bitboard fwd_moves = current_fwd_moves_table[from_sq] & empty_sqs;
        if (fwd_moves) { 
            int to_sq = get_lsb_index(fwd_moves); 
            BoardLocation to_loc = from_sq_idx(to_sq);
            bool is_promotion = (check_row_for_promo && to_loc.row == promotion_target_coord) ||
                                (!check_row_for_promo && to_loc.col == promotion_target_coord);
            if (is_promotion) { moves.emplace_back(from_loc, to_loc, PieceType::ROOK); } 
            else { moves.emplace_back(from_loc, to_loc); }
        }
        Bitboard cap_moves = (*current_atk_table_for_player)[from_sq] & opp_pieces;
        Bitboard temp_cap_moves = cap_moves; 
        while (temp_cap_moves) {
            int to_sq = pop_lsb(temp_cap_moves);
            BoardLocation to_loc = from_sq_idx(to_sq);
            bool is_promotion = (check_row_for_promo && to_loc.row == promotion_target_coord) ||
                                (!check_row_for_promo && to_loc.col == promotion_target_coord);
            if (is_promotion) { moves.emplace_back(from_loc, to_loc, PieceType::ROOK); } 
            else { moves.emplace_back(from_loc, to_loc); }
        }
    }
}

// Knight moves (unchanged)
void Board::get_knight_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard knights = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KNIGHT)];
    Bitboard not_my_pieces = ~player_bitboards_[p_idx]; 
    Bitboard temp_knights = knights; 
    while (temp_knights) {
        int from_sq = pop_lsb(temp_knights);
        BoardLocation from_loc = from_sq_idx(from_sq);
        Bitboard possible_moves = knight_attacks_[from_sq] & not_my_pieces;
        Bitboard temp_possible_moves = possible_moves; 
        while (temp_possible_moves) {
            int to_sq = pop_lsb(temp_possible_moves);
            moves.emplace_back(from_loc, from_sq_idx(to_sq));
        }
    }
}

// King moves (unchanged)
void Board::get_king_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard kings = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
    Bitboard not_my_pieces = ~player_bitboards_[p_idx]; 
    if (kings == 0) return; 
    int from_sq = get_lsb_index(kings); 
    BoardLocation from_loc = from_sq_idx(from_sq);
    Bitboard possible_moves = king_attacks_[from_sq] & not_my_pieces;
    Bitboard temp_possible_moves = possible_moves; 
    while (temp_possible_moves) {
        int to_sq = pop_lsb(temp_possible_moves);
        moves.emplace_back(from_loc, from_sq_idx(to_sq));
    }
}

// Old generate_sliding_moves is no longer used for runtime move generation.
// It's effectively replaced by calculate_rook/bishop_attacks_on_the_fly for init.

// --- NEW: Rook moves using Magic Bitboards ---
void Board::get_rook_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard rooks = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::ROOK)];
    Bitboard my_pieces = player_bitboards_[p_idx];
    
    Bitboard temp_rooks = rooks;
    while(temp_rooks) {
        int from_sq = pop_lsb(temp_rooks);
        BoardLocation from_loc = from_sq_idx(from_sq);

        Bitboard blockers = occupied_bitboard_ & rook_masks_[from_sq];
        unsigned int magic_idx = (blockers * RookMagics[from_sq]) >> rook_shift_bits_[from_sq];
        Bitboard possible_moves = rook_attack_table_[rook_attack_offsets_[from_sq] + magic_idx];
        
        possible_moves &= ~my_pieces; // Cannot move to a square occupied by own piece

        Bitboard temp_possible_moves = possible_moves;
        while(temp_possible_moves) {
            int to_sq = pop_lsb(temp_possible_moves);
            moves.emplace_back(from_loc, from_sq_idx(to_sq));
        }
    }
}

// --- NEW: Bishop moves using Magic Bitboards ---
void Board::get_bishop_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard bishops = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::BISHOP)];
    Bitboard my_pieces = player_bitboards_[p_idx];

    Bitboard temp_bishops = bishops;
    while(temp_bishops) {
        int from_sq = pop_lsb(temp_bishops); // from_sq is defined here
        BoardLocation from_loc = from_sq_idx(from_sq);

        Bitboard blockers = occupied_bitboard_ & bishop_masks_[from_sq];
        unsigned int magic_idx = (blockers * BishopMagics[from_sq]) >> bishop_shift_bits_[from_sq]; 
        Bitboard possible_moves = bishop_attack_table_[bishop_attack_offsets_[from_sq] + magic_idx];

        possible_moves &= ~my_pieces; // Cannot move to a square occupied by own piece

        Bitboard temp_possible_moves = possible_moves;
        while(temp_possible_moves) {
            int to_sq = pop_lsb(temp_possible_moves);
            moves.emplace_back(from_loc, from_sq_idx(to_sq));
        }
    }
}


// --- Move Execution (make_move, make_move_for_mcts) (Largely unchanged logic, bitboard ops correct) ---
std::optional<Piece> Board::make_move(const Move &move) {
  UndoInfo undo_info;
  undo_info.original_piece_bitboards = piece_bitboards_; undo_info.original_player_bitboards = player_bitboards_; undo_info.original_occupied_bitboard = occupied_bitboard_;
  undo_info.move = move; undo_info.original_player = current_player_; undo_info.original_full_move_number = full_move_number_;
  undo_info.original_move_number_of_last_reset = move_number_of_last_reset_; undo_info.eliminated_player = std::nullopt;
  undo_info.was_history_cleared = false; undo_info.previous_hash = current_hash_; 
  const auto& zobrist_data = get_zobrist_data();
  int fr = move.from_loc.row, fc = move.from_loc.col; int tr = move.to_loc.row, tc = move.to_loc.col;
  int from_sq_idx = Board::to_sq_idx(fr, fc); int to_sq_idx = Board::to_sq_idx(tr, tc); int moving_player_idx = static_cast<int>(current_player_);
  std::optional<Piece> moving_piece_opt = get_piece_at_sq(from_sq_idx); 
  if (!moving_piece_opt) { throw std::runtime_error("Attempting to move from an empty square in make_move. From sq: " + std::to_string(from_sq_idx)); }
  if (moving_piece_opt->player != current_player_) { throw std::runtime_error("Attempting to move opponent's piece."); }
  Piece moving_piece_obj = *moving_piece_opt; undo_info.original_moving_piece_type = moving_piece_obj.piece_type;
  int moving_pt_bb_idx = piece_type_to_bb_idx(moving_piece_obj.piece_type);
  undo_info.captured_piece = get_piece_at_sq(to_sq_idx); 
  bool is_capture = undo_info.captured_piece.has_value(); bool is_pawn_move = (moving_piece_obj.piece_type == PieceType::PAWN);
  bool is_resetting_move = is_pawn_move || is_capture; 
  current_hash_ ^= zobrist_data.get_piece_key(moving_piece_obj.piece_type, moving_piece_obj.player, from_sq_idx);
  clear_bit(piece_bitboards_[moving_player_idx][moving_pt_bb_idx], from_sq_idx); clear_bit(player_bitboards_[moving_player_idx], from_sq_idx); clear_bit(occupied_bitboard_, from_sq_idx); 
  if (is_capture) {
      const Piece& captured = undo_info.captured_piece.value();
      current_hash_ ^= zobrist_data.get_piece_key(captured.piece_type, captured.player, to_sq_idx);
      int captured_player_idx = static_cast<int>(captured.player); int captured_pt_bb_idx = piece_type_to_bb_idx(captured.piece_type);
      clear_bit(piece_bitboards_[captured_player_idx][captured_pt_bb_idx], to_sq_idx); clear_bit(player_bitboards_[captured_player_idx], to_sq_idx);
  }
  PieceType final_piece_type = moving_piece_obj.piece_type; if (move.promotion_piece_type) { final_piece_type = move.promotion_piece_type.value(); }
  int final_pt_bb_idx = piece_type_to_bb_idx(final_piece_type);
  set_bit(piece_bitboards_[moving_player_idx][final_pt_bb_idx], to_sq_idx); set_bit(player_bitboards_[moving_player_idx], to_sq_idx); set_bit(occupied_bitboard_, to_sq_idx); 
  current_hash_ ^= zobrist_data.get_piece_key(final_piece_type, moving_piece_obj.player, to_sq_idx);
  if (is_capture) {
    const Piece &captured = undo_info.captured_piece.value(); player_points_[moving_piece_obj.player] += get_piece_capture_value(captured);
    if (captured.piece_type == PieceType::KING) { eliminate_player(captured.player); undo_info.eliminated_player = captured.player; }
  }
  Player player_who_moved = current_player_; Player last_active_player_in_sequence = get_last_active_player();
  bool was_last_player_turn = (player_who_moved == last_active_player_in_sequence);
  if (was_last_player_turn) full_move_number_++; 
  if (is_resetting_move) { move_number_of_last_reset_ = full_move_number_; position_history_.clear(); undo_info.was_history_cleared = true; } 
  else { undo_info.was_history_cleared = false; }
  undo_stack_.push_back(undo_info); 
  advance_turn();                   
  position_history_.push_back(get_position_key()); 
  is_game_over(); 
  return undo_info.captured_piece; 
}

std::optional<Piece> Board::make_move_for_mcts(const Move &move) {
  const auto& zobrist_data = get_zobrist_data();
  int fr = move.from_loc.row, fc = move.from_loc.col; int tr = move.to_loc.row, tc = move.to_loc.col;
  int from_sq_idx = Board::to_sq_idx(fr, fc); int to_sq_idx = Board::to_sq_idx(tr, tc); int moving_player_idx = static_cast<int>(current_player_);
  std::optional<Piece> moving_piece_opt = get_piece_at_sq(from_sq_idx);
  if (!moving_piece_opt) { throw std::runtime_error("MCTS: Attempting to move from an empty square. From sq: " + std::to_string(from_sq_idx)); }
  if (moving_piece_opt->player != current_player_) { throw std::runtime_error("MCTS: Attempting to move opponent's piece."); }
  Piece moving_piece_obj = *moving_piece_opt; int moving_pt_bb_idx = piece_type_to_bb_idx(moving_piece_obj.piece_type);
  std::optional<Piece> captured_piece_opt = get_piece_at_sq(to_sq_idx); 
  bool is_capture = captured_piece_opt.has_value(); bool is_pawn_move = (moving_piece_obj.piece_type == PieceType::PAWN);
  bool is_resetting_move = is_pawn_move || is_capture;
  current_hash_ ^= zobrist_data.get_piece_key(moving_piece_obj.piece_type, moving_piece_obj.player, from_sq_idx);
  clear_bit(piece_bitboards_[moving_player_idx][moving_pt_bb_idx], from_sq_idx); clear_bit(player_bitboards_[moving_player_idx], from_sq_idx); clear_bit(occupied_bitboard_, from_sq_idx);
  if (is_capture) {
      const Piece& captured = captured_piece_opt.value();
      current_hash_ ^= zobrist_data.get_piece_key(captured.piece_type, captured.player, to_sq_idx);
      int captured_player_idx = static_cast<int>(captured.player); int captured_pt_bb_idx = piece_type_to_bb_idx(captured.piece_type);
      clear_bit(piece_bitboards_[captured_player_idx][captured_pt_bb_idx], to_sq_idx); clear_bit(player_bitboards_[captured_player_idx], to_sq_idx);
  }
  PieceType final_piece_type = moving_piece_obj.piece_type; if (move.promotion_piece_type) { final_piece_type = move.promotion_piece_type.value(); }
  int final_pt_bb_idx = piece_type_to_bb_idx(final_piece_type);
  set_bit(piece_bitboards_[moving_player_idx][final_pt_bb_idx], to_sq_idx); set_bit(player_bitboards_[moving_player_idx], to_sq_idx); set_bit(occupied_bitboard_, to_sq_idx);
  current_hash_ ^= zobrist_data.get_piece_key(final_piece_type, moving_piece_obj.player, to_sq_idx);
  if (is_capture) {
    const Piece &captured_val = captured_piece_opt.value(); player_points_[moving_piece_obj.player] += get_piece_capture_value(captured_val);
    if (captured_val.piece_type == PieceType::KING) { eliminate_player(captured_val.player); }
  }
  Player player_who_moved = current_player_; Player last_active_player_in_sequence = get_last_active_player();
  bool was_last_player_turn = (player_who_moved == last_active_player_in_sequence);
  if (was_last_player_turn) full_move_number_++;
  if (is_resetting_move) { move_number_of_last_reset_ = full_move_number_; }
  advance_turn(); 
  is_game_over(); 
  return captured_piece_opt; 
}

// --- Undo Move (Unchanged) ---
void Board::undo_move() {
  if (undo_stack_.empty()) { throw std::runtime_error("No previous state available to undo."); }
  UndoInfo undo_info = undo_stack_.back(); undo_stack_.pop_back();
  piece_bitboards_ = undo_info.original_piece_bitboards; player_bitboards_ = undo_info.original_player_bitboards; occupied_bitboard_ = undo_info.original_occupied_bitboard;
  current_hash_ = undo_info.previous_hash;
  current_player_ = undo_info.original_player; full_move_number_ = undo_info.original_full_move_number; move_number_of_last_reset_ = undo_info.original_move_number_of_last_reset;
  bool is_resignation_undo = (undo_info.move.from_loc.row == -1); 
  if (!is_resignation_undo) { if (!position_history_.empty()) { position_history_.pop_back(); } }
  if (undo_info.eliminated_player) { Player player_to_revive = *undo_info.eliminated_player; active_players_.insert(player_to_revive); }
  if (!is_resignation_undo && undo_info.captured_piece) {
    const Piece &captured = undo_info.captured_piece.value();
    player_points_[undo_info.original_player] -= get_piece_capture_value(captured);
  }
  termination_reason_ = std::nullopt; 
}

// --- Player Elimination (Unchanged) ---
void Board::eliminate_player(Player player) {
  if (active_players_.count(player)) {
    const auto& zobrist_data = get_zobrist_data();
    current_hash_ ^= zobrist_data.get_active_player_status_key(player);
    active_players_.erase(player); 
  }
}

// --- Bitboard Accessors (Unchanged) ---
Bitboard Board::get_occupied_bitboard() const { return occupied_bitboard_; }
Bitboard Board::get_player_bitboard(Player p) const { return player_bitboards_[static_cast<int>(p)]; }
Bitboard Board::get_piece_bitboard(Player p, PieceType pt) const {
    return piece_bitboards_[static_cast<int>(p)][piece_type_to_bb_idx(pt)];
}
void Board::print_bitboard(Bitboard bb, const std::string& label) {
    std::cout << "Bitboard: " << label << " (0x" << std::hex << bb << std::dec << ")" << std::endl;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r, c);
            std::cout << (get_bit(bb, sq_idx) ? "1 " : ". ");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// --- Game State Accessors (Unchanged) ---
const Board::ActivePlayerSet &Board::get_active_players() const { return active_players_; }
const Board::PlayerPointMap &Board::get_player_points() const { return player_points_; }
Player Board::get_current_player() const { return current_player_; }
int Board::get_full_move_number() const { return full_move_number_; }
int Board::get_move_number_of_last_reset() const { return move_number_of_last_reset_; }
const std::optional<std::string> &Board::get_termination_reason() const { return termination_reason_; }
const Board::PositionHistory &Board::get_position_history() const { return position_history_; }
Player Board::get_last_active_player() const {
  if (active_players_.empty()) return Player::RED; 
  Player last_player = Player::RED; int max_val = -1;
  for (Player p : active_players_) { if (static_cast<int>(p) > max_val) { max_val = static_cast<int>(p); last_player = p; } }
  return last_player;
}

// --- Game Status (is_game_over, get_game_result, get_winner) (Unchanged) ---
bool Board::is_game_over() const {
  if (termination_reason_) return true; 
  if (active_players_.size() <= 1) { termination_reason_ = "elimination"; return true; }
  int moves_since_last_reset = full_move_number_ - move_number_of_last_reset_;
  if (moves_since_last_reset >= 50) { 
    if (!undo_stack_.empty()) { 
      Player player_who_just_moved = undo_stack_.back().original_player;
      if (player_who_just_moved == get_last_active_player()) { termination_reason_ = "fifty_move_rule"; return true; }
    }
  }
  int count = 0; for (const auto &key : position_history_) if (key == current_hash_) count++;
  if (count >= 3) { termination_reason_ = "threefold_repetition"; return true; }
  return false;
}
Board::PlayerPointMap Board::get_game_result() const {
  PlayerPointMap results = player_points_; int num_kings_of_inactive_players = 0;
  for (int p_idx_loop = 0; p_idx_loop < NUM_PLAYERS_BB; ++p_idx_loop) {
      Player p_enum = static_cast<Player>(p_idx_loop);
      if (!active_players_.count(p_enum)) { 
          if (piece_bitboards_[p_idx_loop][Board::piece_type_to_bb_idx(PieceType::KING)] != 0ULL) {
              num_kings_of_inactive_players++;
          }
      }
  }
  int num_active_players = active_players_.size();
  if (termination_reason_) { 
    const std::string &reason = *termination_reason_;
    if (reason == "fifty_move_rule" || reason == "threefold_repetition") {
      if (num_active_players > 0) { 
        int dead_king_bonus_per_player = (num_kings_of_inactive_players > 0) ? static_cast<int>(std::ceil(3.0 * num_kings_of_inactive_players / num_active_players)) : 0;
        for (Player p : active_players_) { results[p] += (2 + dead_king_bonus_per_player); }
      }
    } else if (reason == "elimination") {
      if (num_active_players == 1 && num_kings_of_inactive_players > 0) {
        results[*active_players_.begin()] += (3 * num_kings_of_inactive_players);
      }
    }
  }
  return results;
}
std::optional<Player> Board::get_winner() const {
  if (!termination_reason_) return std::nullopt; 
  PlayerPointMap final_scores = get_game_result();
  auto winner_it = std::max_element(final_scores.begin(), final_scores.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
  if (winner_it == final_scores.end()) return std::nullopt; 
  return (winner_it == final_scores.end()) ? std::nullopt : std::optional<Player>(winner_it->first);
}

// --- Piece Values (Unchanged) ---
int Board::get_piece_value(const Piece& piece) const {
  switch (piece.piece_type) {
  case PieceType::PAWN: return 1; case PieceType::KNIGHT: return 3; case PieceType::BISHOP: return 5; case PieceType::ROOK: return 5; case PieceType::KING: return 3; default: return 0; 
  }
}
int Board::get_piece_capture_value(const Piece& piece) const {
    if (!active_players_.count(piece.player)) { return (piece.piece_type == PieceType::KING) ? 3 : 0; }
    switch (piece.piece_type) {
        case PieceType::PAWN: return 1; case PieceType::KNIGHT: return 3; case PieceType::BISHOP: return 5; case PieceType::ROOK: return 5; case PieceType::KING: return 3; default: return 0;
    }
}

// --- Board Evaluation Function (evaluate) (Unchanged) ---
Board::PlayerPointMap Board::evaluate() const {
  PlayerPointMap scores; for (int i = 0; i < 4; ++i) scores[static_cast<Player>(i)] = 0.0; 
  std::map<Player, BoardLocation> king_coords; std::map<Player, bool> king_present; for (int i = 0; i < 4; ++i) king_present[static_cast<Player>(i)] = false;
  for (int p_idx = 0; p_idx < NUM_PLAYERS_BB; ++p_idx) {
      Player p_enum = static_cast<Player>(p_idx); Bitboard king_bb = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
      if (king_bb != 0ULL) { int king_sq = get_lsb_index(king_bb); king_coords[p_enum] = from_sq_idx(king_sq); king_present[p_enum] = true; } 
      else { king_present[p_enum] = false; }
  }
  for (int sq_idx = 0; sq_idx < NUM_SQUARES_BB; ++sq_idx) {
      BoardLocation loc = from_sq_idx(sq_idx); int r = loc.row; int c = loc.col;
      std::optional<Piece> piece_opt = get_piece_at_sq(sq_idx); 
      if (piece_opt) { 
        const Piece &piece = *piece_opt; Player player = piece.player;
        if (active_players_.count(player)) {
          scores[player] += get_piece_value(piece);
          if (piece.piece_type == PieceType::KNIGHT || piece.piece_type == PieceType::BISHOP) {
            if (((player == Player::RED && r == 7) || (player == Player::YELLOW && r == 0) || (player == Player::GREEN && c == 7) || (player == Player::BLUE && c == 0))) { scores[player] -= 0.4; }
          }
          if (piece.piece_type == PieceType::KING) {
            for (const auto &dir : KING_DIRS_EVAL) { 
              int nr = r + dir.first; int nc = c + dir.second;
              if (is_valid_square(nr, nc)) { 
                std::optional<Piece> adjacent_opt = get_piece_at_sq(to_sq_idx(nr, nc));
                if (adjacent_opt) { 
                  if (adjacent_opt->player == player) { scores[player] += (adjacent_opt->piece_type == PieceType::PAWN ? 0.2 : 0.05); } 
                  else { if (!active_players_.count(adjacent_opt->player)) { scores[player] += 0.15; } else { scores[player] -= 0.15; } }
                }
              }
            }
          } 
          if (piece.piece_type == PieceType::PAWN) {
            int dr = 0, dc = 0; int cap_r1 = 0, cap_c1 = 0, cap_r2 = 0, cap_c2 = 0; 
            switch (player) {
            case Player::RED:    scores[player] += 0.2 * (6 - r); dr = -1; dc = 0; cap_r1 = -1; cap_c1 = -1; cap_r2 = -1; cap_c2 = 1; break;
            case Player::BLUE:   scores[player] += 0.2 * (c - 1); dr = 0; dc = 1; cap_r1 = -1; cap_c1 = 1; cap_r2 = 1; cap_c2 = 1; break;
            case Player::YELLOW: scores[player] += 0.2 * (r - 1); dr = 1; dc = 0; cap_r1 = 1; cap_c1 = -1; cap_r2 = 1; cap_c2 = 1; break;
            case Player::GREEN:  scores[player] += 0.2 * (6 - c); dr = 0; dc = -1; cap_r1 = -1; cap_c1 = -1; cap_r2 = 1; cap_c2 = -1; break;
            }
            if (is_valid_square(r + dr, c + dc) && get_piece_at_sq(to_sq_idx(r + dr, c + dc))) scores[player] -= 0.2;
            for (const auto &cap_delta : {std::make_pair(cap_r1, cap_c1), std::make_pair(cap_r2, cap_c2)}) {
              int cap_r = r + cap_delta.first; int cap_c = c + cap_delta.second;
              if (is_valid_square(cap_r, cap_c)) {
                  std::optional<Piece> target_opt = get_piece_at_sq(to_sq_idx(cap_r, cap_c));
                  if (target_opt) { 
                    const auto &target = *target_opt;
                    if (target.player == player) { if (target.piece_type == PieceType::BISHOP || target.piece_type == PieceType::KNIGHT) { scores[player] += 0.2; } } 
                    else { scores[player] += 0.2; if (target.piece_type == PieceType::KING && active_players_.count(target.player)) { scores[player] += 0.1; scores[target.player] -= 0.5; } }
                  }
              }
            }
          } 
        } 
      } 
    } 
  for (int i = 0; i < 4; ++i) {
    Player p = static_cast<Player>(i);
    if (active_players_.count(p) && !king_present[p]) scores[p] = -999.0; 
    scores[p] += player_points_.at(p); 
    scores[p] -= 20;                   
  }
  return scores;
}

// --- Player Actions (resign, advance_turn) (Unchanged) ---
void Board::resign() {
  Player resigning_player = current_player_; 
  if (active_players_.count(resigning_player)) {
    UndoInfo resign_undo_info;
    resign_undo_info.original_piece_bitboards = piece_bitboards_; resign_undo_info.original_player_bitboards = player_bitboards_; resign_undo_info.original_occupied_bitboard = occupied_bitboard_;
    resign_undo_info.original_player = resigning_player; resign_undo_info.original_full_move_number = full_move_number_;
    resign_undo_info.original_move_number_of_last_reset = move_number_of_last_reset_; resign_undo_info.previous_hash = current_hash_;
    resign_undo_info.eliminated_player = resigning_player; resign_undo_info.was_history_cleared = false;         
    resign_undo_info.move.from_loc = {-1,-1}; resign_undo_info.captured_piece = std::nullopt; 
    eliminate_player(resigning_player); 
    if (active_players_.size() <= 1) { 
        const auto& zobrist_data = get_zobrist_data();
        current_hash_ ^= zobrist_data.get_turn_key(resigning_player); 
        is_game_over(); 
    } else { advance_turn(); }
    undo_stack_.push_back(resign_undo_info); 
  }
}
void Board::advance_turn() {
  const auto& zobrist_data = get_zobrist_data(); Player old_player = current_player_;
  current_player_ = static_cast<Player>((static_cast<int>(current_player_) + 1) % 4);
  while (active_players_.find(current_player_) == active_players_.end()) {
    if (active_players_.size() <= 1) break; 
    current_player_ = static_cast<Player>((static_cast<int>(current_player_) + 1) % 4);
  }
  if (!active_players_.empty()) { 
      current_hash_ ^= zobrist_data.get_turn_key(old_player); 
      if(active_players_.count(current_player_)){ current_hash_ ^= zobrist_data.get_turn_key(current_player_); }
  }
}

// --- ANSI Color Codes and Unicode Symbols (for print_board) (Unchanged) ---
const std::string ANSI_RESET_BB = "\033[0m"; 
const std::string ANSI_RED_BB = "\033[31m"; const std::string ANSI_GREEN_BB = "\033[32m";
const std::string ANSI_YELLOW_BB = "\033[33m"; const std::string ANSI_BLUE_BB = "\033[34m";
const std::string UNICODE_KING_BB = "♔"; const std::string UNICODE_ROOK_BB = "♖";
const std::string UNICODE_BISHOP_BB = "♗"; const std::string UNICODE_KNIGHT_BB = "♘";
const std::string UNICODE_PAWN_BB = "♙";

// --- Utility: Print Board to Console (print_board) (Unchanged) ---
void Board::print_board() const {
  std::cout << "   a  b  c  d  e  f  g  h" << std::endl; 
  for (int r = 0; r < BOARD_SIZE; ++r) {
    std::cout << 8 - r << " "; 
    for (int c = 0; c < BOARD_SIZE; ++c) {
      int sq_idx = to_sq_idx(r, c); std::optional<Piece> piece_opt = get_piece_at_sq(sq_idx); 
      std::string symbol_str = " "; 
      if (piece_opt) {
        const Piece &p = *piece_opt; bool display_as_inactive = !active_players_.count(p.player); 
        const std::string* base_symbol = nullptr;
        switch (p.piece_type) {
            case PieceType::PAWN:   base_symbol = &UNICODE_PAWN_BB;   break; case PieceType::KNIGHT: base_symbol = &UNICODE_KNIGHT_BB; break;
            case PieceType::BISHOP: base_symbol = &UNICODE_BISHOP_BB; break; case PieceType::ROOK:   base_symbol = &UNICODE_ROOK_BB;   break;
            case PieceType::KING:   base_symbol = &UNICODE_KING_BB;   break;
        }
        if (base_symbol) { 
            if (display_as_inactive) { symbol_str = *base_symbol; } 
            else { 
                const std::string* color_code = nullptr;
                switch (p.player) {
                    case Player::RED:    color_code = &ANSI_RED_BB;    break; case Player::BLUE:   color_code = &ANSI_BLUE_BB;   break;
                    case Player::YELLOW: color_code = &ANSI_YELLOW_BB; break; case Player::GREEN:  color_code = &ANSI_GREEN_BB;  break;
                }
                if (color_code) { symbol_str = *color_code + *base_symbol + ANSI_RESET_BB; } 
                else { symbol_str = *base_symbol; }
            }
        }
      }
      std::cout << "[" << symbol_str << "]"; 
    }
    std::cout << std::endl; 
  }
  std::cout << "Turn: ";
  switch (current_player_) {
  case Player::RED:    std::cout << ANSI_RED_BB << "RED" << ANSI_RESET_BB; break; case Player::BLUE:   std::cout << ANSI_BLUE_BB << "BLUE" << ANSI_RESET_BB; break;
  case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "YELLOW" << ANSI_RESET_BB; break; case Player::GREEN:  std::cout << ANSI_GREEN_BB << "GREEN" << ANSI_RESET_BB; break;
  }
  std::cout << std::endl; std::cout << "Active Players: ";
  for(Player active_p : active_players_){
      switch(active_p){
          case Player::RED: std::cout << ANSI_RED_BB << "R " << ANSI_RESET_BB; break; case Player::BLUE: std::cout << ANSI_BLUE_BB << "B " << ANSI_RESET_BB; break;
          case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "Y " << ANSI_RESET_BB; break; case Player::GREEN: std::cout << ANSI_GREEN_BB << "G " << ANSI_RESET_BB; break;
      }
  }
  std::cout << std::endl; std::cout << "Points: ";
  for(const auto& pt_pair : player_points_){ 
      switch(pt_pair.first){ 
          case Player::RED: std::cout << ANSI_RED_BB << "R:" << pt_pair.second << ANSI_RESET_BB << " "; break; case Player::BLUE: std::cout << ANSI_BLUE_BB << "B:" << pt_pair.second << ANSI_RESET_BB << " "; break;
          case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "Y:" << pt_pair.second << ANSI_RESET_BB << " "; break; case Player::GREEN: std::cout << ANSI_GREEN_BB << "G:" << pt_pair.second << ANSI_RESET_BB << " "; break;
      }
  }
  std::cout << std::endl; if(termination_reason_) std::cout << "Game Over: " << *termination_reason_ << std::endl;
}

// --- Zobrist Position Key Accessor (Unchanged) ---
Board::PositionKey Board::get_position_key() const { return current_hash_; }

} // namespace chaturaji_cpp