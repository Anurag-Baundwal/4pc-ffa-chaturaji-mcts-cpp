// board.cpp
#include "board.h"
#include <algorithm> // For std::find, std::max_element, std::copy
#include <array>     // For Zobrist key storage and bitboard arrays
#include <cmath>     // For std::ceil, std::round (used in evaluate, get_game_result)
#include <cstdint>   // For ZobristKey (uint64_t), Bitboard (uint64_t)
#include <iostream>  // For print_board, print_bitboard
#include <limits>    // For numeric_limits (Zobrist key generation)
#include <numeric>   // For std::accumulate (optional, not currently used but good include)
#include <random>    // For Zobrist key generation (std::mt19937_64)
#include <sstream>   // Potentially for string conversions (not directly used here)
#include <stdexcept> // For std::out_of_range, std::runtime_error
#include <utility>   // For std::move, std::pair
#include <vector>    // For move lists, undo stack, directional constants

#ifdef _MSC_VER
#include <intrin.h> // For MSVC specific intrinsics like _BitScanForward64 (used by pop_lsb/get_lsb_index)
#endif

namespace chaturaji_cpp {

// Anonymous namespace for Zobrist, internal constants, and lookup table initializers
namespace { 
// --- Zobrist Hashing Constants ---
const int NUM_PIECE_TYPES_FOR_HASH = 5; // P, N, B, R, K (for Zobrist keys)
// --- Bitboard Related Constants ---
const int NUM_BB_PIECE_TYPES = 5;       // P, N, B, R, K (for bitboard array indexing)
const int NUM_PLAYERS_BB = 4;           // Number of players (for bitboard array indexing)

// --- DIRECTIONAL CONSTANTS (primarily for evaluate() ) ---
// These are used by the evaluate() function.
// Bitboard move generation uses its own lookup tables or specific bitwise logic.
const std::vector<std::pair<int, int>> BISHOP_DIRS_EVAL = { {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
const std::vector<std::pair<int, int>> ROOK_DIRS_EVAL = { {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
const std::vector<std::pair<int, int>> KING_DIRS_EVAL = { {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
const std::vector<std::pair<int, int>> KNIGHT_MOVES_EVAL = { {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
// --- END ADDED DIRECTIONAL CONSTANTS ---


// Helper to map PieceType to bitboard array index (0-4)
int piece_type_to_bb_idx_internal(PieceType pt) {
    // PieceType is PAWN=1 ... KING=5. Map to 0-4.
    int val = static_cast<int>(pt) - 1; 
    if (val < 0 || val >= NUM_BB_PIECE_TYPES) {
        throw std::out_of_range("Invalid PieceType for bitboard index.");
    }
    return val;
}

// --- Zobrist Hashing Data Structure and Initialization ---
struct ZobristData {
  // piece_keys[piece_type_idx][player_idx][square_idx]
  std::array<std::array<std::array<ZobristKey, NUM_SQUARES_BB>, NUM_PLAYERS_BB>, NUM_PIECE_TYPES_FOR_HASH> piece_keys;
  std::array<ZobristKey, NUM_PLAYERS_BB> turn_keys;
  std::array<ZobristKey, NUM_PLAYERS_BB> active_player_status_keys;
  // Add keys for castling, en passant if needed in a different game variant

  ZobristData() {
    // Use a high-quality random number generator
    std::mt19937_64 rng(0xBADFACE); // Fixed seed for reproducibility
    std::uniform_int_distribution<ZobristKey> dist(0, std::numeric_limits<ZobristKey>::max());

    // Generate keys for each piece type, player, and square
    for (int type_idx = 0; type_idx < NUM_PIECE_TYPES_FOR_HASH; ++type_idx) {
      for (int player_idx = 0; player_idx < NUM_PLAYERS_BB; ++player_idx) {
          for (int sq_idx = 0; sq_idx < NUM_SQUARES_BB; ++sq_idx) {
            piece_keys[type_idx][player_idx][sq_idx] = dist(rng);
          }
      }
    }

    // Generate keys for whose turn it is
    for (int player_idx = 0; player_idx < NUM_PLAYERS_BB; ++player_idx) {
      turn_keys[player_idx] = dist(rng);
    }

    // Generate keys for player active status
    for (int player_idx = 0; player_idx < NUM_PLAYERS_BB; ++player_idx) {
        active_player_status_keys[player_idx] = dist(rng);
    }
  }

  // Helper to get piece key safely, mapping PieceType to array index
  ZobristKey get_piece_key(PieceType type, Player player, int square_index) const {
    if (square_index < 0 || square_index >= NUM_SQUARES_BB) {
      throw std::out_of_range("Square index out of range for Zobrist key lookup.");
    }
    int type_idx = static_cast<int>(type) - 1; // PieceType is PAWN=1 ... KING=5. Map to 0-4.
    if (type_idx < 0 || type_idx >= NUM_PIECE_TYPES_FOR_HASH) {
      throw std::out_of_range("PieceType out of range for Zobrist key lookup.");
    }
    int player_idx = static_cast<int>(player); // Player enum 0..3 maps directly
    if (player_idx < 0 || player_idx >= NUM_PLAYERS_BB) {
      throw std::out_of_range("Player out of range for Zobrist key lookup.");
    }
    return piece_keys[type_idx][player_idx][square_index]; 
    }

  // Helper to get turn key safely
  ZobristKey get_turn_key(Player player) const {
    int player_idx = static_cast<int>(player);
    if (player_idx < 0 || player_idx >= NUM_PLAYERS_BB) {
      throw std::out_of_range("Player out of range for Zobrist key lookup.");
    }
    return turn_keys[player_idx];
  }

  // Helper to get active player status key safely
  ZobristKey get_active_player_status_key(Player player) const {
    int player_idx = static_cast<int>(player);
    if (player_idx < 0 || player_idx >= NUM_PLAYERS_BB) {
        throw std::out_of_range("Player out of range for Zobrist active status key lookup.");
    }
    return active_player_status_keys[player_idx];
  }
};

// Meyers' Singleton: Ensures safe initialization of ZobristData on first access
const ZobristData &get_zobrist_data() {
  static const ZobristData instance; // Initialized only once
  return instance;
}

// --- Bitboard constants for pawn move generation (file checks) ---
const Bitboard FILE_A_BB = 0x0101010101010101ULL; // Bitboard representing file A
const Bitboard FILE_H_BB = 0x8080808080808080ULL; // Bitboard representing file H

// --- Pawn Promotion Target Coordinates (Bitboard version) ---
// These define the row or column index a pawn must reach for promotion.
const int PROMOTION_ROW_RED_BB = 0;    // Red pawns promote on row 0
const int PROMOTION_COL_BLUE_BB = 7;   // Blue pawns promote on col 7
const int PROMOTION_ROW_YELLOW_BB = 7; // Yellow pawns promote on row 7
const int PROMOTION_COL_GREEN_BB = 0;  // Green pawns promote on col 0
} // end anonymous namespace


// --- Static Lookup Tables for Bitboard Move Generation ---
// These are initialized once by StaticInitializer.
std::array<Bitboard, NUM_SQUARES_BB> Board::knight_attacks_; // knight_attacks_[sq_idx] = bitboard of squares a knight can attack from sq_idx
std::array<Bitboard, NUM_SQUARES_BB> Board::king_attacks_;   // king_attacks_[sq_idx] = bitboard of squares a king can attack from sq_idx

// Pawn attacks: pawn_attacks_PLAYERCOLOR_[player_enum_val][sq_idx]
// Note: The player_enum_val index here is a bit redundant in naming but clarifies structure.
// e.g. pawn_attacks_red_[Player::RED_idx][sq_idx] for red pawn attacks from sq_idx.
// The first dimension [player_enum_val] will only have data for that specific player,
// e.g., pawn_attacks_red_ will only have meaningful data in pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx].
std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> Board::pawn_attacks_red_;
std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> Board::pawn_attacks_blue_;
std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> Board::pawn_attacks_yellow_;
std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> Board::pawn_attacks_green_;

// Pawn forward moves (non-capture): pawn_fwd_moves_PLAYERCOLOR_[sq_idx]
std::array<Bitboard, NUM_SQUARES_BB> Board::pawn_fwd_moves_red_;
std::array<Bitboard, NUM_SQUARES_BB> Board::pawn_fwd_moves_blue_;
std::array<Bitboard, NUM_SQUARES_BB> Board::pawn_fwd_moves_yellow_;
std::array<Bitboard, NUM_SQUARES_BB> Board::pawn_fwd_moves_green_;

// Sliding piece rays: piece_rays_[sq_idx][direction_idx]
// Note: These are initialized but currently NOT USED by get_rook_moves_bb/get_bishop_moves_bb.
// Those functions use on-the-fly ray casting in generate_sliding_moves.
std::array<std::array<Bitboard, 4>, NUM_SQUARES_BB> Board::rook_rays_; 
std::array<std::array<Bitboard, 4>, NUM_SQUARES_BB> Board::bishop_rays_;

// Static initializer trick to call initialize_lookup_tables() before main()
Board::StaticInitializer Board::static_initializer_; 


// --- Bitboard Helper Functions (Public Static) ---
// Maps PieceType enum to an index suitable for piece_bitboards_ array (0-4)
int Board::piece_type_to_bb_idx(PieceType pt) {
    return piece_type_to_bb_idx_internal(pt); // Calls internal helper
}
// Checks if a square index (0-63) is valid
bool Board::is_valid_sq_idx(int sq_idx) {
    return sq_idx >= 0 && sq_idx < NUM_SQUARES_BB;
}
// Converts (row, col) to a square index (0-63)
int Board::to_sq_idx(int r, int c) {
    return r * BOARD_SIZE + c; // BOARD_SIZE is 8
}
// Converts a square index (0-63) to (row, col)
BoardLocation Board::from_sq_idx(int sq_idx) {
    return {sq_idx / BOARD_SIZE, sq_idx % BOARD_SIZE};
}

// --- Lookup Table Initialization ---
void Board::initialize_lookup_tables() {
    // --- Knight Attacks ---
    const int kn_moves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                {1, -2},  {1, 2},  {2, -1},  {2, 1}};
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r, c);
            knight_attacks_[sq_idx] = 0ULL;
            for (auto& move : kn_moves) {
                int nr = r + move[0];
                int nc = c + move[1];
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    set_bit(knight_attacks_[sq_idx], to_sq_idx(nr, nc));
                }
            }
        }
    }
    // --- King Attacks ---
    const int ki_moves[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                {0, 1},   {1, -1}, {1, 0},  {1, 1}};
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r, c);
            king_attacks_[sq_idx] = 0ULL;
            for (auto& move : ki_moves) {
                int nr = r + move[0];
                int nc = c + move[1];
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    set_bit(king_attacks_[sq_idx], to_sq_idx(nr, nc));
                }
            }
        }
    }

    // --- Pawn Forward Moves and Attacks (for each player color/direction) ---
    // Red Pawns (move -1 in row)
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c);
            pawn_fwd_moves_red_[sq_idx] = 0ULL;
            pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx] = 0ULL; 
            if (r > 0) { // Can move forward
                set_bit(pawn_fwd_moves_red_[sq_idx], to_sq_idx(r-1, c));
                // Captures: not on file A for left capture, not on file H for right capture
                if (c > 0 && !(FILE_A_BB & (1ULL << sq_idx))) set_bit(pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx], to_sq_idx(r-1, c-1));
                if (c < BOARD_SIZE - 1 && !(FILE_H_BB & (1ULL << sq_idx))) set_bit(pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx], to_sq_idx(r-1, c+1));
            }
        }
    }
    // Blue Pawns (move +1 in col)
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c);
            pawn_fwd_moves_blue_[sq_idx] = 0ULL;
            pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx] = 0ULL;
             if (c < BOARD_SIZE -1) { // Can move forward
                set_bit(pawn_fwd_moves_blue_[sq_idx], to_sq_idx(r, c+1));
                if (r > 0) set_bit(pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx], to_sq_idx(r-1, c+1));
                if (r < BOARD_SIZE - 1) set_bit(pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx], to_sq_idx(r+1, c+1));
            }
        }
    }
    // Yellow Pawns (move +1 in row)
     for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c);
            pawn_fwd_moves_yellow_[sq_idx] = 0ULL;
            pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx] = 0ULL;
             if (r < BOARD_SIZE -1) { // Can move forward
                set_bit(pawn_fwd_moves_yellow_[sq_idx], to_sq_idx(r+1, c));
                if (c > 0 && !(FILE_A_BB & (1ULL << sq_idx))) set_bit(pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx], to_sq_idx(r+1, c-1));
                if (c < BOARD_SIZE - 1 && !(FILE_H_BB & (1ULL << sq_idx))) set_bit(pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx], to_sq_idx(r+1, c+1));
            }
        }
    }
    // Green Pawns (move -1 in col)
     for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c);
            pawn_fwd_moves_green_[sq_idx] = 0ULL;
            pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx] = 0ULL;
             if (c > 0) { // Can move forward
                set_bit(pawn_fwd_moves_green_[sq_idx], to_sq_idx(r, c-1));
                if (r > 0) set_bit(pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx], to_sq_idx(r-1, c-1));
                if (r < BOARD_SIZE - 1) set_bit(pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx], to_sq_idx(r+1, c-1));
            }
        }
    }

    // --- Sliding Piece Rays (Rook & Bishop) ---
    // Note: These are initialized but currently NOT USED by get_rook_moves_bb/get_bishop_moves_bb.
    // Rook Rays (0: North, 1: East, 2: South, 3: West)
    for (int r_start = 0; r_start < BOARD_SIZE; ++r_start) {
        for (int c_start = 0; c_start < BOARD_SIZE; ++c_start) {
            int start_sq = to_sq_idx(r_start, c_start);
            rook_rays_[start_sq][0] = 0ULL; rook_rays_[start_sq][1] = 0ULL; 
            rook_rays_[start_sq][2] = 0ULL; rook_rays_[start_sq][3] = 0ULL; 
            for (int r_ray = r_start - 1; r_ray >= 0; --r_ray) set_bit(rook_rays_[start_sq][0], to_sq_idx(r_ray, c_start)); // North
            for (int c_ray = c_start + 1; c_ray < BOARD_SIZE; ++c_ray) set_bit(rook_rays_[start_sq][1], to_sq_idx(r_start, c_ray)); // East
            for (int r_ray = r_start + 1; r_ray < BOARD_SIZE; ++r_ray) set_bit(rook_rays_[start_sq][2], to_sq_idx(r_ray, c_start)); // South
            for (int c_ray = c_start - 1; c_ray >= 0; --c_ray) set_bit(rook_rays_[start_sq][3], to_sq_idx(r_start, c_ray)); // West
        }
    }
    // Bishop Rays (0: NE, 1: SE, 2: SW, 3: NW)
     for (int r_start = 0; r_start < BOARD_SIZE; ++r_start) {
        for (int c_start = 0; c_start < BOARD_SIZE; ++c_start) {
            int start_sq = to_sq_idx(r_start, c_start);
            bishop_rays_[start_sq][0] = 0ULL; bishop_rays_[start_sq][1] = 0ULL; 
            bishop_rays_[start_sq][2] = 0ULL; bishop_rays_[start_sq][3] = 0ULL; 
            for (int i = 1; r_start - i >= 0 && c_start + i < BOARD_SIZE; ++i) set_bit(bishop_rays_[start_sq][0], to_sq_idx(r_start - i, c_start + i)); // NE
            for (int i = 1; r_start + i < BOARD_SIZE && c_start + i < BOARD_SIZE; ++i) set_bit(bishop_rays_[start_sq][1], to_sq_idx(r_start + i, c_start + i)); // SE
            for (int i = 1; r_start + i < BOARD_SIZE && c_start - i >= 0; ++i) set_bit(bishop_rays_[start_sq][2], to_sq_idx(r_start + i, c_start - i)); // SW
            for (int i = 1; r_start - i >= 0 && c_start - i >= 0; ++i) set_bit(bishop_rays_[start_sq][3], to_sq_idx(r_start - i, c_start - i)); // NW
        }
    }
}

// --- Constructor ---
Board::Board()
    : current_player_(Player::RED), full_move_number_(1),
      move_number_of_last_reset_(0), termination_reason_(std::nullopt) {
  // Initialize bitboards to all 0s (empty)
  for (auto& player_bb_array : piece_bitboards_) {
      player_bb_array.fill(0ULL);
  }
  player_bitboards_.fill(0ULL);
  occupied_bitboard_ = 0ULL;

  // Initialize player points and active players
  for (int i = 0; i < 4; ++i) {
    Player p = static_cast<Player>(i);
    player_points_[p] = 0;
    active_players_.insert(p);
  }
  // Setup initial piece positions (only on bitboards now)
  setup_initial_board(); 

  // --- Calculate Initial Zobrist Hash --- 
  const auto& zobrist_data = get_zobrist_data();
  current_hash_ = 0; // Start fresh

  // Hash pieces from bitboards
  for (int p_idx = 0; p_idx < NUM_PLAYERS_BB; ++p_idx) {
      Player player = static_cast<Player>(p_idx);
      for (int pt_bb_idx = 0; pt_bb_idx < NUM_BB_PIECE_TYPES; ++pt_bb_idx) {
          PieceType piece_type = static_cast<PieceType>(pt_bb_idx + 1); // Map 0-4 to PieceType 1-5
          Bitboard current_piece_bb = piece_bitboards_[p_idx][pt_bb_idx];
          Bitboard temp_bb = current_piece_bb;
          while(temp_bb) { // Iterate over set bits in the current piece bitboard
              int sq_idx = pop_lsb(temp_bb); // Get and remove one piece's square
              current_hash_ ^= zobrist_data.get_piece_key(piece_type, player, sq_idx);
          }
      }
  }
  // Hash current player's turn
  current_hash_ ^= zobrist_data.get_turn_key(current_player_);
  // Hash active player statuses
  for (Player p : active_players_) { // active_players_ is initialized with all players
      current_hash_ ^= zobrist_data.get_active_player_status_key(p);
  }
  // Add initial position hash to history
  position_history_.push_back(current_hash_);
}

// --- Copy Constructor ---
Board::Board(const Board &other)
    : // REMOVED: board_(other.board_), (Mailbox array `board_` is no longer a member)
      active_players_(other.active_players_),
      player_points_(other.player_points_),
      current_player_(other.current_player_),
      position_history_(other.position_history_),
      full_move_number_(other.full_move_number_),
      move_number_of_last_reset_(other.move_number_of_last_reset_),
      termination_reason_(other.termination_reason_),
      current_hash_(other.current_hash_),
      undo_stack_(other.undo_stack_), // Deep copy handled by stack's copy constructor/assignment
      // Copy bitboard states
      piece_bitboards_(other.piece_bitboards_),
      player_bitboards_(other.player_bitboards_),
      occupied_bitboard_(other.occupied_bitboard_)
       {}

// --- Move Constructor ---
Board::Board(Board &&other) noexcept
    : // REMOVED: board_(std::move(other.board_)), (Mailbox array `board_` is no longer a member)
      active_players_(std::move(other.active_players_)),
      player_points_(std::move(other.player_points_)),
      current_player_(other.current_player_),
      position_history_(std::move(other.position_history_)),
      full_move_number_(other.full_move_number_),
      move_number_of_last_reset_(other.move_number_of_last_reset_),
      termination_reason_(std::move(other.termination_reason_)),
      current_hash_(other.current_hash_),
      undo_stack_(std::move(other.undo_stack_)),
      // Move bitboard states
      piece_bitboards_(std::move(other.piece_bitboards_)),
      player_bitboards_(std::move(other.player_bitboards_)),
      occupied_bitboard_(other.occupied_bitboard_)
       {
  // Reset moved-from object's state
  other.full_move_number_ = 1; 
  other.move_number_of_last_reset_ = 0;
  other.current_hash_ = 0; 
  // Reset moved-from object's bitboards
  other.occupied_bitboard_ = 0ULL; 
  for(auto& arr : other.piece_bitboards_) arr.fill(0ULL);
  other.player_bitboards_.fill(0ULL);
}

// --- Copy Assignment Operator ---
Board &Board::operator=(const Board &other) {
  if (this != &other) { // Self-assignment check
    // REMOVED: board_ = other.board_; (Mailbox array `board_` is no longer a member)
    active_players_ = other.active_players_;
    player_points_ = other.player_points_;
    current_player_ = other.current_player_;
    position_history_ = other.position_history_;
    full_move_number_ = other.full_move_number_;
    move_number_of_last_reset_ = other.move_number_of_last_reset_;
    termination_reason_ = other.termination_reason_;
    current_hash_ = other.current_hash_;
    undo_stack_ = other.undo_stack_; // Deep copy handled by stack's op=
    // Copy bitboard states
    piece_bitboards_ = other.piece_bitboards_;
    player_bitboards_ = other.player_bitboards_;
    occupied_bitboard_ = other.occupied_bitboard_;
  }
  return *this;
}

// --- Move Assignment Operator ---
Board &Board::operator=(Board &&other) noexcept {
  if (this != &other) { // Self-assignment check
    // REMOVED: board_ = std::move(other.board_); (Mailbox array `board_` is no longer a member)
    active_players_ = std::move(other.active_players_);
    player_points_ = std::move(other.player_points_);
    current_player_ = other.current_player_; // Enum copy is fine
    position_history_ = std::move(other.position_history_);
    full_move_number_ = other.full_move_number_;
    move_number_of_last_reset_ = other.move_number_of_last_reset_;
    termination_reason_ = std::move(other.termination_reason_);
    current_hash_ = other.current_hash_;
    undo_stack_ = std::move(other.undo_stack_);
    // Move bitboard states
    piece_bitboards_ = std::move(other.piece_bitboards_);
    player_bitboards_ = std::move(other.player_bitboards_);
    occupied_bitboard_ = other.occupied_bitboard_;

    // Reset moved-from object's state
    other.full_move_number_ = 1;
    other.move_number_of_last_reset_ = 0;
    other.current_hash_ = 0;
    // Reset moved-from object's bitboards
    other.occupied_bitboard_ = 0ULL; 
    for(auto& arr : other.piece_bitboards_) arr.fill(0ULL);
    other.player_bitboards_.fill(0ULL);
  }
  return *this;
}

// --- MCTS Child Board Creation ---
// Creates a child board state by applying a move to a copy of the parent's state.
// Designed for MCTS to explore moves without modifying the parent node's board.
Board Board::create_mcts_child_board(const Board& parent_board, const Move& move) {
  // 1. Create a new board object. Default constructor initializes histories/undo empty.
  Board child_board; 

  // 2. Copy essential current state from the parent
  // REMOVED: child_board.board_ = parent_board.board_; (Mailbox array `board_` is no longer a member)
  child_board.active_players_ = parent_board.active_players_;
  child_board.player_points_ = parent_board.player_points_;
  child_board.current_player_ = parent_board.current_player_; // Player *before* the move
  child_board.full_move_number_ = parent_board.full_move_number_;
  child_board.move_number_of_last_reset_ = parent_board.move_number_of_last_reset_;
  child_board.current_hash_ = parent_board.current_hash_; // Hash *before* the move

  // Copy bitboard states from parent
  child_board.piece_bitboards_ = parent_board.piece_bitboards_;
  child_board.player_bitboards_ = parent_board.player_bitboards_;
  child_board.occupied_bitboard_ = parent_board.occupied_bitboard_;
  
  // termination_reason_ is not copied; child node determines its own termination.
  // position_history_ and undo_stack_ start empty for the child and are populated by make_move.

  // 3. Apply the move to the child board's copied state using the lightweight function.
  // make_move_for_mcts will update: current_player_, full_move_number_, move_number_of_last_reset_,
  // current_hash_, bitboards.
  // It will NOT push to undo_stack_ or position_history_ of the child_board.
  child_board.make_move_for_mcts(move);

  // The state of child_board is now the state *after* 'move' was applied.
  // Its history/undo stack only contains information about *that single move*.
  return child_board; // Return the newly created and updated board state
}

// --- Helper to get piece at a square using bitboards ---
// Iterates through player and piece bitboards to find the piece at a given square.
// Returns std::nullopt if the square is empty or invalid.
std::optional<Piece> Board::get_piece_at_sq(int sq_idx) const {
    if (!is_valid_sq_idx(sq_idx)) return std::nullopt; // Invalid square
    if (!get_bit(occupied_bitboard_, sq_idx)) return std::nullopt; // Optimization: if not occupied, no piece there

    // Iterate through each player
    for (int p_idx = 0; p_idx < NUM_PLAYERS_BB; ++p_idx) {
        if (get_bit(player_bitboards_[p_idx], sq_idx)) { // If this player has a piece on this square
            Player player = static_cast<Player>(p_idx);
            // Iterate through each piece type for this player
            for (int pt_bb_idx = 0; pt_bb_idx < NUM_BB_PIECE_TYPES; ++pt_bb_idx) {
                if (get_bit(piece_bitboards_[p_idx][pt_bb_idx], sq_idx)) {
                    PieceType pt = static_cast<PieceType>(pt_bb_idx + 1); // Map 0-4 to PieceType enum 1-5
                    return Piece(player, pt); // Found the piece
                }
            }
            // This state implies an inconsistency if player_bitboards_[p_idx] was set
            // but no specific piece_bitboards_ was set for that player at that square.
            // This should not happen with correct bitboard management.
            throw std::runtime_error("Bitboard inconsistency in get_piece_at_sq: Player bit set, but no piece type bit.");
        }
    }
    // This implies occupied_bitboard_ was set, but no player_bitboards_ was set for that square.
    // This should ideally not happen if bitboards are managed correctly.
    // Returning nullopt is safer than throwing if such an intermediate state could (erroneously) exist,
    // though the previous check for player_bitboards should cover most cases.
    return std::nullopt; // Should be unreachable if occupied_bitboard_ implies a player's bit is also set.
}

// --- Initial Board Setup ---
void Board::setup_initial_board() {
  // Clear existing bitboards (redundant if called from constructor, but safe)
  for (auto& player_bbs : piece_bitboards_) player_bbs.fill(0ULL);
  player_bitboards_.fill(0ULL);
  occupied_bitboard_ = 0ULL;
  // REMOVED: for (auto& row : board_) row.fill(std::nullopt); (Mailbox array removal)

  // Helper lambda to place a piece on the bitboards
  auto place_piece = [&](Player p, PieceType pt, int r, int c) {
      // REMOVED: board_[r][c].emplace(p, pt); (Mailbox array removal)
      int sq_idx = to_sq_idx(r, c);
      int player_idx = static_cast<int>(p);
      int pt_bb_idx = piece_type_to_bb_idx(pt);
      // Update bitboards
      set_bit(piece_bitboards_[player_idx][pt_bb_idx], sq_idx); // Specific piece type for player
      set_bit(player_bitboards_[player_idx], sq_idx);           // All pieces for player
      set_bit(occupied_bitboard_, sq_idx);                      // All occupied squares
  };

  // Place Red pieces
  place_piece(Player::RED, PieceType::ROOK, 7, 0);
  place_piece(Player::RED, PieceType::KNIGHT, 7, 1);
  place_piece(Player::RED, PieceType::BISHOP, 7, 2);
  place_piece(Player::RED, PieceType::KING, 7, 3);
  for (int col = 0; col < 4; ++col) place_piece(Player::RED, PieceType::PAWN, 6, col);

  // Place Blue pieces
  place_piece(Player::BLUE, PieceType::ROOK, 0, 0);
  place_piece(Player::BLUE, PieceType::KNIGHT, 1, 0);
  place_piece(Player::BLUE, PieceType::BISHOP, 2, 0);
  place_piece(Player::BLUE, PieceType::KING, 3, 0);
  for (int row = 0; row < 4; ++row) place_piece(Player::BLUE, PieceType::PAWN, row, 1);

  // Place Yellow pieces
  place_piece(Player::YELLOW, PieceType::ROOK, 0, 7);
  place_piece(Player::YELLOW, PieceType::KNIGHT, 0, 6);
  place_piece(Player::YELLOW, PieceType::BISHOP, 0, 5);
  place_piece(Player::YELLOW, PieceType::KING, 0, 4);
  for (int col = 4; col < 8; ++col) place_piece(Player::YELLOW, PieceType::PAWN, 1, col);

  // Place Green pieces
  place_piece(Player::GREEN, PieceType::KING, 4, 7);
  place_piece(Player::GREEN, PieceType::BISHOP, 5, 7);
  place_piece(Player::GREEN, PieceType::KNIGHT, 6, 7);
  place_piece(Player::GREEN, PieceType::ROOK, 7, 7);
  for (int row = 4; row < 8; ++row) place_piece(Player::GREEN, PieceType::PAWN, row, 6);
}

// --- Square Validity Check (Array Context) ---
// Used mainly by functions that still iterate r,c like evaluate() or print_board(),
// or for on-the-fly sliding move generation.
bool Board::is_valid_square(int row, int col) const {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

// --- Pseudo-Legal Move Generation (Master Function) ---
std::vector<Move> Board::get_pseudo_legal_moves(Player player) const {
  std::vector<Move> pseudo_legal_moves;
  pseudo_legal_moves.reserve(128); // Pre-allocate reasonable space
  
  // Return empty if player is not active (e.g., eliminated)
  if (!active_players_.count(player)) {
      return pseudo_legal_moves; 
  }
  // Call bitboard-based move generation functions for each piece type
  get_pawn_moves_bb(player, pseudo_legal_moves);
  get_knight_moves_bb(player, pseudo_legal_moves);
  get_bishop_moves_bb(player, pseudo_legal_moves); // Uses on-the-fly ray casting
  get_rook_moves_bb(player, pseudo_legal_moves);   // Uses on-the-fly ray casting
  get_king_moves_bb(player, pseudo_legal_moves);
  return pseudo_legal_moves;
}

// --- Bitboard-Based Move Generation Helpers ---

// Generates pawn moves using bitboards and lookup tables
void Board::get_pawn_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard pawns = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::PAWN)];
    Bitboard my_pieces = player_bitboards_[p_idx];
    Bitboard opp_pieces = occupied_bitboard_ & ~my_pieces; // Squares occupied by any opponent
    Bitboard empty_sqs = ~occupied_bitboard_;             // All empty squares

    // Pointers to the correct lookup tables for the current player
    const Bitboard* current_fwd_moves_table = nullptr; 
    const std::array<Bitboard, NUM_SQUARES_BB>* current_atk_table_for_player = nullptr;

    int promotion_target_coord = -1; // Row or Col index for promotion
    bool check_row_for_promo = false;  // True if promotion depends on row, false if on col

    // Select appropriate lookup tables and promotion criteria based on player
    switch (player) {
        case Player::RED:    current_fwd_moves_table = &pawn_fwd_moves_red_[0];    current_atk_table_for_player = &pawn_attacks_red_[p_idx];    promotion_target_coord = PROMOTION_ROW_RED_BB;   check_row_for_promo = true; break;
        case Player::BLUE:   current_fwd_moves_table = &pawn_fwd_moves_blue_[0];   current_atk_table_for_player = &pawn_attacks_blue_[p_idx];   promotion_target_coord = PROMOTION_COL_BLUE_BB;  check_row_for_promo = false; break;
        case Player::YELLOW: current_fwd_moves_table = &pawn_fwd_moves_yellow_[0]; current_atk_table_for_player = &pawn_attacks_yellow_[p_idx]; promotion_target_coord = PROMOTION_ROW_YELLOW_BB; check_row_for_promo = true; break;
        case Player::GREEN:  current_fwd_moves_table = &pawn_fwd_moves_green_[0];  current_atk_table_for_player = &pawn_attacks_green_[p_idx];  promotion_target_coord = PROMOTION_COL_GREEN_BB; check_row_for_promo = false; break;
    }
    if (!current_fwd_moves_table || !current_atk_table_for_player) return; // Should not happen
    
    Bitboard temp_pawns = pawns; // Iterate over current player's pawns
    while (temp_pawns) {
        int from_sq = pop_lsb(temp_pawns); // Get and remove one pawn's square
        BoardLocation from_loc = from_sq_idx(from_sq);

        // 1. Forward moves (non-capture)
        Bitboard fwd_moves = current_fwd_moves_table[from_sq] & empty_sqs;
        if (fwd_moves) { // Pawns can only move one step forward (no double push in Chaturaji)
            int to_sq = get_lsb_index(fwd_moves); // Should be only one bit set if any
            BoardLocation to_loc = from_sq_idx(to_sq);
            bool is_promotion = (check_row_for_promo && to_loc.row == promotion_target_coord) ||
                                (!check_row_for_promo && to_loc.col == promotion_target_coord);
            if (is_promotion) {
                moves.emplace_back(from_loc, to_loc, PieceType::ROOK); // Default promotion to Rook
                // Add other promotions if rules allow (e.g. Knight, Bishop)
            } else {
                moves.emplace_back(from_loc, to_loc);
            }
        }

        // 2. Capture moves
        Bitboard cap_moves = (*current_atk_table_for_player)[from_sq] & opp_pieces;
        Bitboard temp_cap_moves = cap_moves; // Iterate over possible capture squares
        while (temp_cap_moves) {
            int to_sq = pop_lsb(temp_cap_moves);
            BoardLocation to_loc = from_sq_idx(to_sq);
            bool is_promotion = (check_row_for_promo && to_loc.row == promotion_target_coord) ||
                                (!check_row_for_promo && to_loc.col == promotion_target_coord);
            if (is_promotion) {
                moves.emplace_back(from_loc, to_loc, PieceType::ROOK); // Default promotion to Rook
            } else {
                moves.emplace_back(from_loc, to_loc);
            }
        }
    }
}

// Generates knight moves using bitboards and lookup table
void Board::get_knight_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard knights = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KNIGHT)];
    Bitboard not_my_pieces = ~player_bitboards_[p_idx]; // Can move to empty or opponent's square
    
    Bitboard temp_knights = knights; // Iterate over current player's knights
    while (temp_knights) {
        int from_sq = pop_lsb(temp_knights);
        BoardLocation from_loc = from_sq_idx(from_sq);
        Bitboard possible_moves = knight_attacks_[from_sq] & not_my_pieces;
        
        Bitboard temp_possible_moves = possible_moves; // Iterate over target squares
        while (temp_possible_moves) {
            int to_sq = pop_lsb(temp_possible_moves);
            moves.emplace_back(from_loc, from_sq_idx(to_sq));
        }
    }
}

// Generates king moves using bitboards and lookup table
void Board::get_king_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard kings = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
    Bitboard not_my_pieces = ~player_bitboards_[p_idx]; // Can move to empty or opponent's square

    if (kings == 0) return; // No king for this player (should not happen if active)
    
    int from_sq = get_lsb_index(kings); // Assuming only one king per player
    BoardLocation from_loc = from_sq_idx(from_sq);
    Bitboard possible_moves = king_attacks_[from_sq] & not_my_pieces;

    Bitboard temp_possible_moves = possible_moves; // Iterate over target squares
    while (temp_possible_moves) {
        int to_sq = pop_lsb(temp_possible_moves);
        moves.emplace_back(from_loc, from_sq_idx(to_sq));
    }
}

// Helper for generating sliding piece moves (Rook, Bishop) using on-the-fly ray casting.
// Note: This does NOT use the precomputed rook_rays_/bishop_rays_ lookup tables.
void Board::generate_sliding_moves(Player p, int from_sq, PieceType pt, const std::vector<std::pair<int,int>>& directions, std::vector<Move>& moves) const {
    BoardLocation from_loc = from_sq_idx(from_sq);
    Bitboard my_pieces = player_bitboards_[static_cast<int>(p)];
    Bitboard opp_pieces = occupied_bitboard_ & ~my_pieces; // Squares occupied by any opponent

    for (const auto& dir_pair : directions) { // For each direction (e.g., North, NE)
        int dr = dir_pair.first; int dc = dir_pair.second;
        int r = from_loc.row + dr; int c = from_loc.col + dc;
        while (is_valid_square(r,c)) { // While on the board
            int to_sq = to_sq_idx(r,c);
            BoardLocation to_loc(r,c);
            if (get_bit(my_pieces, to_sq)) break; // Blocked by own piece
            if (get_bit(opp_pieces, to_sq)) {      // Can capture opponent piece
                moves.emplace_back(from_loc, to_loc); 
                break; // Blocked by opponent piece (after capture)
            }
            moves.emplace_back(from_loc, to_loc); // Empty square, can move
            r += dr; c += dc; // Continue along the ray
        }
    }
}

// Generates rook moves using the generate_sliding_moves helper.
void Board::get_rook_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard rooks = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::ROOK)];
    const std::vector<std::pair<int, int>> DIRS = {{-1,0}, {0,1}, {1,0}, {0,-1}}; // N, E, S, W
    
    Bitboard temp_rooks = rooks; // Iterate over current player's rooks
    while(temp_rooks) {
        int from_sq = pop_lsb(temp_rooks);
        generate_sliding_moves(player, from_sq, PieceType::ROOK, DIRS, moves);
    }
}

// Generates bishop moves using the generate_sliding_moves helper.
void Board::get_bishop_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard bishops = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::BISHOP)];
    const std::vector<std::pair<int, int>> DIRS = {{-1,1}, {1,1}, {1,-1}, {-1,-1}}; // NE, SE, SW, NW
    
    Bitboard temp_bishops = bishops; // Iterate over current player's bishops
    while(temp_bishops) {
        int from_sq = pop_lsb(temp_bishops);
        generate_sliding_moves(player, from_sq, PieceType::BISHOP, DIRS, moves);
    }
}


// --- Move Execution ---
std::optional<Piece> Board::make_move(const Move &move) {
  // --- Setup Undo Information ---
  UndoInfo undo_info;
  // Store original bitboard states for undo
  undo_info.original_piece_bitboards = piece_bitboards_;
  undo_info.original_player_bitboards = player_bitboards_;
  undo_info.original_occupied_bitboard = occupied_bitboard_;
  // Store other game state for undo
  undo_info.move = move;
  undo_info.original_player = current_player_;
  undo_info.original_full_move_number = full_move_number_;
  undo_info.original_move_number_of_last_reset = move_number_of_last_reset_;
  undo_info.eliminated_player = std::nullopt;
  undo_info.was_history_cleared = false;
  undo_info.previous_hash = current_hash_; // Store hash BEFORE any changes

  const auto& zobrist_data = get_zobrist_data();
  int fr = move.from_loc.row, fc = move.from_loc.col;
  int tr = move.to_loc.row, tc = move.to_loc.col;
  int from_sq_idx = Board::to_sq_idx(fr, fc);
  int to_sq_idx = Board::to_sq_idx(tr, tc);
  int moving_player_idx = static_cast<int>(current_player_);

  // --- Validate Moving Piece (from bitboards) ---
  std::optional<Piece> moving_piece_opt = get_piece_at_sq(from_sq_idx); // Get piece info from bitboards
  if (!moving_piece_opt) { 
    throw std::runtime_error("Attempting to move from an empty square in make_move (checked via bitboards). From sq: " + std::to_string(from_sq_idx));
  }
  if (moving_piece_opt->player != current_player_) { // Check if the piece belongs to the current player
      throw std::runtime_error("Attempting to move opponent's piece. Mover: " + 
                               std::to_string(static_cast<int>(current_player_)) + 
                               ", Piece Owner: " + std::to_string(static_cast<int>(moving_piece_opt->player)));
  }
  Piece moving_piece_obj = *moving_piece_opt; // Get copy of moving piece
  undo_info.original_moving_piece_type = moving_piece_obj.piece_type;
  int moving_pt_bb_idx = piece_type_to_bb_idx(moving_piece_obj.piece_type);

  // --- Store Captured Piece Info (from bitboards) ---
  undo_info.captured_piece = get_piece_at_sq(to_sq_idx); // Get piece info from bitboards
  bool is_capture = undo_info.captured_piece.has_value();
  bool is_pawn_move = (moving_piece_obj.piece_type == PieceType::PAWN);
  bool is_resetting_move = is_pawn_move || is_capture; // Move that resets 50-move counter and history

  // --- ZOBRIST UPDATE & BITBOARD UPDATE: Part 1 (Remove pieces from old state) ---
  // 1a. XOR out moving piece from Zobrist hash & clear from bitboards
  current_hash_ ^= zobrist_data.get_piece_key(moving_piece_obj.piece_type, moving_piece_obj.player, from_sq_idx);
  clear_bit(piece_bitboards_[moving_player_idx][moving_pt_bb_idx], from_sq_idx);
  clear_bit(player_bitboards_[moving_player_idx], from_sq_idx);
  clear_bit(occupied_bitboard_, from_sq_idx); // Clear from general occupied too
  
  // 1b. If capture, XOR out captured piece from Zobrist & clear from bitboards
  if (is_capture) {
      const Piece& captured = undo_info.captured_piece.value();
      current_hash_ ^= zobrist_data.get_piece_key(captured.piece_type, captured.player, to_sq_idx);
      int captured_player_idx = static_cast<int>(captured.player);
      int captured_pt_bb_idx = piece_type_to_bb_idx(captured.piece_type);
      // Clear captured piece from its bitboards
      clear_bit(piece_bitboards_[captured_player_idx][captured_pt_bb_idx], to_sq_idx);
      clear_bit(player_bitboards_[captured_player_idx], to_sq_idx);
      // Note: occupied_bitboard_ at to_sq_idx will be set again by the moving piece later.
      // If it was not cleared here, it would remain set if the target square was occupied by the captured piece.
      // The moving piece will set it again anyway.
  }
  // REMOVED: Board Array Changes (e.g. board_[fr][fc] = std::nullopt;) (Mailbox array removal)
  // 1c. Zobrist turn key XORing is handled by advance_turn().

  // Handle Promotion (update piece type before placing)
  PieceType final_piece_type = moving_piece_obj.piece_type;
  if (move.promotion_piece_type) {
    final_piece_type = move.promotion_piece_type.value();
  }
  // REMOVED: Board Array Changes (e.g. board_[tr][tc] = Piece(...)) (Mailbox array removal)
  
  // --- ZOBRIST UPDATE & BITBOARD UPDATE: Part 2 (Add piece to new state) ---
  // 2a. XOR in final piece at destination & set in bitboards
  int final_pt_bb_idx = piece_type_to_bb_idx(final_piece_type);
  set_bit(piece_bitboards_[moving_player_idx][final_pt_bb_idx], to_sq_idx);
  set_bit(player_bitboards_[moving_player_idx], to_sq_idx);
  set_bit(occupied_bitboard_, to_sq_idx); // Set destination as occupied
  current_hash_ ^= zobrist_data.get_piece_key(final_piece_type, moving_piece_obj.player, to_sq_idx);
  
  // --- Handle Captures, Points & Elimination ---
  if (is_capture) {
    const Piece &captured = undo_info.captured_piece.value();
    player_points_[moving_piece_obj.player] += get_piece_capture_value(captured);
    // If a King was captured, eliminate the player
    if (captured.piece_type == PieceType::KING) {
        eliminate_player(captured.player); // Handles Zobrist for active status & bitboard clearing
        undo_info.eliminated_player = captured.player;
    }
  }
  // --- Update Game State Counters & History ---
  Player player_who_moved = current_player_;
  Player last_active_player_in_sequence = get_last_active_player();
  bool was_last_player_turn = (player_who_moved == last_active_player_in_sequence);

  if (was_last_player_turn) full_move_number_++; // Increment full move if last player in round moved

  if (is_resetting_move) { // Pawn move or capture
    move_number_of_last_reset_ = full_move_number_; // Reset 50-move counter base
    position_history_.clear();                      // Reset repetition history
    undo_info.was_history_cleared = true;
  } else {
    undo_info.was_history_cleared = false; 
  }

  // --- Final Steps ---
  undo_stack_.push_back(undo_info); // Push undo info (includes previous hash and bitboards)
  advance_turn();                   // Advances turn and updates Zobrist hash for player change
  
  // Push the hash of the *resulting* state (including the next player's turn) for repetition checks
  position_history_.push_back(get_position_key()); 
  
  is_game_over(); // Call to update termination_reason_ if game ended
  return undo_info.captured_piece; // Return the captured piece, if any
}

// --- Lightweight Move Execution for MCTS ---
// Does NOT create UndoInfo, push to undo_stack_, or modify position_history_ directly related to the move.
std::optional<Piece> Board::make_move_for_mcts(const Move &move) {
  const auto& zobrist_data = get_zobrist_data();
  int fr = move.from_loc.row, fc = move.from_loc.col;
  int tr = move.to_loc.row, tc = move.to_loc.col;
  int from_sq_idx = Board::to_sq_idx(fr, fc);
  int to_sq_idx = Board::to_sq_idx(tr, tc);
  int moving_player_idx = static_cast<int>(current_player_);

  // --- Validate Moving Piece (from bitboards) ---
  std::optional<Piece> moving_piece_opt = get_piece_at_sq(from_sq_idx);
  if (!moving_piece_opt) {
    throw std::runtime_error("MCTS: Attempting to move from an empty square. From sq: " + std::to_string(from_sq_idx));
  }
  if (moving_piece_opt->player != current_player_) {
      throw std::runtime_error("MCTS: Attempting to move opponent's piece. Mover: " +
                               std::to_string(static_cast<int>(current_player_)) +
                               ", Piece Owner: " + std::to_string(static_cast<int>(moving_piece_opt->player)));
  }
  Piece moving_piece_obj = *moving_piece_opt;
  int moving_pt_bb_idx = piece_type_to_bb_idx(moving_piece_obj.piece_type);

  // --- Determine Captured Piece (from bitboards) ---
  std::optional<Piece> captured_piece_opt = get_piece_at_sq(to_sq_idx); // Renamed to avoid conflict
  bool is_capture = captured_piece_opt.has_value();
  bool is_pawn_move = (moving_piece_obj.piece_type == PieceType::PAWN);
  bool is_resetting_move = is_pawn_move || is_capture;

  // --- ZOBRIST UPDATE & BITBOARD UPDATE: Part 1 (Remove pieces from old state) ---
  // 1a. XOR out moving piece from Zobrist hash & clear from bitboards
  current_hash_ ^= zobrist_data.get_piece_key(moving_piece_obj.piece_type, moving_piece_obj.player, from_sq_idx);
  clear_bit(piece_bitboards_[moving_player_idx][moving_pt_bb_idx], from_sq_idx);
  clear_bit(player_bitboards_[moving_player_idx], from_sq_idx);
  clear_bit(occupied_bitboard_, from_sq_idx);

  // 1b. If capture, XOR out captured piece from Zobrist & clear from bitboards
  if (is_capture) {
      const Piece& captured = captured_piece_opt.value();
      current_hash_ ^= zobrist_data.get_piece_key(captured.piece_type, captured.player, to_sq_idx);
      int captured_player_idx = static_cast<int>(captured.player);
      int captured_pt_bb_idx = piece_type_to_bb_idx(captured.piece_type);
      clear_bit(piece_bitboards_[captured_player_idx][captured_pt_bb_idx], to_sq_idx);
      clear_bit(player_bitboards_[captured_player_idx], to_sq_idx);
  }

  // Handle Promotion (update piece type before placing)
  PieceType final_piece_type = moving_piece_obj.piece_type;
  if (move.promotion_piece_type) {
    final_piece_type = move.promotion_piece_type.value();
  }

  // --- ZOBRIST UPDATE & BITBOARD UPDATE: Part 2 (Add piece to new state) ---
  int final_pt_bb_idx = piece_type_to_bb_idx(final_piece_type);
  set_bit(piece_bitboards_[moving_player_idx][final_pt_bb_idx], to_sq_idx);
  set_bit(player_bitboards_[moving_player_idx], to_sq_idx);
  set_bit(occupied_bitboard_, to_sq_idx);
  current_hash_ ^= zobrist_data.get_piece_key(final_piece_type, moving_piece_obj.player, to_sq_idx);

  // --- Handle Captures, Points & Elimination ---
  if (is_capture) {
    const Piece &captured_val = captured_piece_opt.value();
    player_points_[moving_piece_obj.player] += get_piece_capture_value(captured_val);
    if (captured_val.piece_type == PieceType::KING) {
        eliminate_player(captured_val.player); // Handles Zobrist for active status & bitboard clearing
    }
  }

  // --- Update Game State Counters ---
  // position_history_ and undo_stack_ are NOT updated here.
  Player player_who_moved = current_player_; // current_player_ before advance_turn()
  Player last_active_player_in_sequence = get_last_active_player();
  bool was_last_player_turn = (player_who_moved == last_active_player_in_sequence);

  if (was_last_player_turn) full_move_number_++;

  if (is_resetting_move) {
    move_number_of_last_reset_ = full_move_number_;
    // DO NOT clear position_history_ or set undo_info.was_history_cleared.
  }

  // --- Final Steps ---
  // NO undo_stack_.push_back()
  advance_turn(); // Advances turn and updates Zobrist hash for player change
  // NO position_history_.push_back()

  is_game_over(); // Call to update termination_reason_ if game ended
  return captured_piece_opt; // Return the captured piece, if any
}

// --- Undo Last Move ---
void Board::undo_move() {
  if (undo_stack_.empty()) {
    throw std::runtime_error("No previous state available to undo.");
  }
  // Pop Undo Information
  UndoInfo undo_info = undo_stack_.back();
  undo_stack_.pop_back();

  // 1. Restore Bitboard State
  piece_bitboards_ = undo_info.original_piece_bitboards;
  player_bitboards_ = undo_info.original_player_bitboards;
  occupied_bitboard_ = undo_info.original_occupied_bitboard;

  // 2. Restore Zobrist Hash (to state *before* the undone move)
  current_hash_ = undo_info.previous_hash;

  // 3. Restore Player Turn, Game State Counters
  current_player_ = undo_info.original_player;
  full_move_number_ = undo_info.original_full_move_number;
  move_number_of_last_reset_ = undo_info.original_move_number_of_last_reset;

  // 4. Restore Position History
  // A resignation undo doesn't pop history, as no new state was pushed for it.
  // A regular move undo needs to pop the state that was just made.
  bool is_resignation_undo = (undo_info.move.from_loc.row == -1); // Heuristic: resign uses invalid from_loc
  if (!is_resignation_undo) {
    if (!position_history_.empty()) { // Pop the hash of the state we are undoing from
        position_history_.pop_back();
    }
    // If history was cleared by the move being undone, it needs to be restored.
    // This is complex as the actual history items are gone. For now, this simplistic
    // pop is what the code does. True restoration of cleared history would need more stored in UndoInfo.
    // The current implementation relies on previous_hash.
  }
  
  // REMOVED: Reverse Board Array Piece Changes (Mailbox array removal)
  // The bitboard restoration in step 1 handles piece positions and types.

  // 6. Reverse Elimination (Restore Player and their Zobrist active status key)
  // Bitboards for the revived player were already restored in step 1.
  // Zobrist active status was XORed out by eliminate_player, and previous_hash restored it.
  // We just need to add them back to the active_players_ set.
  if (undo_info.eliminated_player) {
    Player player_to_revive = *undo_info.eliminated_player;
    active_players_.insert(player_to_revive); 
    // Zobrist hash for active status is implicitly restored by current_hash_ = undo_info.previous_hash;
  }

  // 7. Reverse Point Changes (Only for regular moves with captures)
  if (!is_resignation_undo && undo_info.captured_piece) {
    const Piece &captured = undo_info.captured_piece.value();
    // Subtract points from the player who made the original move
    player_points_[undo_info.original_player] -= get_piece_capture_value(captured);
  }

  // 8. Clear Termination Reason
  termination_reason_ = std::nullopt; // State may no longer be terminal
}

// --- Player Elimination ---
void Board::eliminate_player(Player player) {
  if (active_players_.count(player)) {
    const auto& zobrist_data = get_zobrist_data();
    // Zobrist: XOR out the active status key for the player being eliminated
    current_hash_ ^= zobrist_data.get_active_player_status_key(player);
    active_players_.erase(player); // Remove from active set
    
    // Note: Zobrist keys for the actual pieces of the eliminated player are NOT XORed out here.
    // The pieces are still on the board. Only their "active status" Zobrist key is removed.
  }
}


// --- Bitboard Accessors ---
Bitboard Board::get_occupied_bitboard() const { return occupied_bitboard_; }
Bitboard Board::get_player_bitboard(Player p) const { return player_bitboards_[static_cast<int>(p)]; }
Bitboard Board::get_piece_bitboard(Player p, PieceType pt) const {
    return piece_bitboards_[static_cast<int>(p)][piece_type_to_bb_idx(pt)];
}

// Utility to print a bitboard for debugging
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

// --- Game State Accessors ---
// REMOVED: const Board::BoardGrid &Board::get_board_grid() const { return board_; } (Mailbox array removal)
const Board::ActivePlayerSet &Board::get_active_players() const { return active_players_; }
const Board::PlayerPointMap &Board::get_player_points() const { return player_points_; }
Player Board::get_current_player() const { return current_player_; }
int Board::get_full_move_number() const { return full_move_number_; }
int Board::get_move_number_of_last_reset() const { return move_number_of_last_reset_; }
const std::optional<std::string> &Board::get_termination_reason() const { return termination_reason_; }
const Board::PositionHistory &Board::get_position_history() const { return position_history_; }

// Helper to find the last player in sequence (numerically highest enum value) among active players
Player Board::get_last_active_player() const {
  if (active_players_.empty()) return Player::RED; // Default or throw if game should not proceed
  Player last_player = Player::RED; 
  int max_val = -1;
  for (Player p : active_players_) {
    if (static_cast<int>(p) > max_val) {
      max_val = static_cast<int>(p);
      last_player = p;
    }
  }
  return last_player;
}

// --- Game Status ---
// Checks if the game is over and sets termination_reason_ if so.
bool Board::is_game_over() const {
  if (termination_reason_) return true; // Already determined

  // 1. Elimination: If 1 or 0 players are active
  if (active_players_.size() <= 1) { 
    termination_reason_ = "elimination"; 
    return true; 
  }

  // 2. Fifty-Move Rule: 50 full moves since last capture or pawn move
  int moves_since_last_reset = full_move_number_ - move_number_of_last_reset_;
  // The rule implies 50 moves by *each* player if all 4 are active,
  // but here it's 50 *full_move_number* increments.
  // A full move completes when the last player in sequence moves.
  if (moves_since_last_reset >= 50) { // 50 full moves (e.g. R->B->Y->G is one full move)
    if (!undo_stack_.empty()) { // Need to know who just moved to apply rule correctly
      Player player_who_just_moved = undo_stack_.back().original_player;
      // Rule triggers if the 50th non-resetting move was by the last player in the turn sequence
      if (player_who_just_moved == get_last_active_player()) {
        termination_reason_ = "fifty_move_rule"; 
        return true;
      }
    }
  }

  // 3. Threefold Repetition
  // Current hash is already in position_history_ (added at end of make_move)
  // So, 3 occurrences mean the position has repeated 3 times.
  int count = 0;
  for (const auto &key : position_history_) if (key == current_hash_) count++;
  if (count >= 3) { 
    termination_reason_ = "threefold_repetition"; 
    return true; 
  }
  // No game-ending condition met
  return false;
}

// Calculates final game scores based on termination reason and captured points.
Board::PlayerPointMap Board::get_game_result() const {
  PlayerPointMap results = player_points_; // Start with points from captures during the game
  int num_kings_of_inactive_players = 0;

  // Count kings of INACTIVE players still on the board (via bitboards)
  // These kings provide bonus points to active players in draw scenarios,
  // or to the winner in an elimination scenario.
  for (int p_idx_loop = 0; p_idx_loop < NUM_PLAYERS_BB; ++p_idx_loop) {
      Player p_enum = static_cast<Player>(p_idx_loop);
      if (!active_players_.count(p_enum)) { // If player is inactive
          // Check their king bitboard for a remaining king
          if (piece_bitboards_[p_idx_loop][Board::piece_type_to_bb_idx(PieceType::KING)] != 0ULL) {
              num_kings_of_inactive_players++;
          }
      }
  }

  int num_active_players = active_players_.size();

  // Apply Bonuses based on Termination Reason
  if (termination_reason_) { 
    const std::string &reason = *termination_reason_;
    // Draw Scenarios (50-move or 3-fold repetition)
    if (reason == "fifty_move_rule" || reason == "threefold_repetition") {
      if (num_active_players > 0) { 
        // Bonus for kings of inactive players, distributed among active players
        int dead_king_bonus_per_player = (num_kings_of_inactive_players > 0) ? 
            static_cast<int>(std::ceil(3.0 * num_kings_of_inactive_players / num_active_players)) : 0;
        for (Player p : active_players_) { 
            results[p] += (2 + dead_king_bonus_per_player); // Base +2 for draw, plus share of dead kings
        }
      }
    } 
    // Elimination Scenario (Last Man Standing)
    else if (reason == "elimination") {
      // Winner (if one exists) gets points for kings of all inactive players
      if (num_active_players == 1 && num_kings_of_inactive_players > 0) {
        results[*active_players_.begin()] += (3 * num_kings_of_inactive_players);
      }
      // No base +2 bonus for elimination winner, only dead king points.
    }
  }
  return results;
}

// Determines the winner of the game, if any.
std::optional<Player> Board::get_winner() const {
  if (!termination_reason_) return std::nullopt; // Game not over or reason not set

  PlayerPointMap final_scores = get_game_result();
  // Find player with the maximum score
  auto winner_it = std::max_element(final_scores.begin(), final_scores.end(),
                       [](const auto &a, const auto &b) { return a.second < b.second; });
  
  if (winner_it == final_scores.end()) return std::nullopt; // Should not happen if players exist

  // Could add tie-breaking rules here if needed. Currently returns first player with max score.
  return (winner_it == final_scores.end()) ? std::nullopt : std::optional<Player>(winner_it->first);
}

// --- Piece Values (for evaluation and capture points) ---
// Base material value of a piece (used in evaluation)
int Board::get_piece_value(const Piece& piece) const {
  // This function only depends on piece.piece_type, so it's fine.
  switch (piece.piece_type) {
  case PieceType::PAWN: return 1; case PieceType::KNIGHT: return 3;
  case PieceType::BISHOP: return 5; case PieceType::ROOK: return 5;
  case PieceType::KING: return 3; default: return 0; // Should not happen
  }
}
// Points awarded for capturing a piece
int Board::get_piece_capture_value(const Piece& piece) const {
    // If captured piece's owner is inactive:
    if (!active_players_.count(piece.player)) {
        return (piece.piece_type == PieceType::KING) ? 3 : 0; // King of inactive player is worth 3, others 0
    }
    // If captured piece's owner is active, use standard values:
    switch (piece.piece_type) {
        case PieceType::PAWN: return 1; case PieceType::KNIGHT: return 3;
        case PieceType::BISHOP: return 5; case PieceType::ROOK: return 5;
        case PieceType::KING: return 3; // Capturing an active King (leads to elimination)
        default: return 0;
    }
}

// --- Board Evaluation Function (Rewritten for Bitboards) ---
// Calculates a score for each player based on material, position, safety, etc.
// Uses get_piece_at_sq to retrieve piece information from bitboards.
Board::PlayerPointMap Board::evaluate() const {
  PlayerPointMap scores; // Note: PlayerPointMap is std::map<Player, int> in board.h
                         // but this function uses double for intermediate scores.
  for (int i = 0; i < 4; ++i) scores[static_cast<Player>(i)] = 0.0; // Use double for intermediate scores

  std::map<Player, BoardLocation> king_coords; // To store king locations for safety checks
  std::map<Player, bool> king_present;         // To check if active players have their king
  for (int i = 0; i < 4; ++i) king_present[static_cast<Player>(i)] = false;

  // Pre-populate king locations and presence from bitboards
  for (int p_idx = 0; p_idx < NUM_PLAYERS_BB; ++p_idx) {
      Player p_enum = static_cast<Player>(p_idx);
      Bitboard king_bb = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
      if (king_bb != 0ULL) {
          int king_sq = get_lsb_index(king_bb); // Assumes one king per player
          king_coords[p_enum] = from_sq_idx(king_sq);
          king_present[p_enum] = true;
      } else {
          king_present[p_enum] = false; // Explicitly mark king as not present
      }
  }
  
  // Iterate over all squares using square indices
  for (int sq_idx = 0; sq_idx < NUM_SQUARES_BB; ++sq_idx) {
      BoardLocation loc = from_sq_idx(sq_idx);
      int r = loc.row;
      int c = loc.col;
      std::optional<Piece> piece_opt = get_piece_at_sq(sq_idx); // Get piece from bitboards

      if (piece_opt) { // If a piece is on this square
        const Piece &piece = *piece_opt;
        Player player = piece.player;

        // Only evaluate pieces of active players for material/positional value
        if (active_players_.count(player)) {
          // Base material score
          scores[player] += get_piece_value(piece);

          // Penalties/Bonuses for piece development and position
          if (piece.piece_type == PieceType::KNIGHT || piece.piece_type == PieceType::BISHOP) {
            // Penalty for undeveloped minor pieces on starting row/col
            if (((player == Player::RED && r == 7) || (player == Player::YELLOW && r == 0) ||
                 (player == Player::GREEN && c == 7) || (player == Player::BLUE && c == 0))) {
              scores[player] -= 0.4; 
            }
          }
          // King-specific evaluation
          if (piece.piece_type == PieceType::KING) {
            // king_present and king_coords already populated from the pre-population step
            // King safety: check adjacent squares
            for (const auto &dir : KING_DIRS_EVAL) { // Use _EVAL version of dirs
              int nr = r + dir.first; int nc = c + dir.second;
              if (is_valid_square(nr, nc)) { // Check if adjacent square is on board
                std::optional<Piece> adjacent_opt = get_piece_at_sq(to_sq_idx(nr, nc));
                if (adjacent_opt) { // If adjacent square is occupied
                  if (adjacent_opt->player == player) { // Friendly piece nearby
                    scores[player] += (adjacent_opt->piece_type == PieceType::PAWN ? 0.2 : 0.05); // Pawn shelter > other piece
                  } else { // Opponent piece nearby
                    if (!active_players_.count(adjacent_opt->player)) { // Piece of inactive player (less threat/potential shelter)
                        scores[player] += 0.15;        
                    } else { // Active opponent (danger)
                        scores[player] -= 0.15;        
                    }
                  }
                }
              }
            }
          } 
          // Pawn-specific evaluation
          if (piece.piece_type == PieceType::PAWN) {
            int dr = 0, dc = 0; // Pawn forward move direction
            int cap_r1 = 0, cap_c1 = 0, cap_r2 = 0, cap_c2 = 0; // Pawn capture directions
            // Determine pawn direction based on player
            switch (player) {
            case Player::RED:    scores[player] += 0.2 * (6 - r); dr = -1; dc = 0; cap_r1 = -1; cap_c1 = -1; cap_r2 = -1; cap_c2 = 1; break;
            case Player::BLUE:   scores[player] += 0.2 * (c - 1); dr = 0; dc = 1; cap_r1 = -1; cap_c1 = 1; cap_r2 = 1; cap_c2 = 1; break;
            case Player::YELLOW: scores[player] += 0.2 * (r - 1); dr = 1; dc = 0; cap_r1 = 1; cap_c1 = -1; cap_r2 = 1; cap_c2 = 1; break;
            case Player::GREEN:  scores[player] += 0.2 * (6 - c); dr = 0; dc = -1; cap_r1 = -1; cap_c1 = -1; cap_r2 = 1; cap_c2 = -1; break;
            }
            // Penalty for blocked pawn (if any piece is directly in front)
            if (is_valid_square(r + dr, c + dc) && get_piece_at_sq(to_sq_idx(r + dr, c + dc))) scores[player] -= 0.2;
            
            // Check pawn attacks/support
            for (const auto &cap_delta : {std::make_pair(cap_r1, cap_c1), std::make_pair(cap_r2, cap_c2)}) {
              int cap_r = r + cap_delta.first; int cap_c = c + cap_delta.second;
              if (is_valid_square(cap_r, cap_c)) {
                  std::optional<Piece> target_opt = get_piece_at_sq(to_sq_idx(cap_r, cap_c));
                  if (target_opt) { // If capture square is occupied
                    const auto &target = *target_opt;
                    if (target.player == player) { // Supporting a friendly piece
                      if (target.piece_type == PieceType::BISHOP || target.piece_type == PieceType::KNIGHT) {
                        scores[player] += 0.2; // Bonus for pawns supporting minor pieces (outpost-like)
                      }
                    } else { // Attacking an enemy piece
                      scores[player] += 0.2; // General bonus for attacking
                      if (target.piece_type == PieceType::KING && active_players_.count(target.player)) { // Attacking an active king
                        scores[player] += 0.1; // Extra bonus for attacking king
                        scores[target.player] -= 0.5; // Penalty for king being attacked by pawn
                      }
                    }
                  }
              }
            }
          } // End Pawn specific
        } // End if piece's player is active
      } // End if piece_opt has value (square is occupied)
    } // End sq_idx loop
  

  // Final adjustments to scores
  for (int i = 0; i < 4; ++i) {
    Player p = static_cast<Player>(i);
    // Massive penalty if an active player has no king (should imply imminent loss)
    if (active_players_.count(p) && !king_present[p]) scores[p] = -999.0; 
    scores[p] += player_points_.at(p); // Add points from captures during the game
    scores[p] -= 20;                   // Base score adjustment (from Python version)
    // Optional: Rounding scores[p] = std::round(scores[p] * 100.0) / 100.0;
  }
  return scores;
}

// --- Player Actions ---
// Handles player resignation.
void Board::resign() {
  Player resigning_player = current_player_; 
  if (active_players_.count(resigning_player)) {
    // --- Create Undo Info for Resignation ---
    UndoInfo resign_undo_info;
    // Store bitboard state for undo
    resign_undo_info.original_piece_bitboards = piece_bitboards_;
    resign_undo_info.original_player_bitboards = player_bitboards_;
    resign_undo_info.original_occupied_bitboard = occupied_bitboard_;
    // Store other game state
    resign_undo_info.original_player = resigning_player; 
    resign_undo_info.original_full_move_number = full_move_number_;
    resign_undo_info.original_move_number_of_last_reset = move_number_of_last_reset_;
    resign_undo_info.previous_hash = current_hash_;
    resign_undo_info.eliminated_player = resigning_player; // Mark the player who resigned
    resign_undo_info.was_history_cleared = false;          // Resign doesn't clear history
    resign_undo_info.move.from_loc = {-1,-1}; // Sentinel for resignation move in undo
    resign_undo_info.captured_piece = std::nullopt; 

    // --- Perform the elimination ---
    eliminate_player(resigning_player); // Handles Zobrist, active_players set, and player's bitboards

    // --- Advance turn or end game ---
    if (active_players_.size() <= 1) { // If resignation ends the game (0 or 1 active players left)
        const auto& zobrist_data = get_zobrist_data();
        // XOR out the resigning player's turn key from hash as no new player takes turn
        current_hash_ ^= zobrist_data.get_turn_key(resigning_player); 
        is_game_over(); // Sets termination_reason_ to "elimination"
    } else {
        advance_turn(); // Game continues, advance to next active player (handles Zobrist turn key)
    }
    undo_stack_.push_back(resign_undo_info); // Push resignation undo info
  }
}

// Advances to the next active player and updates Zobrist hash for turn.
void Board::advance_turn() {
  const auto& zobrist_data = get_zobrist_data(); 
  Player old_player = current_player_;

  // Cycle to the next player (RED->BLUE->YELLOW->GREEN->RED)
  current_player_ = static_cast<Player>((static_cast<int>(current_player_) + 1) % 4);
  
  // Skip eliminated players
  while (active_players_.find(current_player_) == active_players_.end()) {
    if (active_players_.size() <= 1) break; // Avoid infinite loop if game is ending/ended
    current_player_ = static_cast<Player>((static_cast<int>(current_player_) + 1) % 4);
  }

  // Update Zobrist hash for turn change
  if (!active_players_.empty()) { // Only update turn hash if game isn't completely empty
      current_hash_ ^= zobrist_data.get_turn_key(old_player); // XOR out old player's turn
      // XOR in new player's turn, but only if they are active and the game isn't over
      // in a way that no one has a turn (e.g. all but one eliminated, old_player was last to move).
      if(active_players_.count(current_player_)){ 
         current_hash_ ^= zobrist_data.get_turn_key(current_player_); 
      }
  }
  // If game ends because the last player was eliminated (e.g. by king capture), old_player's turn key 
  // would have been XORed out during make_move (if king captured) or by resign(). 
  // If advance_turn is called after such an event (e.g., if the game could theoretically continue), 
  // it handles the turn transition correctly.
}

// --- ANSI Color Codes and Unicode Symbols (for print_board) ---
const std::string ANSI_RESET_BB = "\033[0m"; 
const std::string ANSI_RED_BB = "\033[31m"; const std::string ANSI_GREEN_BB = "\033[32m";
const std::string ANSI_YELLOW_BB = "\033[33m"; const std::string ANSI_BLUE_BB = "\033[34m";
const std::string UNICODE_KING_BB = "♔"; const std::string UNICODE_ROOK_BB = "♖";
const std::string UNICODE_BISHOP_BB = "♗"; const std::string UNICODE_KNIGHT_BB = "♘";
const std::string UNICODE_PAWN_BB = "♙";

// --- Utility: Print Board to Console (Rewritten for Bitboards) ---
// Uses get_piece_at_sq (which queries bitboards) for printing.
void Board::print_board() const {
  std::cout << "   a  b  c  d  e  f  g  h" << std::endl; // Column labels
  for (int r = 0; r < BOARD_SIZE; ++r) {
    std::cout << 8 - r << " "; // Row labels (chess style: 8 down to 1)
    for (int c = 0; c < BOARD_SIZE; ++c) {
      int sq_idx = to_sq_idx(r, c);
      std::optional<Piece> piece_opt = get_piece_at_sq(sq_idx); // Use bitboard helper to get piece
      std::string symbol_str = " "; // Default for empty square
      if (piece_opt) {
        const Piece &p = *piece_opt;
        bool display_as_inactive = !active_players_.count(p.player); // Display inactive player pieces without color

        const std::string* base_symbol = nullptr;
        // Select base Unicode symbol for the piece type
        switch (p.piece_type) {
            case PieceType::PAWN:   base_symbol = &UNICODE_PAWN_BB;   break;
            case PieceType::KNIGHT: base_symbol = &UNICODE_KNIGHT_BB; break;
            case PieceType::BISHOP: base_symbol = &UNICODE_BISHOP_BB; break;
            case PieceType::ROOK:   base_symbol = &UNICODE_ROOK_BB;   break;
            case PieceType::KING:   base_symbol = &UNICODE_KING_BB;   break;
        }

        if (base_symbol) { // If a valid piece type
            if (display_as_inactive) { // Inactive players' pieces are shown without color
                symbol_str = *base_symbol;
            } else { // Active players' pieces are colored
                const std::string* color_code = nullptr;
                // Select ANSI color code for the player
                switch (p.player) {
                    case Player::RED:    color_code = &ANSI_RED_BB;    break;
                    case Player::BLUE:   color_code = &ANSI_BLUE_BB;   break;
                    case Player::YELLOW: color_code = &ANSI_YELLOW_BB; break;
                    case Player::GREEN:  color_code = &ANSI_GREEN_BB;  break;
                }
                if (color_code) {
                    symbol_str = *color_code + *base_symbol + ANSI_RESET_BB; // Combine color, symbol, and reset
                } else { // Fallback if color code is somehow not found (should not happen)
                    symbol_str = *base_symbol;
                }
            }
        }
      }
      std::cout << "[" << symbol_str << "]"; // Print piece or empty space in brackets
    }
    std::cout << std::endl; 
  }
  // Print game info below the board
  std::cout << "Turn: ";
  switch (current_player_) {
  case Player::RED:    std::cout << ANSI_RED_BB << "RED" << ANSI_RESET_BB; break;
  case Player::BLUE:   std::cout << ANSI_BLUE_BB << "BLUE" << ANSI_RESET_BB; break;
  case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "YELLOW" << ANSI_RESET_BB; break;
  case Player::GREEN:  std::cout << ANSI_GREEN_BB << "GREEN" << ANSI_RESET_BB; break;
  }
  std::cout << std::endl;
  std::cout << "Active Players: ";
  for(Player active_p : active_players_){
      switch(active_p){
          case Player::RED: std::cout << ANSI_RED_BB << "R " << ANSI_RESET_BB; break;
          case Player::BLUE: std::cout << ANSI_BLUE_BB << "B " << ANSI_RESET_BB; break;
          case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "Y " << ANSI_RESET_BB; break;
          case Player::GREEN: std::cout << ANSI_GREEN_BB << "G " << ANSI_RESET_BB; break;
      }
  }
  std::cout << std::endl;
  std::cout << "Points: ";
  for(const auto& pt_pair : player_points_){ // pt_pair is std::pair<const Player, int>
      switch(pt_pair.first){ 
          case Player::RED: std::cout << ANSI_RED_BB << "R:" << pt_pair.second << ANSI_RESET_BB << " "; break;
          case Player::BLUE: std::cout << ANSI_BLUE_BB << "B:" << pt_pair.second << ANSI_RESET_BB << " "; break;
          case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "Y:" << pt_pair.second << ANSI_RESET_BB << " "; break;
          case Player::GREEN: std::cout << ANSI_GREEN_BB << "G:" << pt_pair.second << ANSI_RESET_BB << " "; break;
      }
  }
  std::cout << std::endl;
  if(termination_reason_) std::cout << "Game Over: " << *termination_reason_ << std::endl;
  // std::cout << "Current Hash: 0x" << std::hex << current_hash_ << std::dec << std::endl; // For debugging Zobrist
  // std::cout << "FullMove: " << full_move_number_ << ", LastReset: " << move_number_of_last_reset_ << std::endl; // For debugging 50-move
}

// --- Zobrist Position Key Accessor ---
Board::PositionKey Board::get_position_key() const { return current_hash_; }

} // namespace chaturaji_cpp