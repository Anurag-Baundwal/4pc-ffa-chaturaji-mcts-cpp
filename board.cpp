#include "board.h"
#include "magic_utils.h" // Include for magic_utils:: functions and constants
#include "types.h"
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
  std::array<std::array<std::array<ZobristKey, magic_utils::NUM_SQUARES>, NUM_PLAYERS_BB>, NUM_PIECE_TYPES_FOR_HASH> piece_keys;
  std::array<ZobristKey, NUM_PLAYERS_BB> turn_keys;
  std::array<ZobristKey, NUM_PLAYERS_BB> active_player_status_keys;
  ZobristData() {
    // Use a high-quality random number generator
    std::mt19937_64 rng(0xBADFACE); // Fixed seed for reproducibility
    std::uniform_int_distribution<ZobristKey> dist(0, std::numeric_limits<ZobristKey>::max());

    // Generate keys for each piece type, player, and square
    for (int type_idx = 0; type_idx < NUM_PIECE_TYPES_FOR_HASH; ++type_idx) {
      for (int player_idx = 0; player_idx < NUM_PLAYERS_BB; ++player_idx) {
          for (int sq_idx = 0; sq_idx < magic_utils::NUM_SQUARES; ++sq_idx) {
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
    if (square_index < 0 || square_index >= magic_utils::NUM_SQUARES) {
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
// Using magic_utils::NUM_SQUARES for array sizes
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::knight_attacks_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::king_attacks_;

std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> Board::pawn_attacks_red_;
std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> Board::pawn_attacks_blue_;
std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> Board::pawn_attacks_yellow_;
std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> Board::pawn_attacks_green_;

std::array<Bitboard, magic_utils::NUM_SQUARES> Board::pawn_fwd_moves_red_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::pawn_fwd_moves_blue_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::pawn_fwd_moves_yellow_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::pawn_fwd_moves_green_;

// --- Magic Bitboard related static members ---
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::rook_masks_;
std::array<Bitboard, magic_utils::NUM_SQUARES> Board::bishop_masks_;
std::array<int, magic_utils::NUM_SQUARES> Board::rook_shift_bits_;
std::array<int, magic_utils::NUM_SQUARES> Board::bishop_shift_bits_;
std::vector<Bitboard> Board::rook_attack_table_;
std::vector<Bitboard> Board::bishop_attack_table_;
std::array<unsigned int, magic_utils::NUM_SQUARES> Board::rook_attack_offsets_;
std::array<unsigned int, magic_utils::NUM_SQUARES> Board::bishop_attack_offsets_;
// --- END Magic Bitboard related static members ---

// Static initializer trick to call initialize_lookup_tables() before main()
Board::StaticInitializer Board::static_initializer_; 


// --- Bitboard Helper Functions (Public Static) ---
// Maps PieceType enum to an index suitable for piece_bitboards_ array (0-4)
int Board::piece_type_to_bb_idx(PieceType pt) {
    return piece_type_to_bb_idx_internal(pt); // Calls internal helper
}
// Checks if a square index (0-63) is valid
bool Board::is_valid_sq_idx(int sq_idx) {
    return sq_idx >= 0 && sq_idx < magic_utils::NUM_SQUARES;
}
// to_sq_idx and from_sq_idx are now in magic_utils, accessed via magic_utils::to_sq_idx, etc.

// --- Lookup Table Initialization ---
void Board::initialize_lookup_tables() {
    // --- Knight Attacks ---
    const int kn_moves[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                                {1, -2},  {1, 2},  {2, -1},  {2, 1}};
    for (int r = 0; r < magic_utils::BOARD_SIZE; ++r) {
        for (int c = 0; c < magic_utils::BOARD_SIZE; ++c) {
            int sq_idx = magic_utils::to_sq_idx(r, c);
            knight_attacks_[sq_idx] = 0ULL;
            for (auto& move : kn_moves) {
                int nr = r + move[0];
                int nc = c + move[1];
                if (nr >= 0 && nr < magic_utils::BOARD_SIZE && nc >= 0 && nc < magic_utils::BOARD_SIZE) {
                    magic_utils::set_bit(knight_attacks_[sq_idx], magic_utils::to_sq_idx(nr, nc));
                }
            }
        }
    }
    // --- King Attacks ---
    const int ki_moves[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                {0, 1},   {1, -1}, {1, 0},  {1, 1}};
    for (int r = 0; r < magic_utils::BOARD_SIZE; ++r) {
        for (int c = 0; c < magic_utils::BOARD_SIZE; ++c) {
            int sq_idx = magic_utils::to_sq_idx(r, c);
            king_attacks_[sq_idx] = 0ULL;
            for (auto& move : ki_moves) {
                int nr = r + move[0];
                int nc = c + move[1];
                if (nr >= 0 && nr < magic_utils::BOARD_SIZE && nc >= 0 && nc < magic_utils::BOARD_SIZE) {
                    magic_utils::set_bit(king_attacks_[sq_idx], magic_utils::to_sq_idx(nr, nc));
                }
            }
        }
    }

    // --- Pawn Forward Moves and Attacks (for each player color/direction) ---
    // Red Pawns (move -1 in row)
    for (int r = 0; r < magic_utils::BOARD_SIZE; ++r) {
        for (int c = 0; c < magic_utils::BOARD_SIZE; ++c) {
            int sq_idx = magic_utils::to_sq_idx(r,c);
            pawn_fwd_moves_red_[sq_idx] = 0ULL;
            pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx] = 0ULL; 
            if (r > 0) { // Can move forward
                magic_utils::set_bit(pawn_fwd_moves_red_[sq_idx], magic_utils::to_sq_idx(r-1, c));
                if (c > 0) magic_utils::set_bit(pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx], magic_utils::to_sq_idx(r-1, c-1));
                if (c < magic_utils::BOARD_SIZE - 1) magic_utils::set_bit(pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx], magic_utils::to_sq_idx(r-1, c+1));
            }
        }
    }
    // Blue Pawns (move +1 in col)
    for (int r = 0; r < magic_utils::BOARD_SIZE; ++r) {
        for (int c = 0; c < magic_utils::BOARD_SIZE; ++c) {
            int sq_idx = magic_utils::to_sq_idx(r,c);
            pawn_fwd_moves_blue_[sq_idx] = 0ULL;
            pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx] = 0ULL;
             if (c < magic_utils::BOARD_SIZE -1) { // Can move forward
                magic_utils::set_bit(pawn_fwd_moves_blue_[sq_idx], magic_utils::to_sq_idx(r, c+1));
                if (r > 0) magic_utils::set_bit(pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx], magic_utils::to_sq_idx(r-1, c+1));
                if (r < magic_utils::BOARD_SIZE - 1) magic_utils::set_bit(pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx], magic_utils::to_sq_idx(r+1, c+1));
            }
        }
    }
    // Yellow Pawns (move +1 in row)
     for (int r = 0; r < magic_utils::BOARD_SIZE; ++r) {
        for (int c = 0; c < magic_utils::BOARD_SIZE; ++c) {
            int sq_idx = magic_utils::to_sq_idx(r,c);
            pawn_fwd_moves_yellow_[sq_idx] = 0ULL;
            pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx] = 0ULL;
             if (r < magic_utils::BOARD_SIZE -1) { // Can move forward
                magic_utils::set_bit(pawn_fwd_moves_yellow_[sq_idx], magic_utils::to_sq_idx(r+1, c));
                if (c > 0) magic_utils::set_bit(pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx], magic_utils::to_sq_idx(r+1, c-1));
                if (c < magic_utils::BOARD_SIZE - 1) magic_utils::set_bit(pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx], magic_utils::to_sq_idx(r+1, c+1));
            }
        }
    }
    // Green Pawns (move -1 in col)
     for (int r = 0; r < magic_utils::BOARD_SIZE; ++r) {
        for (int c = 0; c < magic_utils::BOARD_SIZE; ++c) {
            int sq_idx = magic_utils::to_sq_idx(r,c);
            pawn_fwd_moves_green_[sq_idx] = 0ULL;
            pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx] = 0ULL;
             if (c > 0) { // Can move forward
                magic_utils::set_bit(pawn_fwd_moves_green_[sq_idx], magic_utils::to_sq_idx(r, c-1));
                if (r > 0) magic_utils::set_bit(pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx], magic_utils::to_sq_idx(r-1, c-1));
                if (r < magic_utils::BOARD_SIZE - 1) magic_utils::set_bit(pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx], magic_utils::to_sq_idx(r+1, c-1));
            }
        }
    }
    
    // --- Magic Bitboard Initialization ---
    // 1. Generate Blocker Masks using magic_utils functions
    for (int sq = 0; sq < magic_utils::NUM_SQUARES; ++sq) {
        rook_masks_[sq] = magic_utils::generate_rook_mask(sq);
        bishop_masks_[sq] = magic_utils::generate_bishop_mask(sq);
    }
    
    // 2. Calculate shift bits and total table sizes using magic_utils constants
    unsigned int total_rook_table_entries = 0;
    unsigned int total_bishop_table_entries = 0;
    for (int sq = 0; sq < magic_utils::NUM_SQUARES; ++sq) {
        // Copy the pre-generated shifts from magic_utils to Board's member variables
        rook_shift_bits_[sq] = magic_utils::RookShifts[sq]; 
        bishop_shift_bits_[sq] = magic_utils::BishopShifts[sq];
        
        rook_attack_offsets_[sq] = total_rook_table_entries;
        total_rook_table_entries += (1ULL << magic_utils::pop_count(rook_masks_[sq]));
        
        bishop_attack_offsets_[sq] = total_bishop_table_entries;
        total_bishop_table_entries += (1ULL << magic_utils::pop_count(bishop_masks_[sq]));
    }
    rook_attack_table_.resize(total_rook_table_entries);
    bishop_attack_table_.resize(total_bishop_table_entries);

    // 3. Populate Attack Tables using magic_utils functions and constants
    for (int sq = 0; sq < magic_utils::NUM_SQUARES; ++sq) {
        // Rooks
        Bitboard r_mask = rook_masks_[sq];
        int r_num_mask_bits = magic_utils::pop_count(r_mask);
        unsigned int r_num_entries_for_sq = (1ULL << r_num_mask_bits);
        for (unsigned int i = 0; i < r_num_entries_for_sq; ++i) {
            Bitboard occupancy = magic_utils::get_occupancy_subset(i, r_num_mask_bits, r_mask);
            Bitboard attacks = magic_utils::calculate_rook_attacks_on_the_fly(sq, occupancy);
            // Use magic numbers and shifts from magic_utils
            unsigned int magic_idx = (occupancy * magic_utils::RookMagics[sq]) >> magic_utils::RookShifts[sq];
            rook_attack_table_[rook_attack_offsets_[sq] + magic_idx] = attacks;
        }

        // Bishops
        Bitboard b_mask = bishop_masks_[sq];
        int b_num_mask_bits = magic_utils::pop_count(b_mask);
        unsigned int b_num_entries_for_sq = (1ULL << b_num_mask_bits);
        for (unsigned int i = 0; i < b_num_entries_for_sq; ++i) {
            Bitboard occupancy = magic_utils::get_occupancy_subset(i, b_num_mask_bits, b_mask);
            Bitboard attacks = magic_utils::calculate_bishop_attacks_on_the_fly(sq, occupancy);
            // Use magic numbers and shifts from magic_utils
            unsigned int magic_idx = (occupancy * magic_utils::BishopMagics[sq]) >> magic_utils::BishopShifts[sq];
            bishop_attack_table_[bishop_attack_offsets_[sq] + magic_idx] = attacks;
        }
    }
}

// --- Constructor ---
Board::Board()
    : active_mask_(0xF), // Default 00001111 (All 4 players active)
      current_player_(Player::RED), 
      full_move_number_(1),
      move_number_of_last_reset_(0), 
      termination_reason_(std::nullopt),
      current_hash_(0) 
{
  // Initialize bitboards to all 0s (empty)
  for (auto& player_bb_array : piece_bitboards_) {
      player_bb_array.fill(0ULL);
  }
  player_bitboards_.fill(0ULL);
  occupied_bitboard_ = 0ULL;

  // Initialize player points to 0
  player_points_.fill(0);

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
              int sq_idx = magic_utils::pop_lsb(temp_bb); // Get and remove one piece's square
              current_hash_ ^= zobrist_data.get_piece_key(piece_type, player, sq_idx);
          }
      }
  }
  // Hash current player's turn
  current_hash_ ^= zobrist_data.get_turn_key(current_player_);
  
  // Hash active player statuses
  for (int i = 0; i < 4; ++i) {
      if (active_mask_ & (1 << i)) {
          current_hash_ ^= zobrist_data.get_active_player_status_key(static_cast<Player>(i));
      }
  }
  
  // Add initial position hash to history
  position_history_.push_back(current_hash_);
}

// --- Copy Constructor ---
Board::Board(const Board &other)
    : active_mask_(other.active_mask_),
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

// 
Board::Board(const Board &other, MCTSChildCopyTag)
    : active_mask_(other.active_mask_),
      player_points_(other.player_points_),
      current_player_(other.current_player_),
      position_history_(other.position_history_), // << Deep copy this
      full_move_number_(other.full_move_number_),
      move_number_of_last_reset_(other.move_number_of_last_reset_),
      termination_reason_(std::nullopt),          // << Initialize as nullopt
      current_hash_(other.current_hash_),
      // undo_stack_ is default-initialized (empty)    // << Skip copying this
      piece_bitboards_(other.piece_bitboards_),
      player_bitboards_(other.player_bitboards_),
      occupied_bitboard_(other.occupied_bitboard_)
       {}

// --- Move Constructor ---
Board::Board(Board &&other) noexcept
    : active_mask_(other.active_mask_),
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
      occupied_bitboard_(std::move(other.occupied_bitboard_))
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
    active_mask_ = other.active_mask_;
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
    active_mask_ = other.active_mask_;
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
    occupied_bitboard_ = std::move(other.occupied_bitboard_);

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
Board Board::create_mcts_child_board(const Board& parent_board, const Move& move) {
  Board child_board(parent_board, MCTSChildCopyTag{}); 
  child_board.make_move_for_mcts(move); 
  return child_board;
}

// --- Helper to get piece at a square using bitboards ---
std::optional<Piece> Board::get_piece_at_sq(int sq_idx) const {
    if (!is_valid_sq_idx(sq_idx)) return std::nullopt;
    if (!magic_utils::get_bit(occupied_bitboard_, sq_idx)) return std::nullopt;

    for (int p_idx = 0; p_idx < NUM_PLAYERS_BB; ++p_idx) {
        if (magic_utils::get_bit(player_bitboards_[p_idx], sq_idx)) {
            Player player = static_cast<Player>(p_idx);
            for (int pt_bb_idx = 0; pt_bb_idx < NUM_BB_PIECE_TYPES; ++pt_bb_idx) {
                if (magic_utils::get_bit(piece_bitboards_[p_idx][pt_bb_idx], sq_idx)) {
                    PieceType pt = static_cast<PieceType>(pt_bb_idx + 1);
                    return Piece(player, pt);
                }
            }
            throw std::runtime_error("Bitboard inconsistency in get_piece_at_sq: Player bit set, but no piece type bit.");
        }
    }
    return std::nullopt;
}

// --- Initial Board Setup ---
void Board::setup_initial_board() {
  for (auto& player_bbs : piece_bitboards_) player_bbs.fill(0ULL);
  player_bitboards_.fill(0ULL);
  occupied_bitboard_ = 0ULL;

  auto place_piece = [&](Player p, PieceType pt, int r, int c) {
      int sq_idx = magic_utils::to_sq_idx(r, c);
      int player_idx = static_cast<int>(p);
      int pt_bb_idx = piece_type_to_bb_idx(pt);
      magic_utils::set_bit(piece_bitboards_[player_idx][pt_bb_idx], sq_idx);
      magic_utils::set_bit(player_bitboards_[player_idx], sq_idx);
      magic_utils::set_bit(occupied_bitboard_, sq_idx);
  };

  place_piece(Player::RED, PieceType::ROOK, 7, 0);
  place_piece(Player::RED, PieceType::KNIGHT, 7, 1);
  place_piece(Player::RED, PieceType::BISHOP, 7, 2);
  place_piece(Player::RED, PieceType::KING, 7, 3);
  for (int col = 0; col < 4; ++col) place_piece(Player::RED, PieceType::PAWN, 6, col);

  place_piece(Player::BLUE, PieceType::ROOK, 0, 0);
  place_piece(Player::BLUE, PieceType::KNIGHT, 1, 0);
  place_piece(Player::BLUE, PieceType::BISHOP, 2, 0);
  place_piece(Player::BLUE, PieceType::KING, 3, 0);
  for (int row = 0; row < 4; ++row) place_piece(Player::BLUE, PieceType::PAWN, row, 1);

  place_piece(Player::YELLOW, PieceType::ROOK, 0, 7);
  place_piece(Player::YELLOW, PieceType::KNIGHT, 0, 6);
  place_piece(Player::YELLOW, PieceType::BISHOP, 0, 5);
  place_piece(Player::YELLOW, PieceType::KING, 0, 4);
  for (int col = 4; col < 8; ++col) place_piece(Player::YELLOW, PieceType::PAWN, 1, col);

  place_piece(Player::GREEN, PieceType::KING, 4, 7);
  place_piece(Player::GREEN, PieceType::BISHOP, 5, 7);
  place_piece(Player::GREEN, PieceType::KNIGHT, 6, 7);
  place_piece(Player::GREEN, PieceType::ROOK, 7, 7);
  for (int row = 4; row < 8; ++row) place_piece(Player::GREEN, PieceType::PAWN, row, 6);
}

// --- Square Validity Check (Array Context) ---
bool Board::is_valid_square(int row, int col) const {
  return row >= 0 && row < magic_utils::BOARD_SIZE && col >= 0 && col < magic_utils::BOARD_SIZE;
}

// --- Pseudo-Legal Move Generation (Master Function) ---
void Board::get_pseudo_legal_moves(Player player, MoveList& moves) const {
  moves.clear();
  
  // Check if player is active using bitmask
  if (!(active_mask_ & (1 << static_cast<int>(player)))) {
      return; 
  }

  // --- Resignation is always a valid option ---
  // Strategic reasons:
  // 1. Claim Win: End game while ahead (in 2-player endgame).
  // 2. Defensive: Prevent opponents from farming points from my pieces.
  moves.push_back(Move::Resign());

  get_pawn_moves_bb(player, moves);
  get_knight_moves_bb(player, moves);
  get_bishop_moves_bb(player, moves);
  get_rook_moves_bb(player, moves);
  get_king_moves_bb(player, moves);
}

// --- Bitboard-Based Move Generation Helpers ---
void Board::get_pawn_moves_bb(Player player, MoveList& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard pawns = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::PAWN)];
    Bitboard my_pieces = player_bitboards_[p_idx];
    Bitboard opp_pieces = occupied_bitboard_ & ~my_pieces;
    Bitboard empty_sqs = ~occupied_bitboard_;

    const Bitboard* current_fwd_moves_table = nullptr; 
    const std::array<Bitboard, magic_utils::NUM_SQUARES>* current_atk_table_for_player = nullptr;
    int promotion_target_coord = -1;
    bool check_row_for_promo = false;

    switch (player) {
        case Player::RED:    current_fwd_moves_table = &pawn_fwd_moves_red_[0];    current_atk_table_for_player = &pawn_attacks_red_[p_idx];    promotion_target_coord = PROMOTION_ROW_RED_BB;   check_row_for_promo = true; break;
        case Player::BLUE:   current_fwd_moves_table = &pawn_fwd_moves_blue_[0];   current_atk_table_for_player = &pawn_attacks_blue_[p_idx];   promotion_target_coord = PROMOTION_COL_BLUE_BB;  check_row_for_promo = false; break;
        case Player::YELLOW: current_fwd_moves_table = &pawn_fwd_moves_yellow_[0]; current_atk_table_for_player = &pawn_attacks_yellow_[p_idx]; promotion_target_coord = PROMOTION_ROW_YELLOW_BB; check_row_for_promo = true; break;
        case Player::GREEN:  current_fwd_moves_table = &pawn_fwd_moves_green_[0];  current_atk_table_for_player = &pawn_attacks_green_[p_idx];  promotion_target_coord = PROMOTION_COL_GREEN_BB; check_row_for_promo = false; break;
    }
    if (!current_fwd_moves_table || !current_atk_table_for_player) return;
    
    Bitboard temp_pawns = pawns;
    while (temp_pawns) {
        int from_sq = magic_utils::pop_lsb(temp_pawns);
        BoardLocation from_loc = magic_utils::from_sq_idx(from_sq);

        Bitboard fwd_moves = current_fwd_moves_table[from_sq] & empty_sqs;
        if (fwd_moves) {
            int to_sq = magic_utils::get_lsb_index(fwd_moves);
            BoardLocation to_loc = magic_utils::from_sq_idx(to_sq);
            bool is_promotion = (check_row_for_promo && to_loc.row == promotion_target_coord) ||
                                (!check_row_for_promo && to_loc.col == promotion_target_coord);
            if (is_promotion) {
                moves.emplace_back(from_loc, to_loc, PieceType::ROOK);
            } else {
                moves.emplace_back(from_loc, to_loc);
            }
        }

        Bitboard cap_moves = (*current_atk_table_for_player)[from_sq] & opp_pieces;
        Bitboard temp_cap_moves = cap_moves;
        while (temp_cap_moves) {
            int to_sq = magic_utils::pop_lsb(temp_cap_moves);
            BoardLocation to_loc = magic_utils::from_sq_idx(to_sq);
            bool is_promotion = (check_row_for_promo && to_loc.row == promotion_target_coord) ||
                                (!check_row_for_promo && to_loc.col == promotion_target_coord);
            if (is_promotion) {
                moves.emplace_back(from_loc, to_loc, PieceType::ROOK);
            } else {
                moves.emplace_back(from_loc, to_loc);
            }
        }
    }
}

void Board::get_knight_moves_bb(Player player, MoveList& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard knights = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KNIGHT)];
    Bitboard not_my_pieces = ~player_bitboards_[p_idx];
    
    Bitboard temp_knights = knights;
    while (temp_knights) {
        int from_sq = magic_utils::pop_lsb(temp_knights);
        BoardLocation from_loc = magic_utils::from_sq_idx(from_sq);
        Bitboard possible_moves = knight_attacks_[from_sq] & not_my_pieces;
        
        Bitboard temp_possible_moves = possible_moves;
        while (temp_possible_moves) {
            int to_sq = magic_utils::pop_lsb(temp_possible_moves);
            moves.emplace_back(from_loc, magic_utils::from_sq_idx(to_sq));
        }
    }
}

void Board::get_king_moves_bb(Player player, MoveList& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard kings = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
    Bitboard not_my_pieces = ~player_bitboards_[p_idx];

    if (kings == 0) return;
    
    int from_sq = magic_utils::get_lsb_index(kings);
    BoardLocation from_loc = magic_utils::from_sq_idx(from_sq);
    Bitboard possible_moves = king_attacks_[from_sq] & not_my_pieces;

    Bitboard temp_possible_moves = possible_moves;
    while (temp_possible_moves) {
        int to_sq = magic_utils::pop_lsb(temp_possible_moves);
        moves.emplace_back(from_loc, magic_utils::from_sq_idx(to_sq));
    }
}

void Board::get_rook_moves_bb(Player player, MoveList& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard rooks = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::ROOK)];
    Bitboard my_pieces = player_bitboards_[p_idx];
    
    Bitboard temp_rooks = rooks;
    while(temp_rooks) {
        int from_sq = magic_utils::pop_lsb(temp_rooks);
        BoardLocation from_loc = magic_utils::from_sq_idx(from_sq);

        Bitboard blockers = occupied_bitboard_ & rook_masks_[from_sq]; 
        unsigned int magic_idx = (blockers * magic_utils::RookMagics[from_sq]) >> rook_shift_bits_[from_sq];
        Bitboard possible_moves = rook_attack_table_[rook_attack_offsets_[from_sq] + magic_idx];
        
        possible_moves &= ~my_pieces;

        Bitboard temp_possible_moves = possible_moves;
        while(temp_possible_moves) {
            int to_sq = magic_utils::pop_lsb(temp_possible_moves);
            moves.emplace_back(from_loc, magic_utils::from_sq_idx(to_sq));
        }
    }
}

void Board::get_bishop_moves_bb(Player player, MoveList& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard bishops = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::BISHOP)];
    Bitboard my_pieces = player_bitboards_[p_idx];

    Bitboard temp_bishops = bishops;
    while(temp_bishops) {
        int from_sq = magic_utils::pop_lsb(temp_bishops); 
        BoardLocation from_loc = magic_utils::from_sq_idx(from_sq);

        Bitboard blockers = occupied_bitboard_ & bishop_masks_[from_sq];
        unsigned int magic_idx = (blockers * magic_utils::BishopMagics[from_sq]) >> bishop_shift_bits_[from_sq]; 
        Bitboard possible_moves = bishop_attack_table_[bishop_attack_offsets_[from_sq] + magic_idx];

        possible_moves &= ~my_pieces;

        Bitboard temp_possible_moves = possible_moves;
        while(temp_possible_moves) {
            int to_sq = magic_utils::pop_lsb(temp_possible_moves);
            moves.emplace_back(from_loc, magic_utils::from_sq_idx(to_sq));
        }
    }
}


// --- Move Execution ---
std::optional<Piece> Board::make_move(const Move &move) {
  if (move.is_resignation()) {
      resign();
      return std::nullopt;
  }

  UndoInfo undo_info;

  undo_info.move = move;
  undo_info.original_active_mask = active_mask_;
  undo_info.original_player = current_player_;
  undo_info.original_full_move_number = full_move_number_;
  undo_info.original_move_number_of_last_reset = move_number_of_last_reset_;
  undo_info.eliminated_player = std::nullopt;
  undo_info.was_history_cleared = false;
  undo_info.previous_hash = current_hash_;
  undo_info.check_bonus_points = 0;

  // --- STEP 1: PRE-MOVE CHECK DETECTION ---
  struct EnemyKingState {
      Player player;
      int sq_idx;
      Bitboard attackers_before;
  };
  std::vector<EnemyKingState> enemy_king_states;
  enemy_king_states.reserve(3);

  // Iterate 0..3 for active players using bitmask
  for (int i = 0; i < 4; ++i) {
      if (active_mask_ & (1 << i)) {
          Player p = static_cast<Player>(i);
          if (p != current_player_) {
              int p_idx = static_cast<int>(p);
              Bitboard king_bb = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
              if (king_bb) {
                  int k_sq = magic_utils::get_lsb_index(king_bb);
                  // Store who attacks this king BEFORE the move
                  enemy_king_states.push_back({p, k_sq, get_attackers_on_sq(k_sq)});
              }
          }
      }
  }
  // ----------------------------------------

  const auto& zobrist_data = get_zobrist_data();
  int fr = move.from_loc.row, fc = move.from_loc.col;
  int tr = move.to_loc.row, tc = move.to_loc.col;
  int from_sq_idx = magic_utils::to_sq_idx(fr, fc);
  int to_sq_idx = magic_utils::to_sq_idx(tr, tc);
  int moving_player_idx = static_cast<int>(current_player_);

  std::optional<Piece> moving_piece_opt = get_piece_at_sq(from_sq_idx);
  if (!moving_piece_opt) { 
    throw std::runtime_error("Attempting to move from an empty square in make_move. From sq: " + std::to_string(from_sq_idx));
  }
  if (moving_piece_opt->player != current_player_) {
      throw std::runtime_error("Attempting to move opponent's piece.");
  }
  Piece moving_piece_obj = *moving_piece_opt;
  int moving_pt_bb_idx = piece_type_to_bb_idx(moving_piece_obj.piece_type);

  undo_info.captured_piece = get_piece_at_sq(to_sq_idx);
  bool is_capture = undo_info.captured_piece.has_value();
  bool is_pawn_move = (moving_piece_obj.piece_type == PieceType::PAWN);
  bool is_resetting_move = is_pawn_move || is_capture;

  current_hash_ ^= zobrist_data.get_piece_key(moving_piece_obj.piece_type, moving_piece_obj.player, from_sq_idx);
  magic_utils::clear_bit(piece_bitboards_[moving_player_idx][moving_pt_bb_idx], from_sq_idx);
  magic_utils::clear_bit(player_bitboards_[moving_player_idx], from_sq_idx);
  magic_utils::clear_bit(occupied_bitboard_, from_sq_idx);
  
  if (is_capture) {
      const Piece& captured = undo_info.captured_piece.value();
      current_hash_ ^= zobrist_data.get_piece_key(captured.piece_type, captured.player, to_sq_idx);
      int captured_player_idx = static_cast<int>(captured.player);
      int captured_pt_bb_idx = piece_type_to_bb_idx(captured.piece_type);
      magic_utils::clear_bit(piece_bitboards_[captured_player_idx][captured_pt_bb_idx], to_sq_idx);
      magic_utils::clear_bit(player_bitboards_[captured_player_idx], to_sq_idx);
  }
  
  PieceType final_piece_type = moving_piece_obj.piece_type;
  if (move.promotion_piece_type) {
    final_piece_type = move.promotion_piece_type.value();
  }
  
  int final_pt_bb_idx = piece_type_to_bb_idx(final_piece_type);
  magic_utils::set_bit(piece_bitboards_[moving_player_idx][final_pt_bb_idx], to_sq_idx);
  magic_utils::set_bit(player_bitboards_[moving_player_idx], to_sq_idx);
  magic_utils::set_bit(occupied_bitboard_, to_sq_idx);
  current_hash_ ^= zobrist_data.get_piece_key(final_piece_type, moving_piece_obj.player, to_sq_idx);
  
  if (is_capture) {
    const Piece &captured = undo_info.captured_piece.value();
    player_points_[static_cast<int>(moving_piece_obj.player)] += get_piece_capture_value(captured);
    if (captured.piece_type == PieceType::KING) {
        eliminate_player(captured.player);
        undo_info.eliminated_player = captured.player;
    }
  }

  // --- STEP 2: POST-MOVE CHECK SCORING ---
  int kings_checked_count = 0;

  for (const auto& king_state : enemy_king_states) {
      // Logic: If player is still active AND King still exists
      if (active_mask_ & (1 << static_cast<int>(king_state.player))) {
          int p_idx = static_cast<int>(king_state.player);
          Bitboard king_bb = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
          
          if (king_bb) {
               int k_sq = magic_utils::get_lsb_index(king_bb);
               Bitboard attackers_after = get_attackers_on_sq(k_sq);

               // Logic: (after & ~before) reveals purely NEW attackers.
               // This covers discovered checks (blocking piece moved) and direct checks.
               if ((attackers_after & ~king_state.attackers_before) != 0ULL) {
                   kings_checked_count++;
               }
          }
      }
  }

  // Award Bonus and Store in UndoInfo
  if (kings_checked_count == 2) {
      player_points_[static_cast<int>(current_player_)] += 1;
      undo_info.check_bonus_points = 1;
  } else if (kings_checked_count == 3) {
      player_points_[static_cast<int>(current_player_)] += 5;
      undo_info.check_bonus_points = 5;
  }
  // ---------------------------------------

  Player player_who_moved = current_player_;
  Player last_active_player_in_sequence = get_last_active_player();
  bool was_last_player_turn = (player_who_moved == last_active_player_in_sequence);

  if (was_last_player_turn) full_move_number_++;

  if (is_resetting_move) {
    move_number_of_last_reset_ = full_move_number_;
    position_history_.clear();
    undo_info.was_history_cleared = true;
  } else {
    undo_info.was_history_cleared = false; 
  }

  undo_stack_.push_back(undo_info);
  advance_turn();
  position_history_.push_back(get_position_key()); 
  is_game_over();
  return undo_info.captured_piece;
}

// --- Lightweight Move Execution for MCTS ---
std::optional<Piece> Board::make_move_for_mcts(const Move &move) {
  if (move.is_resignation()) {
      resign();
      return std::nullopt;
  }

  // --- STEP 1: PRE-MOVE CHECK DETECTION ---
  struct EnemyKingState {
      Player player;
      int sq_idx;
      Bitboard attackers_before;
  };
  std::vector<EnemyKingState> enemy_king_states;
  enemy_king_states.reserve(3);
  
  for (int i = 0; i < 4; ++i) {
      if (active_mask_ & (1 << i)) {
          Player p = static_cast<Player>(i);
          if (p != current_player_) {
              int p_idx = static_cast<int>(p);
              Bitboard king_bb = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
              if (king_bb) {
                  int k_sq = magic_utils::get_lsb_index(king_bb);
                  enemy_king_states.push_back({p, k_sq, get_attackers_on_sq(k_sq)});
              }
          }
      }
  }
  // ----------------------------------------

  const auto& zobrist_data = get_zobrist_data();
  int fr = move.from_loc.row, fc = move.from_loc.col;
  int tr = move.to_loc.row, tc = move.to_loc.col;
  int from_sq_idx = magic_utils::to_sq_idx(fr, fc);
  int to_sq_idx = magic_utils::to_sq_idx(tr, tc);
  int moving_player_idx = static_cast<int>(current_player_);

  std::optional<Piece> moving_piece_opt = get_piece_at_sq(from_sq_idx);
  if (!moving_piece_opt) {
    throw std::runtime_error("MCTS: Attempting to move from an empty square. From sq: " + std::to_string(from_sq_idx));
  }
  if (moving_piece_opt->player != current_player_) {
      throw std::runtime_error("MCTS: Attempting to move opponent's piece.");
  }
  Piece moving_piece_obj = *moving_piece_opt;
  int moving_pt_bb_idx = piece_type_to_bb_idx(moving_piece_obj.piece_type);

  std::optional<Piece> captured_piece_opt = get_piece_at_sq(to_sq_idx);
  bool is_capture = captured_piece_opt.has_value();
  bool is_pawn_move = (moving_piece_obj.piece_type == PieceType::PAWN);
  bool is_resetting_move = is_pawn_move || is_capture;

  current_hash_ ^= zobrist_data.get_piece_key(moving_piece_obj.piece_type, moving_piece_obj.player, from_sq_idx);
  magic_utils::clear_bit(piece_bitboards_[moving_player_idx][moving_pt_bb_idx], from_sq_idx);
  magic_utils::clear_bit(player_bitboards_[moving_player_idx], from_sq_idx);
  magic_utils::clear_bit(occupied_bitboard_, from_sq_idx);

  if (is_capture) {
      const Piece& captured = captured_piece_opt.value();
      current_hash_ ^= zobrist_data.get_piece_key(captured.piece_type, captured.player, to_sq_idx);
      int captured_player_idx = static_cast<int>(captured.player);
      int captured_pt_bb_idx = piece_type_to_bb_idx(captured.piece_type);
      magic_utils::clear_bit(piece_bitboards_[captured_player_idx][captured_pt_bb_idx], to_sq_idx);
      magic_utils::clear_bit(player_bitboards_[captured_player_idx], to_sq_idx);
  }

  PieceType final_piece_type = moving_piece_obj.piece_type;
  if (move.promotion_piece_type) {
    final_piece_type = move.promotion_piece_type.value();
  }

  int final_pt_bb_idx = piece_type_to_bb_idx(final_piece_type);
  magic_utils::set_bit(piece_bitboards_[moving_player_idx][final_pt_bb_idx], to_sq_idx);
  magic_utils::set_bit(player_bitboards_[moving_player_idx], to_sq_idx);
  magic_utils::set_bit(occupied_bitboard_, to_sq_idx);
  current_hash_ ^= zobrist_data.get_piece_key(final_piece_type, moving_piece_obj.player, to_sq_idx);

  if (is_capture) {
    const Piece &captured_val = captured_piece_opt.value();
    player_points_[static_cast<int>(moving_piece_obj.player)] += get_piece_capture_value(captured_val);
    if (captured_val.piece_type == PieceType::KING) {
        eliminate_player(captured_val.player);
    }
  }

  // --- STEP 2: POST-MOVE CHECK SCORING (MCTS) ---
  int kings_checked_count = 0;
  for (const auto& king_state : enemy_king_states) {
      if (active_mask_ & (1 << static_cast<int>(king_state.player))) {
          int p_idx = static_cast<int>(king_state.player);
          Bitboard king_bb = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
          if (king_bb) {
               int k_sq = magic_utils::get_lsb_index(king_bb);
               Bitboard attackers_after = get_attackers_on_sq(k_sq);
               if ((attackers_after & ~king_state.attackers_before) != 0ULL) {
                   kings_checked_count++;
               }
          }
      }
  }

  if (kings_checked_count == 2) {
      player_points_[static_cast<int>(moving_piece_obj.player)] += 1;
  } else if (kings_checked_count == 3) {
      player_points_[static_cast<int>(moving_piece_obj.player)] += 5;
  }
  // ----------------------------------------------

  Player player_who_moved = current_player_;
  Player last_active_player_in_sequence = get_last_active_player();
  bool was_last_player_turn = (player_who_moved == last_active_player_in_sequence);

  if (was_last_player_turn) full_move_number_++;

  if (is_resetting_move) {
    move_number_of_last_reset_ = full_move_number_;
    position_history_.clear();
  }
  position_history_.push_back(current_hash_);

  advance_turn();
  is_game_over();
  return captured_piece_opt;
}

// --- Undo Last Move ---
void Board::undo_move() {
  if (undo_stack_.empty()) {
    throw std::runtime_error("No previous state available to undo.");
  }
  UndoInfo undo_info = undo_stack_.back();
  undo_stack_.pop_back();

  // 1. Restore Scalars and History
  active_mask_ = undo_info.original_active_mask;
  current_hash_ = undo_info.previous_hash;
  current_player_ = undo_info.original_player;
  full_move_number_ = undo_info.original_full_move_number;
  move_number_of_last_reset_ = undo_info.original_move_number_of_last_reset;

  if (!position_history_.empty()) {
      position_history_.pop_back();
  }

  // 2. Restore Points (Logic remains the same)
  if (!undo_info.move.is_resignation() && undo_info.captured_piece) {
    const Piece &captured = undo_info.captured_piece.value();
    player_points_[static_cast<int>(undo_info.original_player)] -= get_piece_capture_value(captured);
  }
  if (undo_info.check_bonus_points > 0) {
      player_points_[static_cast<int>(undo_info.original_player)] -= undo_info.check_bonus_points;
  }

  termination_reason_ = std::nullopt;

  // 3. Incremental Bitboard Restoration
  // We must strictly reverse the operations done in make_move.
  
  // A. Resignation: Nothing to move back, just state restoration (handled above).
  if (undo_info.move.is_resignation()) {
      return; 
  }

  const Move& move = undo_info.move;
  int from_sq = magic_utils::to_sq_idx(move.from_loc.row, move.from_loc.col);
  int to_sq = magic_utils::to_sq_idx(move.to_loc.row, move.to_loc.col);
  int player_idx = static_cast<int>(undo_info.original_player);

  // --- Step 3a: Move the piece BACK (To -> From) ---
  
  // Determine what piece type is currently at 'to_sq' (the piece that moved)
  // If it was a promotion, it is currently the Promoted Type (e.g. ROOK).
  // If not, it is the original type.
  PieceType piece_on_dest_sq = PieceType::PAWN; // Default, will find below
  
  // Optimization: We know the player, so we only scan their bitboards
  bool found = false;
  for(int pt = 1; pt <= 5; ++pt) {
      PieceType pt_enum = static_cast<PieceType>(pt);
      int bb_idx = piece_type_to_bb_idx(pt_enum);
      if (magic_utils::get_bit(piece_bitboards_[player_idx][bb_idx], to_sq)) {
          piece_on_dest_sq = pt_enum;
          found = true;
          break;
      }
  }

  if (!found) {
      // Should never happen in a bug-free engine
      throw std::runtime_error("Critical Undo Error: Piece missing at destination.");
  }

  // Remove the piece from destination (To)
  int moving_pt_bb_idx = piece_type_to_bb_idx(piece_on_dest_sq);
  magic_utils::clear_bit(piece_bitboards_[player_idx][moving_pt_bb_idx], to_sq);
  magic_utils::clear_bit(player_bitboards_[player_idx], to_sq);
  magic_utils::clear_bit(occupied_bitboard_, to_sq);

  // Place the piece back at source (From)
  // CRITICAL: If it was a promotion, 'piece_on_dest_sq' is ROOK, but we must put back a PAWN.
  PieceType original_piece_type = piece_on_dest_sq;
  if (move.promotion_piece_type.has_value()) {
      // In Chaturaji, pawns usually promote to Rooks.
      // If we see a promotion recorded, the original piece was a PAWN.
      original_piece_type = PieceType::PAWN;
  }

  int original_pt_bb_idx = piece_type_to_bb_idx(original_piece_type);
  magic_utils::set_bit(piece_bitboards_[player_idx][original_pt_bb_idx], from_sq);
  magic_utils::set_bit(player_bitboards_[player_idx], from_sq);
  magic_utils::set_bit(occupied_bitboard_, from_sq);

  // --- Step 3b: Restore Captured Piece (if any) ---
  if (undo_info.captured_piece.has_value()) {
      const Piece& captured = undo_info.captured_piece.value();
      int cap_player_idx = static_cast<int>(captured.player);
      int cap_pt_bb_idx = piece_type_to_bb_idx(captured.piece_type);

      // Place the captured piece back at 'to_sq'
      magic_utils::set_bit(piece_bitboards_[cap_player_idx][cap_pt_bb_idx], to_sq);
      magic_utils::set_bit(player_bitboards_[cap_player_idx], to_sq);
      magic_utils::set_bit(occupied_bitboard_, to_sq);
  }
}

// --- Player Elimination ---
void Board::eliminate_player(Player player) {
  int pi = static_cast<int>(player);
  if (active_mask_ & (1 << pi)) {
    const auto& zobrist_data = get_zobrist_data();
    current_hash_ ^= zobrist_data.get_active_player_status_key(player);
    active_mask_ &= ~(1 << pi);
  }
}

// --- Bitboard Accessors ---
Bitboard Board::get_occupied_bitboard() const { return occupied_bitboard_; }
Bitboard Board::get_player_bitboard(Player p) const { return player_bitboards_[static_cast<int>(p)]; }
Bitboard Board::get_piece_bitboard(Player p, PieceType pt) const {
    return piece_bitboards_[static_cast<int>(p)][piece_type_to_bb_idx(pt)];
}

Bitboard Board::get_squares_attacked_by(Player player) const {
    // Eliminated players do not attack squares.
    if (!(active_mask_ & (1 << static_cast<int>(player)))) {
        return 0ULL;
    }

    Bitboard attacks = 0ULL;
    int p_idx = static_cast<int>(player);
    Bitboard occ = occupied_bitboard_;

    // 1. Pawns
    // Note: We use the attack tables. Pawn attacks are static based on the square.
    Bitboard pawns = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::PAWN)];
    while (pawns) {
        int sq = magic_utils::pop_lsb(pawns);
        switch (player) {
            case Player::RED:    attacks |= pawn_attacks_red_[static_cast<int>(Player::RED)][sq]; break;
            case Player::BLUE:   attacks |= pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq]; break;
            case Player::YELLOW: attacks |= pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq]; break;
            case Player::GREEN:  attacks |= pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq]; break;
        }
    }

    // 2. Knights
    Bitboard knights = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KNIGHT)];
    while (knights) {
        int sq = magic_utils::pop_lsb(knights);
        attacks |= knight_attacks_[sq];
    }

    // 3. Kings
    Bitboard king = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::KING)];
    if (king) {
        int sq = magic_utils::get_lsb_index(king);
        attacks |= king_attacks_[sq];
    }

    // 4. Rooks (Sliding)
    Bitboard rooks = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::ROOK)];
    while (rooks) {
        int sq = magic_utils::pop_lsb(rooks);
        // Look up attacks using Magic Bitboards (same logic as get_rook_moves but accumulating attacks)
        Bitboard blockers = occ & rook_masks_[sq];
        unsigned int magic_idx = (blockers * magic_utils::RookMagics[sq]) >> rook_shift_bits_[sq];
        attacks |= rook_attack_table_[rook_attack_offsets_[sq] + magic_idx];
    }

    // 5. Bishops (Sliding)
    Bitboard bishops = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::BISHOP)];
    while (bishops) {
        int sq = magic_utils::pop_lsb(bishops);
        Bitboard blockers = occ & bishop_masks_[sq];
        unsigned int magic_idx = (blockers * magic_utils::BishopMagics[sq]) >> bishop_shift_bits_[sq];
        attacks |= bishop_attack_table_[bishop_attack_offsets_[sq] + magic_idx];
    }

    return attacks;
}

Bitboard Board::get_attackers_on_sq(int sq_idx) const {
    Bitboard attackers = 0ULL;
    Bitboard occ = occupied_bitboard_;

    // Helper lambda to check if a player is active
    auto is_active = [&](int p_idx) {
        return (active_mask_ & (1 << p_idx));
    };

    // 1. Knights (symmetric attacks)
    Bitboard all_knights = 0ULL;
    for(int i=0; i<4; ++i) {
        if (is_active(i)) all_knights |= piece_bitboards_[i][piece_type_to_bb_idx(PieceType::KNIGHT)];
    }
    attackers |= (knight_attacks_[sq_idx] & all_knights);

    // 2. Kings (symmetric attacks)
    Bitboard all_kings = 0ULL;
    for(int i=0; i<4; ++i) {
        if (is_active(i)) all_kings |= piece_bitboards_[i][piece_type_to_bb_idx(PieceType::KING)];
    }
    attackers |= (king_attacks_[sq_idx] & all_kings);

    // 3. Rooks (symmetric sliding)
    Bitboard all_rooks = 0ULL;
    for(int i=0; i<4; ++i) {
        if (is_active(i)) all_rooks |= piece_bitboards_[i][piece_type_to_bb_idx(PieceType::ROOK)];
    }
    
    Bitboard r_blockers = occ & rook_masks_[sq_idx];
    unsigned int r_magic_idx = (r_blockers * magic_utils::RookMagics[sq_idx]) >> rook_shift_bits_[sq_idx];
    attackers |= (rook_attack_table_[rook_attack_offsets_[sq_idx] + r_magic_idx] & all_rooks);

    // 4. Bishops (symmetric sliding)
    Bitboard all_bishops = 0ULL;
    for(int i=0; i<4; ++i) {
        if (is_active(i)) all_bishops |= piece_bitboards_[i][piece_type_to_bb_idx(PieceType::BISHOP)];
    }
    
    Bitboard b_blockers = occ & bishop_masks_[sq_idx];
    unsigned int b_magic_idx = (b_blockers * magic_utils::BishopMagics[sq_idx]) >> bishop_shift_bits_[sq_idx];
    attackers |= (bishop_attack_table_[bishop_attack_offsets_[sq_idx] + b_magic_idx] & all_bishops);

    // 5. Pawns (Asymmetric - Look "backwards" from target to potential attacker)
    BoardLocation loc = magic_utils::from_sq_idx(sq_idx);
    int r = loc.row; 
    int c = loc.col;

    // Check against RED Pawns (Attack from Row + 1)
    if (r < 7 && (active_mask_ & (1 << static_cast<int>(Player::RED)))) {
        Bitboard red_pawns = piece_bitboards_[static_cast<int>(Player::RED)][piece_type_to_bb_idx(PieceType::PAWN)];
        if (c > 0 && magic_utils::get_bit(red_pawns, magic_utils::to_sq_idx(r + 1, c - 1))) 
            magic_utils::set_bit(attackers, magic_utils::to_sq_idx(r + 1, c - 1));
        if (c < 7 && magic_utils::get_bit(red_pawns, magic_utils::to_sq_idx(r + 1, c + 1))) 
            magic_utils::set_bit(attackers, magic_utils::to_sq_idx(r + 1, c + 1));
    }

    // Check against BLUE Pawns (Attack from Col - 1)
    if (c > 0 && (active_mask_ & (1 << static_cast<int>(Player::BLUE)))) {
        Bitboard blue_pawns = piece_bitboards_[static_cast<int>(Player::BLUE)][piece_type_to_bb_idx(PieceType::PAWN)];
        if (r > 0 && magic_utils::get_bit(blue_pawns, magic_utils::to_sq_idx(r - 1, c - 1))) 
            magic_utils::set_bit(attackers, magic_utils::to_sq_idx(r - 1, c - 1));
        if (r < 7 && magic_utils::get_bit(blue_pawns, magic_utils::to_sq_idx(r + 1, c - 1))) 
            magic_utils::set_bit(attackers, magic_utils::to_sq_idx(r + 1, c - 1));
    }

    // Check against YELLOW Pawns (Attack from Row - 1)
    if (r > 0 && (active_mask_ & (1 << static_cast<int>(Player::YELLOW)))) {
        Bitboard yellow_pawns = piece_bitboards_[static_cast<int>(Player::YELLOW)][piece_type_to_bb_idx(PieceType::PAWN)];
        if (c > 0 && magic_utils::get_bit(yellow_pawns, magic_utils::to_sq_idx(r - 1, c - 1))) 
            magic_utils::set_bit(attackers, magic_utils::to_sq_idx(r - 1, c - 1));
        if (c < 7 && magic_utils::get_bit(yellow_pawns, magic_utils::to_sq_idx(r - 1, c + 1))) 
            magic_utils::set_bit(attackers, magic_utils::to_sq_idx(r - 1, c + 1));
    }

    // Check against GREEN Pawns (Attack from Col + 1)
    if (c < 7 && (active_mask_ & (1 << static_cast<int>(Player::GREEN)))) {
        Bitboard green_pawns = piece_bitboards_[static_cast<int>(Player::GREEN)][piece_type_to_bb_idx(PieceType::PAWN)];
        if (r > 0 && magic_utils::get_bit(green_pawns, magic_utils::to_sq_idx(r - 1, c + 1))) 
            magic_utils::set_bit(attackers, magic_utils::to_sq_idx(r - 1, c + 1));
        if (r < 7 && magic_utils::get_bit(green_pawns, magic_utils::to_sq_idx(r + 1, c + 1))) 
            magic_utils::set_bit(attackers, magic_utils::to_sq_idx(r + 1, c + 1));
    }

    return attackers;
}

// Utility to print a bitboard for debugging
void Board::print_bitboard(Bitboard bb, const std::string& label) {
    std::cout << "Bitboard: " << label << " (0x" << std::hex << bb << std::dec << ")" << std::endl;
    for (int r = 0; r < magic_utils::BOARD_SIZE; ++r) {
        for (int c = 0; c < magic_utils::BOARD_SIZE; ++c) {
            int sq_idx = magic_utils::to_sq_idx(r, c);
            std::cout << (magic_utils::get_bit(bb, sq_idx) ? "1 " : ". ");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// --- Game State Accessors ---
const std::set<Player> Board::get_active_players() const {
    std::set<Player> res;
    for (int i = 0; i < 4; ++i) {
        if (active_mask_ & (1 << i)) {
            res.insert(static_cast<Player>(i));
        }
    }
    return res;
}

const PlayerPointMap &Board::get_player_points() const { return player_points_; }
Player Board::get_current_player() const { return current_player_; }
int Board::get_full_move_number() const { return full_move_number_; }
int Board::get_move_number_of_last_reset() const { return move_number_of_last_reset_; }
const std::optional<TerminationReason> &Board::get_termination_reason() const { return termination_reason_; }
const Board::PositionHistory &Board::get_position_history() const { return position_history_; }

Player Board::get_last_active_player() const {
  if (active_mask_ == 0) return Player::RED;
  for (int i = 3; i >= 0; --i) {
      if (active_mask_ & (1 << i)) {
          return static_cast<Player>(i);
      }
  }
  return Player::RED;
}

// --- Game Status ---
bool Board::is_game_over() const {
  if (termination_reason_) return true;

  // --- Autoclaim Logic (Referee Enforcement) ---
  // This logic detects if the game MUST end to protect the player who is in 2nd place.
  int active_count = magic_utils::pop_count(static_cast<Bitboard>(active_mask_));
  
  if (active_count == 2) {
    Player p_curr = current_player_;
    Player p_opp;
    for (int i = 0; i < 4; ++i) {
        if (active_mask_ & (1 << i) && static_cast<Player>(i) != p_curr) {
            p_opp = static_cast<Player>(i);
            break;
        }
    }

    // 1. Calculate points transfer if current player resigns/claims now
    int dead_kings_on_board = 0;
    for (int i = 0; i < 4; ++i) {
      if (!(active_mask_ & (1 << i))) {
        if (piece_bitboards_[i][piece_type_to_bb_idx(PieceType::KING)] != 0ULL) {
          dead_kings_on_board++;
        }
      }
    }
    int potential_transfer = 3 + (3 * dead_kings_on_board);

    // 2. Check scores
    int score_curr = player_points_[static_cast<int>(p_curr)];
    int score_opp_after_transfer = player_points_[static_cast<int>(p_opp)] + potential_transfer;
    
    // Check if Current Player is the Absolute Winner (cannot be overtaken)
    bool is_absolute_winner = (score_curr > score_opp_after_transfer);
    if (is_absolute_winner) {
        for (int i = 0; i < 4; ++i) {
            Player p_loop = static_cast<Player>(i);
            if (p_loop != p_curr && p_loop != p_opp && player_points_[i] >= score_curr) {
                is_absolute_winner = false;
                break;
            }
        }
    }

    if (is_absolute_winner) {
      // Find the score of the inactive player currently in 2nd place
      int best_inactive_score = -1;
      for (int i = 0; i < 4; ++i) {
        if (!(active_mask_ & (1 << i))) {
          if (player_points_[i] > best_inactive_score) best_inactive_score = player_points_[i];
        }
      }

      // 3. Autoclaim Trigger
      // If the inactive player has more points than the active opponent (is currently 2nd),
      // AND the active opponent cannot catch up even if we give them the resignation bonus,
      // then we FORCE the claim to freeze the standings.
      if (best_inactive_score > player_points_[static_cast<int>(p_opp)]) {
        if (score_opp_after_transfer <= best_inactive_score) {
          termination_reason_ = TerminationReason::AUTOCLAIM;
          return true;
        }
      }
    }
  }

  // --- Standard Termination Checks ---
  if (active_count <= 1) { 
    termination_reason_ = TerminationReason::ELIMINATION; 
    return true; 
  }

  int moves_since_last_reset = full_move_number_ - move_number_of_last_reset_;
  if (moves_since_last_reset >= 50) {
    // If we are in the middle of a real game (undo_stack exists)
    if (!undo_stack_.empty()) {
      Player player_who_just_moved = undo_stack_.back().original_player;
      if (player_who_just_moved == get_last_active_player()) {
        termination_reason_ = TerminationReason::FIFTY_MOVE_RULE; 
        return true;
      }
    } 
    // If we are in MCTS (no undo_stack)
    else {
      // In MCTS, we skip the undo stack for performance, so we cannot verify if 
      // the 50-move cycle ended on exactly the last active player's turn.
      // However, we MUST terminate the search here to ensure simulations do not 
      // exceed the 256-ply capacity of position_history_, which would cause
      // out-of-bounds memory access and crashes.
      termination_reason_ = TerminationReason::FIFTY_MOVE_RULE;
      return true;
    }
  }

  int count = 0;
  for (const auto &key : position_history_) if (key == current_hash_) count++;
  if (count >= 3) { 
    termination_reason_ = TerminationReason::THREEFOLD_REPETITION; 
    return true; 
  }
  return false;
}

PlayerPointMap Board::get_game_result() const {
  PlayerPointMap results = player_points_; // Copy current points
  
  // Calculate bonus points based on dead kings for standard draws (50-move/repetition)
  int num_kings_of_inactive_players = 0;
  for (int i = 0; i < 4; ++i) {
      if (!(active_mask_ & (1 << i))) {
          if (piece_bitboards_[i][Board::piece_type_to_bb_idx(PieceType::KING)] != 0ULL) {
              num_kings_of_inactive_players++;
          }
      }
  }

  int num_active = magic_utils::pop_count(static_cast<Bitboard>(active_mask_));
  
  if (termination_reason_) {
    TerminationReason reason = *termination_reason_;
    if (reason == TerminationReason::AUTOCLAIM) {
      // We manually simulate the points transfer that would happen if p_curr resigned.
      // (Winner keeps their points, Opponent gets the bonus).
      Player p_curr = current_player_;
      Player p_opp = Player::RED;
      for (int i=0; i<4; ++i) {
        if ((active_mask_ & (1 << i)) && static_cast<Player>(i) != p_curr) { 
        p_opp = static_cast<Player>(i); break;
        }
      }
      // Opponent gets: 3 (Winner's King) + 3 * (Dead Kings on Board)
      int points_to_add = 3; 
      for (int i = 0; i < 4; ++i) {
        if (!(active_mask_ & (1 << i))) {
          if (piece_bitboards_[i][piece_type_to_bb_idx(PieceType::KING)] != 0ULL) {
              points_to_add += 3;
          }
        }
      }
      results[static_cast<int>(p_opp)] += points_to_add;
    } 
    else if (reason == TerminationReason::FIFTY_MOVE_RULE || reason == TerminationReason::THREEFOLD_REPETITION) {
      if (num_active > 0) { 
        int bonus = (num_kings_of_inactive_players > 0) ? 
          static_cast<int>(std::ceil(3.0 * num_kings_of_inactive_players / num_active)) : 0;
        for (int i=0; i<4; ++i) { 
          if(active_mask_ & (1 << i)) results[i] += (2 + bonus);
        }
      }
    } 
    else if (reason == TerminationReason::ELIMINATION) {
      if (num_active == 1 && num_kings_of_inactive_players > 0) {
        for(int i=0; i<4; ++i) {
          if(active_mask_ & (1<<i)) {
            results[i] += (3 * num_kings_of_inactive_players);
            break;
          }
        }
      }
    }
  }
  return results;
}

std::optional<Player> Board::get_winner() const {
  if (!termination_reason_) return std::nullopt;
  
  PlayerPointMap final_scores = get_game_result();
  
  // Find the iterator to the maximum score
  auto winner_it = std::max_element(final_scores.begin(), final_scores.end());
  
  if (winner_it == final_scores.end()) return std::nullopt;
  
  // Calculate the index (which corresponds to the Player enum)
  int p_idx = std::distance(final_scores.begin(), winner_it);
  return static_cast<Player>(p_idx);
}

int Board::get_piece_value(const Piece& piece) const {
  switch (piece.piece_type) {
  case PieceType::PAWN: return 1; case PieceType::KNIGHT: return 3;
  case PieceType::BISHOP: return 5; case PieceType::ROOK: return 5;
  case PieceType::KING: return 3; default: return 0;
  }
}
int Board::get_piece_capture_value(const Piece& piece) const {
    if (!(active_mask_ & (1 << static_cast<int>(piece.player)))) {
        return (piece.piece_type == PieceType::KING) ? 3 : 0;
    }
    switch (piece.piece_type) {
        case PieceType::PAWN: return 1; case PieceType::KNIGHT: return 3;
        case PieceType::BISHOP: return 5; case PieceType::ROOK: return 5;
        case PieceType::KING: return 3;
        default: return 0;
    }
}

std::map<Player, int> Board::evaluate() const {
  std::map<Player, int> scores_map;
  
  for (int i = 0; i < 4; ++i) {
      scores_map[static_cast<Player>(i)] = player_points_[i];
  }
  
  for (int p_idx = 0; p_idx < 4; ++p_idx) {
      Player p_enum = static_cast<Player>(p_idx);
      // If inactive and no king, harsh penalty (though points stay)
      if ((active_mask_ & (1 << p_idx)) == 0 && piece_bitboards_[p_idx][4] == 0) {
          scores_map[p_enum] = -999;
          continue;
      }
      
      // Count material
      for (int pt = 0; pt < 5; ++pt) {
          Bitboard bb = piece_bitboards_[p_idx][pt];
          int count = magic_utils::pop_count(bb);
          Piece p(p_enum, static_cast<PieceType>(pt + 1));
          scores_map[p_enum] += count * get_piece_value(p);
      }
  }
  
  return scores_map;
}

// --- Player Actions ---
void Board::resign() {
  Player resigning_player = current_player_; 
  if (active_mask_ & (1 << static_cast<int>(resigning_player))) {
    UndoInfo resign_undo_info;
    resign_undo_info.original_active_mask = active_mask_;
    resign_undo_info.original_player = resigning_player; 
    resign_undo_info.original_full_move_number = full_move_number_;
    resign_undo_info.original_move_number_of_last_reset = move_number_of_last_reset_;
    resign_undo_info.previous_hash = current_hash_;
    resign_undo_info.eliminated_player = resigning_player;
    resign_undo_info.was_history_cleared = false;
    resign_undo_info.move = Move::Resign();
    resign_undo_info.captured_piece = std::nullopt; 

    eliminate_player(resigning_player);

    if (magic_utils::pop_count(static_cast<Bitboard>(active_mask_)) <= 1) {
        const auto& zobrist_data = get_zobrist_data();
        current_hash_ ^= zobrist_data.get_turn_key(resigning_player); 
        is_game_over();
    } else {
        advance_turn();
    }
    undo_stack_.push_back(resign_undo_info);
    position_history_.push_back(current_hash_);
  }
}

void Board::advance_turn() {
  if (active_mask_ == 0) return;
  const auto& zobrist_data = get_zobrist_data(); 
  Player old_player = current_player_;
  current_player_ = static_cast<Player>((static_cast<int>(current_player_) + 1) % 4);
  
  while (!(active_mask_ & (1 << static_cast<int>(current_player_)))) {
    current_player_ = static_cast<Player>((static_cast<int>(current_player_) + 1) % 4);
  }

  current_hash_ ^= zobrist_data.get_turn_key(old_player);
  current_hash_ ^= zobrist_data.get_turn_key(current_player_); 
}

const std::string ANSI_RESET_BB = "\033[0m"; 
const std::string ANSI_RED_BB = "\033[31m"; const std::string ANSI_GREEN_BB = "\033[32m";
const std::string ANSI_YELLOW_BB = "\033[33m"; const std::string ANSI_BLUE_BB = "\033[34m";
const std::string UNICODE_KING_BB = "♔"; const std::string UNICODE_ROOK_BB = "♖";
const std::string UNICODE_BISHOP_BB = "♗"; const std::string UNICODE_KNIGHT_BB = "♘";
const std::string UNICODE_PAWN_BB = "♙";

void Board::print_board() const {
  std::cout << "   a  b  c  d  e  f  g  h" << std::endl;
  for (int r = 0; r < magic_utils::BOARD_SIZE; ++r) {
    std::cout << 8 - r << " ";
    for (int c = 0; c < magic_utils::BOARD_SIZE; ++c) {
      int sq_idx = magic_utils::to_sq_idx(r, c);
      std::optional<Piece> piece_opt = get_piece_at_sq(sq_idx);
      std::string symbol_str = " ";
      if (piece_opt) {
        const Piece &p = *piece_opt;
        bool display_as_inactive = !(active_mask_ & (1 << static_cast<int>(p.player)));
        const std::string* base_symbol = nullptr;
        switch (p.piece_type) {
            case PieceType::PAWN:   base_symbol = &UNICODE_PAWN_BB;   break;
            case PieceType::KNIGHT: base_symbol = &UNICODE_KNIGHT_BB; break;
            case PieceType::BISHOP: base_symbol = &UNICODE_BISHOP_BB; break;
            case PieceType::ROOK:   base_symbol = &UNICODE_ROOK_BB;   break;
            case PieceType::KING:   base_symbol = &UNICODE_KING_BB;   break;
        }
        if (base_symbol) {
            if (display_as_inactive) {
                symbol_str = *base_symbol;
            } else {
                const std::string* color_code = nullptr;
                switch (p.player) {
                    case Player::RED:    color_code = &ANSI_RED_BB;    break;
                    case Player::BLUE:   color_code = &ANSI_BLUE_BB;   break;
                    case Player::YELLOW: color_code = &ANSI_YELLOW_BB; break;
                    case Player::GREEN:  color_code = &ANSI_GREEN_BB;  break;
                }
                if (color_code) {
                    symbol_str = *color_code + *base_symbol + ANSI_RESET_BB;
                } else {
                    symbol_str = *base_symbol;
                }
            }
        }
      }
      std::cout << "[" << symbol_str << "]";
    }
    std::cout << std::endl; 
  }
  std::cout << "Turn: ";
  switch (current_player_) {
  case Player::RED:    std::cout << ANSI_RED_BB << "RED" << ANSI_RESET_BB; break;
  case Player::BLUE:   std::cout << ANSI_BLUE_BB << "BLUE" << ANSI_RESET_BB; break;
  case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "YELLOW" << ANSI_RESET_BB; break;
  case Player::GREEN:  std::cout << ANSI_GREEN_BB << "GREEN" << ANSI_RESET_BB; break;
  }
  std::cout << std::endl;
  std::cout << "Active Players: ";
  for(int i=0; i<4; ++i){
      if (active_mask_ & (1 << i)) {
          switch(static_cast<Player>(i)){
              case Player::RED: std::cout << ANSI_RED_BB << "R " << ANSI_RESET_BB; break;
              case Player::BLUE: std::cout << ANSI_BLUE_BB << "B " << ANSI_RESET_BB; break;
              case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "Y " << ANSI_RESET_BB; break;
              case Player::GREEN: std::cout << ANSI_GREEN_BB << "G " << ANSI_RESET_BB; break;
          }
      }
  }
  std::cout << std::endl;
  std::cout << "Points: ";
  for(int i=0; i<4; ++i){
      switch(static_cast<Player>(i)){ 
          case Player::RED: std::cout << ANSI_RED_BB << "R:" << player_points_[i] << ANSI_RESET_BB << " "; break;
          case Player::BLUE: std::cout << ANSI_BLUE_BB << "B:" << player_points_[i] << ANSI_RESET_BB << " "; break;
          case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "Y:" << player_points_[i] << ANSI_RESET_BB << " "; break;
          case Player::GREEN: std::cout << ANSI_GREEN_BB << "G:" << player_points_[i] << ANSI_RESET_BB << " "; break;
      }
  }
  std::cout << std::endl;
  if(termination_reason_) std::cout << "Game Over: " << chaturaji_cpp::to_string(*termination_reason_) << std::endl;
}

Board::PositionKey Board::get_position_key() const { return current_hash_; }

} // namespace chaturaji_cpp