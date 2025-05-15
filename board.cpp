// board2.cpp - newest 
#include "board.h"
#include <algorithm> 
#include <array>     
#include <cmath>     
#include <cstdint>   
#include <iostream>
#include <limits>  
#include <numeric> 
#include <random>  
#include <sstream>
#include <stdexcept>
#include <utility> 
#include <vector>

#ifdef _MSC_VER
#include <intrin.h> 
#endif

namespace chaturaji_cpp {

// Anonymous namespace for Zobrist and other internal constants
namespace { 
const int NUM_PIECE_TYPES_FOR_HASH = 5; 
const int NUM_BB_PIECE_TYPES = 5;       
const int NUM_PLAYERS_BB = 4;

// --- ADDED DIRECTIONAL CONSTANTS (needed by evaluate() and old move gen if still used) ---
const std::vector<std::pair<int, int>> BISHOP_DIRS_EVAL = { {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
const std::vector<std::pair<int, int>> ROOK_DIRS_EVAL = { {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
const std::vector<std::pair<int, int>> KING_DIRS_EVAL = { {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
const std::vector<std::pair<int, int>> KNIGHT_MOVES_EVAL = { {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
// --- END ADDED DIRECTIONAL CONSTANTS ---


int piece_type_to_bb_idx_internal(PieceType pt) {
    int val = static_cast<int>(pt) - 1;
    if (val < 0 || val >= NUM_BB_PIECE_TYPES) {
        throw std::out_of_range("Invalid PieceType for bitboard index.");
    }
    return val;
}

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

const Bitboard FILE_A_BB = 0x0101010101010101ULL; // Renamed to avoid conflict
const Bitboard FILE_H_BB = 0x8080808080808080ULL; // Renamed

const int PROMOTION_ROW_RED_BB = 0;    
const int PROMOTION_COL_BLUE_BB = 7;   
const int PROMOTION_ROW_YELLOW_BB = 7; 
const int PROMOTION_COL_GREEN_BB = 0;  
} // end anonymous namespace


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
std::array<std::array<Bitboard, 4>, NUM_SQUARES_BB> Board::rook_rays_; 
std::array<std::array<Bitboard, 4>, NUM_SQUARES_BB> Board::bishop_rays_;
Board::StaticInitializer Board::static_initializer_; 


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

void Board::initialize_lookup_tables() {
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

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c);
            pawn_fwd_moves_red_[sq_idx] = 0ULL;
            pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx] = 0ULL; 
            if (r > 0) { 
                set_bit(pawn_fwd_moves_red_[sq_idx], to_sq_idx(r-1, c));
                if (c > 0 && !(FILE_A_BB & (1ULL << sq_idx))) set_bit(pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx], to_sq_idx(r-1, c-1));
                if (c < BOARD_SIZE - 1 && !(FILE_H_BB & (1ULL << sq_idx))) set_bit(pawn_attacks_red_[static_cast<int>(Player::RED)][sq_idx], to_sq_idx(r-1, c+1));
            }
        }
    }
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c);
            pawn_fwd_moves_blue_[sq_idx] = 0ULL;
            pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx] = 0ULL;
             if (c < BOARD_SIZE -1) { 
                set_bit(pawn_fwd_moves_blue_[sq_idx], to_sq_idx(r, c+1));
                if (r > 0) set_bit(pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx], to_sq_idx(r-1, c+1));
                if (r < BOARD_SIZE - 1) set_bit(pawn_attacks_blue_[static_cast<int>(Player::BLUE)][sq_idx], to_sq_idx(r+1, c+1));
            }
        }
    }
     for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c);
            pawn_fwd_moves_yellow_[sq_idx] = 0ULL;
            pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx] = 0ULL;
             if (r < BOARD_SIZE -1) { 
                set_bit(pawn_fwd_moves_yellow_[sq_idx], to_sq_idx(r+1, c));
                if (c > 0 && !(FILE_A_BB & (1ULL << sq_idx))) set_bit(pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx], to_sq_idx(r+1, c-1));
                if (c < BOARD_SIZE - 1 && !(FILE_H_BB & (1ULL << sq_idx))) set_bit(pawn_attacks_yellow_[static_cast<int>(Player::YELLOW)][sq_idx], to_sq_idx(r+1, c+1));
            }
        }
    }
     for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int sq_idx = to_sq_idx(r,c);
            pawn_fwd_moves_green_[sq_idx] = 0ULL;
            pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx] = 0ULL;
             if (c > 0) { 
                set_bit(pawn_fwd_moves_green_[sq_idx], to_sq_idx(r, c-1));
                if (r > 0) set_bit(pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx], to_sq_idx(r-1, c-1));
                if (r < BOARD_SIZE - 1) set_bit(pawn_attacks_green_[static_cast<int>(Player::GREEN)][sq_idx], to_sq_idx(r+1, c-1));
            }
        }
    }
    for (int r_start = 0; r_start < BOARD_SIZE; ++r_start) {
        for (int c_start = 0; c_start < BOARD_SIZE; ++c_start) {
            int start_sq = to_sq_idx(r_start, c_start);
            rook_rays_[start_sq][0] = 0ULL; rook_rays_[start_sq][1] = 0ULL; 
            rook_rays_[start_sq][2] = 0ULL; rook_rays_[start_sq][3] = 0ULL; 
            for (int r = r_start - 1; r >= 0; --r) set_bit(rook_rays_[start_sq][0], to_sq_idx(r, c_start)); 
            for (int c = c_start + 1; c < BOARD_SIZE; ++c) set_bit(rook_rays_[start_sq][1], to_sq_idx(r_start, c)); 
            for (int r = r_start + 1; r < BOARD_SIZE; ++r) set_bit(rook_rays_[start_sq][2], to_sq_idx(r, c_start)); 
            for (int c = c_start - 1; c >= 0; --c) set_bit(rook_rays_[start_sq][3], to_sq_idx(r_start, c)); 
        }
    }
     for (int r_start = 0; r_start < BOARD_SIZE; ++r_start) {
        for (int c_start = 0; c_start < BOARD_SIZE; ++c_start) {
            int start_sq = to_sq_idx(r_start, c_start);
            bishop_rays_[start_sq][0] = 0ULL; bishop_rays_[start_sq][1] = 0ULL; 
            bishop_rays_[start_sq][2] = 0ULL; bishop_rays_[start_sq][3] = 0ULL; 
            for (int i = 1; r_start - i >= 0 && c_start + i < BOARD_SIZE; ++i) set_bit(bishop_rays_[start_sq][0], to_sq_idx(r_start - i, c_start + i)); 
            for (int i = 1; r_start + i < BOARD_SIZE && c_start + i < BOARD_SIZE; ++i) set_bit(bishop_rays_[start_sq][1], to_sq_idx(r_start + i, c_start + i)); 
            for (int i = 1; r_start + i < BOARD_SIZE && c_start - i >= 0; ++i) set_bit(bishop_rays_[start_sq][2], to_sq_idx(r_start + i, c_start - i)); 
            for (int i = 1; r_start - i >= 0 && c_start - i >= 0; ++i) set_bit(bishop_rays_[start_sq][3], to_sq_idx(r_start - i, c_start - i)); 
        }
    }
}

Board::Board()
    : current_player_(Player::RED), full_move_number_(1),
      move_number_of_last_reset_(0), termination_reason_(std::nullopt) {
  for (auto &row : board_) {
    row.fill(std::nullopt);
  }
  for (auto& player_bb_array : piece_bitboards_) {
      player_bb_array.fill(0ULL);
  }
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
  for (int r = 0; r < BOARD_SIZE; ++r) {
      for (int c = 0; c < BOARD_SIZE; ++c) {
          if (board_[r][c]) {
              const Piece& piece = *board_[r][c];
              int sq_idx = to_sq_idx(r,c);
              current_hash_ ^= zobrist_data.get_piece_key(piece.piece_type, piece.player, sq_idx);
          }
      }
  }
  current_hash_ ^= zobrist_data.get_turn_key(current_player_);
  for (Player p : active_players_) {
      current_hash_ ^= zobrist_data.get_active_player_status_key(p);
  }
  position_history_.push_back(current_hash_);
}

Board::Board(const Board &other)
    : board_(other.board_), active_players_(other.active_players_),
      player_points_(other.player_points_),
      current_player_(other.current_player_),
      position_history_(other.position_history_),
      full_move_number_(other.full_move_number_),
      move_number_of_last_reset_(other.move_number_of_last_reset_),
      termination_reason_(other.termination_reason_),
      current_hash_(other.current_hash_),
      undo_stack_(other.undo_stack_),
      piece_bitboards_(other.piece_bitboards_),
      player_bitboards_(other.player_bitboards_),
      occupied_bitboard_(other.occupied_bitboard_)
       {}

Board::Board(Board &&other) noexcept
    : board_(std::move(other.board_)),
      active_players_(std::move(other.active_players_)),
      player_points_(std::move(other.player_points_)),
      current_player_(other.current_player_),
      position_history_(std::move(other.position_history_)),
      full_move_number_(other.full_move_number_),
      move_number_of_last_reset_(other.move_number_of_last_reset_),
      termination_reason_(std::move(other.termination_reason_)),
      current_hash_(other.current_hash_),
      undo_stack_(std::move(other.undo_stack_)),
      piece_bitboards_(std::move(other.piece_bitboards_)),
      player_bitboards_(std::move(other.player_bitboards_)),
      occupied_bitboard_(other.occupied_bitboard_)
       {
  other.full_move_number_ = 1; 
  other.move_number_of_last_reset_ = 0;
  other.current_hash_ = 0; 
  other.occupied_bitboard_ = 0ULL; 
  for(auto& arr : other.piece_bitboards_) arr.fill(0ULL);
  other.player_bitboards_.fill(0ULL);
}

Board &Board::operator=(const Board &other) {
  if (this != &other) { 
    board_ = other.board_;
    active_players_ = other.active_players_;
    player_points_ = other.player_points_;
    current_player_ = other.current_player_;
    position_history_ = other.position_history_;
    full_move_number_ = other.full_move_number_;
    move_number_of_last_reset_ = other.move_number_of_last_reset_;
    termination_reason_ = other.termination_reason_;
    current_hash_ = other.current_hash_;
    undo_stack_ = other.undo_stack_; 
    piece_bitboards_ = other.piece_bitboards_;
    player_bitboards_ = other.player_bitboards_;
    occupied_bitboard_ = other.occupied_bitboard_;
  }
  return *this;
}

Board &Board::operator=(Board &&other) noexcept {
  if (this != &other) { 
    board_ = std::move(other.board_);
    active_players_ = std::move(other.active_players_);
    player_points_ = std::move(other.player_points_);
    current_player_ = other.current_player_;
    position_history_ = std::move(other.position_history_);
    full_move_number_ = other.full_move_number_;
    move_number_of_last_reset_ = other.move_number_of_last_reset_;
    termination_reason_ = std::move(other.termination_reason_);
    current_hash_ = other.current_hash_;
    undo_stack_ = std::move(other.undo_stack_);
    piece_bitboards_ = std::move(other.piece_bitboards_);
    player_bitboards_ = std::move(other.player_bitboards_);
    occupied_bitboard_ = other.occupied_bitboard_;

    other.full_move_number_ = 1;
    other.move_number_of_last_reset_ = 0;
    other.current_hash_ = 0;
    other.occupied_bitboard_ = 0ULL; 
    for(auto& arr : other.piece_bitboards_) arr.fill(0ULL);
    other.player_bitboards_.fill(0ULL);
  }
  return *this;
}

// --- ADDED IMPLEMENTATION for create_mcts_child_board ---
Board Board::create_mcts_child_board(const Board& parent_board, const Move& move) {
  Board child_board; // Default constructor initializes histories/undo empty

  // Copy essential current state from parent
  child_board.board_ = parent_board.board_; // Array board
  child_board.active_players_ = parent_board.active_players_;
  child_board.player_points_ = parent_board.player_points_;
  child_board.current_player_ = parent_board.current_player_; // Player *before* the move
  child_board.full_move_number_ = parent_board.full_move_number_;
  child_board.move_number_of_last_reset_ = parent_board.move_number_of_last_reset_;
  child_board.current_hash_ = parent_board.current_hash_; // Hash *before* the move

  // Copy bitboard states
  child_board.piece_bitboards_ = parent_board.piece_bitboards_;
  child_board.player_bitboards_ = parent_board.player_bitboards_;
  child_board.occupied_bitboard_ = parent_board.occupied_bitboard_;
  
  // termination_reason_ is not copied for MCTS child

  // Apply the move to the child board's state
  // make_move will update current_player_, full_move_number_,
  // move_number_of_last_reset_, current_hash_, bitboards, board_ array,
  // and push ONE UndoInfo onto child_board.undo_stack_.
  // It also adds the new hash to child_board.position_history_.
  child_board.make_move(move);

  return child_board;
}
// --- END create_mcts_child_board ---


void Board::setup_initial_board() {
  for (auto& player_bbs : piece_bitboards_) player_bbs.fill(0ULL);
  player_bitboards_.fill(0ULL);
  occupied_bitboard_ = 0ULL;
  for (auto& row : board_) row.fill(std::nullopt);

  auto place_piece = [&](Player p, PieceType pt, int r, int c) {
      board_[r][c].emplace(p, pt);
      int sq_idx = to_sq_idx(r, c);
      int player_idx = static_cast<int>(p);
      int pt_bb_idx = piece_type_to_bb_idx(pt);
      set_bit(piece_bitboards_[player_idx][pt_bb_idx], sq_idx);
      set_bit(player_bitboards_[player_idx], sq_idx);
      set_bit(occupied_bitboard_, sq_idx);
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

bool Board::is_valid_square(int row, int col) const {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

std::vector<Move> Board::get_pseudo_legal_moves(Player player) const {
  std::vector<Move> pseudo_legal_moves;
  pseudo_legal_moves.reserve(128); 
  if (!active_players_.count(player)) {
      return pseudo_legal_moves; 
  }
  get_pawn_moves_bb(player, pseudo_legal_moves);
  get_knight_moves_bb(player, pseudo_legal_moves);
  get_bishop_moves_bb(player, pseudo_legal_moves);
  get_rook_moves_bb(player, pseudo_legal_moves);
  get_king_moves_bb(player, pseudo_legal_moves);
  return pseudo_legal_moves;
}

void Board::get_pawn_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard pawns = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::PAWN)];
    Bitboard my_pieces = player_bitboards_[p_idx];
    Bitboard opp_pieces = occupied_bitboard_ & ~my_pieces; 
    Bitboard empty_sqs = ~occupied_bitboard_;
    const Bitboard* current_fwd_moves_table = nullptr; // Use const Bitboard*
    const std::array<Bitboard, NUM_SQUARES_BB>* current_atk_table_for_player = nullptr; // Pointer to the specific player's attack table

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
        int from_sq = pop_lsb(temp_pawns);
        BoardLocation from_loc = from_sq_idx(from_sq);
        Bitboard fwd_moves = current_fwd_moves_table[from_sq] & empty_sqs;
        if (fwd_moves) { 
            int to_sq = get_lsb_index(fwd_moves); 
            BoardLocation to_loc = from_sq_idx(to_sq);
            bool is_promotion = (check_row_for_promo && to_loc.row == promotion_target_coord) ||
                                (!check_row_for_promo && to_loc.col == promotion_target_coord);
            if (is_promotion) moves.emplace_back(from_loc, to_loc, PieceType::ROOK);
            else moves.emplace_back(from_loc, to_loc);
        }
        Bitboard cap_moves = (*current_atk_table_for_player)[from_sq] & opp_pieces;
        Bitboard temp_cap_moves = cap_moves;
        while (temp_cap_moves) {
            int to_sq = pop_lsb(temp_cap_moves);
            BoardLocation to_loc = from_sq_idx(to_sq);
            bool is_promotion = (check_row_for_promo && to_loc.row == promotion_target_coord) ||
                                (!check_row_for_promo && to_loc.col == promotion_target_coord);
            if (is_promotion) moves.emplace_back(from_loc, to_loc, PieceType::ROOK);
            else moves.emplace_back(from_loc, to_loc);
        }
    }
}
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
void Board::generate_sliding_moves(Player p, int from_sq, PieceType pt, const std::vector<std::pair<int,int>>& directions, std::vector<Move>& moves) const {
    BoardLocation from_loc = from_sq_idx(from_sq);
    Bitboard my_pieces = player_bitboards_[static_cast<int>(p)];
    Bitboard opp_pieces = occupied_bitboard_ & ~my_pieces;
    for (const auto& dir_pair : directions) {
        int dr = dir_pair.first; int dc = dir_pair.second;
        int r = from_loc.row + dr; int c = from_loc.col + dc;
        while (is_valid_square(r,c)) {
            int to_sq = to_sq_idx(r,c);
            BoardLocation to_loc(r,c);
            if (get_bit(my_pieces, to_sq)) break;
            if (get_bit(opp_pieces, to_sq)) { moves.emplace_back(from_loc, to_loc); break; }
            moves.emplace_back(from_loc, to_loc);
            r += dr; c += dc;
        }
    }
}
void Board::get_rook_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard rooks = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::ROOK)];
    const std::vector<std::pair<int, int>> DIRS = {{-1,0}, {0,1}, {1,0}, {0,-1}};
    Bitboard temp_rooks = rooks;
    while(temp_rooks) {
        int from_sq = pop_lsb(temp_rooks);
        generate_sliding_moves(player, from_sq, PieceType::ROOK, DIRS, moves);
    }
}
void Board::get_bishop_moves_bb(Player player, std::vector<Move>& moves) const {
    int p_idx = static_cast<int>(player);
    Bitboard bishops = piece_bitboards_[p_idx][piece_type_to_bb_idx(PieceType::BISHOP)];
    const std::vector<std::pair<int, int>> DIRS = {{-1,1}, {1,1}, {1,-1}, {-1,-1}};
    Bitboard temp_bishops = bishops;
    while(temp_bishops) {
        int from_sq = pop_lsb(temp_bishops);
        generate_sliding_moves(player, from_sq, PieceType::BISHOP, DIRS, moves);
    }
}

std::optional<Piece> Board::make_move(const Move &move) {
  UndoInfo undo_info;
  undo_info.original_piece_bitboards = piece_bitboards_;
  undo_info.original_player_bitboards = player_bitboards_;
  undo_info.original_occupied_bitboard = occupied_bitboard_;
  undo_info.move = move;
  undo_info.original_player = current_player_;
  undo_info.original_full_move_number = full_move_number_;
  undo_info.original_move_number_of_last_reset = move_number_of_last_reset_;
  undo_info.eliminated_player = std::nullopt;
  undo_info.was_history_cleared = false;
  undo_info.previous_hash = current_hash_; 
  const auto& zobrist_data = get_zobrist_data();
  int fr = move.from_loc.row, fc = move.from_loc.col;
  int tr = move.to_loc.row, tc = move.to_loc.col;
  int from_sq_idx = Board::to_sq_idx(fr, fc);
  int to_sq_idx = Board::to_sq_idx(tr, tc);
  int moving_player_idx = static_cast<int>(current_player_);
  if (!board_[fr][fc]) {
    throw std::runtime_error("Attempting to move from an empty square in make_move.");
  }
  Piece moving_piece_obj = board_[fr][fc].value(); 
  undo_info.original_moving_piece_type = moving_piece_obj.piece_type;
  int moving_pt_bb_idx = piece_type_to_bb_idx(moving_piece_obj.piece_type);
  undo_info.captured_piece = board_[tr][tc];
  bool is_capture = undo_info.captured_piece.has_value();
  bool is_pawn_move = (moving_piece_obj.piece_type == PieceType::PAWN);
  bool is_resetting_move = is_pawn_move || is_capture;
  current_hash_ ^= zobrist_data.get_piece_key(moving_piece_obj.piece_type, moving_piece_obj.player, from_sq_idx);
  clear_bit(piece_bitboards_[moving_player_idx][moving_pt_bb_idx], from_sq_idx);
  clear_bit(player_bitboards_[moving_player_idx], from_sq_idx);
  clear_bit(occupied_bitboard_, from_sq_idx); 
  if (is_capture) {
      const Piece& captured = undo_info.captured_piece.value();
      current_hash_ ^= zobrist_data.get_piece_key(captured.piece_type, captured.player, to_sq_idx);
      int captured_player_idx = static_cast<int>(captured.player);
      int captured_pt_bb_idx = piece_type_to_bb_idx(captured.piece_type);
      clear_bit(piece_bitboards_[captured_player_idx][captured_pt_bb_idx], to_sq_idx);
      clear_bit(player_bitboards_[captured_player_idx], to_sq_idx);
  }
  board_[fr][fc] = std::nullopt; 
  PieceType final_piece_type = moving_piece_obj.piece_type;
  if (move.promotion_piece_type) {
    final_piece_type = move.promotion_piece_type.value();
  }
  board_[tr][tc] = Piece(moving_piece_obj.player, final_piece_type); 
  int final_pt_bb_idx = piece_type_to_bb_idx(final_piece_type);
  set_bit(piece_bitboards_[moving_player_idx][final_pt_bb_idx], to_sq_idx);
  set_bit(player_bitboards_[moving_player_idx], to_sq_idx);
  set_bit(occupied_bitboard_, to_sq_idx);
  if (is_capture) {
    const Piece &captured = undo_info.captured_piece.value();
    player_points_[moving_piece_obj.player] += get_piece_capture_value(captured);
    if (captured.piece_type == PieceType::KING) {
        eliminate_player(captured.player); 
        undo_info.eliminated_player = captured.player;
    }
  }
  current_hash_ ^= zobrist_data.get_piece_key(final_piece_type, moving_piece_obj.player, to_sq_idx);
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

void Board::undo_move() {
  if (undo_stack_.empty()) {
    throw std::runtime_error("No previous state available to undo.");
  }
  UndoInfo undo_info = undo_stack_.back();
  undo_stack_.pop_back();
  piece_bitboards_ = undo_info.original_piece_bitboards;
  player_bitboards_ = undo_info.original_player_bitboards;
  occupied_bitboard_ = undo_info.original_occupied_bitboard;
  current_hash_ = undo_info.previous_hash;
  current_player_ = undo_info.original_player;
  full_move_number_ = undo_info.original_full_move_number;
  move_number_of_last_reset_ = undo_info.original_move_number_of_last_reset;
  bool is_resignation_undo = (undo_info.move.from_loc.row == -1); 
  if (!is_resignation_undo) {
    if (!position_history_.empty()) {
        position_history_.pop_back();
    }
  }
  if (!is_resignation_undo) { 
    const Move &move = undo_info.move;
    int fr = move.from_loc.row; int fc = move.from_loc.col;
    int tr = move.to_loc.row; int tc = move.to_loc.col;
    board_[fr][fc] = Piece(undo_info.original_player, undo_info.original_moving_piece_type);
    board_[tr][tc] = undo_info.captured_piece;
  }
  if (undo_info.eliminated_player) {
    Player player_to_revive = *undo_info.eliminated_player;
    active_players_.insert(player_to_revive); 
  }
  if (!is_resignation_undo && undo_info.captured_piece) {
    const Piece &captured = undo_info.captured_piece.value();
    player_points_[undo_info.original_player] -= get_piece_capture_value(captured);
  }
  termination_reason_ = std::nullopt; 
}

void Board::eliminate_player(Player player) {
  if (active_players_.count(player)) {
    const auto& zobrist_data = get_zobrist_data();
    current_hash_ ^= zobrist_data.get_active_player_status_key(player);
    active_players_.erase(player);
    int p_idx = static_cast<int>(player);
    for (int pt_bb_idx = 0; pt_bb_idx < NUM_BB_PIECE_TYPES; ++pt_bb_idx) {
        occupied_bitboard_ &= ~piece_bitboards_[p_idx][pt_bb_idx]; 
        piece_bitboards_[p_idx][pt_bb_idx] = 0ULL; 
    }
    player_bitboards_[p_idx] = 0ULL; 
  }
}

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

const Board::BoardGrid &Board::get_board_grid() const { return board_; }
const Board::ActivePlayerSet &Board::get_active_players() const { return active_players_; }
const Board::PlayerPointMap &Board::get_player_points() const { return player_points_; }
Player Board::get_current_player() const { return current_player_; }
int Board::get_full_move_number() const { return full_move_number_; }
int Board::get_move_number_of_last_reset() const { return move_number_of_last_reset_; }
const std::optional<std::string> &Board::get_termination_reason() const { return termination_reason_; }
const Board::PositionHistory &Board::get_position_history() const { return position_history_; }

Player Board::get_last_active_player() const {
  if (active_players_.empty()) return Player::RED; 
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
bool Board::is_game_over() const {
  if (termination_reason_) return true; 
  if (active_players_.size() <= 1) { termination_reason_ = "elimination"; return true; }
  int moves_since_last_reset = full_move_number_ - move_number_of_last_reset_;
  if (moves_since_last_reset >= 50) {
    if (!undo_stack_.empty()) {
      Player player_who_just_moved = undo_stack_.back().original_player;
      if (player_who_just_moved == get_last_active_player()) {
        termination_reason_ = "fifty_move_rule"; return true;
      }
    }
  }
  int count = 0;
  for (const auto &key : position_history_) if (key == current_hash_) count++;
  if (count >= 3) { termination_reason_ = "threefold_repetition"; return true; }
  return false;
}
Board::PlayerPointMap Board::get_game_result() const {
  PlayerPointMap results = player_points_; 
  int num_kings_of_inactive_players = 0;

  // Iterate over all players
  for (int p_idx_loop = 0; p_idx_loop < NUM_PLAYERS_BB; ++p_idx_loop) {
      Player p_enum = static_cast<Player>(p_idx_loop);
      if (!active_players_.count(p_enum)) { // If player is inactive
          // Check their king bitboard
          // *** CORRECTED LOGIC FOR KING CHECK ***
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
        int dead_king_bonus_per_player = (num_kings_of_inactive_players > 0) ? 
            static_cast<int>(std::ceil(3.0 * num_kings_of_inactive_players / num_active_players)) : 0;
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
  auto winner_it = std::max_element(final_scores.begin(), final_scores.end(),
                       [](const auto &a, const auto &b) { return a.second < b.second; });
  return (winner_it == final_scores.end()) ? std::nullopt : std::optional<Player>(winner_it->first);
}
int Board::get_piece_value(const Piece& piece) const {
  switch (piece.piece_type) {
  case PieceType::PAWN: return 1; case PieceType::KNIGHT: return 3;
  case PieceType::BISHOP: return 5; case PieceType::ROOK: return 5;
  case PieceType::KING: return 3; default: return 0;
  }
}
int Board::get_piece_capture_value(const Piece& piece) const {
    if (!active_players_.count(piece.player)) {
        return (piece.piece_type == PieceType::KING) ? 3 : 0;
    }
    switch (piece.piece_type) {
        case PieceType::PAWN: return 1; case PieceType::KNIGHT: return 3;
        case PieceType::BISHOP: return 5; case PieceType::ROOK: return 5;
        case PieceType::KING: return 3; default: return 0;
    }
}
Board::PlayerPointMap Board::evaluate() const {
  PlayerPointMap scores;
  for (int i = 0; i < 4; ++i) scores[static_cast<Player>(i)] = 0;
  std::map<Player, BoardLocation> king_coords;
  std::map<Player, bool> king_present;
  for (int i = 0; i < 4; ++i) king_present[static_cast<Player>(i)] = false;

  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const auto &piece_opt = board_[r][c]; 
      if (piece_opt) {
        const Piece &piece = *piece_opt;
        Player player = piece.player;
        if (active_players_.count(player)) {
          scores[player] += get_piece_value(piece);
          if (piece.piece_type == PieceType::KNIGHT || piece.piece_type == PieceType::BISHOP) {
            if (((player == Player::RED && r == 7) || (player == Player::YELLOW && r == 0) ||
                 (player == Player::GREEN && c == 7) || (player == Player::BLUE && c == 0))) {
              scores[player] -= 0.4; 
            }
          }
          if (piece.piece_type == PieceType::KING) {
            king_present[player] = true;
            king_coords[player] = BoardLocation(r, c);
            // Use KING_DIRS_EVAL here
            for (const auto &dir : KING_DIRS_EVAL) { 
              int nr = r + dir.first; int nc = c + dir.second;
              if (is_valid_square(nr, nc)) {
                const auto &adjacent_opt = board_[nr][nc];
                if (adjacent_opt) {
                  if (adjacent_opt->player == player) { 
                    scores[player] += (adjacent_opt->piece_type == PieceType::PAWN ? 0.2 : 0.05);
                  } else { 
                    if (!active_players_.count(adjacent_opt->player)) { scores[player] += 0.15; } 
                    else { scores[player] -= 0.15; }
                  }
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
            if (is_valid_square(r + dr, c + dc) && board_[r + dr][c + dc]) scores[player] -= 0.2;
            for (const auto &cap_delta : {std::make_pair(cap_r1, cap_c1), std::make_pair(cap_r2, cap_c2)}) {
              int cap_r = r + cap_delta.first; int cap_c = c + cap_delta.second;
              if (is_valid_square(cap_r, cap_c) && board_[cap_r][cap_c]) {
                const auto &target = *board_[cap_r][cap_c];
                if (target.player == player) { 
                  if (target.piece_type == PieceType::BISHOP || target.piece_type == PieceType::KNIGHT) scores[player] += 0.2; 
                } else { 
                  scores[player] += 0.2;
                  if (target.piece_type == PieceType::KING && active_players_.count(target.player)) { 
                    scores[player] += 0.1; scores[target.player] -= 0.5; 
                  }
                }
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

void Board::resign() {
  Player resigning_player = current_player_; 
  if (active_players_.count(resigning_player)) {
    UndoInfo resign_undo_info;
    resign_undo_info.original_piece_bitboards = piece_bitboards_;
    resign_undo_info.original_player_bitboards = player_bitboards_;
    resign_undo_info.original_occupied_bitboard = occupied_bitboard_;
    resign_undo_info.original_player = resigning_player; 
    resign_undo_info.original_full_move_number = full_move_number_;
    resign_undo_info.original_move_number_of_last_reset = move_number_of_last_reset_;
    resign_undo_info.previous_hash = current_hash_;
    resign_undo_info.eliminated_player = resigning_player; 
    resign_undo_info.was_history_cleared = false; 
    resign_undo_info.captured_piece = std::nullopt; 
    eliminate_player(resigning_player); 
    if (active_players_.size() <= 1) {
        const auto& zobrist_data = get_zobrist_data();
        current_hash_ ^= zobrist_data.get_turn_key(resigning_player);
        is_game_over(); 
    } else {
        advance_turn(); 
    }
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
      if(active_players_.count(current_player_)) current_hash_ ^= zobrist_data.get_turn_key(current_player_); 
  }
}

const std::string ANSI_RESET_BB = "\033[0m"; 
const std::string ANSI_RED_BB = "\033[31m"; const std::string ANSI_GREEN_BB = "\033[32m";
const std::string ANSI_YELLOW_BB = "\033[33m"; const std::string ANSI_BLUE_BB = "\033[34m";
const std::string UNICODE_KING_BB = "♔"; const std::string UNICODE_ROOK_BB = "♖";
const std::string UNICODE_BISHOP_BB = "♗"; const std::string UNICODE_KNIGHT_BB = "♘";
const std::string UNICODE_PAWN_BB = "♙";

void Board::print_board() const {
  // ... (Full print_board implementation as before, using the color/unicode constants above)
  // This function primarily uses board_[r][c] so it doesn't need significant changes
  // due to bitboards, as long as board_ is kept in sync.
  // For brevity, the full symbol selection logic is not repeated here but assumed to be correct from your previous version.
  std::cout << "   a  b  c  d  e  f  g  h" << std::endl;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    std::cout << 8 - r << " ";
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const auto &piece_opt = board_[r][c];
      std::string symbol_str = " ";
      if (piece_opt) {
        const Piece &p = *piece_opt;
        bool display_as_inactive = !active_players_.count(p.player);
        const std::string& current_pawn_sym = display_as_inactive ? UNICODE_PAWN_BB : (p.player == Player::RED ? ANSI_RED_BB + UNICODE_PAWN_BB + ANSI_RESET_BB : p.player == Player::BLUE ? ANSI_BLUE_BB + UNICODE_PAWN_BB + ANSI_RESET_BB : p.player == Player::YELLOW ? ANSI_YELLOW_BB + UNICODE_PAWN_BB + ANSI_RESET_BB : ANSI_GREEN_BB + UNICODE_PAWN_BB + ANSI_RESET_BB);
        const std::string& current_knight_sym = display_as_inactive ? UNICODE_KNIGHT_BB : (p.player == Player::RED ? ANSI_RED_BB + UNICODE_KNIGHT_BB + ANSI_RESET_BB : p.player == Player::BLUE ? ANSI_BLUE_BB + UNICODE_KNIGHT_BB + ANSI_RESET_BB : p.player == Player::YELLOW ? ANSI_YELLOW_BB + UNICODE_KNIGHT_BB + ANSI_RESET_BB : ANSI_GREEN_BB + UNICODE_KNIGHT_BB + ANSI_RESET_BB);
        const std::string& current_bishop_sym = display_as_inactive ? UNICODE_BISHOP_BB : (p.player == Player::RED ? ANSI_RED_BB + UNICODE_BISHOP_BB + ANSI_RESET_BB : p.player == Player::BLUE ? ANSI_BLUE_BB + UNICODE_BISHOP_BB + ANSI_RESET_BB : p.player == Player::YELLOW ? ANSI_YELLOW_BB + UNICODE_BISHOP_BB + ANSI_RESET_BB : ANSI_GREEN_BB + UNICODE_BISHOP_BB + ANSI_RESET_BB);
        const std::string& current_rook_sym = display_as_inactive ? UNICODE_ROOK_BB : (p.player == Player::RED ? ANSI_RED_BB + UNICODE_ROOK_BB + ANSI_RESET_BB : p.player == Player::BLUE ? ANSI_BLUE_BB + UNICODE_ROOK_BB + ANSI_RESET_BB : p.player == Player::YELLOW ? ANSI_YELLOW_BB + UNICODE_ROOK_BB + ANSI_RESET_BB : ANSI_GREEN_BB + UNICODE_ROOK_BB + ANSI_RESET_BB);
        const std::string& current_king_sym = display_as_inactive ? UNICODE_KING_BB : (p.player == Player::RED ? ANSI_RED_BB + UNICODE_KING_BB + ANSI_RESET_BB : p.player == Player::BLUE ? ANSI_BLUE_BB + UNICODE_KING_BB + ANSI_RESET_BB : p.player == Player::YELLOW ? ANSI_YELLOW_BB + UNICODE_KING_BB + ANSI_RESET_BB : ANSI_GREEN_BB + UNICODE_KING_BB + ANSI_RESET_BB);
        switch (p.piece_type) {
          case PieceType::PAWN:   symbol_str = current_pawn_sym;   break;
          case PieceType::KNIGHT: symbol_str = current_knight_sym; break;
          case PieceType::BISHOP: symbol_str = current_bishop_sym; break;
          case PieceType::ROOK:   symbol_str = current_rook_sym;   break;
          case PieceType::KING:   symbol_str = current_king_sym;   break;
        }
      }
      std::cout << "[" << symbol_str << "]";
    }
    std::cout << std::endl; 
  }
  std::cout << "Turn: ";
  switch (current_player_) {
  case Player::RED:    std::cout << ANSI_RED_BB << "RED" << ANSI_RESET_BB; break;
  // ... other players
  case Player::BLUE:   std::cout << ANSI_BLUE_BB << "BLUE" << ANSI_RESET_BB; break;
  case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "YELLOW" << ANSI_RESET_BB; break;
  case Player::GREEN:  std::cout << ANSI_GREEN_BB << "GREEN" << ANSI_RESET_BB; break;
  }
  std::cout << std::endl;
  // ... print active players, points, termination reason
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
  for(const auto& pt_pair : player_points_){
      switch(pt_pair.first){
          case Player::RED: std::cout << ANSI_RED_BB << "R:" << pt_pair.second << ANSI_RESET_BB << " "; break;
          case Player::BLUE: std::cout << ANSI_BLUE_BB << "B:" << pt_pair.second << ANSI_RESET_BB << " "; break;
          case Player::YELLOW: std::cout << ANSI_YELLOW_BB << "Y:" << pt_pair.second << ANSI_RESET_BB << " "; break;
          case Player::GREEN: std::cout << ANSI_GREEN_BB << "G:" << pt_pair.second << ANSI_RESET_BB << " "; break;
      }
  }
  std::cout << std::endl;
  if(termination_reason_) std::cout << "Game Over: " << *termination_reason_ << std::endl;
  // std::cout << "Occ: " << std::hex << occupied_bitboard_ << std::dec << std::endl; // DEBUG
}
Board::PositionKey Board::get_position_key() const { return current_hash_; }

} // namespace chaturaji_cpp