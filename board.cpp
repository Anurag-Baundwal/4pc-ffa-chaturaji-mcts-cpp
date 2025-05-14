#include "board.h"
#include <algorithm> // For std::find, std::max_element
#include <array>     // For Zobrist key storage
#include <cmath>     // For std::round
#include <cstdint>   // For ZobristKey
#include <iostream>
#include <limits>  // For numeric_limits in get_last_active_player
#include <numeric> // For std::accumulate (optional)
#include <random>  // For Zobrist key generation
#include <sstream>
#include <stdexcept>
#include <utility> // For std::move
#include <vector>

namespace chaturaji_cpp {

namespace { // Anonymous namespace to limit scope to this file

const int NUM_PIECE_TYPES_FOR_HASH = 5; // P, N, B, R, K
const int NUM_PLAYERS = 4;
const int NUM_SQUARES = 64;

struct ZobristData {
  std::array<std::array<std::array<ZobristKey, NUM_SQUARES>, NUM_PLAYERS>, NUM_PIECE_TYPES_FOR_HASH> piece_keys; //
  std::array<ZobristKey, NUM_PLAYERS> turn_keys;
  std::array<ZobristKey, NUM_PLAYERS> active_player_status_keys;

    
  // Add keys for castling, en passant if needed in a different game

  ZobristData() {
    // Use a high-quality random number generator
    std::mt19937_64 rng(0xBADFACE); // Fixed seed for reproducibility
    std::uniform_int_distribution<ZobristKey> dist(0, std::numeric_limits<ZobristKey>::max());

    // Generate keys for each piece type, player, and square
    for (int type_idx = 0; type_idx < NUM_PIECE_TYPES_FOR_HASH; ++type_idx) {
      for (int player_idx = 0; player_idx < NUM_PLAYERS; ++player_idx) {
          for (int sq_idx = 0; sq_idx < NUM_SQUARES; ++sq_idx) {
            piece_keys[type_idx][player_idx][sq_idx] = dist(rng);
          }
      }
    }

    // Generate keys for whose turn it is
    for (int player_idx = 0; player_idx < NUM_PLAYERS; ++player_idx) {
      turn_keys[player_idx] = dist(rng);
    }

    // Generate keys for player active status
    for (int player_idx = 0; player_idx < NUM_PLAYERS; ++player_idx) {
        active_player_status_keys[player_idx] = dist(rng);
    }
  }

  // Helper to get piece key safely, mapping PieceType to array index
  ZobristKey get_piece_key(PieceType type, Player player, int square_index) const {
    if (square_index < 0 || square_index >= NUM_SQUARES) {
      throw std::out_of_range(
          "Square index out of range for Zobrist key lookup.");
    }
    int type_idx = static_cast<int>(type) - 1; // PieceType is PAWN=1 ... KING=5. Map to 0-4.
    if (type_idx < 0 || type_idx >= NUM_PIECE_TYPES_FOR_HASH) {
      throw std::out_of_range("PieceType out of range for Zobrist key lookup.");
    }
    int player_idx = static_cast<int>(player); // Player enum 0..3 maps directly
    if (player_idx < 0 || player_idx >= NUM_PLAYERS) {
      throw std::out_of_range("Player out of range for Zobrist key lookup.");
    }
    return piece_keys[type_idx][player_idx][square_index]; 
    }

  // Helper to get turn key safely
  ZobristKey get_turn_key(Player player) const {
    int player_idx = static_cast<int>(player);
    if (player_idx < 0 || player_idx >= NUM_PLAYERS) {
      throw std::out_of_range("Player out of range for Zobrist key lookup.");
    }
    return turn_keys[player_idx];
  }
  ZobristKey get_active_player_status_key(Player player) const {
    int player_idx = static_cast<int>(player);
    if (player_idx < 0 || player_idx >= NUM_PLAYERS) {
        throw std::out_of_range("Player out of range for Zobrist active status key lookup.");
    }
    return active_player_status_keys[player_idx];
  }
};

// Meyers' Singleton: Ensures safe initialization on first access
const ZobristData &get_zobrist_data() {
  static const ZobristData instance; // Initialized only once
  return instance;
}

} // end anonymous namespace

// --- Static Data ---
// Helper for iterating directions (used in Bishop, Rook, King moves)
const std::vector<std::pair<int, int>> BISHOP_DIRS = {
    {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
const std::vector<std::pair<int, int>> ROOK_DIRS = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
const std::vector<std::pair<int, int>> KING_DIRS = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
const std::vector<std::pair<int, int>> KNIGHT_MOVES = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};

// Promotion rows/cols
const int PROMOTION_ROW_RED = 0;
const int PROMOTION_COL_BLUE = 7;
const int PROMOTION_ROW_YELLOW = 7;
const int PROMOTION_COL_GREEN = 0;

// --- Constructor ---
Board::Board()
    : current_player_(Player::RED), full_move_number_(1),
      move_number_of_last_reset_(0), termination_reason_(std::nullopt) {
  // Initialize board_ to all nullopt (empty squares)
  for (auto &row : board_) {
    row.fill(std::nullopt);
  }
  // Initialize player points and active players
  for (int i = 0; i < 4; ++i) {
    Player p = static_cast<Player>(i);
    player_points_[p] = 0;
    active_players_.insert(p);
  }
  setup_initial_board();

  // --- Calculate Initial Zobrist Hash --- 
  const auto& zobrist_data = get_zobrist_data();
  current_hash_ = 0; // Start fresh

  // Hash pieces
  for (int r = 0; r < BOARD_SIZE; ++r) {
      for (int c = 0; c < BOARD_SIZE; ++c) {
          if (board_[r][c]) {
              const Piece& piece = *board_[r][c];
              int sq_idx = r * BOARD_SIZE + c;
              current_hash_ ^= zobrist_data.get_piece_key(piece.piece_type, piece.player, sq_idx);
          }
      }
  }

  // Hash current player's turn
  current_hash_ ^= zobrist_data.get_turn_key(current_player_);
  
  // Hash active player statuses
  for (Player p : active_players_) { // active_players_ is initialized with all players
      current_hash_ ^= zobrist_data.get_active_player_status_key(p);
  }
  // --- END Initial Zobrist Hash Calculation ---

  // Add initial position hash to history
  position_history_.push_back(current_hash_);
}

// --- Copy Constructor ---
Board::Board(const Board &other)
    : board_(other.board_), active_players_(other.active_players_),
      player_points_(other.player_points_),
      current_player_(other.current_player_),
      position_history_(other.position_history_),
      full_move_number_(other.full_move_number_),
      move_number_of_last_reset_(other.move_number_of_last_reset_),
      termination_reason_(other.termination_reason_),
      current_hash_(other.current_hash_),
      undo_stack_(other.undo_stack_) {}

// --- Move Constructor ---
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
      undo_stack_(std::move(other.undo_stack_)) {
  // Reset other state after move if necessary
  other.full_move_number_ = 1; // Reset moved-from object
  other.move_number_of_last_reset_ = 0;
  other.current_hash_ = 0; // RESET moved-from hash
}

// --- Copy Assignment Operator ---
Board &Board::operator=(const Board &other) {
  if (this != &other) { // Self-assignment check
    board_ = other.board_;
    active_players_ = other.active_players_;
    player_points_ = other.player_points_;
    current_player_ = other.current_player_;
    position_history_ = other.position_history_;
    full_move_number_ = other.full_move_number_;
    move_number_of_last_reset_ = other.move_number_of_last_reset_;
    termination_reason_ = other.termination_reason_;
    current_hash_ = other.current_hash_;
    undo_stack_ = other.undo_stack_; // Deep copy handled by stack's op=
  }
  return *this;
}

// --- Move Assignment Operator ---
Board &Board::operator=(Board &&other) noexcept {
  if (this != &other) { // Self-assignment check
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

    // Reset other state if necessary
    other.full_move_number_ = 1;
    other.move_number_of_last_reset_ = 0;
    other.current_hash_ = 0;
  }
  return *this;
}

Board Board::create_mcts_child_board(const Board& parent_board, const Move& move) {
  // 1. Create a new board object. Default constructor ensures vectors are empty.
  Board child_board; // This board starts "clean" in terms of history/undo

  // 2. Copy essential state from the parent
  child_board.board_ = parent_board.board_; // Deep copy of the grid (fixed size, fast)
  child_board.active_players_ = parent_board.active_players_; // Copy the set
  child_board.player_points_ = parent_board.player_points_; // Copy the map
  child_board.current_player_ = parent_board.current_player_; // Player *before* the move
  child_board.full_move_number_ = parent_board.full_move_number_;
  child_board.move_number_of_last_reset_ = parent_board.move_number_of_last_reset_;
  child_board.current_hash_ = parent_board.current_hash_; // Hash *before* the move

  // termination_reason_ and game_history_ (if kept) are not copied for MCTS nodes

  // 3. Apply the move to the newly copied state
  // make_move will update current_player_, full_move_number_,
  // move_number_of_last_reset_, current_hash_, and push ONE UndoInfo
  // onto child_board.undo_stack_ (which was empty).
  // It also adds the new hash to child_board.position_history_ (which was empty).
  child_board.make_move(move);

  // The state of child_board is now the state *after* 'move' was applied,
  // but its history/undo stack only contains information about *that single move*
  // (and the resulting hash in position_history).

  return child_board; // Return the newly created and updated board state
}

// --- Helper to find the last player in sequence ---
Player Board::get_last_active_player() const {
  if (active_players_.empty()) {
    // This case should ideally not happen in a valid game state
    // Return a default or throw, depending on desired error handling
    // Returning RED as a placeholder, but behavior is undefined.
    // Consider throwing std::runtime_error("No active players found.");
    return Player::RED; // Or throw
  }
  // Find the player with the maximum enum value (GREEN > YELLOW > BLUE > RED)
  Player last_player = Player::RED; // Start with the lowest
  int max_val = -1;
  for (Player p : active_players_) {
    if (static_cast<int>(p) > max_val) {
      max_val = static_cast<int>(p);
      last_player = p;
    }
  }
  return last_player;
}

// --- Core Game Logic ---

void Board::setup_initial_board() {
  // Clear board first (optional, constructor already does)
  // for (auto& row : board_) row.fill(std::nullopt);

  // Red pieces
  board_[7][0].emplace(Player::RED, PieceType::ROOK);
  board_[7][1].emplace(Player::RED, PieceType::KNIGHT);
  board_[7][2].emplace(Player::RED, PieceType::BISHOP);
  board_[7][3].emplace(Player::RED, PieceType::KING);
  for (int col = 0; col < 4; ++col) {
    board_[6][col].emplace(Player::RED, PieceType::PAWN);
  }

  // Blue pieces
  board_[0][0].emplace(Player::BLUE, PieceType::ROOK);
  board_[1][0].emplace(Player::BLUE, PieceType::KNIGHT);
  board_[2][0].emplace(Player::BLUE, PieceType::BISHOP);
  board_[3][0].emplace(Player::BLUE, PieceType::KING);
  for (int row = 0; row < 4; ++row) {
    board_[row][1].emplace(Player::BLUE, PieceType::PAWN);
  }

  // Yellow pieces
  board_[0][7].emplace(Player::YELLOW, PieceType::ROOK);
  board_[0][6].emplace(Player::YELLOW, PieceType::KNIGHT);
  board_[0][5].emplace(Player::YELLOW, PieceType::BISHOP);
  board_[0][4].emplace(Player::YELLOW, PieceType::KING);
  for (int col = 4; col < 8; ++col) {
    board_[1][col].emplace(Player::YELLOW, PieceType::PAWN);
  }

  // Green pieces
  board_[4][7].emplace(Player::GREEN, PieceType::KING);
  board_[5][7].emplace(Player::GREEN, PieceType::BISHOP);
  board_[6][7].emplace(Player::GREEN, PieceType::KNIGHT);
  board_[7][7].emplace(Player::GREEN, PieceType::ROOK);
  for (int row = 4; row < 8; ++row) {
    board_[row][6].emplace(Player::GREEN, PieceType::PAWN);
  }
}

bool Board::is_valid_square(int row, int col) const {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

std::vector<Move> Board::get_pseudo_legal_moves(Player player) const {
  std::vector<Move> pseudo_legal_moves;
  pseudo_legal_moves.reserve(64); // Pre-allocate some space
  // Ensure the player for whom moves are generated is active.
  // This check should ideally be done by the caller (e.g., MCTS ensuring current_player_ is active).
  // If we assume 'player' is current_player_ and current_player_ is always active if game not over.
  if (!active_players_.count(player)) {
      return pseudo_legal_moves; // Return empty if player is not active
  }

  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const auto &piece_opt = board_[r][c];
      if (piece_opt && piece_opt->player == player) {
        switch (piece_opt->piece_type) {
        case PieceType::PAWN:
          get_pawn_moves(r, c, pseudo_legal_moves);
          break;
        case PieceType::KNIGHT:
          get_knight_moves(r, c, pseudo_legal_moves);
          break;
        case PieceType::BISHOP:
          get_bishop_moves(r, c, pseudo_legal_moves);
          break;
        case PieceType::ROOK:
          get_rook_moves(r, c, pseudo_legal_moves);
          break;
        case PieceType::KING:
          get_king_moves(r, c, pseudo_legal_moves);
          break;
        default:
          break; // Should not happen for valid types
        }
      }
    }
  }
  return pseudo_legal_moves;
}

// --- Private Move Generation Helpers ---

void Board::get_pawn_moves(int row, int col, std::vector<Move> &moves) const {
  const auto &moving_piece = board_[row][col]; // Should always have value here
  if (!moving_piece) return;                   // Safety check
  Player player = moving_piece->player;
  BoardLocation from(row, col);

  int dr = 0, dc = 0;              // Direction of forward move
  int promotion_target_coord = -1; // Row or Col for promotion
  bool check_row_for_promo = false;

  // Determine direction and promotion target based on player
  switch (player) {
  case Player::RED:
    dr = -1;
    dc = 0;
    promotion_target_coord = PROMOTION_ROW_RED;
    check_row_for_promo = true;
    break;
  case Player::BLUE:
    dr = 0;
    dc = 1;
    promotion_target_coord = PROMOTION_COL_BLUE;
    check_row_for_promo = false;
    break;
  case Player::YELLOW:
    dr = 1;
    dc = 0;
    promotion_target_coord = PROMOTION_ROW_YELLOW;
    check_row_for_promo = true;
    break;
  case Player::GREEN:
    dr = 0;
    dc = -1;
    promotion_target_coord = PROMOTION_COL_GREEN;
    check_row_for_promo = false;
    break;
  }

  // 1. Forward move (non-capture)
  int to_row_fwd = row + dr;
  int to_col_fwd = col + dc;
  if (is_valid_square(to_row_fwd, to_col_fwd) &&
      !board_[to_row_fwd][to_col_fwd]) {
    BoardLocation to(to_row_fwd, to_col_fwd);
    bool is_promotion =
        (check_row_for_promo && to_row_fwd == promotion_target_coord) ||
        (!check_row_for_promo && to_col_fwd == promotion_target_coord);
    if (is_promotion) {
      // Only Rook promotion is specified in the Python code
      moves.emplace_back(from, to, PieceType::ROOK);
      // Add other promotions here if needed (e.g., KNIGHT, BISHOP)
      // moves.emplace_back(from, to, PieceType::KNIGHT);
      // moves.emplace_back(from, to, PieceType::BISHOP);
    } else {
      moves.emplace_back(from, to);
    }
  }

  // 2. Capture moves
  int cap_dr1 = 0, cap_dc1 = 0; // Capture direction 1 delta
  int cap_dr2 = 0, cap_dc2 = 0; // Capture direction 2 delta

  switch (player) {
  case Player::RED:
    cap_dr1 = -1;
    cap_dc1 = -1;
    cap_dr2 = -1;
    cap_dc2 = 1;
    break;
  case Player::BLUE:
    cap_dr1 = -1;
    cap_dc1 = 1;
    cap_dr2 = 1;
    cap_dc2 = 1;
    break;
  case Player::YELLOW:
    cap_dr1 = 1;
    cap_dc1 = -1;
    cap_dr2 = 1;
    cap_dc2 = 1;
    break;
  case Player::GREEN:
    cap_dr1 = -1;
    cap_dc1 = -1;
    cap_dr2 = 1;
    cap_dc2 = -1;
    break;
  }

  for (const auto &cap_delta :
       {std::make_pair(cap_dr1, cap_dc1), std::make_pair(cap_dr2, cap_dc2)}) {
    int to_row_cap = row + cap_delta.first;
    int to_col_cap = col + cap_delta.second;

    if (is_valid_square(to_row_cap, to_col_cap)) {
      const auto &target_piece = board_[to_row_cap][to_col_cap];
      if (target_piece &&
          target_piece->player != player) { // Capture requires opponent piece
        BoardLocation to(to_row_cap, to_col_cap);
        bool is_promotion =
            (check_row_for_promo && to_row_cap == promotion_target_coord) ||
            (!check_row_for_promo && to_col_cap == promotion_target_coord);
        if (is_promotion) {
          moves.emplace_back(from, to, PieceType::ROOK);
          // Add other promotions here if needed
          // moves.emplace_back(from, to, PieceType::KNIGHT);
          // moves.emplace_back(from, to, PieceType::BISHOP);
        } else {
          moves.emplace_back(from, to);
        }
      }
    }
  }
}

void Board::get_knight_moves(int row, int col, std::vector<Move> &moves) const {
  const auto &moving_piece = board_[row][col];
  if (!moving_piece) return;
  Player player = moving_piece->player;
  BoardLocation from(row, col);

  for (const auto &d : KNIGHT_MOVES) {
    int r = row + d.first;
    int c = col + d.second;
    if (is_valid_square(r, c)) {
      const auto &target = board_[r][c];
      if (!target || target->player != player) { // Empty or opponent piece
        moves.emplace_back(from, BoardLocation(r, c));
      }
    }
  }
}

void Board::get_bishop_moves(int row, int col, std::vector<Move> &moves) const {
  const auto &moving_piece = board_[row][col];
  if (!moving_piece) return;
  Player player = moving_piece->player;
  BoardLocation from(row, col);

  for (const auto &dir : BISHOP_DIRS) {
    int dr = dir.first;
    int dc = dir.second;
    int r = row + dr;
    int c = col + dc;
    while (is_valid_square(r, c)) {
      const auto &target = board_[r][c];
      if (!target) { // Empty square
        moves.emplace_back(from, BoardLocation(r, c));
      } else {                          // Piece encountered
        if (target->player != player) { // Opponent piece
          moves.emplace_back(from, BoardLocation(r, c));
        }
        break; // Stop searching in this direction (blocked)
      }
      r += dr;
      c += dc;
    }
  }
}

void Board::get_rook_moves(int row, int col, std::vector<Move> &moves) const {
  const auto &moving_piece = board_[row][col];
  if (!moving_piece) return;
  Player player = moving_piece->player;
  BoardLocation from(row, col);

  for (const auto &dir : ROOK_DIRS) {
    int dr = dir.first;
    int dc = dir.second;
    int r = row + dr;
    int c = col + dc;
    while (is_valid_square(r, c)) {
      const auto &target = board_[r][c];
      if (!target) { // Empty square
        moves.emplace_back(from, BoardLocation(r, c));
      } else {                          // Piece encountered
        if (target->player != player) { // Opponent piece
          moves.emplace_back(from, BoardLocation(r, c));
        }
        break; // Stop searching in this direction (blocked)
      }
      r += dr;
      c += dc;
    }
  }
}

void Board::get_king_moves(int row, int col, std::vector<Move> &moves) const {
  const auto &moving_piece = board_[row][col];
  if (!moving_piece) return;
  Player player = moving_piece->player;
  BoardLocation from(row, col);

  for (const auto &dir : KING_DIRS) {
    int r = row + dir.first;
    int c = col + dir.second;
    if (is_valid_square(r, c)) {
      const auto &target = board_[r][c];
      if (!target || target->player != player) { // Empty or opponent piece
        moves.emplace_back(from, BoardLocation(r, c));
      }
    }
  }
}

// --- Move Execution ---

std::optional<Piece> Board::make_move(const Move &move) {
  // --- Setup ---
  UndoInfo undo_info;
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
  int from_sq_idx = fr * BOARD_SIZE + fc;
  int to_sq_idx = tr * BOARD_SIZE + tc;

  // --- Validate Moving Piece ---
  if (!board_[fr][fc]) {
    throw std::runtime_error("Attempting to move from an empty square in make_move.");
  }
  Piece moving_piece = board_[fr][fc].value(); // Make a copy to potentially modify
  undo_info.original_moving_piece_type = moving_piece.piece_type;

  // --- Store Captured Piece Info ---
  undo_info.captured_piece = board_[tr][tc];
  bool is_capture = undo_info.captured_piece.has_value();
  bool is_pawn_move = (moving_piece.piece_type == PieceType::PAWN);
  bool is_resetting_move = is_pawn_move || is_capture;

  // --- ZOBRIST UPDATE: Part 1 (Remove pieces/turn from old state) ---
  // 1a. XOR out the moving piece from its original square. It's always alive here.
  current_hash_ ^= zobrist_data.get_piece_key(moving_piece.piece_type, moving_piece.player, from_sq_idx);

  // 1b. XOR out the captured piece (if any) from the destination square.
  if (is_capture) {
      const Piece& captured = undo_info.captured_piece.value();
      current_hash_ ^= zobrist_data.get_piece_key(captured.piece_type, captured.player, to_sq_idx);
  }
  // 1c. Turn XOR is handled by advance_turn later.
  // --- END ZOBRIST UPDATE Part 1 ---

  // --- Perform Board Changes ---
  board_[fr][fc] = std::nullopt; // Clear the 'from' square

  // Handle Promotion (Modify the *copy* before placing it)
  if (move.promotion_piece_type) {
    moving_piece.piece_type = move.promotion_piece_type.value(); // Update type on the copy
  }

  // Place the (potentially promoted) piece on the 'to' square
  board_[tr][tc] = moving_piece;
  // Get a reference *to the piece now on the board*
  Piece& final_piece_at_to = board_[tr][tc].value();

  // --- Handle Captures & Elimination ---
  if (is_capture) {
    const Piece &captured = undo_info.captured_piece.value();
      player_points_[moving_piece.player] += get_piece_capture_value(captured);

      // If a King was captured, eliminate the player
      if (captured.piece_type == PieceType::KING) {
        eliminate_player(captured.player);
        undo_info.eliminated_player = captured.player;
      }
    }

  // --- ZOBRIST UPDATE: Part 2 (Add piece to new state) ---
  // 2a. XOR in the final piece (potentially promoted) at its destination square. It's always alive here.
  current_hash_ ^= zobrist_data.get_piece_key(final_piece_at_to.piece_type, final_piece_at_to.player, to_sq_idx);
  // 2b. Turn XOR is handled by advance_turn later.
  // --- END ZOBRIST UPDATE Part 2 ---

  // --- Update Game State Counters & History ---
  Player player_who_moved = current_player_;
  Player last_active_player_in_sequence = get_last_active_player();
  bool was_last_player_turn = (player_who_moved == last_active_player_in_sequence);

  if (was_last_player_turn) {
    full_move_number_++;
  }

  if (is_resetting_move) {
    move_number_of_last_reset_ = full_move_number_;
    position_history_.clear(); // Reset repetition history
    undo_info.was_history_cleared = true;
  } else {
    undo_info.was_history_cleared = false; // Explicitly set to false
  }

  // --- Final Steps ---
  undo_stack_.push_back(undo_info); // Push undo info including previous hash
  advance_turn();                   // Advances turn and updates hash for player change

  
  // --- NOW Push the hash of the *resulting* state (including the next player's turn) ---
  // This is the state the repetition check needs to look for.
  position_history_.push_back(get_position_key()); // get_position_key() returns current_hash_

  
  // Check for game over *after* move is fully made and turn advanced
  // Note: is_game_over will check the *current* hash against the history *just added*.
  // This might be slightly off compared to some rulesets that check *before* adding the latest hash.
  // However, for a simple back-and-forth repetition, this should work as the state will appear 3 times *in* the history including the current state.
  // If a rule requires checking history *excluding* the current state, this logic would need adjustment.
  // Let's stick to the simpler "count occurrences in history including current state" interpretation for now.
  is_game_over(); // Call to update termination_reason_ if needed
  return undo_info.captured_piece;
}

void Board::undo_move() {
  if (undo_stack_.empty()) {
    throw std::runtime_error("No previous state available to undo.");
  }

  // --- Pop Undo Information ---
  UndoInfo undo_info = undo_stack_.back();
  undo_stack_.pop_back();

  // --- 5. Restore Zobrist Hash First ---
  // This sets the hash to the exact value it had *before* the move was made (including the old player's turn).
  current_hash_ = undo_info.previous_hash;

  // --- 1. Restore Player Turn, Game State Counters, and Histories ---
  // These are independent of the board state itself and restored first.
  current_player_ = undo_info.original_player;
  full_move_number_ = undo_info.original_full_move_number;
  move_number_of_last_reset_ = undo_info.original_move_number_of_last_reset;

  // History restoration (only pop if it wasn't a resignation or clearing move)
  // We need a way to distinguish resignation undo_info. Maybe check if move is default?
  // Let's assume resignation undo_info has default Move (from/to = -1,-1).
  bool is_resignation_undo = (undo_info.move.from_loc.row == -1); // Heuristic
  if (!is_resignation_undo) {
    if (!position_history_.empty()) {
        position_history_.pop_back();
    }
  }

  // --- 2. Reverse Board Piece Changes (ONLY for regular moves) ---
  if (!is_resignation_undo) { // Check if it's NOT a resignation undo
    const Move &move = undo_info.move;
    int fr = move.from_loc.row;
    int fc = move.from_loc.col;
    int tr = move.to_loc.row;
    int tc = move.to_loc.col;

    // This part should only execute for actual moves.
    if (board_[tr][tc]) { // Check if 'to' square has a piece (it should for a move)
        Piece piece_that_moved = board_[tr][tc].value();
        piece_that_moved.piece_type = undo_info.original_moving_piece_type;
        board_[fr][fc] = piece_that_moved;
        board_[tr][tc] = undo_info.captured_piece;
    } else {
        // This might indicate an issue if we expected a piece on the 'to' square
        // For now, just handle the potential nullopt on 'to' gracefully.
         if (undo_info.captured_piece) { // If there *was* a capture
              board_[tr][tc] = undo_info.captured_piece; // Put captured piece back
               board_[fr][fc] = Piece(undo_info.original_player, undo_info.original_moving_piece_type); // Reconstruct moving piece? Needs care. Best to ensure board_[tr][tc] isn't nullopt above. Let's refine.

               // Safer approach: Assume tr/tc must contain the moved piece
               throw std::runtime_error("Undo error: Expected moved piece on target square, found none.");
          } else {
              // No capture, moving piece wasn't on target? Error.
               throw std::runtime_error("Undo error: Expected moved piece on target square for non-capture.");
          }
          // Let's simplify assuming board_[tr][tc] always holds the piece after a valid move.
          // The initial check 'if (board_[tr][tc])' should be sufficient if make_move is correct.
          // Error if it's nullopt:
          // throw std::runtime_error("Undo error: Target square is unexpectedly empty.");
    }
  }
  // If it *is* a resignation undo, we skip piece movement reversal.

  // --- 3. Reverse Elimination (Restore Player and Piece States) ---
  // This must happen *before* hash restoration, as it changes the board state
  // that the restored hash needs to match.
  if (undo_info.eliminated_player) {
    Player player_to_revive = *undo_info.eliminated_player;
    active_players_.insert(player_to_revive); // Add player back to active set
    // ZOBRIST: XOR in the active status key for the revived player
    // current_hash_ ^= get_zobrist_data().get_active_player_status_key(player_to_revive);

    // No explicit hash changes needed here; the hash restoration below handles it.
  }

  // --- 4. Reverse Point Changes (Only for regular (non-resignation) moves) ---
  if (!is_resignation_undo && undo_info.captured_piece) {
    const Piece &captured = undo_info.captured_piece.value();

    // Subtract points from the player who made the original move
    player_points_[undo_info.original_player] -= get_piece_capture_value(captured);
  }

  // --- 6. Clear Termination Reason ---
  termination_reason_ = std::nullopt; // State may no longer be terminal
}

// --- advance_turn() Correctly handles turn XORing ---
void Board::advance_turn() {
  const auto& zobrist_data = get_zobrist_data();
  Player old_player = current_player_;

  int next_player_val = (static_cast<int>(current_player_) + 1) % 4;
  current_player_ = static_cast<Player>(next_player_val);

  // Skip eliminated players
  while (active_players_.find(current_player_) == active_players_.end()) {
    if (active_players_.size() <= 1) break; // Avoid infinite loop if only 0 or 1 players left
    next_player_val = (static_cast<int>(current_player_) + 1) % 4;
    current_player_ = static_cast<Player>(next_player_val);
  }

  // Update hash for turn change, carefully handling game end states
  if (active_players_.size() > 0) { // Only update turn hash if the game isn't completely empty
      current_hash_ ^= zobrist_data.get_turn_key(old_player); // XOR out old player
      // Only XOR in new player if they are actually active (handles game ending exactly on turn change)
      if(active_players_.count(current_player_)){
         current_hash_ ^= zobrist_data.get_turn_key(current_player_); // XOR in new player
      }
  }
  // Note: If game ends because the last player was eliminated, old_player was XORed out
  // by eliminate_player, and no new player is XORed in here. Correct.
}

// --- Game State Accessors --- (Implement getters)
const Board::BoardGrid &Board::get_board_grid() const { return board_; }
const Board::ActivePlayerSet &Board::get_active_players() const {
  return active_players_;
}
const Board::PlayerPointMap &Board::get_player_points() const {
  return player_points_;
}
Player Board::get_current_player() const { return current_player_; }
int Board::get_full_move_number() const { return full_move_number_; }
int Board::get_move_number_of_last_reset() const {
  return move_number_of_last_reset_;
}

const std::optional<std::string> &Board::get_termination_reason() const {
  return termination_reason_;
}
const Board::PositionHistory &Board::get_position_history() const {
  return position_history_;
}

// --- Game Status ---

bool Board::is_game_over() const {
  if (termination_reason_) return true; // Already determined

  if (active_players_.size() <= 1) {
    termination_reason_ = "elimination";
    return true;
  }

  // --- NEW 50-Move Rule Check ---
  int moves_since_last_reset = full_move_number_ - move_number_of_last_reset_;

  if (moves_since_last_reset >= 50) {
    // Rule check triggers only if the 50th move was just completed by the last
    // player in sequence. We need the player who *made* the move that
    // potentially triggers the check. This state is available in the last entry
    // of the undo stack.
    if (!undo_stack_.empty()) {
      Player player_who_just_moved = undo_stack_.back().original_player;
      Player last_active_player =
          get_last_active_player(); // Check against current active players

      if (player_who_just_moved == last_active_player) {
        termination_reason_ = "fifty_move_rule";
        return true;
      }
    } else {
      // Cannot check who moved if undo stack is empty (e.g., checking before
      // first move?) This shouldn't happen if called after make_move.
    }
  }
  // --- End 50-Move Rule Check ---

  // --- Threefold Repetition Check ---
  // Note: The current hash (current_key) is *already* in the history
  // because it was added at the end of make_move *before* is_game_over might be called.
  // So we need to find >= 3 occurrences.
  PositionKey current_key = current_hash_; // Get the current hash value
  int count = 0;
  for (const auto &key : position_history_) {
    if (key == current_key) {
      count++;
    }
  }
  // Standard rule: third occurrence triggers draw
  if (count >= 3) {
    termination_reason_ = "threefold_repetition";
    return true;
  }
  // --- End Threefold Repetition Check ---

  return false;
}

Board::PlayerPointMap Board::get_game_result() const {
  PlayerPointMap results = player_points_; // Start with current capture points

  // --- Count Kings of INACTIVE players ---
  int num_kings_of_inactive_players = 0;
  for (const auto &row : board_) {
    for (const auto &piece_opt : row) {
      if (piece_opt && piece_opt->piece_type == PieceType::KING && 
      !active_players_.count(piece_opt->player)) { // Check if King's owner is inactive
        num_kings_of_inactive_players++;
      }
    }
  }

  int num_active_players = active_players_.size();

  // --- Apply Bonuses based on Termination Reason ---
  if (termination_reason_) { // Check if the game is actually over and reason is
                             // set
    const std::string &reason = *termination_reason_;

    // --- Draw Scenarios (50-move or 3-fold) ---
    if (reason == "fifty_move_rule" || reason == "threefold_repetition") {
      if (num_active_players >
          0) { // Should always be > 0 for a draw, but safety check
        // Calculate how many points each active player gets based on the number of kings of inactive players
        // Basically, kings are worth 3 points each, and if the player they belong to is inactive, other players get those points at the end of the game
        int dead_king_bonus_per_player = 0;
        if (num_kings_of_inactive_players > 0) {
          // Use floating-point division before ceiling
          dead_king_bonus_per_player = static_cast<int>(
              std::ceil(3.0 * num_kings_of_inactive_players / num_active_players));
        }

        // Apply base draw bonus (+2) and inactive player king bonus to each active player
        for (Player p : active_players_) {
          results[p] += 2;                          // Base +2 bonus for draw
          results[p] += dead_king_bonus_per_player; // Add dead king bonus
        }
      }
    }
    // --- Elimination Scenario (Last Man Standing) ---
    else if (reason == "elimination") {
      // This condition implies num_active_players should be 1 (or 0 if somehow
      // everyone resigns simultaneously?) We apply the bonus only if there's
      // exactly one winner left.
      if (num_active_players == 1 && num_kings_of_inactive_players > 0) {
        Player winner = *active_players_.begin(); // Get the single remaining player
        int dead_king_bonus =
            3 * num_kings_of_inactive_players; // Simpler calculation for 1 active player
        results[winner] += dead_king_bonus;
      }
      // No base bonus for elimination, only the potential dead king bonus for
      // the winner.
    }
    // --- Other termination reasons (e.g., resignation leading to elimination)
    // --- If a player resigns and causes elimination, the "elimination" logic
    // above handles the bonus for the winner. If resignation leads to a draw
    // scenario (e.g., only 2 players left, one resigns, other cannot force
    // win?), the termination reason might need specific handling or might
    // default to 'elimination' if it leaves one player. The current logic
    // assumes 'elimination' is set correctly when only one player remains.
  }
  // If termination_reason_ is not set, just return the base capture points.

  return results;
}

std::optional<Player> Board::get_winner() const {
  if (!termination_reason_) { // Game must be over
    // Optionally call is_game_over() here to ensure termination state is set
    // if (!const_cast<Board*>(this)->is_game_over()) return std::nullopt;
    // The above const_cast is ugly. Better to ensure is_game_over() is called
    // before get_winner().
    return std::nullopt;
  }

  PlayerPointMap final_scores = get_game_result();

  // Find player with max score
  auto winner_it =
      std::max_element(final_scores.begin(), final_scores.end(),
                       [](const auto &a, const auto &b) {
                         return a.second < b.second; // Compare by points
                       });

  if (winner_it == final_scores.end()) {
    return std::nullopt; // Should not happen if there are players
  }

  // Check for ties (if multiple players have the same max score)
  // For now, return the first one found with max score, consistent with
  // Python's sorted list approach.
  return winner_it->first;
}

// --- Evaluation ---

int Board::get_piece_value(const Piece& piece) const {
  switch (piece.piece_type) {
  case PieceType::PAWN: return 1;
  case PieceType::KNIGHT: return 3;
  case PieceType::BISHOP: return 5;
  case PieceType::ROOK: return 5;
  case PieceType::KING: return 3; 
  default: return 0;
  }
}

int Board::get_piece_capture_value(const Piece& piece) const {
    // Check if the captured piece's owner is active
    if (!active_players_.count(piece.player)) {
        // Owner is inactive
        if (piece.piece_type == PieceType::KING) {
            return 3; // Capturing a King of an inactive player (simulates old DEAD_KING capture value)
        }
        return 0; // Other pieces of inactive players are worth 0
    }
    // Owner is active, use standard values
    switch (piece.piece_type) {
        case PieceType::PAWN: return 1;
        case PieceType::KNIGHT: return 3;
        case PieceType::BISHOP: return 5;
        case PieceType::ROOK: return 5;
        case PieceType::KING: return 3; // Capturing an active King
        default: return 0;
    }
}

Board::PlayerPointMap Board::evaluate() const {
  PlayerPointMap scores;
  for (int i = 0; i < 4; ++i)
    scores[static_cast<Player>(i)] = 0;

  // Temporary structure to hold intermediate eval data like in Python
  std::map<Player, BoardLocation> king_coords;
  std::map<Player, bool> king_present;
  for (int i = 0; i < 4; ++i)
    king_present[static_cast<Player>(i)] = false;

  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const auto &piece_opt = board_[r][c];
      if (piece_opt) {
        const Piece &piece = *piece_opt;
        Player player = piece.player;

        // Only evaluate piece of active players for material/positional value
        if (active_players_.count(player)) {
          // Base material score
          scores[player] += get_piece_value(piece);

          // Penalties/Bonuses 
          if (piece.piece_type == PieceType::KNIGHT ||
              piece.piece_type == PieceType::BISHOP) {
            if (((player == Player::RED && r == 7) ||
                 (player == Player::YELLOW && r == 0) ||
                 (player == Player::GREEN && c == 7) ||
                 (player == Player::BLUE && c == 0))) {
              scores[player] -= 0.4; // Undeveloped penalty
            }
          }

          if (piece.piece_type == PieceType::KING) {
            king_present[player] = true;
            king_coords[player] = BoardLocation(r, c);
            // King safety check
            for (const auto &dir : KING_DIRS) {
              int nr = r + dir.first;
              int nc = c + dir.second;
              if (is_valid_square(nr, nc)) {
                const auto &adjacent_opt = board_[nr][nc];
                if (adjacent_opt) {
                  if (adjacent_opt->player == player) { // Friendly piece
                    scores[player] += (adjacent_opt->piece_type == PieceType::PAWN ? 0.2 : 0.05);
                  } else { // Opponent piece
                    if (!active_players_.count(
                            adjacent_opt->player)) { // Piece of inactive player
                      scores[player] += 0.15;        // Shelter bonus
                    } else {                         // Active opponent
                      scores[player] -= 0.15;        // Danger penalty
                    }
                  }
                }
              }
            }
          } // End King specific

          if (piece.piece_type == PieceType::PAWN) {
            // Pawn advancement and structure checks
            int dr = 0, dc = 0, adv_row = r, adv_col = c;
            int cap_r1 = 0, cap_c1 = 0, cap_r2 = 0,
                cap_c2 = 0; // Capture checks
            switch (player) {
            case Player::RED:
              scores[player] += 0.2 * (6 - r);
              dr = -1;
              dc = 0;
              cap_r1 = -1;
              cap_c1 = -1;
              cap_r2 = -1;
              cap_c2 = 1;
              break;
            case Player::BLUE:
              scores[player] += 0.2 * (c - 1);
              dr = 0;
              dc = 1;
              cap_r1 = -1;
              cap_c1 = 1;
              cap_r2 = 1;
              cap_c2 = 1;
              break;
            case Player::YELLOW:
              scores[player] += 0.2 * (r - 1);
              dr = 1;
              dc = 0;
              cap_r1 = 1;
              cap_c1 = -1;
              cap_r2 = 1;
              cap_c2 = 1;
              break;
            case Player::GREEN:
              scores[player] += 0.2 * (6 - c);
              dr = 0;
              dc = -1;
              cap_r1 = -1;
              cap_c1 = -1;
              cap_r2 = 1;
              cap_c2 = -1;
              break;
            }
            // Check blocked
            int block_r = r + dr;
            int block_c = c + dc;
            if (is_valid_square(block_r, block_c) && board_[block_r][block_c]) {
              // Python code checks if blocking piece is not same player, but
              // standard blocked is just any piece. Sticking to Python's logic:
              // Penalty if blocked by *opponent*? No, python code seems to
              // check if *not* empty. Correction: Python checks
              // `board[row-1][col] != None and board[row-1][col].player !=
              // piece.player` for RED. This seems wrong. A blocked pawn is
              // blocked regardless of who blocks it. Let's apply penalty if
              // *any* piece is directly in front.
              scores[player] -= 0.2;
            }

            // Check attacks/support
            for (const auto &cap_delta : {std::make_pair(cap_r1, cap_c1),
                                          std::make_pair(cap_r2, cap_c2)}) {
              int cap_r = r + cap_delta.first;
              int cap_c = c + cap_delta.second;
              if (is_valid_square(cap_r, cap_c) && board_[cap_r][cap_c]) {
                const auto &target = *board_[cap_r][cap_c];
                if (target.player == player) { // Supporting friendly piece
                  if (target.piece_type == PieceType::BISHOP ||
                      target.piece_type == PieceType::KNIGHT) {
                    scores[player] += 0.2; // Outpost bonus
                  }
                } else { // Attacking enemy piece
                  scores[player] += 0.2;
                  if (target.piece_type == PieceType::KING &&
                      active_players_.count(
                          target.player)) { // Attacking active king
                    scores[player] += 0.1;
                    scores[target.player] -=
                        0.5; // Penalty for king being attacked by pawn
                  }
                }
              }
            }

          } // End Pawn specific

        } // End if piece is active and not dead
      } // End if piece_opt has value
    } // End col loop
  } // End row loop

  // Final adjustments
  for (int i = 0; i < 4; ++i) {
    Player p = static_cast<Player>(i);
    if (active_players_.count(p) && !king_present[p]) {
      scores[p] = -999.0; // No king penalty (eliminated but not yet processed?)
    }
    scores[p] += player_points_.at(
        p);          // Add captured points (Use .at() for const map access)
    scores[p] -= 20; // Base score adjustment from Python code

    // Rounding? Python used round(, 2). C++ standard rounding works
    // differently. We can keep it as double or round if needed. scores[p] =
    // std::round(scores[p] * 100.0) / 100.0;
  }

  return scores;
}

// --- Player Actions ---

void Board::eliminate_player(Player player) {
  if (active_players_.count(player)) {
    const auto& zobrist_data = get_zobrist_data();
    // Zobrist: XOR out the active status key for the player being eliminated
    current_hash_ ^= zobrist_data.get_active_player_status_key(player);
    active_players_.erase(player); 
    // Note: Turn hash update is handled by advance_turn if called (e.g., in resign)
  }
}

void Board::resign() {
  Player resigning_player = current_player_; // Store before potentially changing turn
  if (active_players_.count(resigning_player)) {
    // --- Create Undo Info for Resignation ---
    UndoInfo resign_undo_info;
    // Store info *before* any changes
    resign_undo_info.original_player = resigning_player; // Player whose turn it was
    resign_undo_info.original_full_move_number = full_move_number_;
    resign_undo_info.original_move_number_of_last_reset = move_number_of_last_reset_;
    resign_undo_info.previous_hash = current_hash_;
    resign_undo_info.eliminated_player = resigning_player; // Mark the player who resigned/was eliminated
    resign_undo_info.was_history_cleared = false; // Resign doesn't clear history
    // Use a special sentinel value for the move or leave it default? Let's leave default for now.
    // resign_undo_info.move = Move(); // Default move
    resign_undo_info.captured_piece = std::nullopt; // No capture involved
    // original_moving_piece_type isn't relevant for resign, leave default
    // resign_undo_info.original_moving_piece_type = PieceType::PAWN; // Default

    // --- Now perform the elimination ---
    eliminate_player(resigning_player);

    // Check if the game ends *immediately* due to this resignation
    // (i.e., only 1 player remains active AFTER elimination)
    bool game_just_ended = (active_players_.size() <= 1);

    if (!game_just_ended) {
        advance_turn(); // handles turn hash update
    } else {
        // Game ended. XOR out the resigning player's turn key.
        const auto& zobrist_data = get_zobrist_data();
        current_hash_ ^= zobrist_data.get_turn_key(resigning_player);

        is_game_over(); // Sets termination reason
    }
    // --- Push the resignation-specific undo info ---
    undo_stack_.push_back(resign_undo_info);
  }
}

// --- Utility ---

// --- ANSI Color Codes ---
const std::string ANSI_RESET = "\033[0m";
const std::string ANSI_RED = "\033[31m";
const std::string ANSI_GREEN = "\033[32m";
const std::string ANSI_YELLOW = "\033[33m";
const std::string ANSI_BLUE = "\033[34m";

// --- Unicode Chess Symbols (as UTF-8 strings) ---
// Ensure your terminal supports UTF-8 and these symbols
const std::string UNICODE_KING = "♔";
const std::string UNICODE_ROOK = "♖";
const std::string UNICODE_BISHOP = "♗";
const std::string UNICODE_KNIGHT = "♘";
const std::string UNICODE_PAWN = "♙";

void Board::print_board() const {
  // Define colored piece strings (combine color, symbol, reset)
  // Red pieces
  const std::string red_king = ANSI_RED + UNICODE_KING + ANSI_RESET;
  const std::string red_rook = ANSI_RED + UNICODE_ROOK + ANSI_RESET;
  const std::string red_bishop = ANSI_RED + UNICODE_BISHOP + ANSI_RESET;
  const std::string red_knight = ANSI_RED + UNICODE_KNIGHT + ANSI_RESET;
  const std::string red_pawn = ANSI_RED + UNICODE_PAWN + ANSI_RESET;

  // Yellow pieces
  const std::string yellow_king = ANSI_YELLOW + UNICODE_KING + ANSI_RESET;
  const std::string yellow_rook = ANSI_YELLOW + UNICODE_ROOK + ANSI_RESET;
  const std::string yellow_bishop = ANSI_YELLOW + UNICODE_BISHOP + ANSI_RESET;
  const std::string yellow_knight = ANSI_YELLOW + UNICODE_KNIGHT + ANSI_RESET;
  const std::string yellow_pawn = ANSI_YELLOW + UNICODE_PAWN + ANSI_RESET;

  // Blue pieces
  const std::string blue_king = ANSI_BLUE + UNICODE_KING + ANSI_RESET;
  const std::string blue_rook = ANSI_BLUE + UNICODE_ROOK + ANSI_RESET;
  const std::string blue_bishop = ANSI_BLUE + UNICODE_BISHOP + ANSI_RESET;
  const std::string blue_knight = ANSI_BLUE + UNICODE_KNIGHT + ANSI_RESET;
  const std::string blue_pawn = ANSI_BLUE + UNICODE_PAWN + ANSI_RESET;

  // Green pieces
  const std::string green_king = ANSI_GREEN + UNICODE_KING + ANSI_RESET;
  const std::string green_rook = ANSI_GREEN + UNICODE_ROOK + ANSI_RESET;
  const std::string green_bishop = ANSI_GREEN + UNICODE_BISHOP + ANSI_RESET;
  const std::string green_knight = ANSI_GREEN + UNICODE_KNIGHT + ANSI_RESET;
  const std::string green_pawn = ANSI_GREEN + UNICODE_PAWN + ANSI_RESET;

  // Dead pieces (no color)
  const std::string dead_king = UNICODE_KING;
  const std::string dead_rook = UNICODE_ROOK;
  const std::string dead_bishop = UNICODE_BISHOP;
  const std::string dead_knight = UNICODE_KNIGHT;
  const std::string dead_pawn = UNICODE_PAWN;

  // Print header row (column numbers 0-7) - Matching python's "   0  1  2  3  4
  // 5  6  7"
  std::cout << "   a  b  c  d  e  f  g  h" << std::endl;

  // Print board rows
  for (int r = 0; r < BOARD_SIZE; ++r) {
    // Print row number (0-7) followed by a space
    std::cout << 8 - r << " ";
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const auto &piece_opt = board_[r][c];
      std::string symbol = " "; // Default empty square content

      if (piece_opt) {
        const Piece &p = *piece_opt;
        bool display_as_inactive = !active_players_.count(p.player);

        // Use uncolored symbols for pieces of inactive players
        const std::string& current_pawn_sym = display_as_inactive ? UNICODE_PAWN : (
            p.player == Player::RED ? red_pawn : 
            p.player == Player::BLUE ? blue_pawn :
            p.player == Player::YELLOW ? yellow_pawn : green_pawn);
        // ... similar logic for other piece types ...
        const std::string& current_knight_sym = display_as_inactive ? UNICODE_KNIGHT : (
            p.player == Player::RED ? red_knight :
            p.player == Player::BLUE ? blue_knight :
            p.player == Player::YELLOW ? yellow_knight : green_knight);
        const std::string& current_bishop_sym = display_as_inactive ? UNICODE_BISHOP : (
            p.player == Player::RED ? red_bishop :
            p.player == Player::BLUE ? blue_bishop :
            p.player == Player::YELLOW ? yellow_bishop : green_bishop);
        const std::string& current_rook_sym = display_as_inactive ? UNICODE_ROOK : (
            p.player == Player::RED ? red_rook :
            p.player == Player::BLUE ? blue_rook :
            p.player == Player::YELLOW ? yellow_rook : green_rook);
        const std::string& current_king_sym = display_as_inactive ? UNICODE_KING : (
            p.player == Player::RED ? red_king :
            p.player == Player::BLUE ? blue_king :
            p.player == Player::YELLOW ? yellow_king : green_king);

        switch (p.piece_type) {
          case PieceType::PAWN:   symbol = current_pawn_sym;   break;
          case PieceType::KNIGHT: symbol = current_knight_sym; break;
          case PieceType::BISHOP: symbol = current_bishop_sym; break;
          case PieceType::ROOK:   symbol = current_rook_sym;   break;
          case PieceType::KING:   symbol = current_king_sym;   break;
          // No DEAD_KING
        }
      }
      std::cout << "[" << symbol << "]";
    }
    std::cout << std::endl; 
  }
  // Removed the bottom coordinate line and separator line

  // --- Print existing game info (kept below the board) ---
  std::cout << "Turn: ";
  switch (current_player_) {
  case Player::RED:
    std::cout << ANSI_RED << "RED" << ANSI_RESET;
    break;
  case Player::BLUE:
    std::cout << ANSI_BLUE << "BLUE" << ANSI_RESET;
    break;
  case Player::YELLOW:
    std::cout << ANSI_YELLOW << "YELLOW" << ANSI_RESET;
    break;
  case Player::GREEN:
    std::cout << ANSI_GREEN << "GREEN" << ANSI_RESET;
    break;
  }
  std::cout << std::endl;
  std::cout << "Active Players: ";
  for (Player p : active_players_) {
    switch (p) {
    case Player::RED:
      std::cout << ANSI_RED << "R " << ANSI_RESET;
      break;
    case Player::BLUE:
      std::cout << ANSI_BLUE << "B " << ANSI_RESET;
      break;
    case Player::YELLOW:
      std::cout << ANSI_YELLOW << "Y " << ANSI_RESET;
      break;
    case Player::GREEN:
      std::cout << ANSI_GREEN << "G " << ANSI_RESET;
      break;
    }
  }
  std::cout << std::endl;
  std::cout << "Points: ";
  bool first_point = true;
  for (const auto &pair : player_points_) {
    if (!first_point) std::cout << " ";
    first_point = false;
    switch (pair.first) {
    case Player::RED:
      std::cout << ANSI_RED << "R:" << pair.second << ANSI_RESET;
      break;
    case Player::BLUE:
      std::cout << ANSI_BLUE << "B:" << pair.second << ANSI_RESET;
      break;
    case Player::YELLOW:
      std::cout << ANSI_YELLOW << "Y:" << pair.second << ANSI_RESET;
      break;
    case Player::GREEN:
      std::cout << ANSI_GREEN << "G:" << pair.second << ANSI_RESET;
      break;
    }
  }
  std::cout << std::endl;
  // std::cout << "50-move counter: " << fifty_move_counter_ << std::endl;
  std::cout << std::endl;
  if (termination_reason_) {
    std::cout << "Game Over: " << *termination_reason_ << std::endl;
  }
}

Board::PositionKey Board::get_position_key() const {
    return current_hash_;
}

} // namespace chaturaji_cpp