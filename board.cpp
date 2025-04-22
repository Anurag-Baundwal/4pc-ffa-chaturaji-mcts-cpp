#include "board.h"
#include <algorithm> // For std::find, std::max_element
#include <cmath>     // For std::round
#include <iostream>
#include <numeric> // For std::accumulate (optional)
#include <sstream>
#include <stdexcept>
#include <utility> // For std::move
#include <vector>

namespace chaturaji_cpp {

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
    : current_player_(Player::RED), fifty_move_counter_(0),
      termination_reason_(std::nullopt) {
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
  // Add initial position hash? Maybe not needed if first move resets counter.
  // position_history_.push_back(get_position_key());
}

// --- Copy Constructor ---
Board::Board(const Board &other)
    : board_(other.board_), active_players_(other.active_players_),
      player_points_(other.player_points_),
      current_player_(other.current_player_),
      position_history_(other.position_history_),
      fifty_move_counter_(other.fifty_move_counter_),
      termination_reason_(other.termination_reason_),
      game_history_(other.game_history_),
      undo_stack_(other.undo_stack_) {}

// --- Move Constructor ---
Board::Board(Board &&other) noexcept
    : board_(std::move(other.board_)),
      active_players_(std::move(other.active_players_)),
      player_points_(std::move(other.player_points_)),
      current_player_(other.current_player_),
      position_history_(std::move(other.position_history_)),
      fifty_move_counter_(other.fifty_move_counter_),
      termination_reason_(std::move(other.termination_reason_)),
      game_history_(std::move(other.game_history_)),
      // --- FIX: Move the undo_stack_ instead of previous_board_state_ ---
      undo_stack_(std::move(other.undo_stack_))
{
  // Reset other state after move if necessary (unique_ptr handles its resource)
  other.fifty_move_counter_ = 0;
}

// --- Copy Assignment Operator ---
Board &Board::operator=(const Board &other) {
  if (this != &other) { // Self-assignment check
    board_ = other.board_;
    active_players_ = other.active_players_;
    player_points_ = other.player_points_;
    current_player_ = other.current_player_;
    position_history_ = other.position_history_;
    fifty_move_counter_ = other.fifty_move_counter_;
    termination_reason_ = other.termination_reason_;
    game_history_ = other.game_history_;
    // Deep copy the previous state
    undo_stack_ = other.undo_stack_;
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
    fifty_move_counter_ = other.fifty_move_counter_;
    termination_reason_ = std::move(other.termination_reason_);
    game_history_ = std::move(other.game_history_);
    // --- FIX: Move assign the undo_stack_ ---
    undo_stack_ = std::move(other.undo_stack_);

    // Reset other state if necessary
    other.fifty_move_counter_ = 0;
  }
  return *this;
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

  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const auto &piece_opt = board_[r][c];
      if (piece_opt && piece_opt->player == player && !piece_opt->is_dead) {
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
        // Add cases for QUEEN, ONE_POINT_QUEEN if they become playable
        case PieceType::QUEEN: // Combine Rook and Bishop logic if needed
          get_rook_moves(r, c, pseudo_legal_moves);
          get_bishop_moves(r, c, pseudo_legal_moves);
          break;
        case PieceType::ONE_POINT_QUEEN: // Same movement as Queen
          get_rook_moves(r, c, pseudo_legal_moves);
          get_bishop_moves(r, c, pseudo_legal_moves);
          break;
        case PieceType::DEAD_KING: // Dead kings don't move
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

// Inside board.cpp

std::optional<Piece> Board::make_move(const Move &move) {
  // --- Create Undo Information ---
  UndoInfo undo_info;
  undo_info.move = move;
  undo_info.original_player = current_player_;
  undo_info.original_fifty_move_counter = fifty_move_counter_;
  undo_info.eliminated_player = std::nullopt; // Assume no elimination initially
  undo_info.was_history_cleared = false; // Assume history not cleared initially

  // --- Get Move Details ---
  int fr = move.from_loc.row;
  int fc = move.from_loc.col;
  int tr = move.to_loc.row;
  int tc = move.to_loc.col;

  // --- Validate Moving Piece ---
  if (!board_[fr][fc]) {
    // This check should ideally be done before calling make_move
    // But adding robustness here.
    throw std::runtime_error(
        "Attempting to move from an empty square in make_move.");
  }
  Piece moving_piece = board_[fr][fc].value(); // Get a copy of the piece
  undo_info.original_moving_piece_type =
      moving_piece.piece_type; // Store pre-promotion type

  // --- Store Captured Piece Info ---
  undo_info.captured_piece =
      board_[tr][tc]; // Store potential captured piece (or nullopt)
  bool is_capture = undo_info.captured_piece.has_value();
  bool is_pawn_move = (moving_piece.piece_type == PieceType::PAWN);

  // --- Perform Move on Board ---
  board_[fr][fc] = std::nullopt; // Clear the 'from' square
  board_[tr][tc] = moving_piece; // Place the moving piece on the 'to' square

  // --- Handle Promotion ---
  if (move.promotion_piece_type) {
    board_[tr][tc]->piece_type = move.promotion_piece_type.value();
  }

  // --- Handle Capture Logic & Scoring ---
  if (undo_info.captured_piece) {
    const Piece &captured = undo_info.captured_piece.value();
    if (!captured.is_dead) { // Points only for 'live' captures? No, python code
                             // adds for dead too.
      player_points_[moving_piece.player] += get_piece_capture_value(captured);

      // Check for King capture and elimination
      if (captured.piece_type == PieceType::KING) {
        eliminate_player(
            captured.player); // Modifies board_ and active_players_
        undo_info.eliminated_player = captured.player; // Record elimination
      }
    } else { // Handle capturing dead pieces (including DEAD_KING)
      player_points_[moving_piece.player] += get_piece_capture_value(captured);
      // No elimination needed if capturing a dead piece
    }
  }

  // --- Update Fifty-Move Counter & Position History ---
  if (is_pawn_move || is_capture) {
    fifty_move_counter_ = 0;
    position_history_.clear(); // Reset history on irreversible moves
    undo_info.was_history_cleared = true;
  } else {
    fifty_move_counter_++;
  }

  // --- Update Histories ---
  position_history_.push_back(get_position_key()); // Add current state hash
  game_history_.push_back(move);                   // Add move to game log

  // --- Push Undo Info ---
  undo_stack_.push_back(
      undo_info); // Add undo info AFTER state changes are complete

  // --- Advance Turn ---
  advance_turn(); // Advances current_player_

  // --- Return Captured Piece ---
  return undo_info.captured_piece; // Return the piece captured in this move
}

// Inside board.cpp

void Board::undo_move() {
  if (undo_stack_.empty()) {
    throw std::runtime_error("No previous state available to undo.");
  }

  // --- Pop Undo Information ---
  UndoInfo undo_info = undo_stack_.back();
  undo_stack_.pop_back();

  // --- Retrieve Move Details ---
  const Move &move = undo_info.move;
  int fr = move.from_loc.row;
  int fc = move.from_loc.col;
  int tr = move.to_loc.row;
  int tc = move.to_loc.col;

  // --- Restore Player Turn ---
  // Note: This needs to be done before potentially un-eliminating players
  current_player_ = undo_info.original_player;

  // --- Restore Histories ---
  if (!game_history_.empty()) {
    game_history_.pop_back(); // Remove last move log entry
  }
  // Position history restoration is tricky if it was cleared.
  // The standard approach only requires popping the last hash.
  // If was_history_cleared is true, the repetition count might be slightly off
  // compared to the deep copy version for positions *before* the irreversible
  // move. This is a common trade-off for efficiency.
  if (!position_history_.empty()) {
    position_history_.pop_back();
  }
  // TODO: If perfect history restoration after clearing is needed,
  // the UndoInfo would need to store the *entire* previous history,
  // partially defeating the purpose. Or, change the repetition check logic.

  // --- Restore Fifty-Move Counter ---
  fifty_move_counter_ = undo_info.original_fifty_move_counter;

  // --- Reverse Board Changes ---
  // Get the piece that ended up on the 'to' square (might be promoted)
  Piece moving_piece = board_[tr][tc].value();
  // Restore its original type if it was promoted
  moving_piece.piece_type = undo_info.original_moving_piece_type;
  // Place it back on the 'from' square
  board_[fr][fc] = moving_piece;
  // Restore the 'to' square (put back captured piece or make it empty)
  board_[tr][tc] = undo_info.captured_piece;

  // --- Reverse Elimination (if necessary) ---
  if (undo_info.eliminated_player) {
    Player player_to_revive = *undo_info.eliminated_player;
    active_players_.insert(player_to_revive);

    // Find the Dead King and revert it, and revert other dead pieces
    for (int r = 0; r < BOARD_SIZE; ++r) {
      for (int c = 0; c < BOARD_SIZE; ++c) {
        auto &piece_opt = board_[r][c];
        if (piece_opt && piece_opt->player == player_to_revive) {
          if (piece_opt->piece_type == PieceType::DEAD_KING) {
            // Check if this dead king is the one captured by this move.
            // This check prevents accidental revival if multiple dead kings
            // of the same player existed (highly unlikely but possible).
            // The captured piece was at [tr][tc] before this undo restored it.
            if (r == tr && c == tc) {
              piece_opt->piece_type = PieceType::KING;
            }
            // Keep is_dead = false for the King/Dead King itself
          } else {
            // Revive other pieces of that player
            piece_opt->is_dead = false;
          }
        }
      }
    }
  }

  // --- Reverse Point Changes (if capture occurred) ---
  if (undo_info.captured_piece) {
    const Piece &captured = undo_info.captured_piece.value();
    // Subtract the points that were added
    // Points were added to the player whose turn it *was* (original_player)
    player_points_[undo_info.original_player] -=
        get_piece_capture_value(captured);
  }

  // --- Clear Termination Reason (state might no longer be terminal) ---
  termination_reason_ = std::nullopt;
}

void Board::advance_turn() {
  int next_player_val = (static_cast<int>(current_player_) + 1) % 4;
  current_player_ = static_cast<Player>(next_player_val);

  // Skip eliminated players
  while (active_players_.find(current_player_) == active_players_.end()) {
    // Check if only one player is left (should be caught by is_game_over, but
    // safety check)
    if (active_players_.size() <= 1) break;
    next_player_val = (static_cast<int>(current_player_) + 1) % 4;
    current_player_ = static_cast<Player>(next_player_val);
  }
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
int Board::get_fifty_move_counter() const { return fifty_move_counter_; }
const std::optional<std::string> &Board::get_termination_reason() const {
  return termination_reason_;
}
const Board::GameHistory &Board::get_game_history() const {
  return game_history_;
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

  if (fifty_move_counter_ >=
      100) { // Standard chess is 50 moves by *each* player
             // Python code increments per *player turn*. So 50 turns = 100
             // half-moves for 2p. For 4p, 50 turns means 50 increments. Let's
             // stick to Python logic first: 50 increments. If standard rule
             // interpretation needed, change condition to fifty_move_counter_
             // >= 50 * active_players_.size() maybe? Reverting to Python's
             // direct counter >= 50 for now.
    termination_reason_ = "fifty_move_rule";
    return true;
  }

  PositionKey current_key = get_position_key();
  int count = 0;
  for (const auto &key : position_history_) {
    if (key == current_key) {
      count++;
    }
  }
  if (count >= 3) {
    termination_reason_ = "threefold_repetition";
    return true;
  }

  // Check for stalemate (no legal moves for current player) - Optional but good
  // NOTE: The Python code doesn't explicitly check stalemate before declaring
  // game over, it seems to rely on the engine returning no moves and then
  // resigning. We can add an explicit check here if desired. if
  // (get_pseudo_legal_moves(current_player_).empty()) {
  //     // This could be stalemate or checkmate. Need check detection for
  //     checkmate.
  //     // For now, let's follow Python and let the engine handle no-move
  //     scenarios.
  // }

  return false;
}

Board::PlayerPointMap Board::get_game_result() const {
  PlayerPointMap results = player_points_; // Start with current capture points

  // --- Count Dead Kings on the board ---
  int num_dead_kings = 0;
  for (const auto &row : board_) {
    for (const auto &piece_opt : row) {
      if (piece_opt && piece_opt->piece_type == PieceType::DEAD_KING) {
        num_dead_kings++;
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
        // Calculate Dead King bonus per active player for draws
        int dead_king_bonus_per_player = 0;
        if (num_dead_kings > 0) {
          // Use floating-point division before ceiling
          dead_king_bonus_per_player = static_cast<int>(
              std::ceil(3.0 * num_dead_kings / num_active_players));
        }

        // Apply base draw bonus (+2) and dead king bonus to each active player
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
      if (num_active_players == 1 && num_dead_kings > 0) {
        Player winner =
            *active_players_.begin(); // Get the single remaining player
        int dead_king_bonus =
            3 * num_dead_kings; // Simpler calculation for 1 active player
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

int Board::get_piece_value(const Piece &piece) const {
  // Values from Python code's evaluate function
  switch (piece.piece_type) {
  case PieceType::PAWN:
    return 1;
  case PieceType::KNIGHT:
    return 3;
  case PieceType::BISHOP:
    return 5;
  case PieceType::ROOK:
    return 5;
  case PieceType::QUEEN:
    return 9; // Included though not in setup
  case PieceType::ONE_POINT_QUEEN:
    return 11; // Included
  case PieceType::KING:
    return 3; // Value used in evaluation
  case PieceType::DEAD_KING:
    return 0; // Dead King has no intrinsic value in eval
  default:
    return 0;
  }
}

int Board::get_piece_capture_value(const Piece &piece) const {
  // Values from Python code's get_piece_capture_value function
  switch (piece.piece_type) {
  case PieceType::PAWN:
    return 1;
  case PieceType::KNIGHT:
    return 3;
  case PieceType::BISHOP:
    return 5;
  case PieceType::ROOK:
    return 5;
  case PieceType::QUEEN:
    return 9; // Included
  case PieceType::ONE_POINT_QUEEN:
    return 1; // Value from Python
  case PieceType::KING:
    return 3; // Value from Python
  case PieceType::DEAD_KING:
    return 3; // Value from Python
  default:
    return 0;
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

        // Only evaluate active, non-dead pieces for material/positional value
        if (!piece.is_dead && active_players_.count(player)) {
          // Base material score
          scores[player] += get_piece_value(piece);

          // Penalties/Bonuses from Python eval
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
                  if (adjacent_opt->player == player) {
                    scores[player] +=
                        (adjacent_opt->piece_type == PieceType::PAWN ? 0.2
                                                                     : 0.05);
                  } else { // Opponent piece
                    if (!active_players_.count(
                            adjacent_opt->player)) { // Dead opponent
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
    for (int r = 0; r < BOARD_SIZE; ++r) {
      for (int c = 0; c < BOARD_SIZE; ++c) {
        auto &piece_opt = board_[r][c];
        if (piece_opt && piece_opt->player == player && !piece_opt->is_dead) {
          if (piece_opt->piece_type == PieceType::KING) {
            piece_opt->piece_type = PieceType::DEAD_KING;
            // Keep is_dead = false for the Dead King itself, as it stays on
            // board
          } else {
            piece_opt->is_dead = true;
          }
        }
      }
    }
    active_players_.erase(player);
    // No need to adjust points here, points are for captures
  }
}

void Board::resign() {
  if (active_players_.count(current_player_)) {
    eliminate_player(current_player_);
    // Only advance turn if game is not over
    if (active_players_.size() > 1) {
      advance_turn();
    } else {
      // If resigning makes the game end, set termination reason
      is_game_over(); // This should set the reason to elimination
    }
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
const std::string UNICODE_QUEEN = "♕";
const std::string UNICODE_ROOK = "♖";
const std::string UNICODE_BISHOP = "♗";
const std::string UNICODE_KNIGHT = "♘";
const std::string UNICODE_PAWN = "♙";

void Board::print_board() const {
    // Define colored piece strings (combine color, symbol, reset)
    // Red pieces
    const std::string red_king = ANSI_RED + UNICODE_KING + ANSI_RESET;
    const std::string red_queen = ANSI_RED + UNICODE_QUEEN + ANSI_RESET;
    const std::string red_rook = ANSI_RED + UNICODE_ROOK + ANSI_RESET;
    const std::string red_bishop = ANSI_RED + UNICODE_BISHOP + ANSI_RESET;
    const std::string red_knight = ANSI_RED + UNICODE_KNIGHT + ANSI_RESET;
    const std::string red_pawn = ANSI_RED + UNICODE_PAWN + ANSI_RESET;

    // Yellow pieces
    const std::string yellow_king = ANSI_YELLOW + UNICODE_KING + ANSI_RESET;
    const std::string yellow_queen = ANSI_YELLOW + UNICODE_QUEEN + ANSI_RESET;
    const std::string yellow_rook = ANSI_YELLOW + UNICODE_ROOK + ANSI_RESET;
    const std::string yellow_bishop = ANSI_YELLOW + UNICODE_BISHOP + ANSI_RESET;
    const std::string yellow_knight = ANSI_YELLOW + UNICODE_KNIGHT + ANSI_RESET;
    const std::string yellow_pawn = ANSI_YELLOW + UNICODE_PAWN + ANSI_RESET;

    // Blue pieces
    const std::string blue_king = ANSI_BLUE + UNICODE_KING + ANSI_RESET;
    const std::string blue_queen = ANSI_BLUE + UNICODE_QUEEN + ANSI_RESET;
    const std::string blue_rook = ANSI_BLUE + UNICODE_ROOK + ANSI_RESET;
    const std::string blue_bishop = ANSI_BLUE + UNICODE_BISHOP + ANSI_RESET;
    const std::string blue_knight = ANSI_BLUE + UNICODE_KNIGHT + ANSI_RESET;
    const std::string blue_pawn = ANSI_BLUE + UNICODE_PAWN + ANSI_RESET;

    // Green pieces
    const std::string green_king = ANSI_GREEN + UNICODE_KING + ANSI_RESET;
    const std::string green_queen = ANSI_GREEN + UNICODE_QUEEN + ANSI_RESET;
    const std::string green_rook = ANSI_GREEN + UNICODE_ROOK + ANSI_RESET;
    const std::string green_bishop = ANSI_GREEN + UNICODE_BISHOP + ANSI_RESET;
    const std::string green_knight = ANSI_GREEN + UNICODE_KNIGHT + ANSI_RESET;
    const std::string green_pawn = ANSI_GREEN + UNICODE_PAWN + ANSI_RESET;

    // Dead pieces (no color)
    const std::string dead_king = UNICODE_KING;
    const std::string dead_queen = UNICODE_QUEEN;
    const std::string dead_rook = UNICODE_ROOK;
    const std::string dead_bishop = UNICODE_BISHOP;
    const std::string dead_knight = UNICODE_KNIGHT;
    const std::string dead_pawn = UNICODE_PAWN;

    // Print header row (column numbers 0-7) - Matching python's "   0  1  2  3  4  5  6  7"
    std::cout << "   a  b  c  d  e  f  g  h" << std::endl;

    // Print board rows
    for (int r = 0; r < BOARD_SIZE; ++r) {
        // Print row number (0-7) followed by a space - Matching python's `print(row, end=" ")`
        std::cout << 8-r << " ";
        for (int c = 0; c < BOARD_SIZE; ++c) {
            const auto& piece_opt = board_[r][c];
            std::string symbol = " "; // Default empty square content

            if (piece_opt) {
                const Piece& p = *piece_opt;
                bool use_dead_symbol = p.is_dead || p.piece_type == PieceType::DEAD_KING;

                if (use_dead_symbol) {
                    // Select dead symbol (no color)
                    switch (p.piece_type) {
                        case PieceType::PAWN:           symbol = dead_pawn; break;
                        case PieceType::KNIGHT:         symbol = dead_knight; break;
                        case PieceType::BISHOP:         symbol = dead_bishop; break;
                        case PieceType::ROOK:           symbol = dead_rook; break;
                        case PieceType::QUEEN:          symbol = dead_queen; break;
                        case PieceType::KING:           symbol = dead_king; break;
                        case PieceType::ONE_POINT_QUEEN:symbol = dead_queen; break;
                        case PieceType::DEAD_KING:      symbol = dead_king; break;
                    }
                } else {
                    // Select colored symbol for active pieces
                    switch (p.player) {
                        case Player::RED:
                            switch (p.piece_type) {
                                case PieceType::PAWN:   symbol = red_pawn; break;
                                case PieceType::KNIGHT: symbol = red_knight; break;
                                case PieceType::BISHOP: symbol = red_bishop; break;
                                case PieceType::ROOK:   symbol = red_rook; break;
                                case PieceType::QUEEN:  symbol = red_queen; break;
                                case PieceType::KING:   symbol = red_king; break;
                                case PieceType::ONE_POINT_QUEEN: symbol = red_queen; break;
                            }
                            break;
                        case Player::BLUE:
                             switch (p.piece_type) {
                                case PieceType::PAWN:   symbol = blue_pawn; break;
                                case PieceType::KNIGHT: symbol = blue_knight; break;
                                case PieceType::BISHOP: symbol = blue_bishop; break;
                                case PieceType::ROOK:   symbol = blue_rook; break;
                                case PieceType::QUEEN:  symbol = blue_queen; break;
                                case PieceType::KING:   symbol = blue_king; break;
                                case PieceType::ONE_POINT_QUEEN: symbol = blue_queen; break;
                            }
                            break;
                        case Player::YELLOW:
                             switch (p.piece_type) {
                                case PieceType::PAWN:   symbol = yellow_pawn; break;
                                case PieceType::KNIGHT: symbol = yellow_knight; break;
                                case PieceType::BISHOP: symbol = yellow_bishop; break;
                                case PieceType::ROOK:   symbol = yellow_rook; break;
                                case PieceType::QUEEN:  symbol = yellow_queen; break;
                                case PieceType::KING:   symbol = yellow_king; break;
                                case PieceType::ONE_POINT_QUEEN: symbol = yellow_queen; break;
                            }
                            break;
                        case Player::GREEN:
                             switch (p.piece_type) {
                                case PieceType::PAWN:   symbol = green_pawn; break;
                                case PieceType::KNIGHT: symbol = green_knight; break;
                                case PieceType::BISHOP: symbol = green_bishop; break;
                                case PieceType::ROOK:   symbol = green_rook; break;
                                case PieceType::QUEEN:  symbol = green_queen; break;
                                case PieceType::KING:   symbol = green_king; break;
                                case PieceType::ONE_POINT_QUEEN: symbol = green_queen; break;
                            }
                            break;
                    }
                }
            }
            // Print the square content with brackets - Matching python's `print(f"[{symbol}]", end="")`
            std::cout << "[" << symbol << "]";
        }
        std::cout << std::endl; // Newline after each row
        // Removed the horizontal separator line here
    }
     // Removed the bottom coordinate line and separator line

    // --- Print existing game info (kept below the board) ---
    std::cout << "Turn: ";
    switch (current_player_) {
    case Player::RED:    std::cout << ANSI_RED << "RED" << ANSI_RESET; break;
    case Player::BLUE:   std::cout << ANSI_BLUE << "BLUE" << ANSI_RESET; break;
    case Player::YELLOW: std::cout << ANSI_YELLOW << "YELLOW" << ANSI_RESET; break;
    case Player::GREEN:  std::cout << ANSI_GREEN << "GREEN" << ANSI_RESET; break;
    }
    std::cout << std::endl;
    std::cout << "Active Players: ";
    for (Player p : active_players_) {
        switch (p) {
            case Player::RED:    std::cout << ANSI_RED << "R " << ANSI_RESET; break;
            case Player::BLUE:   std::cout << ANSI_BLUE << "B " << ANSI_RESET; break;
            case Player::YELLOW: std::cout << ANSI_YELLOW << "Y " << ANSI_RESET; break;
            case Player::GREEN:  std::cout << ANSI_GREEN << "G " << ANSI_RESET; break;
        }
    }
    std::cout << std::endl;
    std::cout << "Points: ";
    bool first_point = true;
    for (const auto& pair : player_points_) {
        if (!first_point) std::cout << " ";
        first_point = false;
        switch (pair.first) {
            case Player::RED:    std::cout << ANSI_RED << "R:" << pair.second << ANSI_RESET; break;
            case Player::BLUE:   std::cout << ANSI_BLUE << "B:" << pair.second << ANSI_RESET; break;
            case Player::YELLOW: std::cout << ANSI_YELLOW << "Y:" << pair.second << ANSI_RESET; break;
            case Player::GREEN:  std::cout << ANSI_GREEN << "G:" << pair.second << ANSI_RESET; break;
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
  // Simple string representation for hashing
  // Format: Board_Grid/Current_Player/Active_Players/Fifty_Counter
  // NOTE: Including fifty_move_counter is NOT standard for threefold
  // repetition. Repetition normally only cares about the piece positions and
  // castling/enpassant rights (and whose turn it is). Sticking closely to
  // Python code which includes it implicitly by adding to history every turn.
  // For a more standard check, only hash board + current player + potentially
  // irreversible move rights.
  std::stringstream ss;
  for (int r = 0; r < BOARD_SIZE; ++r) {
    for (int c = 0; c < BOARD_SIZE; ++c) {
      const auto &p = board_[r][c];
      if (!p) {
        ss << '.';
      } else {
        char piece_char = '?';
        switch (p->piece_type) {
        case PieceType::PAWN:
          piece_char = 'P';
          break;
        case PieceType::KNIGHT:
          piece_char = 'N';
          break;
        case PieceType::BISHOP:
          piece_char = 'B';
          break;
        case PieceType::ROOK:
          piece_char = 'R';
          break;
        case PieceType::QUEEN:
          piece_char = 'Q';
          break;
        case PieceType::KING:
          piece_char = 'K';
          break;
        case PieceType::ONE_POINT_QUEEN:
          piece_char = 'O';
          break;
        case PieceType::DEAD_KING:
          piece_char = 'k';
          break; // Distinguish dead king
        }
        // Add player indicator, lowercase for dead non-king pieces
        switch (p->player) {
        case Player::RED:
          ss << (p->is_dead ? (char)std::tolower(piece_char) : piece_char);
          break;
        case Player::BLUE:
          ss << (p->is_dead ? (char)std::tolower(piece_char) : piece_char);
          break; // Need different indicators!
        case Player::YELLOW:
          ss << (p->is_dead ? (char)std::tolower(piece_char) : piece_char);
          break;
        case Player::GREEN:
          ss << (p->is_dead ? (char)std::tolower(piece_char) : piece_char);
          break;
        }
        // Let's use R/B/Y/G prefixes + piece type char + 'd' if dead
        ss.str(""); // Clear stringstream
        ss.clear(); // Clear error flags
        for (int r_ = 0; r_ < BOARD_SIZE; ++r_) {
          for (int c_ = 0; c_ < BOARD_SIZE; ++c_) {
            const auto &p_ = board_[r_][c_];
            if (!p_) {
              ss << "__.";
            } // 3 chars per square
            else {
              char p_char = '?';
              switch (p_->player) {
              case Player::RED:
                p_char = 'R';
                break;
              case Player::BLUE:
                p_char = 'B';
                break;
              case Player::YELLOW:
                p_char = 'Y';
                break;
              case Player::GREEN:
                p_char = 'G';
                break;
              }
              ss << p_char;

              char t_char = '?';
              switch (p_->piece_type) {
              case PieceType::PAWN:
                t_char = 'P';
                break;
              case PieceType::KNIGHT:
                t_char = 'N';
                break;
              case PieceType::BISHOP:
                t_char = 'B';
                break;
              case PieceType::ROOK:
                t_char = 'R';
                break;
              case PieceType::QUEEN:
                t_char = 'Q';
                break;
              case PieceType::KING:
                t_char = 'K';
                break;
              case PieceType::ONE_POINT_QUEEN:
                t_char = 'O';
                break;
              case PieceType::DEAD_KING:
                t_char = 'k';
                break;
              }
              ss << t_char;
              ss << (p_->is_dead ? 'd' : '.'); // Dead marker
            }
          }
          ss << '/'; // Row separator
        }
        ss << '|';                               // Section separator
        ss << static_cast<int>(current_player_); // Whose turn

        // Technically, threefold repetition also depends on irreversible move
        // rights (castling, en passant) which aren't present here. But player
        // elimination *is* irreversible. Include active players in the hash?
        ss << '|';
        for (Player p : active_players_)
          ss << static_cast<int>(p);

        return ss.str(); // Use this comprehensive string
      }
    }
    ss << '/'; // Row separator
  }
  ss << '|';                               // Section separator
  ss << static_cast<int>(current_player_); // Whose turn
  // ss << '|';
  // ss << fifty_move_counter_; // Should this be part of the hash? Python's
  // history implies yes.

  return ss.str();
}

} // namespace chaturaji_cpp