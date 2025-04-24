#pragma once

#include <vector>
#include <set>
#include <map>
#include <optional>
#include <array>
#include <string>
#include <memory>   // For std::unique_ptr if needed later, or std::optional
#include <cstdint>  // For ZobristKey (uint64_t)

#include "types.h"
#include "piece.h"

namespace chaturaji_cpp {

// Define board size constant
constexpr int BOARD_SIZE = 8;

// Forward declaration of Board if UndoInfo is outside the class scope
class Board;

struct UndoInfo {
    Move move;                     // The move that was made
    std::optional<Piece> captured_piece; // The piece that was on the 'to' square (or nullopt)
    PieceType original_moving_piece_type; // Type of the piece *before* potential promotion
    Player original_player;        // Player whose turn it was *before* the move
    int original_full_move_number;
    int original_move_number_of_last_reset;
    bool was_history_cleared;      // Did this move clear the position history?
    std::optional<Player> eliminated_player; // Player eliminated by this move (if any)
    ZobristKey previous_hash;      // Hash *before* the move was made
    // No need to store points delta; can be recalculated from captured_piece
    // No need to store previous position history; just pop the last hash during undo
};

class Board {
public:
    // --- Typedefs for clarity ---
    using BoardGrid = std::array<std::array<std::optional<Piece>, BOARD_SIZE>, BOARD_SIZE>;
    using PositionKey = ZobristKey;
    using PositionHistory = std::vector<PositionKey>;
    using GameHistory = std::vector<Move>;
    using PlayerPointMap = std::map<Player, int>;
    using ActivePlayerSet = std::set<Player>;

    // --- Constructors ---
    Board(); // Default constructor initializes the board
    Board(const Board& other); // Copy constructor
    Board(Board&& other) noexcept; // Move constructor

    // --- Operators ---
    Board& operator=(const Board& other); // Copy assignment
    Board& operator=(Board&& other) noexcept; // Move assignment

    // --- Core Game Logic ---
    void setup_initial_board();
    bool is_valid_square(int row, int col) const;
    std::vector<Move> get_pseudo_legal_moves(Player player) const;
    // Returns captured piece (if any)
    std::optional<Piece> make_move(const Move& move);
    void undo_move(); // Restores from _previous_board_state

    // --- Game State Accessors ---
    const BoardGrid& get_board_grid() const;
    const ActivePlayerSet& get_active_players() const;
    const PlayerPointMap& get_player_points() const;
    Player get_current_player() const;
    int get_full_move_number() const;
    int get_move_number_of_last_reset() const;
    const std::optional<std::string>& get_termination_reason() const;
    const PositionHistory& get_position_history() const; // Added accessor

    // --- Game Status ---
    bool is_game_over() const;             // Checks and sets termination_reason if true
    PlayerPointMap get_game_result() const; // Calculates final scores based on state
    std::optional<Player> get_winner() const; // Determines winner based on game result

    // --- Evaluation ---
    PlayerPointMap evaluate() const; // Hand-crafted evaluation (can be removed later if only NN is used)
    int get_piece_value(const Piece& piece) const;
    int get_piece_capture_value(const Piece& piece) const;

    // --- Player Actions ---
    void eliminate_player(Player player);
    void resign(); // Current player resigns

    // --- Utility ---
    void print_board() const; // Simple text representation
    PositionKey get_position_key() const; // Generate a unique key for the current position

private:
    // --- Internal State ---
    BoardGrid board_;
    ActivePlayerSet active_players_;
    PlayerPointMap player_points_;
    Player current_player_;
    PositionHistory position_history_;
    int full_move_number_;
    int move_number_of_last_reset_;
    mutable std::optional<std::string> termination_reason_;
    ZobristKey current_hash_;

    // --- NEW: Stack for Undo Information ---
    std::vector<UndoInfo> undo_stack_; // Use vector as a stack

    // --- Private Helper Methods for Move Generation ---
    void get_pawn_moves(int row, int col, std::vector<Move>& moves) const;
    void get_knight_moves(int row, int col, std::vector<Move>& moves) const;
    void get_bishop_moves(int row, int col, std::vector<Move>& moves) const;
    void get_rook_moves(int row, int col, std::vector<Move>& moves) const;
    void get_king_moves(int row, int col, std::vector<Move>& moves) const;

    // --- Private Helper for Turn Advancement ---
    void advance_turn();
    Player get_last_active_player() const; // ADD declaration
};

} // namespace chaturaji_cpp