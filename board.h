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
constexpr int NUM_SQUARES_BB = 64;

// Forward declaration
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

    // New fields for bitboard state, no original comments for these specific fields
    std::array<std::array<Bitboard, 5>, 4> original_piece_bitboards;
    std::array<Bitboard, 4> original_player_bitboards;
    Bitboard original_occupied_bitboard;

    // No need to store points delta; can be recalculated from captured_piece
    // No need to store previous position history; just pop the last hash during undo
};

class Board {
public:
    // --- Typedefs for clarity ---
    using PositionKey = ZobristKey;
    using PositionHistory = std::vector<PositionKey>;
    using GameHistory = std::vector<Move>;
    using PlayerPointMap = std::map<Player, int>;
    using ActivePlayerSet = std::set<Player>;

    // --- Constructors ---
    Board(); // Default constructor initializes the board
    Board(const Board& other); // Copy constructor
    Board(Board&& other) noexcept; // Move constructor

    // --- Static Factory for MCTS Child Boards ---
    // Creates a lightweight board state from parent for MCTS expansion.
    // Copies essential state, but initializes history/undo stacks as empty.
    static Board create_mcts_child_board(const Board& parent_board, const Move& move);

    // --- Operators ---
    Board& operator=(const Board& other); // Copy assignment
    Board& operator=(Board&& other) noexcept; // Move assignment

    // --- Core Game Logic ---
    void setup_initial_board();
    bool is_valid_square(int row, int col) const; // Original check based on row/col
    static bool is_valid_sq_idx(int sq_idx);      // New: check based on 0-63 index
    static int to_sq_idx(int r, int c);           // New: convert row/col to 0-63 index
    static BoardLocation from_sq_idx(int sq_idx); // New: convert 0-63 index to row/col
    std::optional<Piece> get_piece_at_sq(int sq_idx) const; // NEW: Get piece from bitboards

    std::vector<Move> get_pseudo_legal_moves(Player player) const;
    std::optional<Piece> make_move(const Move& move);
    std::optional<Piece> make_move_for_mcts(const Move& move);
    void undo_move();

    // --- Game State Accessors ---
    const ActivePlayerSet& get_active_players() const;
    const PlayerPointMap& get_player_points() const;
    Player get_current_player() const;
    int get_full_move_number() const;
    int get_move_number_of_last_reset() const;
    const std::optional<std::string>& get_termination_reason() const;
    const PositionHistory& get_position_history() const;
    Bitboard get_occupied_bitboard() const; // New: get combined occupied bitboard
    Bitboard get_player_bitboard(Player p) const; // New: get bitboard for a specific player's pieces
    Bitboard get_piece_bitboard(Player p, PieceType pt) const; // New: get bitboard for specific player and piece type

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
    void resign();

    // --- Utility ---
    void print_board() const;
    PositionKey get_position_key() const;
    static void print_bitboard(Bitboard bb, const std::string& label = ""); // New: utility to print a bitboard

    // --- MOVED BITBOARD HELPERS TO PUBLIC ---
    static inline void set_bit(Bitboard& bb, int sq_idx) { bb |= (1ULL << sq_idx); }
    static inline void clear_bit(Bitboard& bb, int sq_idx) { bb &= ~(1ULL << sq_idx); }
    static inline bool get_bit(Bitboard bb, int sq_idx) { return (bb >> sq_idx) & 1ULL; }
    static inline int pop_lsb(Bitboard& bb) {
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
    static inline int get_lsb_index(Bitboard bb) {
         if (bb == 0) return -1;
        #ifdef _MSC_VER
            unsigned long index;
            _BitScanForward64(&index, bb);
            return static_cast<int>(index);
        #else
            return __builtin_ctzll(bb);
        #endif
    }
    // Helper to map PieceType to 0-4 index for bitboards_
    static int piece_type_to_bb_idx(PieceType pt);
    // --- END MOVED BITBOARD HELPERS ---

private:
    // --- Internal State ---
    ActivePlayerSet active_players_;
    PlayerPointMap player_points_;
    Player current_player_;
    PositionHistory position_history_;
    int full_move_number_;
    int move_number_of_last_reset_;
    mutable std::optional<std::string> termination_reason_;
    ZobristKey current_hash_;
    std::vector<UndoInfo> undo_stack_;

    // Bitboard representation
    std::array<std::array<Bitboard, 5>, 4> piece_bitboards_; // [player][piece_type_bb_idx]
    std::array<Bitboard, 4> player_bitboards_;              // [player] (all pieces of that player)
    Bitboard occupied_bitboard_;                             // All pieces on the board

    // Precomputed attack/move lookup tables for bitboards
    static std::array<Bitboard, NUM_SQUARES_BB> knight_attacks_;
    static std::array<Bitboard, NUM_SQUARES_BB> king_attacks_;
    // Pawn attacks are direction-dependent per player (and color)
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_red_;    // [direction_idx][square_idx]
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_blue_;   // Red, Blue, Yellow, Green typically have different 'forward'
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_yellow_;
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_green_;
    // Pawn forward (non-capturing) moves
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_red_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_blue_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_yellow_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_green_;

    // Rays for sliding pieces (rooks, bishops)
    // For rooks: 0: North, 1: East, 2: South, 3: West (example, actual indexing might vary)
    // For bishops: 0: NE, 1: SE, 2: SW, 3: NW (example)
    static std::array<std::array<Bitboard, 4>, NUM_SQUARES_BB> rook_rays_;
    static std::array<std::array<Bitboard, 4>, NUM_SQUARES_BB> bishop_rays_;

    // Helper to initialize static lookup tables
    static void initialize_lookup_tables();
    struct StaticInitializer { StaticInitializer() { initialize_lookup_tables(); } };
    static StaticInitializer static_initializer_;

    // --- Private Helper Methods for Move Generation (Bitboard based) ---
    void get_pawn_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_knight_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_bishop_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_rook_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_king_moves_bb(Player player, std::vector<Move>& moves) const;

    // Generic helper for sliding pieces (rooks, bishops) using bitboards
    void generate_sliding_moves(Player p, int from_sq, PieceType pt, const std::vector<std::pair<int,int>>& directions, std::vector<Move>& moves) const; // Note: directions might be implicit via ray lookups

    // Private Helper for Turn Advancement
    void advance_turn();
    Player get_last_active_player() const;
};

} // namespace chaturaji_cpp