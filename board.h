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
    using GameHistory = std::vector<Move>; // Represents a sequence of moves played in a game
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
    bool is_valid_square(int row, int col) const; // Checks if (row, col) is within board boundaries
    static bool is_valid_sq_idx(int sq_idx);      // Checks if a square index (0-63) is valid
    static int to_sq_idx(int r, int c);           // Converts (row, col) to a square index (0-63)
    static BoardLocation from_sq_idx(int sq_idx); // Converts a square index (0-63) to (row, col)
    std::optional<Piece> get_piece_at_sq(int sq_idx) const; // Get piece from bitboards

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
    Bitboard get_occupied_bitboard() const; // Get combined occupied bitboard
    Bitboard get_player_bitboard(Player p) const; // Get bitboard for a specific player's pieces
    Bitboard get_piece_bitboard(Player p, PieceType pt) const; // Get bitboard for specific player and piece type

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
    static void print_bitboard(Bitboard bb, const std::string& label = ""); // Utility to print a bitboard

    // --- Bitboard Manipulation Helpers ---
    static inline void set_bit(Bitboard& bb, int sq_idx) { bb |= (1ULL << sq_idx); }
    static inline void clear_bit(Bitboard& bb, int sq_idx) { bb &= ~(1ULL << sq_idx); }
    static inline bool get_bit(Bitboard bb, int sq_idx) { return (bb >> sq_idx) & 1ULL; }
    static inline int pop_lsb(Bitboard& bb) { // Gets and clears the least significant bit
        if (bb == 0) return -1; // Or throw exception
        #ifdef _MSC_VER
            unsigned long index;
            _BitScanForward64(&index, bb);
            bb &= bb - 1; // Clears the LSB
            return static_cast<int>(index);
        #else // Assuming GCC/Clang
            int index = __builtin_ctzll(bb); // Count trailing zeros (index of LSB)
            bb &= bb - 1; // Clears the LSB
            return index;
        #endif
    }
    static inline int get_lsb_index(Bitboard bb) { // Gets the index of the least significant bit
         if (bb == 0) return -1; // Or throw exception
        #ifdef _MSC_VER
            unsigned long index;
            _BitScanForward64(&index, bb);
            return static_cast<int>(index);
        #else // Assuming GCC/Clang
            return __builtin_ctzll(bb);
        #endif
    }
    // Helper to map PieceType to 0-4 index for bitboards_
    static int piece_type_to_bb_idx(PieceType pt);

private:
    // --- Internal State ---
    ActivePlayerSet active_players_;
    PlayerPointMap player_points_;
    Player current_player_;
    PositionHistory position_history_; // Stores Zobrist keys of past positions
    int full_move_number_;
    int move_number_of_last_reset_; // For 50-move rule
    mutable std::optional<std::string> termination_reason_; // Store reason for game end
    ZobristKey current_hash_; // Current Zobrist hash of the position
    std::vector<UndoInfo> undo_stack_; // Stack to store information for undoing moves

    // Bitboard representation
    std::array<std::array<Bitboard, 5>, 4> piece_bitboards_; // [player][piece_type_bb_idx]
    std::array<Bitboard, 4> player_bitboards_;              // [player] (all pieces of that player)
    Bitboard occupied_bitboard_;                             // All pieces on the board

    // Precomputed attack/move lookup tables for bitboards
    static std::array<Bitboard, NUM_SQUARES_BB> knight_attacks_;
    static std::array<Bitboard, NUM_SQUARES_BB> king_attacks_;
    // Pawn attacks are direction-dependent per player (and color)
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_red_;    // [player_idx][square_idx]
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_blue_;   // Red, Blue, Yellow, Green typically have different 'forward'
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_yellow_;
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_green_;
    // Pawn forward (non-capturing) moves
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_red_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_blue_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_yellow_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_green_;

    // --- Magic Bitboard Data for Sliding Pieces (Rooks, Bishops) ---
    static std::array<Bitboard, NUM_SQUARES_BB> rook_masks_;       // Relevant occupancy bits for rook moves from each square
    static std::array<Bitboard, NUM_SQUARES_BB> bishop_masks_;     // Relevant occupancy bits for bishop moves from each square
    static std::array<int, NUM_SQUARES_BB> rook_shift_bits_;     // Precomputed shift amounts for rook magic indexing
    static std::array<int, NUM_SQUARES_BB> bishop_shift_bits_;   // Precomputed shift amounts for bishop magic indexing
    
    static std::vector<Bitboard> rook_attack_table_;             // Flat lookup table for all possible rook attacks, indexed by magic
    static std::vector<Bitboard> bishop_attack_table_;           // Flat lookup table for all possible bishop attacks, indexed by magic
    static std::array<unsigned int, NUM_SQUARES_BB> rook_attack_offsets_; // Offsets into rook_attack_table_ for each square
    static std::array<unsigned int, NUM_SQUARES_BB> bishop_attack_offsets_;// Offsets into bishop_attack_table_ for each square
    // --- End Magic Bitboard Data ---

    // Helper to initialize static lookup tables (including magic bitboards)
    static void initialize_lookup_tables();
    // Static initializer trick to call initialize_lookup_tables() before main()
    struct StaticInitializer { StaticInitializer() { initialize_lookup_tables(); } };
    static StaticInitializer static_initializer_;

    // --- Private Helper Methods for Move Generation (Bitboard based) ---
    void get_pawn_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_knight_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_bishop_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_rook_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_king_moves_bb(Player player, std::vector<Move>& moves) const;

    // Private Helper for Turn Advancement
    void advance_turn();
    // Helper to find the last player in sequence (numerically highest enum value) among active players
    Player get_last_active_player() const;
};

} // namespace chaturaji_cpp