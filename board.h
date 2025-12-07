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
#include "magic_utils.h" // Include the new magic utilities

namespace chaturaji_cpp {

// Define board size constants
constexpr int BOARD_SIZE_LOCAL = 8; // Renamed to avoid conflict if magic_utils is used directly in header
constexpr int NUM_SQUARES_BB_LOCAL = 64; // Renamed

// Forward declaration
class Board;

// --- TAG FOR MCTS-SPECIFIC COPY CONSTRUCTOR ---
struct MCTSChildCopyTag {}; // Empty struct used purely for tagging/overloading

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

    // Optimized: No heavy bitboard copies here.
};

class Board {
public:
    // --- Typedefs for clarity ---
    using PositionKey = ZobristKey;
    using PositionHistory = std::vector<PositionKey>;
    using GameHistory = std::vector<Move>; // Represents a sequence of moves played in a game
    
    // OPTIMIZATION: Replaced map/set with std::array for O(1) access
    using PlayerPointArray = std::array<int, 4>;
    using ActivePlayerArray = std::array<bool, 4>;
    
    // Compatibility typedefs (though API semantics have changed slightly for speed)
    using PlayerPointMap = PlayerPointArray; 
    using ActivePlayerSet = ActivePlayerArray;

    // --- Constructors ---
    Board(); // Default constructor initializes the board
    Board(const Board& other); // Copy constructor
    Board(Board&& other) noexcept; // Move constructor

    // --- MCTS-SPECIFIC COPY CONSTRUCTOR DECLARATION ---
    /**
     * @brief Special copy constructor for MCTS child node creation.
     */
    Board(const Board& other, MCTSChildCopyTag tag);

    // --- Static Factory for MCTS Child Boards ---
    static Board create_mcts_child_board(const Board& parent_board, const Move& move);

    // --- Operators ---
    Board& operator=(const Board& other); // Copy assignment
    Board& operator=(Board&& other) noexcept; // Move assignment

    // --- Core Game Logic ---
    void setup_initial_board();
    bool is_valid_square(int row, int col) const; // Checks if (row, col) is within board boundaries
    static bool is_valid_sq_idx(int sq_idx);      // Checks if a square index (0-63) is valid
    std::optional<Piece> get_piece_at_sq(int sq_idx) const; // Get piece from bitboards

    // Optimized move generation using MoveList&
    void get_pseudo_legal_moves(Player player, MoveList& moves) const;
    // Wrapper for backward compatibility returning vector
    std::vector<Move> get_pseudo_legal_moves_vec(Player player) const;

    std::optional<Piece> make_move(const Move& move);
    std::optional<Piece> make_move_for_mcts(const Move& move);
    void undo_move();

    // --- Game State Accessors ---
    // Returns reference to array of booleans (index is Player enum)
    const ActivePlayerArray& get_active_players() const;
    bool is_player_active(Player p) const;
    
    const PlayerPointArray& get_player_points() const;
    int get_player_points(Player p) const;

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
    
    // Returns a Map for API compatibility with other parts of the engine expecting maps
    std::map<Player, int> get_game_result() const; 
    
    std::optional<Player> get_winner() const; // Determines winner based on game result

    // --- Evaluation ---
    std::map<Player, double> evaluate() const; // Hand-crafted evaluation
    int get_piece_value(const Piece& piece) const;
    int get_piece_capture_value(const Piece& piece) const;

    // --- Player Actions ---
    void eliminate_player(Player player);
    void resign();

    // --- Utility ---
    void print_board() const;
    PositionKey get_position_key() const;
    static void print_bitboard(Bitboard bb, const std::string& label = ""); 
    static int piece_type_to_bb_idx(PieceType pt);

private:
    // --- Internal State ---
    // OPTIMIZATION: Replaced std::set/map with arrays and counters
    ActivePlayerArray is_active_;
    int active_player_count_;
    PlayerPointArray points_;
    
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
    static std::array<Bitboard, magic_utils::NUM_SQUARES> knight_attacks_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> king_attacks_;
    static std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> pawn_attacks_red_;
    static std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> pawn_attacks_blue_;
    static std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> pawn_attacks_yellow_;
    static std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> pawn_attacks_green_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> pawn_fwd_moves_red_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> pawn_fwd_moves_blue_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> pawn_fwd_moves_yellow_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> pawn_fwd_moves_green_;

    // --- Magic Bitboard Data ---
    static std::array<Bitboard, magic_utils::NUM_SQUARES> rook_masks_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> bishop_masks_;
    static std::array<int, magic_utils::NUM_SQUARES> rook_shift_bits_;
    static std::array<int, magic_utils::NUM_SQUARES> bishop_shift_bits_;
    static std::vector<Bitboard> rook_attack_table_;
    static std::vector<Bitboard> bishop_attack_table_;
    static std::array<unsigned int, magic_utils::NUM_SQUARES> rook_attack_offsets_;
    static std::array<unsigned int, magic_utils::NUM_SQUARES> bishop_attack_offsets_;

    static void initialize_lookup_tables();
    struct StaticInitializer { StaticInitializer() { initialize_lookup_tables(); } };
    static StaticInitializer static_initializer_;

    // --- Private Helper Methods ---
    void get_pawn_moves_bb(Player player, MoveList& moves) const;
    void get_knight_moves_bb(Player player, MoveList& moves) const;
    void get_bishop_moves_bb(Player player, MoveList& moves) const;
    void get_rook_moves_bb(Player player, MoveList& moves) const;
    void get_king_moves_bb(Player player, MoveList& moves) const;

    void advance_turn();
    Player get_last_active_player() const;
    inline void toggle_piece(Player p, PieceType pt, int sq_idx);
};

} // namespace chaturaji_cpp