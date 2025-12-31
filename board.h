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
    int check_bonus_points = 0; 

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

    // --- MCTS-SPECIFIC COPY CONSTRUCTOR DECLARATION ---
    /**
     * @brief Special copy constructor for MCTS child node creation.
     * It performs a deep copy of essential game state (like bitboards, hashes,
     * player info, and crucially, position_history_) but initializes
     * MCTS-transient states like undo_stack_ as empty and termination_reason_
     * as nullopt.
     * @param other The parent board to copy from.
     * @param tag An empty tag struct to differentiate this constructor.
     */
    Board(const Board& other, MCTSChildCopyTag tag);

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
    // to_sq_idx and from_sq_idx are now in magic_utils
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
    
    Bitboard get_squares_attacked_by(Player player) const;
    Bitboard get_attackers_on_sq(int sq_idx) const;


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
    // print_bitboard now uses magic_utils::get_bit and magic_utils::to_sq_idx
    static void print_bitboard(Bitboard bb, const std::string& label = ""); 

    // --- Bitboard Manipulation Helpers removed, use magic_utils::set_bit, etc. ---
    
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
    // These are specific to Board's internal representation and initialization
    static std::array<Bitboard, magic_utils::NUM_SQUARES> knight_attacks_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> king_attacks_;
    // Pawn attacks are direction-dependent per player (and color)
    static std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> pawn_attacks_red_;
    static std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> pawn_attacks_blue_;
    static std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> pawn_attacks_yellow_;
    static std::array<std::array<Bitboard, magic_utils::NUM_SQUARES>, 4> pawn_attacks_green_;
    // Pawn forward (non-capturing) moves
    static std::array<Bitboard, magic_utils::NUM_SQUARES> pawn_fwd_moves_red_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> pawn_fwd_moves_blue_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> pawn_fwd_moves_yellow_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> pawn_fwd_moves_green_;

    // --- Magic Bitboard Data for Sliding Pieces (Rooks, Bishops) ---
    // These are populated by initialize_lookup_tables using magic_utils functions and constants
    static std::array<Bitboard, magic_utils::NUM_SQUARES> rook_masks_;
    static std::array<Bitboard, magic_utils::NUM_SQUARES> bishop_masks_;
    static std::array<int, magic_utils::NUM_SQUARES> rook_shift_bits_;
    static std::array<int, magic_utils::NUM_SQUARES> bishop_shift_bits_;
    
    static std::vector<Bitboard> rook_attack_table_;
    static std::vector<Bitboard> bishop_attack_table_;
    static std::array<unsigned int, magic_utils::NUM_SQUARES> rook_attack_offsets_;
    static std::array<unsigned int, magic_utils::NUM_SQUARES> bishop_attack_offsets_;
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