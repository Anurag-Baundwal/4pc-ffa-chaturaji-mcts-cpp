#pragma once

#include <vector>
#include <set>
#include <map>
#include <optional>
#include <array>
#include <string>
#include <memory>
#include <cstdint>

#include "types.h"
#include "piece.h"
#include "magic_utils.h"

namespace chaturaji_cpp {

constexpr int BOARD_SIZE_LOCAL = 8;
constexpr int NUM_SQUARES_BB_LOCAL = 64;

class Board;

struct MCTSChildCopyTag {};

struct UndoInfo {
    Move move;
    std::optional<Piece> captured_piece;
    PieceType original_moving_piece_type;
    Player original_player;
    int original_full_move_number;
    int original_move_number_of_last_reset;
    bool was_history_cleared;
    std::optional<Player> eliminated_player;
    ZobristKey previous_hash;
    int check_bonus_points = 0; 

    // New fields for bitboard state
    std::array<std::array<Bitboard, 5>, 4> original_piece_bitboards;
    std::array<Bitboard, 4> original_player_bitboards;
    Bitboard original_occupied_bitboard;
    std::array<Bitboard, 5> original_combined_piece_bitboards;
    uint8_t original_active_mask;
    std::array<std::optional<Piece>, 64> original_mailbox;
};

class Board {
public:
    using PositionKey = ZobristKey;
    using PositionHistory = std::vector<PositionKey>;
    using GameHistory = std::vector<Move>;
    using PlayerPointMap = std::array<int, 4>;
    using ActivePlayerSet = uint8_t;

    Board();
    Board(const Board& other);
    Board(Board&& other) noexcept;

    bool is_player_active(Player p) const {
        return active_mask_ & (1 << static_cast<int>(p));
    }

    void set_player_active(Player p, bool active) {
        if (active) active_mask_ |= (1 << static_cast<int>(p));
        else active_mask_ &= ~(1 << static_cast<int>(p));
    }

    Board(const Board& other, MCTSChildCopyTag tag);

    static Board create_mcts_child_board(const Board& parent_board, const Move& move);

    Board& operator=(const Board& other);
    Board& operator=(Board&& other) noexcept;

    void setup_initial_board();
    bool is_valid_square(int row, int col) const;
    static bool is_valid_sq_idx(int sq_idx);
    std::optional<Piece> get_piece_at_sq(int sq_idx) const;

    // Optimized Move Generation using MoveList (Stack-based)
    void get_pseudo_legal_moves(Player player, MoveList& moves) const;

    // Legacy wrapper for compatibility until call sites are updated
    std::vector<Move> get_pseudo_legal_moves(Player player) const;

    std::optional<Piece> make_move(const Move& move);
    std::optional<Piece> make_move_for_mcts(const Move& move);
    void undo_move();

    ActivePlayerSet get_active_mask() const { return active_mask_; }
    const std::set<Player> get_active_players() const;
    const PlayerPointMap& get_player_points() const;
    Player get_current_player() const;
    int get_full_move_number() const;
    int get_move_number_of_last_reset() const;
    const std::optional<std::string>& get_termination_reason() const;
    const PositionHistory& get_position_history() const;
    Bitboard get_occupied_bitboard() const;
    Bitboard get_player_bitboard(Player p) const;
    Bitboard get_piece_bitboard(Player p, PieceType pt) const;
    
    Bitboard get_squares_attacked_by(Player player) const;
    Bitboard get_attackers_on_sq(int sq_idx) const;

    bool is_game_over() const;
    std::map<Player, int> get_game_result() const;
    std::optional<Player> get_winner() const;

    std::map<Player, int> evaluate() const;
    int get_piece_value(const Piece& piece) const;
    int get_piece_capture_value(const Piece& piece) const;

    void eliminate_player(Player player);
    void resign();

    void print_board() const;
    PositionKey get_position_key() const;
    static void print_bitboard(Bitboard bb, const std::string& label = ""); 

    static int piece_type_to_bb_idx(PieceType pt);

private:
    uint8_t active_mask_;
    PlayerPointMap player_points_;
    Player current_player_;
    PositionHistory position_history_;
    int full_move_number_;
    int move_number_of_last_reset_;
    mutable std::optional<std::string> termination_reason_;
    ZobristKey current_hash_;
    std::vector<UndoInfo> undo_stack_;

    std::array<std::array<Bitboard, 5>, 4> piece_bitboards_;
    std::array<Bitboard, 4> player_bitboards_;
    Bitboard occupied_bitboard_;

    std::array<Bitboard, 5> combined_piece_bitboards_;
    std::array<std::optional<Piece>, 64> mailbox_;

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

    // Optimized Move Gen Helpers using MoveList
    void get_pawn_moves_bb(Player player, MoveList& moves) const;
    void get_knight_moves_bb(Player player, MoveList& moves) const;
    void get_bishop_moves_bb(Player player, MoveList& moves) const;
    void get_rook_moves_bb(Player player, MoveList& moves) const;
    void get_king_moves_bb(Player player, MoveList& moves) const;

    void advance_turn();
    Player get_last_active_player() const;
};

} // namespace chaturaji_cpp