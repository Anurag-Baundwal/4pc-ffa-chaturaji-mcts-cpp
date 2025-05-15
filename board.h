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

namespace chaturaji_cpp {

constexpr int BOARD_SIZE = 8;
constexpr int NUM_SQUARES_BB = 64; 

class Board;

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
    
    std::array<std::array<Bitboard, 5>, 4> original_piece_bitboards; 
    std::array<Bitboard, 4> original_player_bitboards; 
    Bitboard original_occupied_bitboard;
};

class Board {
public:
    using BoardGrid = std::array<std::array<std::optional<Piece>, BOARD_SIZE>, BOARD_SIZE>;
    using PositionKey = ZobristKey;
    using PositionHistory = std::vector<PositionKey>;
    using GameHistory = std::vector<Move>;
    using PlayerPointMap = std::map<Player, int>;
    using ActivePlayerSet = std::set<Player>;

    Board(); 
    Board(const Board& other); 
    Board(Board&& other) noexcept; 

    static Board create_mcts_child_board(const Board& parent_board, const Move& move);
    
    Board& operator=(const Board& other); 
    Board& operator=(Board&& other) noexcept; 

    void setup_initial_board();
    bool is_valid_square(int row, int col) const; 
    static bool is_valid_sq_idx(int sq_idx);      
    static int to_sq_idx(int r, int c);
    static BoardLocation from_sq_idx(int sq_idx);

    std::vector<Move> get_pseudo_legal_moves(Player player) const;
    std::optional<Piece> make_move(const Move& move);
    void undo_move();

    const BoardGrid& get_board_grid() const; 
    const ActivePlayerSet& get_active_players() const;
    const PlayerPointMap& get_player_points() const;
    Player get_current_player() const;
    int get_full_move_number() const;
    int get_move_number_of_last_reset() const;
    const std::optional<std::string>& get_termination_reason() const;
    const PositionHistory& get_position_history() const;
    Bitboard get_occupied_bitboard() const;
    Bitboard get_player_bitboard(Player p) const;
    Bitboard get_piece_bitboard(Player p, PieceType pt) const;

    bool is_game_over() const;             
    PlayerPointMap get_game_result() const; 
    std::optional<Player> get_winner() const; 

    PlayerPointMap evaluate() const; 
    int get_piece_value(const Piece& piece) const;
    int get_piece_capture_value(const Piece& piece) const;

    void eliminate_player(Player player);
    void resign();

    void print_board() const;
    PositionKey get_position_key() const;
    static void print_bitboard(Bitboard bb, const std::string& label = "");

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
    BoardGrid board_; 
    ActivePlayerSet active_players_;
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
    
    static std::array<Bitboard, NUM_SQUARES_BB> knight_attacks_;
    static std::array<Bitboard, NUM_SQUARES_BB> king_attacks_;
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_red_;
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_blue_;
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_yellow_;
    static std::array<std::array<Bitboard, NUM_SQUARES_BB>, 4> pawn_attacks_green_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_red_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_blue_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_yellow_;
    static std::array<Bitboard, NUM_SQUARES_BB> pawn_fwd_moves_green_;

    static std::array<std::array<Bitboard, 4>, NUM_SQUARES_BB> rook_rays_; 
    static std::array<std::array<Bitboard, 4>, NUM_SQUARES_BB> bishop_rays_;

    static void initialize_lookup_tables();
    struct StaticInitializer { StaticInitializer() { initialize_lookup_tables(); } };
    static StaticInitializer static_initializer_; 

    void get_pawn_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_knight_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_bishop_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_rook_moves_bb(Player player, std::vector<Move>& moves) const;
    void get_king_moves_bb(Player player, std::vector<Move>& moves) const;
    
    void generate_sliding_moves(Player p, int from_sq, PieceType pt, const std::vector<std::pair<int,int>>& directions, std::vector<Move>& moves) const;

    void advance_turn();
    Player get_last_active_player() const;
};

} // namespace chaturaji_cpp