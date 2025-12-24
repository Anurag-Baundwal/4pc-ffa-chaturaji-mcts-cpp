#include "utils.h"
#include "magic_utils.h" 
#include <stdexcept>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm> 
#include <iostream>

namespace chaturaji_cpp {

// Local helpers for piece ordering
namespace {
    const std::vector<PieceType> UTIL_PIECE_TYPE_ORDER = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK,
        PieceType::KING
    };
}

std::vector<float> board_to_floats(const Board& board) {
    // Dimensions
    constexpr int NUM_ACTUAL_PIECE_TYPES = 5; // P, N, B, R, K
    constexpr int NUM_PIECE_CHANNELS_ONLY = 4 * NUM_ACTUAL_PIECE_TYPES; // 20
    
    // Total channels: 20 (pieces) + 4 (active status) + 4 (player turn) + 4 (points) + 1 (50-move) + 1 (incoming attacks) = 34 channels
    constexpr int NUM_CHANNELS_TOTAL = NUM_PIECE_CHANNELS_ONLY + 4 + 4 + 4 + 1 + 1; // 34
    
    // Ensure our calculation matches the global constant
    static_assert(NUM_CHANNELS_TOTAL == NN_INPUT_CHANNELS, "Calculated channel count in board_to_floats must match NN_INPUT_CHANNELS");

    // Initialize flat vector with zeros. Size: NN_INPUT_SIZE (2176 floats)
    std::vector<float> tensor_data(NN_INPUT_SIZE, 0.0f);

    auto fill_plane = [&](int channel_idx, float value) {
        if (value == 0.0f) return; 
        int offset = channel_idx * BOARD_AREA;
        std::fill_n(tensor_data.begin() + offset, BOARD_AREA, value);
    };

    auto set_pixel = [&](int channel_idx, int sq_idx, float value) {
        int index = (channel_idx * BOARD_AREA) + sq_idx;
        tensor_data[index] = value;
    };

    // Piece Placement Channels (0-19)
    for (int p_idx = 0; p_idx < 4; ++p_idx) { 
        Player player_enum = static_cast<Player>(p_idx);
        for (int pt_idx = 0; pt_idx < NUM_ACTUAL_PIECE_TYPES; ++pt_idx) { 
            PieceType piece_type_enum = UTIL_PIECE_TYPE_ORDER[pt_idx]; 
            Bitboard bb = board.get_piece_bitboard(player_enum, piece_type_enum);
            int channel = p_idx * NUM_ACTUAL_PIECE_TYPES + pt_idx;

            Bitboard temp_bb = bb;
            while(temp_bb) {
                int sq_idx = magic_utils::pop_lsb(temp_bb); 
                if (channel >= 0 && channel < NUM_PIECE_CHANNELS_ONLY) {
                    set_pixel(channel, sq_idx, 1.0f);
                }
            }
        }
    }

    // Active Player Status Channels (20-23)
    const auto& active_players_set = board.get_active_players();
    int active_status_channel_offset = NUM_PIECE_CHANNELS_ONLY; // 20
    for (int i = 0; i < 4; ++i) {
        Player p_iter = static_cast<Player>(i);
        float val_to_fill = active_players_set.count(p_iter) ? 1.0f : 0.0f;
        fill_plane(active_status_channel_offset + i, val_to_fill);
    }

    // Current Player Channels (24-27)
    Player current_player = board.get_current_player();
    int current_player_channel_offset = active_status_channel_offset + 4; // 24
    int current_player_idx_val = static_cast<int>(current_player);
    if (current_player_idx_val >= 0 && current_player_idx_val < 4) {
        fill_plane(current_player_channel_offset + current_player_idx_val, 1.0f);
    }

    // Player Points Channels (28-31)
    const auto& points = board.get_player_points();
    int points_channel_offset = current_player_channel_offset + 4; // 28
    for (int i = 0; i < 4; ++i) {
        Player p_iter = static_cast<Player>(i);
        float player_points_val = 0.0f;
        auto it = points.find(p_iter);
        if(it != points.end()){
            player_points_val = static_cast<float>(it->second);
        }
        fill_plane(points_channel_offset + i, player_points_val / 100.0f);
    }

    // 50-Move Rule Counter Channel (32)
    int counter_channel_idx = points_channel_offset + 4; 
    int moves_since_reset = board.get_full_move_number() - board.get_move_number_of_last_reset();
    float normalized_count = std::max(0.0f, std::min(1.0f, static_cast<float>(moves_since_reset) / 50.0f));
    fill_plane(counter_channel_idx, normalized_count);

    // Incoming Attacks Channel (33)
    int attack_channel_idx = 33;
    
    Player current_p = board.get_current_player();
    const auto& active_players = board.get_active_players();
    
    Bitboard all_enemy_attacks = 0ULL;
    
    for (Player p : active_players) {
        if (p != current_p) {
            // OR together the attacks from all active opponents
            all_enemy_attacks |= board.get_squares_attacked_by(p);
        }
    }
    
    // Fill the plane based on the bitboard
    while (all_enemy_attacks) {
        int sq_idx = magic_utils::pop_lsb(all_enemy_attacks);
        set_pixel(attack_channel_idx, sq_idx, 1.0f);
    }

    return tensor_data;
}

int move_to_policy_index(const Move& move) {
    int fr_row = move.from_loc.row;
    int fr_col = move.from_loc.col;
    int to_row = move.to_loc.row;
    int to_col = move.to_loc.col;

    if (fr_row < 0 || fr_row >= BOARD_DIM || fr_col < 0 || fr_col >= BOARD_DIM ||
        to_row < 0 || to_row >= BOARD_DIM || to_col < 0 || to_col >= BOARD_DIM) {
        throw std::out_of_range("Move coordinates are out of board bounds for policy index.");
    }
    int from_index = fr_row * BOARD_DIM + fr_col;
    int to_index = to_row * BOARD_DIM + to_col;
    return from_index * BOARD_AREA + to_index;
}

Move policy_index_to_move(int index) {
    if (index < 0 || index >= NN_POLICY_SIZE) {
         throw std::out_of_range("Policy index " + std::to_string(index) + " is out of bounds (0-" + std::to_string(NN_POLICY_SIZE - 1) + ").");
    }
    int to_index = index % BOARD_AREA;
    int from_index = index / BOARD_AREA;

    int to_row = to_index / BOARD_DIM;
    int to_col = to_index % BOARD_DIM;
    int from_row = from_index / BOARD_DIM;
    int from_col = from_index % BOARD_DIM;

    return Move(BoardLocation(from_row, from_col), BoardLocation(to_row, to_col), std::nullopt);
}

std::string get_san_string(const Move& move, const Board& board) {
     std::stringstream ss;
     std::optional<Piece> from_piece_opt = board.get_piece_at_sq(magic_utils::to_sq_idx(move.from_loc.row, move.from_loc.col));
     std::optional<Piece> to_piece_opt = board.get_piece_at_sq(magic_utils::to_sq_idx(move.to_loc.row, move.to_loc.col)); 

     if (!from_piece_opt) return "ERROR";
     
     switch(from_piece_opt->piece_type) {
        case PieceType::KNIGHT: ss << 'N'; break;
        case PieceType::BISHOP: ss << 'B'; break;
        case PieceType::ROOK:   ss << 'R'; break;
        case PieceType::KING:   ss << 'K'; break;
        case PieceType::PAWN: break;
        default: ss << '?'; break;
     }
     ss << static_cast<char>('a' + move.from_loc.col);
     ss << (8 - move.from_loc.row);
     if (to_piece_opt) ss << 'x';
     ss << static_cast<char>('a' + move.to_loc.col);
     ss << (8 - move.to_loc.row);
     if (move.promotion_piece_type) {
         ss << '=';
          switch(*move.promotion_piece_type) {
            case PieceType::ROOK:   ss << 'R'; break;
            default: ss << '?'; break; 
         }
     }
     return ss.str();
}

std::string get_uci_string(const Move& move) {
    std::stringstream ss;
    ss << static_cast<char>('a' + move.from_loc.col);
    ss << (8 - move.from_loc.row);
    ss << static_cast<char>('a' + move.to_loc.col);
    ss << (8 - move.to_loc.row);

    if (move.promotion_piece_type) {
          switch(*move.promotion_piece_type) {
            case PieceType::ROOK:   ss << 'r'; break;
            default: break; 
         }
    }
    return ss.str();
}

} // namespace chaturaji_cpp