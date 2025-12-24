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
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::KING
    };
}

std::vector<float> board_to_floats(const Board& board) {
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

    Player current_p = board.get_current_player();
    int cp_idx = static_cast<int>(current_p);

    // --- Relative Feature Encoding ---
    // Inputs are rotated so that index 0 always represents the current player.
    // Mapping: Rel 0 = Current, Rel 1 = Next, Rel 2 = Opposite, Rel 3 = Previous.
    // 1. Piece Placement (0-19) - RELATIVE
    for (int rel_i = 0; rel_i < 4; ++rel_i) { 
        int abs_p_idx = (cp_idx + rel_i) % 4;
        Player p_enum = static_cast<Player>(abs_p_idx);
        for (int pt_idx = 0; pt_idx < 5; ++pt_idx) { 
            PieceType pt_enum = UTIL_PIECE_TYPE_ORDER[pt_idx]; 
            Bitboard bb = board.get_piece_bitboard(p_enum, pt_enum);
            int channel = (rel_i * 5) + pt_idx;
            while(bb) {
                int sq_idx = magic_utils::pop_lsb(bb); 
                set_pixel(channel, sq_idx, 1.0f);
            }
        }
    }

    // 2. Active Status (20-23) - RELATIVE
    const auto& active_set = board.get_active_players();
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        fill_plane(20 + rel_i, active_set.count(static_cast<Player>((cp_idx + rel_i) % 4)) ? 1.0f : 0.0f);
    }

    // 3. Points (24-27) - RELATIVE
    const auto& points = board.get_player_points();
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        float pts = static_cast<float>(points.at(static_cast<Player>((cp_idx + rel_i) % 4)));
        fill_plane(24 + rel_i, pts / 100.0f);
    }

    // 4. 50-Move Clock (28)
    int moves_since_reset = board.get_full_move_number() - board.get_move_number_of_last_reset();
    fill_plane(28, std::min(1.0f, static_cast<float>(moves_since_reset) / 50.0f));

    // 5. Attack Planes (29-32) - RELATIVE
    std::array<Bitboard, 4> all_atks;
    for(int i=0; i<4; ++i) all_atks[i] = board.get_squares_attacked_by(static_cast<Player>(i));
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        Bitboard bb = all_atks[(cp_idx + rel_i) % 4];
        while(bb) set_pixel(29 + rel_i, magic_utils::pop_lsb(bb), 1.0f);
    }

    // 6. In-Check Planes (33-36) - RELATIVE
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        Bitboard king = board.get_piece_bitboard(static_cast<Player>(abs_idx), PieceType::KING);
        Bitboard stressors = 0;
        for(int opp=0; opp<4; ++opp) if(opp != abs_idx) stressors |= all_atks[opp];
        if (king & stressors) fill_plane(33 + rel_i, 1.0f);
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