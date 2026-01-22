#include "utils.h"
#include "magic_utils.h" 
#include <stdexcept>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm> 
#include <iostream>
#include <regex>
#include <cstdio>
#include <memory>
#include <array>

namespace chaturaji_cpp {

// --- RunStats Implementation ---

void RunStats::save(const std::string& filepath) const {
    std::ofstream out(filepath);
    if (out.is_open()) {
        out << "global_iteration=" << global_iteration << "\n";
        out << "total_samples_generated=" << total_samples_generated << "\n";
        out.close();
    }
}

RunStats RunStats::load(const std::string& filepath) {
    RunStats stats;
    std::ifstream in(filepath);
    if (!in.is_open()) return stats; // Return default 0s

    std::string line;
    while (std::getline(in, line)) {
        if (line.find("global_iteration=") == 0) {
            try { stats.global_iteration = std::stoi(line.substr(17)); } catch(...) {}
        }
        else if (line.find("total_samples_generated=") == 0) {
            try { stats.total_samples_generated = std::stoull(line.substr(24)); } catch(...) {}
        }
    }
    return stats;
}

namespace {
    const std::vector<PieceType> UTIL_PIECE_TYPE_ORDER = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::KING
    };

    // Precomputed rotation table: [player][sq_idx] -> relative_sq_idx
    // 4 players * 64 squares = 256 bytes. Highly cache efficient.
    struct RotationTable {
        std::array<std::array<uint8_t, 64>, 4> to_relative;
        std::array<std::array<uint8_t, 64>, 4> to_absolute;

        constexpr RotationTable() : to_relative{}, to_absolute{} {
            for (int p = 0; p < 4; ++p) {
                for (int r = 0; r < 8; ++r) {
                    for (int c = 0; c < 8; ++c) {
                        int abs_sq = r * 8 + c;
                        int rr = r, rc = c;
                        
                        // Logic from original get_rel_loc
                        switch (static_cast<Player>(p)) {
                            case Player::RED:    rr = r; rc = c; break;
                            case Player::BLUE:   rr = 7 - c; rc = r; break;
                            case Player::YELLOW: rr = 7 - r; rc = 7 - c; break;
                            case Player::GREEN:  rr = c; rc = 7 - r; break;
                        }
                        int rel_sq = rr * 8 + rc;
                        
                        to_relative[p][abs_sq] = static_cast<uint8_t>(rel_sq);
                        to_absolute[p][rel_sq] = static_cast<uint8_t>(abs_sq);
                    }
                }
            }
        }
    };

    constexpr RotationTable ROTATION_TABLE;

    // Inline helper for speed
    inline int get_rel_sq_idx(int abs_sq_idx, Player p) {
        return ROTATION_TABLE.to_relative[static_cast<int>(p)][abs_sq_idx];
    }

    inline int get_abs_sq_idx(int rel_sq_idx, Player p) {
        return ROTATION_TABLE.to_absolute[static_cast<int>(p)][rel_sq_idx];
    }
    
    // Legacy helpers using the new table
    BoardLocation get_rel_loc(int r, int c, Player p) {
        int idx = get_rel_sq_idx(r * 8 + c, p);
        return {idx / 8, idx % 8};
    }

    BoardLocation get_abs_loc(int r, int c, Player p) {
        int idx = get_abs_sq_idx(r * 8 + c, p);
        return {idx / 8, idx % 8};
    }
}

void board_to_floats_into(const Board& board, std::vector<float>& tensor_data) {
    if (tensor_data.size() != NN_INPUT_SIZE) {
        tensor_data.resize(NN_INPUT_SIZE);
    }
    // Fast zeroing
    std::fill(tensor_data.begin(), tensor_data.end(), 0.0f);

    Player current_p = board.get_current_player();
    int cp_idx = static_cast<int>(current_p);

    auto fill_plane_fast = [&](int channel_idx, float value) {
        if (value == 0.0f) return;
        float* ptr = tensor_data.data() + (channel_idx * BOARD_AREA);
        std::fill_n(ptr, BOARD_AREA, value);
    };

    // 1. Piece Placement
    for (int rel_i = 0; rel_i < 4; ++rel_i) { 
        int abs_p_idx = (cp_idx + rel_i) % 4;
        Player p_enum = static_cast<Player>(abs_p_idx);
        int base_channel = rel_i * 5;

        for (int pt_idx = 0; pt_idx < 5; ++pt_idx) { 
            PieceType pt_enum = UTIL_PIECE_TYPE_ORDER[pt_idx]; 
            Bitboard bb = board.get_piece_bitboard(p_enum, pt_enum);
            int channel = base_channel + pt_idx;
            int offset = channel * BOARD_AREA;
            float* channel_ptr = tensor_data.data() + offset;

            while(bb) {
                int abs_sq = magic_utils::pop_lsb(bb);
                int rel_sq = get_rel_sq_idx(abs_sq, current_p);
                channel_ptr[rel_sq] = 1.0f;
            }
        }
    }

    // 2. Active Status
    const auto& active_players = board.get_active_players(); // Set version
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        if (active_players.count(static_cast<Player>((cp_idx + rel_i) % 4))) {
            fill_plane_fast(20 + rel_i, 1.0f);
        }
    }

    // 3. Points
    const auto& points = board.get_player_points();
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        float pts = static_cast<float>(points[abs_idx]);
        fill_plane_fast(24 + rel_i, pts / 100.0f);
    }

    // 4. 50-Move Clock
    int moves_since_reset = board.get_full_move_number() - board.get_move_number_of_last_reset();
    fill_plane_fast(28, std::min(1.0f, static_cast<float>(moves_since_reset) / 50.0f));

    // 5. Attack Planes
    std::array<Bitboard, 4> all_atks;
    for(int i=0; i<4; ++i) all_atks[i] = board.get_squares_attacked_by(static_cast<Player>(i));

    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        Bitboard bb = all_atks[abs_idx];
        int channel = 29 + rel_i;
        int offset = channel * BOARD_AREA;
        float* channel_ptr = tensor_data.data() + offset;
        
        while(bb) {
            int abs_sq = magic_utils::pop_lsb(bb);
            int rel_sq = get_rel_sq_idx(abs_sq, current_p);
            channel_ptr[rel_sq] = 1.0f;
        }
    }

    // 6. In-Check Planes
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        Bitboard king = board.get_piece_bitboard(static_cast<Player>(abs_idx), PieceType::KING);
        Bitboard stressors = 0;
        for(int opp=0; opp<4; ++opp) if(opp != abs_idx) stressors |= all_atks[opp];
        if (king & stressors) fill_plane_fast(33 + rel_i, 1.0f);
    }
}

// Wrapper for compatibility
std::vector<float> board_to_floats(const Board& board) {
    std::vector<float> tensor_data(NN_INPUT_SIZE);
    board_to_floats_into(board, tensor_data);
    return tensor_data;
}

Move parse_string_to_move(const Board& board, const std::string& move_str) {
    if (move_str == "R" || move_str == "T" || move_str == "RESIGN") {
        return Move::Resign();
    }

    std::regex move_regex("([a-h][1-8]).*?([a-h][1-8])");
    std::smatch match;

    if (std::regex_search(move_str, match, move_regex)) {
        std::string from_str = match[1].str();
        std::string to_str = match[2].str();

        BoardLocation from_loc = magic_utils::from_sq_idx(
            magic_utils::to_sq_idx(8 - (from_str[1] - '0'), from_str[0] - 'a')
        );
        
        BoardLocation to_loc = magic_utils::from_sq_idx(
            magic_utils::to_sq_idx(8 - (to_str[1] - '0'), to_str[0] - 'a')
        );

        std::vector<Move> legal_moves = board.get_pseudo_legal_moves(board.get_current_player());
        for (const auto& move : legal_moves) {
            if (move.from_loc == from_loc && move.to_loc == to_loc) {
                return move;
            }
        }
    }
    throw std::invalid_argument("Illegal or malformed move string: " + move_str);
}

int move_to_policy_index(const Move& move, Player p) {
    if (move.is_resignation()) return 0;
    BoardLocation rel_from = get_rel_loc(move.from_loc.row, move.from_loc.col, p);
    BoardLocation rel_to   = get_rel_loc(move.to_loc.row, move.to_loc.col, p);
    return (rel_from.row * 8 + rel_from.col) * 64 + (rel_to.row * 8 + rel_to.col);
}

Move policy_index_to_move(int index, Player p) {
    if (index == 0) return Move::Resign();
    int to_rel_idx = index % 64;
    int from_rel_idx = index / 64;
    BoardLocation abs_from = get_abs_loc(from_rel_idx / 8, from_rel_idx % 8, p);
    BoardLocation abs_to   = get_abs_loc(to_rel_idx / 8, to_rel_idx % 8, p);
    return Move(abs_from, abs_to, std::nullopt);
}

std::string get_san_string(const Move& move, const Board& board) {
     if (move.is_resignation()) return "RESIGN";
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
    if (move.is_resignation()) return "RESIGN";
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