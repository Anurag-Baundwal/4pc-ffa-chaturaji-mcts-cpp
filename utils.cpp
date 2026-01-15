#include "utils.h"
#include "magic_utils.h" 
#include <stdexcept>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm> 
#include <iostream>
#include <regex> // Required for robust parsing
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

// Local helpers for piece ordering
namespace {
    const std::vector<PieceType> UTIL_PIECE_TYPE_ORDER = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::KING
    };

    // Helper to rotate absolute coordinates to current player's perspective
    // This makes the current player always appear at the "Bottom" (Row 7)
    BoardLocation get_rel_loc(int r, int c, Player p) {
        switch (p) {
            case Player::RED:    return {r, c};              // No change
            case Player::BLUE:   return {7 - c, r};     // Rotate 90 CW
            case Player::YELLOW: return {7 - r, 7 - c}; // Rotate 180
            case Player::GREEN:  return {c, 7 - r};     // Rotate 90 CCW
            default: return {r, c};
        }
    }

    // Helper to rotate relative coordinates back to absolute board space
    BoardLocation get_abs_loc(int r, int c, Player p) {
        switch (p) {
            case Player::RED:    return {r, c};
            case Player::BLUE:   return {c, 7 - r};
            case Player::YELLOW: return {7 - r, 7 - c};
            case Player::GREEN:  return {7 - c, r};
            default: return {r, c};
        }
    }
}

std::vector<float> board_to_floats(const Board& board) {
    std::vector<float> tensor_data(NN_INPUT_SIZE, 0.0f);
    Player current_p = board.get_current_player();
    int cp_idx = static_cast<int>(current_p);

    auto fill_plane = [&](int channel_idx, float value) {
        if (value == 0.0f) return; 
        int offset = channel_idx * BOARD_AREA;
        std::fill_n(tensor_data.begin() + offset, BOARD_AREA, value);
    };

    auto set_pixel = [&](int channel_idx, int r, int c, float value) {
        // Apply perspective rotation
        BoardLocation rel = get_rel_loc(r, c, current_p);
        int index = (channel_idx * BOARD_AREA) + (rel.row * 8 + rel.col);
        tensor_data[index] = value;
    };

    // --- Relative Feature Encoding ---
    // Inputs are rotated so that index 0 always represents the current player.
    // Mapping: Rel 0 = Current, Rel 1 = Next, Rel 2 = Opposite, Rel 3 = Previous.
    // 1. Piece Placement (0-19)
    for (int rel_i = 0; rel_i < 4; ++rel_i) { 
        int abs_p_idx = (cp_idx + rel_i) % 4;
        Player p_enum = static_cast<Player>(abs_p_idx);
        for (int pt_idx = 0; pt_idx < 5; ++pt_idx) { 
            PieceType pt_enum = UTIL_PIECE_TYPE_ORDER[pt_idx]; 
            Bitboard bb = board.get_piece_bitboard(p_enum, pt_enum);
            int channel = (rel_i * 5) + pt_idx;
            while(bb) {
                int sq_idx = magic_utils::pop_lsb(bb);
                BoardLocation abs_loc = magic_utils::from_sq_idx(sq_idx);
                set_pixel(channel, abs_loc.row, abs_loc.col, 1.0f);
            }
        }
    }

    // 2. Active Status (20-23)
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        fill_plane(20 + rel_i, board.get_active_players().count(static_cast<Player>((cp_idx + rel_i) % 4)) ? 1.0f : 0.0f);
    }

    // 3. Points (24-27)
    const auto& points = board.get_player_points();
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        float pts = static_cast<float>(points.at(static_cast<Player>((cp_idx + rel_i) % 4)));
        fill_plane(24 + rel_i, pts / 100.0f);
    }

    // 4. 50-Move Clock (28)
    int moves_since_reset = board.get_full_move_number() - board.get_move_number_of_last_reset();
    fill_plane(28, std::min(1.0f, static_cast<float>(moves_since_reset) / 50.0f));

    // 5. Attack Planes (29-32)
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        Bitboard bb = board.get_squares_attacked_by(static_cast<Player>((cp_idx + rel_i) % 4));
        while(bb) {
            int sq_idx = magic_utils::pop_lsb(bb);
            BoardLocation abs_loc = magic_utils::from_sq_idx(sq_idx);
            set_pixel(29 + rel_i, abs_loc.row, abs_loc.col, 1.0f);
        }
    }

    // 6. In-Check Planes (33-36)
    std::array<Bitboard, 4> all_atks;
    for(int i=0; i<4; ++i) all_atks[i] = board.get_squares_attacked_by(static_cast<Player>(i));
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        Bitboard king = board.get_piece_bitboard(static_cast<Player>(abs_idx), PieceType::KING);
        Bitboard stressors = 0;
        for(int opp=0; opp<4; ++opp) if(opp != abs_idx) stressors |= all_atks[opp];
        if (king & stressors) fill_plane(33 + rel_i, 1.0f);
    }

    return tensor_data;
}

Move parse_string_to_move(const Board& board, const std::string& move_str) {
    // 1. Handle Resignations / Timeouts explicitly
    if (move_str == "R" || move_str == "T" || move_str == "RESIGN") {
        return Move::Resign();
    }

    // 2. Extract coordinates (e.g., 'c1', 'b2')
    // Matches "a-h" followed by "1-8".
    // Ignore surrounding chars (K, x, +, #, =R).
    // Using simple regex to capture the first two valid squares found.
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

        // 3. Match against legal moves
        // We do NOT strictly compare strings. If from/to match, it's the move.
        // This implicitly handles promotions because in Chaturaji, 
        // a pawn moving to the last rank ONLY has one legal move (promotion to Rook).
        std::vector<Move> legal_moves = board.get_pseudo_legal_moves(board.get_current_player());
        
        for (const auto& move : legal_moves) {
            if (move.from_loc == from_loc && move.to_loc == to_loc) {
                return move;
            }
        }
        
        // Debug info if not found (optional)
        // std::cerr << "Coords found: " << from_str << "->" << to_str << " but not legal." << std::endl;
    }

    throw std::invalid_argument("Illegal or malformed move string: " + move_str);
}

int move_to_policy_index(const Move& move, Player p) {
    if (move.is_resignation()) {
        return 0; // Index 0 represents Claiming the Win (Resignation)
    }

    BoardLocation rel_from = get_rel_loc(move.from_loc.row, move.from_loc.col, p);
    BoardLocation rel_to   = get_rel_loc(move.to_loc.row, move.to_loc.col, p);

    int from_index = rel_from.row * 8 + rel_from.col;
    int to_index = rel_to.row * 8 + rel_to.col;
    return from_index * 64 + to_index;
}

Move policy_index_to_move(int index, Player p) {
    if (index == 0) {
        return Move::Resign();
    }

    int to_rel_idx = index % 64;
    int from_rel_idx = index / 64;

    BoardLocation from_rel(from_rel_idx / 8, from_rel_idx % 8);
    BoardLocation to_rel(to_rel_idx / 8, to_rel_idx % 8);

    BoardLocation abs_from = get_abs_loc(from_rel.row, from_rel.col, p);
    BoardLocation abs_to   = get_abs_loc(to_rel.row, to_rel.col, p);

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