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
#include <cmath>
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

    BoardLocation get_rel_loc(int r, int c, Player p) {
        switch (p) {
            case Player::RED:    return {r, c};
            case Player::BLUE:   return {7 - c, r};
            case Player::YELLOW: return {7 - r, 7 - c};
            case Player::GREEN:  return {c, 7 - r};
            default: return {r, c};
        }
    }

    BoardLocation get_abs_loc(int r, int c, Player p) {
        switch (p) {
            case Player::RED:    return {r, c};
            case Player::BLUE:   return {c, 7 - r};
            case Player::YELLOW: return {7 - r, 7 - c};
            case Player::GREEN:  return {7 - c, r};
            default: return {r, c};
        }
    }

    // --- Heuristic Helpers ---

    // Constants for connectivity bit-flood
    const Bitboard FILE_A = 0x0101010101010101ULL;
    const Bitboard FILE_H = 0x8080808080808080ULL;
    const Bitboard NOT_FILE_A = ~FILE_A;
    const Bitboard NOT_FILE_H = ~FILE_H;

    // Fast bitwise check if all pawns form a single connected component
    // Replaces BFS with iterative flood fill
    bool are_pawns_fully_connected(Bitboard pawns) {
        if (pawns == 0) return false;
        int count = magic_utils::pop_count(pawns);
        if (count <= 1) return true; // Single pawn (or zero) is technically "connected" or trivial

        // 1. Pick a start pawn (LSB)
        Bitboard flood = (pawns & -static_cast<int64_t>(pawns)); 
        Bitboard temp = 0;

        // 2. Flood fill iteratively until stable
        while (flood != temp) {
            temp = flood;
            // Expand in all 8 directions masked by actual pawns
            // Shifts: +1 (East), -1 (West), +8 (South), -8 (North)
            
            // East (+1): Check wrapping H->A (mask NOT_FILE_A)
            Bitboard east = (flood << 1) & NOT_FILE_A;
            // West (-1): Check wrapping A->H (mask NOT_FILE_H)
            Bitboard west = (flood >> 1) & NOT_FILE_H;
            
            Bitboard south = (flood << 8);
            Bitboard north = (flood >> 8);

            // Diagonals
            Bitboard ne = (north << 1) & NOT_FILE_A;
            Bitboard nw = (north >> 1) & NOT_FILE_H;
            Bitboard se = (south << 1) & NOT_FILE_A;
            Bitboard sw = (south >> 1) & NOT_FILE_H;

            flood |= (east | west | north | south | ne | nw | se | sw);
            flood &= pawns; // Constrain to pawns
        }

        return flood == pawns;
    }
}

void board_to_tensors(const Board& board, std::vector<float>& out_planes, std::vector<float>& out_scalars) {
    // 1. Prepare Buffers
    if (out_planes.size() != NN_INPUT_PLANES_SIZE) out_planes.resize(NN_INPUT_PLANES_SIZE);
    if (out_scalars.size() != NN_INPUT_SCALARS) out_scalars.resize(NN_INPUT_SCALARS);
    
    std::fill(out_planes.begin(), out_planes.end(), 0.0f);
    std::fill(out_scalars.begin(), out_scalars.end(), 0.0f);

    Player current_p = board.get_current_player();
    int cp_idx = static_cast<int>(current_p);

    // --- Helpers ---
    auto set_pixel = [&](int plane_idx, int r, int c, float value) {
        BoardLocation rel = get_rel_loc(r, c, current_p);
        int index = (plane_idx * BOARD_AREA) + (rel.row * 8 + rel.col);
        out_planes[index] = value;
    };

    auto set_bitboard_plane = [&](int plane_idx, Bitboard bb) {
        while(bb) {
            int sq_idx = magic_utils::pop_lsb(bb);
            BoardLocation abs_loc = magic_utils::from_sq_idx(sq_idx);
            set_pixel(plane_idx, abs_loc.row, abs_loc.col, 1.0f);
        }
    };

    // ==========================================
    // PART A: SPATIAL PLANES (28 Total)
    // ==========================================
    int current_plane = 0;

    // 1. Piece Placement (20 planes)
    for (int rel_i = 0; rel_i < 4; ++rel_i) { 
        int abs_p_idx = (cp_idx + rel_i) % 4;
        Player p_enum = static_cast<Player>(abs_p_idx);
        for (int pt_idx = 0; pt_idx < 5; ++pt_idx) { 
            PieceType pt_enum = UTIL_PIECE_TYPE_ORDER[pt_idx]; 
            Bitboard bb = board.get_piece_bitboard(p_enum, pt_enum);
            set_bitboard_plane(current_plane++, bb);
        }
    }

    // 2. X-Ray Attacks (4 planes)
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_p_idx = (cp_idx + rel_i) % 4;
        Player p = static_cast<Player>(abs_p_idx);
        
        Bitboard xray = 0ULL;
        Bitboard rooks = board.get_piece_bitboard(p, PieceType::ROOK);
        Bitboard bishops = board.get_piece_bitboard(p, PieceType::BISHOP);
        while(rooks) xray |= magic_utils::STATIC_ROOK_ATTACKS_EMPTY[magic_utils::pop_lsb(rooks)];
        while(bishops) xray |= magic_utils::STATIC_BISHOP_ATTACKS_EMPTY[magic_utils::pop_lsb(bishops)];
        
        set_bitboard_plane(current_plane++, xray);
    }

    // 3. Standard Attacks (4 planes)
    // Pre-calculate attack maps
    std::array<Bitboard, 4> all_attacks;
    for(int i = 0; i < 4; ++i) all_attacks[i] = board.get_squares_attacked_by(static_cast<Player>(i));

    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        set_bitboard_plane(current_plane++, all_attacks[(cp_idx + rel_i) % 4]);
    }
    
    // Check plane count
    if (current_plane != NN_INPUT_PLANES) {
        throw std::runtime_error("Implementation Error: Plane count mismatch.");
    }

    // ==========================================
    // PART B: SCALARS (34 Total)
    // ==========================================
    int current_scalar = 0;

    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_p_idx = (cp_idx + rel_i) % 4;
        Player p = static_cast<Player>(abs_p_idx);

        Bitboard pawns = board.get_piece_bitboard(p, PieceType::PAWN);
        Bitboard king = board.get_piece_bitboard(p, PieceType::KING);
        int pawn_cnt = magic_utils::pop_count(pawns);

        // A. Material (4 scalars)
        int mat = 0;
        mat += pawn_cnt * 1;
        mat += magic_utils::pop_count(board.get_piece_bitboard(p, PieceType::KNIGHT)) * 3;
        mat += magic_utils::pop_count(board.get_piece_bitboard(p, PieceType::BISHOP)) * 5;
        mat += magic_utils::pop_count(board.get_piece_bitboard(p, PieceType::ROOK)) * 5;
        mat += magic_utils::pop_count(board.get_piece_bitboard(p, PieceType::KING)) * 3;
        out_scalars[current_scalar + 0 + rel_i] = static_cast<float>(mat) / 50.0f;

        // B. Pawn Count (4 scalars)
        out_scalars[current_scalar + 4 + rel_i] = static_cast<float>(pawn_cnt) / 4.0f;

        // C. Connected Pawns (4 scalars)
        out_scalars[current_scalar + 8 + rel_i] = are_pawns_fully_connected(pawns) ? 1.0f : 0.0f;

        // D. Avg Pawn Dist (4 scalars)
        float avg_dist = 0.0f;
        if (king && pawn_cnt > 0) {
            int k_sq = magic_utils::get_lsb_index(king);
            int total_dist = 0;
            Bitboard temp_p = pawns;
            while(temp_p) {
                total_dist += magic_utils::CHEBYSHEV_DIST[k_sq][magic_utils::pop_lsb(temp_p)];
            }
            avg_dist = static_cast<float>(total_dist) / pawn_cnt;
        }
        out_scalars[current_scalar + 12 + rel_i] = avg_dist / 8.0f;

        // E. King Safe Moves (4 scalars)
        int safe_moves = 0;
        if (king) {
            int k_sq = magic_utils::get_lsb_index(king);
            Bitboard neighborhood = magic_utils::STATIC_KING_ATTACKS[k_sq];
            Bitboard own_pieces = board.get_player_bitboard(p);
            Bitboard enemy_attacks = 0ULL;
            for(int i = 0; i < 4; ++i) {
                if(i != abs_p_idx) enemy_attacks |= all_attacks[i];
            }
            Bitboard safe_squares = neighborhood & ~own_pieces & ~enemy_attacks;
            safe_moves = magic_utils::pop_count(safe_squares);
        }
        out_scalars[current_scalar + 16 + rel_i] = static_cast<float>(safe_moves) / 8.0f;
    }
    current_scalar += 20;

    // F. Active Status (4 scalars)
    uint8_t mask = board.get_active_mask();
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        out_scalars[current_scalar + rel_i] = (mask & (1 << abs_idx)) ? 1.0f : 0.0f;
    }
    current_scalar += 4;

    // G. Points (4 scalars)
    const auto& points = board.get_player_points();
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        out_scalars[current_scalar + rel_i] = static_cast<float>(points[abs_idx]) / 50.0f;
    }
    current_scalar += 4;

    // H. 50-Move Clock (1 scalar)
    int moves_since_reset = board.get_full_move_number() - board.get_move_number_of_last_reset();
    out_scalars[current_scalar++] = std::min(1.0f, static_cast<float>(moves_since_reset) / 50.0f);

    // I. In-Check (4 scalars)
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        Bitboard king = board.get_piece_bitboard(static_cast<Player>(abs_idx), PieceType::KING);
        Bitboard stressors = 0;
        for(int opp=0; opp<4; ++opp) {
            if(opp != abs_idx) stressors |= all_attacks[opp];
        }
        out_scalars[current_scalar + rel_i] = (king & stressors) ? 1.0f : 0.0f;
    }
    current_scalar += 4;

    // J. Active Opponent Count (1 scalar)
    int total_active = magic_utils::pop_count(static_cast<Bitboard>(board.get_active_mask()));
    out_scalars[current_scalar++] = static_cast<float>(total_active - 1) / 3.0f;

    // Validation
    if (current_scalar != NN_INPUT_SCALARS) {
        throw std::runtime_error("Implementation Error: Scalar count mismatch.");
    }
}

PackedSample create_packed_sample(
    const Board& board, 
    const std::map<Move, double>& policy, 
    const std::array<double, 4>& rewards
) {
    PackedSample sample;
    // Clear memory to ensure padding bytes are 0 (avoids junk data)
    std::memset(&sample, 0, sizeof(PackedSample));

    // 1. Pack Basic Bitboards and Points
    for(int p = 0; p < 4; ++p) {
        Player pl = static_cast<Player>(p);
        for(int t = 0; t < 5; ++t) {
            sample.piece_bitboards[p][t] = board.get_piece_bitboard(pl, static_cast<PieceType>(t + 1));
        }
        sample.attack_bitboards[p] = board.get_squares_attacked_by(pl);
        sample.player_points[p] = board.get_player_points_array()[p];
        sample.values[p] = static_cast<float>(rewards[p]);
    }

    // 2. Pack Game State Scalars
    sample.full_move_number = board.get_full_move_number();
    sample.move_number_last_reset = board.get_move_number_of_last_reset();
    sample.active_mask = board.get_active_mask();
    sample.current_player = static_cast<uint8_t>(board.get_current_player());

    // 3. Calculate and Pack New Features
    // Note: We need attack maps for King Safety
    std::array<Bitboard, 4> all_attacks;
    for(int i = 0; i < 4; ++i) all_attacks[i] = sample.attack_bitboards[i];

    for (int p = 0; p < 4; ++p) {
        Player pl = static_cast<Player>(p);
        
        // F. X-Ray Attacks
        Bitboard xray = 0ULL;
        Bitboard rooks = board.get_piece_bitboard(pl, PieceType::ROOK);
        Bitboard bishops = board.get_piece_bitboard(pl, PieceType::BISHOP);
        
        while(rooks) {
            xray |= magic_utils::STATIC_ROOK_ATTACKS_EMPTY[magic_utils::pop_lsb(rooks)];
        }
        while(bishops) {
            xray |= magic_utils::STATIC_BISHOP_ATTACKS_EMPTY[magic_utils::pop_lsb(bishops)];
        }
        sample.xray_attack_bitboards[p] = xray;

        // A. Total Material
        int mat = 0;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::PAWN)) * 1;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::KNIGHT)) * 3;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::BISHOP)) * 5;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::ROOK)) * 5;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::KING)) * 3;
        sample.material_score[p] = static_cast<float>(mat) / 50.0f;

        // B. Pawn Count
        Bitboard pawns = board.get_piece_bitboard(pl, PieceType::PAWN);
        int p_cnt = magic_utils::pop_count(pawns);
        sample.pawn_count[p] = static_cast<float>(p_cnt) / 4.0f;

        // C. Connected Pawns
        sample.pawns_connected[p] = are_pawns_fully_connected(pawns) ? 1.0f : 0.0f;

        // D. Average Pawn Distance
        Bitboard kbb = board.get_piece_bitboard(pl, PieceType::KING);
        float dist = 0.0f;
        if(kbb && p_cnt > 0) {
            int k_sq = magic_utils::get_lsb_index(kbb);
            int total = 0;
            Bitboard temp_p = pawns;
            while(temp_p) {
                // Optimized: Use Static Lookup for distance
                total += magic_utils::CHEBYSHEV_DIST[k_sq][magic_utils::pop_lsb(temp_p)];
            }
            dist = static_cast<float>(total) / p_cnt;
        }
        sample.avg_pawn_dist[p] = dist / 8.0f;

        // E. King Safe Moves
        int safe = 0;
        if(kbb) {
            int k_sq = magic_utils::get_lsb_index(kbb);
            Bitboard enemy_attacks = 0ULL;
            for(int opp=0; opp<4; ++opp) {
                if(opp != p) enemy_attacks |= all_attacks[opp];
            }
            // Optimized: Use Static Lookup + Bitmasks
            Bitboard neighborhood = magic_utils::STATIC_KING_ATTACKS[k_sq];
            Bitboard own_pieces = board.get_player_bitboard(pl);
            Bitboard safe_mask = neighborhood & ~own_pieces & ~enemy_attacks;
            safe = magic_utils::pop_count(safe_mask);
        }
        sample.king_safe_moves[p] = static_cast<float>(safe) / 8.0f;
    }

    // 4. Pack Policy (Sparse)
    // Convert map to vector of pairs for sorting
    std::vector<std::pair<float, uint16_t>> sorted_policy;
    sorted_policy.reserve(policy.size());

    Player cp = board.get_current_player();
    for(const auto& item : policy) {
        int idx = move_to_policy_index(item.first, cp);
        // Ensure index is valid for our neural net size (4096)
        if(idx >= 0 && idx < 4096) {
            sorted_policy.push_back({static_cast<float>(item.second), static_cast<uint16_t>(idx)});
        }
    }

    // Sort descending by probability to keep the most important moves 
    // if we exceed MAX_STORED_MOVES (shouldn't happen normally)
    std::sort(sorted_policy.begin(), sorted_policy.end(), [](const auto& a, const auto& b){
        return a.first > b.first;
    });

    int count = std::min(static_cast<int>(sorted_policy.size()), MAX_STORED_MOVES);
    sample.num_policy_entries = count;

    for(int i = 0; i < count; ++i) {
        sample.move_probs[i] = sorted_policy[i].first;
        sample.move_indices[i] = sorted_policy[i].second;
    }

    return sample;
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

        MoveList legal_moves;
        board.get_pseudo_legal_moves(board.get_current_player(), legal_moves);
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