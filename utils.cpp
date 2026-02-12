#include "utils.h"
#include "magic_utils.h" 
#include "thread_safe_queue.h"
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
#include <cstring>

namespace chaturaji_cpp {

// --- TensorPool Implementation ---

// Global static queues for recycling
static ThreadSafeQueue<std::unique_ptr<PlanesArray>>& get_planes_pool() {
    static ThreadSafeQueue<std::unique_ptr<PlanesArray>> pool;
    return pool;
}
static ThreadSafeQueue<std::unique_ptr<ScalarsArray>>& get_scalars_pool() {
    static ThreadSafeQueue<std::unique_ptr<ScalarsArray>> pool;
    return pool;
}
static ThreadSafeQueue<std::unique_ptr<PolicyArray>>& get_policy_pool() {
    static ThreadSafeQueue<std::unique_ptr<PolicyArray>> pool;
    return pool;
}
static ThreadSafeQueue<std::unique_ptr<ValueArray>>& get_value_pool() {
    static ThreadSafeQueue<std::unique_ptr<ValueArray>> pool;
    return pool;
}

std::unique_ptr<PlanesArray> TensorPool::acquire_planes() {
    auto item = get_planes_pool().try_pop();
    if (item) return std::move(*item);
    return std::make_unique<PlanesArray>();
}

std::unique_ptr<ScalarsArray> TensorPool::acquire_scalars() {
    auto item = get_scalars_pool().try_pop();
    if (item) return std::move(*item);
    return std::make_unique<ScalarsArray>();
}

std::unique_ptr<PolicyArray> TensorPool::acquire_policy() {
    auto item = get_policy_pool().try_pop();
    if (item) return std::move(*item);
    return std::make_unique<PolicyArray>();
}

std::unique_ptr<ValueArray> TensorPool::acquire_value() {
    auto item = get_value_pool().try_pop();
    if (item) return std::move(*item);
    return std::make_unique<ValueArray>();
}

void TensorPool::release_planes(std::unique_ptr<PlanesArray> ptr) {
    if(ptr) get_planes_pool().push(std::move(ptr));
}
void TensorPool::release_scalars(std::unique_ptr<ScalarsArray> ptr) {
    if(ptr) get_scalars_pool().push(std::move(ptr));
}
void TensorPool::release_policy(std::unique_ptr<PolicyArray> ptr) {
    if(ptr) get_policy_pool().push(std::move(ptr));
}
void TensorPool::release_value(std::unique_ptr<ValueArray> ptr) {
    if(ptr) get_value_pool().push(std::move(ptr));
}

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

    /**
     * @brief Transforms absolute board coordinates to player-relative coordinates.
     * Perspective is rotated so that the active player's "forward" is always 
     * toward row 0 in the relative grid.
     */
    BoardLocation get_rel_loc(int r, int c, Player p) {
        switch (p) {
            case Player::RED:    return {r, c};
            case Player::BLUE:   return {7 - c, r};
            case Player::YELLOW: return {7 - r, 7 - c};
            case Player::GREEN:  return {c, 7 - r};
            default: return {r, c};
        }
    }

    /**
     * @brief Transforms player-relative coordinates back to absolute board coordinates.
     */
    BoardLocation get_abs_loc(int r, int c, Player p) {
        switch (p) {
            case Player::RED:    return {r, c};
            case Player::BLUE:   return {c, 7 - r};
            case Player::YELLOW: return {7 - r, 7 - c};
            case Player::GREEN:  return {7 - c, r};
            default: return {r, c};
        }
    }

    // --- Spatial Move Plane Constants ---
    // Directions for Queen-like moves: N, S, E, W, NE, NW, SE, SW
    const int DR[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int DC[] = {0, 0, 1, -1, 1, -1, 1, -1};
    // Offsets for Knight jumps
    const int KNIGHT_DR[] = {-2, -2, -1, -1, 1, 1, 2, 2};
    const int KNIGHT_DC[] = {1, -1, 2, -2, 2, -2, 1, -1};
}

void board_to_tensors(const Board& board, float* out_planes, float* out_scalars) {
    std::memset(out_planes, 0, NN_INPUT_PLANES_SIZE * sizeof(float));
    std::memset(out_scalars, 0, NN_INPUT_SCALARS * sizeof(float));
    
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
    // PART B: SCALARS (18 Total)
    // ==========================================
    int current_scalar = 0;

    // --- Pre-calculate Material for all 4 players to allow relative comparison ---
    std::array<int, 4> mat_scores_abs;
    for(int p = 0; p < 4; ++p) {
        Player pl = static_cast<Player>(p);
        int mat = 0;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::PAWN)) * 1;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::KNIGHT)) * 3;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::BISHOP)) * 5;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::ROOK)) * 5;
        mat += magic_utils::pop_count(board.get_piece_bitboard(pl, PieceType::KING)) * 3;
        mat_scores_abs[p] = mat;
    }
    int my_mat = mat_scores_abs[cp_idx]; // Current player's material

    // --- Pre-fetch Points for all 4 players ---
    const auto& points_abs = board.get_player_points_array();
    int my_points = points_abs[cp_idx]; // Current player's points

    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_p_idx = (cp_idx + rel_i) % 4;
        Player p = static_cast<Player>(abs_p_idx);

        // A. Material (4 scalars) -> RELATIVE (My - Theirs) / 20.0
        out_scalars[current_scalar + 0 + rel_i] = static_cast<float>(my_mat - mat_scores_abs[abs_p_idx]) / 20.0f;
    }
    current_scalar += 4;

    // F. Active Status (4 scalars)
    uint8_t mask = board.get_active_mask();
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        out_scalars[current_scalar + rel_i] = (mask & (1 << abs_idx)) ? 1.0f : 0.0f;
    }
    current_scalar += 4;

    // G. Points (4 scalars) -> RELATIVE (My - Theirs) / 20.0
    for (int rel_i = 0; rel_i < 4; ++rel_i) {
        int abs_idx = (cp_idx + rel_i) % 4;
        out_scalars[current_scalar + rel_i] = static_cast<float>(my_points - points_abs[abs_idx]) / 20.0f;
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
    const std::array<double, 16>& rewards
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
    }
    
    // Copy all 16 values into the packed sample
    for(int i=0; i<16; ++i) {
        sample.values[i] = static_cast<float>(rewards[i]);
    }

    // 2. Pack Game State Scalars
    sample.full_move_number = board.get_full_move_number();
    sample.move_number_last_reset = board.get_move_number_of_last_reset();
    sample.active_mask = board.get_active_mask();
    sample.current_player = static_cast<uint8_t>(board.get_current_player());

    // 3. Calculate and Pack Supplemental Features
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
        sample.material_score[p] = static_cast<float>(mat) / 20.0f;
    }

    // 4. Pack Policy (Sparse)
    // Convert map to vector of pairs for sorting
    std::vector<std::pair<float, uint16_t>> sorted_policy;
    sorted_policy.reserve(policy.size());

    Player cp = board.get_current_player();
    for(const auto& item : policy) {
        int idx = move_to_policy_index(item.first, cp);
        // Ensure index is valid for our neural net size (4096)
        if(idx >= 0 && idx < NN_POLICY_SIZE) {
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

// --- Move Indexing (Spatial / Convolutional Strategy) ---

/**
 * @brief Encodes a move into a single integer index [0, 4095].
 * 
 * Strategy:
 * The policy head is viewed as a stack of 64 planes, each 8x8.
 * Index = (Plane_Index * 64) + Relative_From_Square_Index.
 * 
 * Planes 0-55: Queen-like slides (8 directions * 7 distances).
 * Planes 56-63: Knight jumps (8 possible jumps).
 */
int move_to_policy_index(const Move& move, Player p) {
    if (move.is_resignation()) return NN_POLICY_INDEX_RESIGN;

    // 1. Perspective Transformation
    BoardLocation rel_from = get_rel_loc(move.from_loc.row, move.from_loc.col, p);
    BoardLocation rel_to   = get_rel_loc(move.to_loc.row, move.to_loc.col, p);

    int dr = rel_to.row - rel_from.row;
    int dc = rel_to.col - rel_from.col;

    int plane_idx = -1;

    // 2. Check Knight Jumps (8 planes: 56-63)
    for (int k = 0; k < 8; ++k) {
        if (dr == KNIGHT_DR[k] && dc == KNIGHT_DC[k]) {
            plane_idx = 56 + k;
            break;
        }
    }

    // 3. Check Sliding Moves (56 planes: 8 dirs * 7 distances)
    if (plane_idx == -1) {
        for (int dir = 0; dir < 8; ++dir) {
            for (int dist = 1; dist <= 7; ++dist) {
                if (dr == DR[dir] * dist && dc == DC[dir] * dist) {
                    plane_idx = (dir * 7) + (dist - 1);
                    goto found;
                }
            }
        }
    }

found:
    if (plane_idx == -1) return 4096; // Fallback

    // Index = (Plane * 64) + (Row * 8 + Col)
    return (plane_idx * 64) + (rel_from.row * 8 + rel_from.col);
}

/**
 * @brief Decodes a spatial policy index back into a Move object.
 */
Move policy_index_to_move(int index, const Board& board) {
    // 1. Handle Resignation escape hatch
    if (index == 4096) return Move::Resign();

    // 2. Safety check for out-of-bounds indices
    if (index < 0 || index > 4096) return Move::Resign();

    Player p = board.get_current_player();

    int plane_idx = index / 64;
    int from_sq_idx = index % 64;

    int rel_from_r = from_sq_idx / 8;
    int rel_from_c = from_sq_idx % 8;
    int dr = 0, dc = 0;

    // 3. Decode Plane to relative displacement
    if (plane_idx < 56) {
        int dir = plane_idx / 7;
        int dist = (plane_idx % 7) + 1;
        dr = DR[dir] * dist;
        dc = DC[dir] * dist;
    } else {
        int k = plane_idx - 56;
        dr = KNIGHT_DR[k];
        dc = KNIGHT_DC[k];
    }

    int rel_to_r = rel_from_r + dr;
    int rel_to_c = rel_from_c + dc;

    // 4. Bound check relative coordinates
    if (rel_to_r < 0 || rel_to_r > 7 || rel_to_c < 0 || rel_to_c > 7) {
        return Move::Resign(); // Or a specific "IllegalMove" constant
    }

    // 5. Convert to Absolute coordinates
    BoardLocation abs_from = get_abs_loc(rel_from_r, rel_from_c, p);
    BoardLocation abs_to   = get_abs_loc(rel_to_r, rel_to_c, p);

    Move m(abs_from, abs_to);

    // 6. PAWN PROMOTION DETECTION
    int abs_from_idx = magic_utils::to_sq_idx(abs_from.row, abs_from.col);
    std::optional<Piece> piece_opt = board.get_piece_at_sq(abs_from_idx);

    // Safety: Only check promotion if there is actually a piece there
    if (piece_opt && piece_opt->piece_type == PieceType::PAWN) {
        bool is_promo = false;
        switch (p) {
            case Player::RED:    if (abs_to.row == 0) is_promo = true; break;
            case Player::BLUE:   if (abs_to.col == 7) is_promo = true; break;
            case Player::YELLOW: if (abs_to.row == 7) is_promo = true; break;
            case Player::GREEN:  if (abs_to.col == 0) is_promo = true; break;
        }
        
        if (is_promo) {
            m.promotion_piece_type = PieceType::ROOK;
        }
    }

    return m;
}

// --- Rank Probability & Expected Value ---

/**
 * @brief Calculates the ground-truth rank probabilities for a finished game.
 * Handles ties by distributing probability mass equally across tied ranks.
 * @param final_points The absolute points of all 4 players.
 * @return std::array<double, 16> Target rank probabilities (One-hot or split).
 */
std::array<double, 16> get_rank_probabilities_target(const std::array<int, 4>& final_points) {
    std::array<double, 16> target;
    target.fill(0.0);

    struct PScore { int p_idx; int score; };
    std::vector<PScore> sorted_scores;
    for(int i=0; i<4; ++i) sorted_scores.push_back({i, final_points[i]});

    // Sort descending by score
    std::sort(sorted_scores.begin(), sorted_scores.end(), [](auto& a, auto& b){ return a.score > b.score; });

    int i = 0;
    while (i < 4) {
        int j = i;
        // Find group of players with tied scores
        while (j < 4 && sorted_scores[j].score == sorted_scores[i].score) j++;

        int num_tied = j - i;
        // Calculate probability mass for this rank group (e.g., if 2 players tie for 1st, they share 1st and 2nd)
        double prob_per_rank = 1.0 / num_tied;

        for (int k = i; k < j; ++k) {
            int p_idx = sorted_scores[k].p_idx;
            for (int r = i; r < j; ++r) {
                target[p_idx * 4 + r] = prob_per_rank;
            }
        }
        i = j;
    }
    return target;
}

// --- Notation Utilities ---

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