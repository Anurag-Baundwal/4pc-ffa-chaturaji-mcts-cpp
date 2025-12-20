#include "utils.h"
#include "magic_utils.h" // Include for magic_utils::BOARD_SIZE and other utilities
#include <stdexcept>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm> // For std::max, std::min, std::fill_n
#include <iostream>

namespace chaturaji_cpp {

// PIECE_TYPE_ORDER_PHASE1 and PIECE_TYPE_TO_INDEX_PHASE1 are now defined in board.cpp's anonymous namespace
// For utils.cpp, we need them too if they are not exposed globally.
const std::vector<PieceType> UTIL_PIECE_TYPE_ORDER = {
    PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK,
    PieceType::KING
};
const std::map<PieceType, int> UTIL_PIECE_TYPE_TO_INDEX = []{
    std::map<PieceType, int> m;
    for(size_t i=0; i<UTIL_PIECE_TYPE_ORDER.size(); ++i) {
        m[UTIL_PIECE_TYPE_ORDER[i]] = static_cast<int>(i);
    }
    return m;
}();


// --- Updated: Returns raw float vector instead of torch::Tensor ---
std::vector<float> board_to_floats(const Board& board) {
    constexpr int BOARD_AREA = magic_utils::BOARD_SIZE * magic_utils::BOARD_SIZE; // 64
    constexpr int NUM_ACTUAL_PIECE_TYPES = 5; // P, N, B, R, K
    constexpr int NUM_PIECE_CHANNELS_ONLY = 4 * NUM_ACTUAL_PIECE_TYPES; // 20
    
    // Total channels: 20 (pieces) + 4 (active status) + 4 (player turn) + 4 (points) + 1 (50-move) = 33 channels
    constexpr int NUM_CHANNELS_TOTAL = NUM_PIECE_CHANNELS_ONLY + 4 + 4 + 4 + 1; 

    // Initialize flat vector with zeros. Size: 33 * 64 = 2112 floats.
    // Layout is NCHW (implicitly [33, 8, 8]), so Channel 0 is indexes 0-63, Channel 1 is 64-127, etc.
    std::vector<float> tensor_data(NUM_CHANNELS_TOTAL * BOARD_AREA, 0.0f);

    // Helper to fill a specific channel plane with a constant value
    auto fill_plane = [&](int channel_idx, float value) {
        if (value == 0.0f) return; // Vector already init to 0.0f
        int offset = channel_idx * BOARD_AREA;
        std::fill_n(tensor_data.begin() + offset, BOARD_AREA, value);
    };

    // Helper to set a specific square in a channel
    auto set_pixel = [&](int channel_idx, int sq_idx, float value) {
        int index = (channel_idx * BOARD_AREA) + sq_idx;
        tensor_data[index] = value;
    };

    // Piece Placement Channels (0-19)
    for (int p_idx = 0; p_idx < 4; ++p_idx) { // Iterate players
        Player player_enum = static_cast<Player>(p_idx);
        for (int pt_idx = 0; pt_idx < NUM_ACTUAL_PIECE_TYPES; ++pt_idx) { // Iterate piece types
            PieceType piece_type_enum = UTIL_PIECE_TYPE_ORDER[pt_idx]; // Get PieceType from index
            Bitboard bb = board.get_piece_bitboard(player_enum, piece_type_enum);
            int channel = p_idx * NUM_ACTUAL_PIECE_TYPES + pt_idx;

            Bitboard temp_bb = bb;
            while(temp_bb) {
                int sq_idx = magic_utils::pop_lsb(temp_bb); // Use magic_utils::pop_lsb
                // Note: sq_idx (0-63) maps directly to the inner dimension of NCHW for 8x8
                if (channel >= 0 && channel < NUM_PIECE_CHANNELS_ONLY) {
                    set_pixel(channel, sq_idx, 1.0f);
                } else {
                    std::cerr << "Warning: Invalid piece channel in board_to_floats: " << channel << std::endl;
                }
            }
        }
    }

    // Active Player Status Channels (20-23)
    const auto& active_players_set = board.get_active_players();
    int active_status_channel_offset = NUM_PIECE_CHANNELS_ONLY; // Starts at 20
    for (int i = 0; i < 4; ++i) {
        Player p_iter = static_cast<Player>(i);
        float val_to_fill = active_players_set.count(p_iter) ? 1.0f : 0.0f;
        fill_plane(active_status_channel_offset + i, val_to_fill);
    }

    // Current Player Channels (24-27)
    Player current_player = board.get_current_player();
    int current_player_channel_offset = active_status_channel_offset + 4; // Starts at 24
    int current_player_idx_val = static_cast<int>(current_player);
    // Note: Vector init to 0.0f, so we only need to set the current player's plane to 1.0f
    if (current_player_idx_val >= 0 && current_player_idx_val < 4) {
        fill_plane(current_player_channel_offset + current_player_idx_val, 1.0f);
    }

    // Player Points Channels (28-31)
    const auto& points = board.get_player_points();
    int points_channel_offset = current_player_channel_offset + 4; // Starts at 28
    for (int i = 0; i < 4; ++i) {
        Player p_iter = static_cast<Player>(i);
        float player_points_val = 0.0f;
        auto it = points.find(p_iter);
        if(it != points.end()){
            player_points_val = static_cast<float>(it->second);
        }
        // Normalize points 
        float val_to_fill = player_points_val / 100.0f; 
        fill_plane(points_channel_offset + i, val_to_fill);
    }

    // 50-Move Rule Counter Channel (32)
    int counter_channel_idx = points_channel_offset + 4; // Starts at 32 (this is the last channel, index 32)
    int moves_since_reset = board.get_full_move_number() - board.get_move_number_of_last_reset();
    float normalized_count = std::max(0.0f, std::min(1.0f, static_cast<float>(moves_since_reset) / 50.0f));
    
    if (counter_channel_idx == NUM_CHANNELS_TOTAL - 1) {
        fill_plane(counter_channel_idx, normalized_count);
    } else {
        throw std::runtime_error("Internal error: Tensor channel dimension mismatch for 50-move counter.");
    }

    return tensor_data;
}


int move_to_policy_index(const Move& move) {
    int fr_row = move.from_loc.row;
    int fr_col = move.from_loc.col;
    int to_row = move.to_loc.row;
    int to_col = move.to_loc.col;

    // Use magic_utils::BOARD_SIZE for bounds checking
    if (fr_row < 0 || fr_row >= magic_utils::BOARD_SIZE || fr_col < 0 || fr_col >= magic_utils::BOARD_SIZE ||
        to_row < 0 || to_row >= magic_utils::BOARD_SIZE || to_col < 0 || to_col >= magic_utils::BOARD_SIZE) {
        throw std::out_of_range("Move coordinates are out of board bounds for policy index.");
    }
    // Use magic_utils::BOARD_SIZE for calculations
    int from_index = fr_row * magic_utils::BOARD_SIZE + fr_col;
    int to_index = to_row * magic_utils::BOARD_SIZE + to_col;
    return from_index * (magic_utils::BOARD_SIZE * magic_utils::BOARD_SIZE) + to_index; // Range 0 to 63*64 + 63 = 4095
}


Move policy_index_to_move(int index) {
    // Use magic_utils::BOARD_SIZE for bounds checking and calculations
    if (index < 0 || index >= (magic_utils::BOARD_SIZE * magic_utils::BOARD_SIZE * magic_utils::BOARD_SIZE * magic_utils::BOARD_SIZE)) { // 64*64 = 4096 possibilities
         throw std::out_of_range("Policy index " + std::to_string(index) + " is out of bounds (0-4095).");
    }
    int to_index = index % (magic_utils::BOARD_SIZE * magic_utils::BOARD_SIZE);
    int from_index = index / (magic_utils::BOARD_SIZE * magic_utils::BOARD_SIZE);

    int to_row = to_index / magic_utils::BOARD_SIZE;
    int to_col = to_index % magic_utils::BOARD_SIZE;
    int from_row = from_index / magic_utils::BOARD_SIZE;
    int from_col = from_index % magic_utils::BOARD_SIZE;

    return Move(BoardLocation(from_row, from_col), BoardLocation(to_row, to_col), std::nullopt);
}

std::string get_san_string(const Move& move, const Board& board) {
     std::stringstream ss;
     // Get piece at 'from' location using bitboards and magic_utils::to_sq_idx
     std::optional<Piece> from_piece_opt = board.get_piece_at_sq(magic_utils::to_sq_idx(move.from_loc.row, move.from_loc.col));
     // Check if 'to' location has a piece (for capture 'x') using bitboards and magic_utils::to_sq_idx
     std::optional<Piece> to_piece_opt = board.get_piece_at_sq(magic_utils::to_sq_idx(move.to_loc.row, move.to_loc.col)); 

     if (!from_piece_opt) {
         return "ERROR_NO_FROM_PIECE"; 
     }
     PieceType from_type = from_piece_opt->piece_type;

     switch(from_type) {
        case PieceType::KNIGHT: ss << 'N'; break;
        case PieceType::BISHOP: ss << 'B'; break;
        case PieceType::ROOK:   ss << 'R'; break;
        case PieceType::KING:   ss << 'K'; break;
        case PieceType::PAWN: break;
        default: ss << '?'; break; // Should not happen with current PieceTypes
     }
     ss << static_cast<char>('a' + move.from_loc.col);
     ss << (magic_utils::BOARD_SIZE - move.from_loc.row); // Use magic_utils::BOARD_SIZE
     if (to_piece_opt) {
         ss << 'x';
     }
     ss << static_cast<char>('a' + move.to_loc.col);
     ss << (magic_utils::BOARD_SIZE - move.to_loc.row); // Use magic_utils::BOARD_SIZE
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
    ss << (magic_utils::BOARD_SIZE - move.from_loc.row); // Use magic_utils::BOARD_SIZE
    ss << static_cast<char>('a' + move.to_loc.col);
    ss << (magic_utils::BOARD_SIZE - move.to_loc.row); // Use magic_utils::BOARD_SIZE

    if (move.promotion_piece_type) {
          switch(*move.promotion_piece_type) {
            case PieceType::ROOK:   ss << 'r'; break;
            default: break; 
         }
    }
    return ss.str();
}

} // namespace chaturaji_cpp