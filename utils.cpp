#include "utils.h"
#include <stdexcept>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>

namespace chaturaji_cpp {

// Define the consistent order of piece types for tensor channels
const std::vector<PieceType> PIECE_TYPE_ORDER = {
    PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK,
    PieceType::QUEEN, PieceType::KING, PieceType::ONE_POINT_QUEEN, PieceType::DEAD_KING
    // Ensure this order matches the Python version (8 types)
};
// Helper map for quick lookup
const std::map<PieceType, int> PIECE_TYPE_TO_INDEX = []{
    std::map<PieceType, int> m;
    for(int i=0; i<PIECE_TYPE_ORDER.size(); ++i) {
        m[PIECE_TYPE_ORDER[i]] = i;
    }
    return m;
}();


torch::Tensor board_to_tensor(const Board& board, torch::Device device) {
    // Create tensor options: float, target device
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    // Initialize zero tensor: shape [41, 8, 8] (Channels, Height, Width)
    constexpr int NUM_CHANNELS = 41; // Define total channels
    torch::Tensor tensor = torch::zeros({NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE}, options);

    const auto& grid = board.get_board_grid();
    const auto& points = board.get_player_points();
    Player current_player = board.get_current_player();

    // --- Piece Placement Channels (0-31) ---
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            const auto& piece_opt = grid[r][c];
            if (piece_opt) {
                const Piece& piece = *piece_opt;
                int player_idx = static_cast<int>(piece.player);

                // Find the index of the piece type in our defined order
                 auto it = PIECE_TYPE_TO_INDEX.find(piece.piece_type);
                 if (it != PIECE_TYPE_TO_INDEX.end()) {
                     int type_idx = it->second;
                    // Calculate channel index: Player offset + Piece type offset
                    int channel = player_idx * PIECE_TYPE_ORDER.size() + type_idx;

                    // Safety check channel bounds
                    if (channel >= 0 && channel < 32) {
                         tensor[channel][r][c] = 1.0f;
                    } else {
                         // Handle error or warning: Invalid channel calculated
                         // std::cerr << "Warning: Invalid piece channel calculated: " << channel << std::endl;
                    }
                 } else {
                      // Handle error or warning: Piece type not found in order map
                      // std::cerr << "Warning: Piece type not found in PIECE_TYPE_ORDER map." << std::endl;
                 }
            }
        }
    }

    // --- Current Player Channels (32-35) ---
    int current_player_channel = 32 + static_cast<int>(current_player);
    if (current_player_channel >= 32 && current_player_channel < 36) {
         tensor[current_player_channel].fill_(1.0f); // Fill entire 8x8 plane
    }

    // --- Player Points Channels (36-39) ---
    for (int i = 0; i < 4; ++i) {
        Player p = static_cast<Player>(i);
        float player_points = 0.0f;
        auto it = points.find(p);
        if(it != points.end()){
            player_points = static_cast<float>(it->second);
        }
        // Normalize points
        tensor[36 + i].fill_(player_points / 100.0f);
    }

        // --- NEW: 50-Move Rule Counter Channel (40) ---
    // Calculate moves since last pawn move or capture
    int moves_since_reset = board.get_full_move_number() - board.get_move_number_of_last_reset();

    // Normalize the count (e.g., scale to 0.0 - 1.0 range based on the 50 move limit)
    // Clip the value between 0.0 and 1.0, as values beyond the limit don't usually carry extra meaning for the NN.
    float normalized_count = std::max(0.0f, std::min(1.0f, static_cast<float>(moves_since_reset) / 50.0f));

    // Check if the tensor has the expected number of channels before accessing channel 40
    // This check uses the constant defined earlier.
    if (tensor.size(0) == NUM_CHANNELS) {
         tensor[40].fill_(normalized_count); // Fill the entire 8x8 plane of the 41st channel (index 40)
    } else {
        // This should not happen if the tensor was initialized correctly above.
        throw std::runtime_error("Internal error: Tensor channel dimension mismatch when adding 50-move counter.");
    }
    // --- END NEW BLOCK ---

    // Add the batch dimension: [1, 41, 8, 8]
    return tensor.unsqueeze(0);
}


int move_to_policy_index(const Move& move) {
    int fr_row = move.from_loc.row;
    int fr_col = move.from_loc.col;
    int to_row = move.to_loc.row;
    int to_col = move.to_loc.col;

    // Validate coordinates (optional but good practice)
    if (fr_row < 0 || fr_row >= BOARD_SIZE || fr_col < 0 || fr_col >= BOARD_SIZE ||
        to_row < 0 || to_row >= BOARD_SIZE || to_col < 0 || to_col >= BOARD_SIZE) {
        throw std::out_of_range("Move coordinates are out of board bounds.");
    }

    // Calculate index: 8*8 possibilities for 'to' square, 8*8 possibilities for 'from' square
    // int index = (fr_row * BOARD_SIZE + fr_col) * (BOARD_SIZE * BOARD_SIZE) + (to_row * BOARD_SIZE + to_col);
    // The python implementation used: fr * 64 + to => fr_row * 8 * 64 + fr_col * 64 + to_row * 8 + to_col = fr_row*512 + fr_col*64 + to_row*8 + to_col
    // Let's double check the python index calculation:
    // fr = move.from_loc.row * 8 + move.from_loc.col -> 0 to 63
    // to = move.to_loc.row * 8 + move.to_loc.col   -> 0 to 63
    // return fr * 64 + to -> (0..63)*64 + (0..63) -> 0 to 4032 + 63 -> 0 to 4095. Correct.
    int from_index = fr_row * BOARD_SIZE + fr_col;
    int to_index = to_row * BOARD_SIZE + to_col;

    return from_index * (BOARD_SIZE * BOARD_SIZE) + to_index;
}


Move policy_index_to_move(int index) {
    if (index < 0 || index >= (BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE)) {
         throw std::out_of_range("Policy index is out of bounds (0-4095).");
    }

    int to_index = index % (BOARD_SIZE * BOARD_SIZE);
    int from_index = index / (BOARD_SIZE * BOARD_SIZE);

    int to_row = to_index / BOARD_SIZE;
    int to_col = to_index % BOARD_SIZE;
    int from_row = from_index / BOARD_SIZE;
    int from_col = from_index % BOARD_SIZE;

    // Promotion is not encoded in this index
    return Move(BoardLocation(from_row, from_col), BoardLocation(to_row, to_col), std::nullopt);
}

std::string get_san_string(const Move& move, const Board& board) {
     std::stringstream ss;
     const auto& from_piece_opt = board.get_board_grid()[move.from_loc.row][move.from_loc.col];
     const auto& to_piece_opt = board.get_board_grid()[move.to_loc.row][move.to_loc.col]; // Piece being captured (if any)

     if (!from_piece_opt) {
         return "ERROR_NO_FROM_PIECE"; // Should not happen for valid moves
     }

     PieceType from_type = from_piece_opt->piece_type;

     // Piece Type (except Pawns)
     switch(from_type) {
        case PieceType::KNIGHT: ss << 'N'; break;
        case PieceType::BISHOP: ss << 'B'; break;
        case PieceType::ROOK:   ss << 'R'; break;
        case PieceType::QUEEN:  ss << 'Q'; break;
        case PieceType::KING:   ss << 'K'; break;
        case PieceType::ONE_POINT_QUEEN: ss << 'O'; break; // Or 'Q'? 'O' used in tensor
        case PieceType::PAWN: /* No letter */ break;
        case PieceType::DEAD_KING: ss << 'k'; break; // Unlikely to move, but...
        default: ss << '?'; break;
     }

     // TODO: Add disambiguation if needed (e.g., Raxd1 vs Rfx1)
     // This requires checking other pieces of the same type that can move to the same square.
     // Skipping disambiguation for now, like the Python version.

     // From square (sometimes omitted in SAN, but Python version included it)
     // Let's follow the python version's explicit format: {Piece}{From}{Capture}{To}{Promo}
     ss << static_cast<char>('a' + move.from_loc.col);
     ss << (BOARD_SIZE - move.from_loc.row);

     // Capture indicator
     if (to_piece_opt) {
         ss << 'x';
     } else {
         // Python used '-' for non-capture, standard SAN doesn't. Let's omit it.
         // ss << '-';
     }

     // To square
     ss << static_cast<char>('a' + move.to_loc.col);
     ss << (BOARD_SIZE - move.to_loc.row);

     // Promotion
     if (move.promotion_piece_type) {
         ss << '=';
          switch(*move.promotion_piece_type) {
            case PieceType::KNIGHT: ss << 'N'; break;
            case PieceType::BISHOP: ss << 'B'; break;
            case PieceType::ROOK:   ss << 'R'; break;
            case PieceType::QUEEN:  ss << 'Q'; break;
            case PieceType::ONE_POINT_QUEEN: ss << 'O'; break; // Or 'Q'?
            default: ss << '?'; break; // Should only promote to certain types
         }
     }

     // TODO: Add check/checkmate indicator (+ or #)
     // This requires checking legality of opponent moves after this move is made. Skipping for now.

     return ss.str();
}

std::string get_uci_string(const Move& move) {
    std::stringstream ss;
    ss << static_cast<char>('a' + move.from_loc.col);
    ss << (BOARD_SIZE - move.from_loc.row);
    ss << static_cast<char>('a' + move.to_loc.col);
    ss << (BOARD_SIZE - move.to_loc.row);

    // Add promotion piece type (lowercase) for UCI
    if (move.promotion_piece_type) {
          switch(*move.promotion_piece_type) {
            case PieceType::KNIGHT: ss << 'n'; break;
            case PieceType::BISHOP: ss << 'b'; break;
            case PieceType::ROOK:   ss << 'r'; break;
            case PieceType::QUEEN:  ss << 'q'; break;
            case PieceType::ONE_POINT_QUEEN: ss << 'o'; break; // Consistent 'o'?
            default: break; // Unknown promotion?
         }
    }
    return ss.str();
}


} // namespace chaturaji_cpp