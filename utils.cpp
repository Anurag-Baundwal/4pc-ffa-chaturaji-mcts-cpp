#include "utils.h"
#include "magic_utils.h" // Include for magic_utils::BOARD_SIZE and other utilities
#include <stdexcept>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm> // For std::max, std::min

namespace chaturaji_cpp {

// PIECE_TYPE_ORDER_PHASE1 and PIECE_TYPE_TO_INDEX_PHASE1 are now defined in board.cpp's anonymous namespace
// For utils.cpp, we need them too if they are not exposed globally.
// Let's redefine them here for clarity, or ensure they are accessible from board.cpp's definitions.
// For now, redefining. Ideally, this would be in a shared constants header or accessible.
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


torch::Tensor board_to_tensor(const Board& board, torch::Device device) {
  // 1. Always create and populate on CPU first
  auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  
  constexpr int NUM_ACTUAL_PIECE_TYPES = 5; // P, N, B, R, K
  constexpr int NUM_PIECE_CHANNELS_ONLY = 4 * NUM_ACTUAL_PIECE_TYPES; // 20
  
  // Total channels: 20 (pieces) + 4 (active status) + 4 (player turn) + 4 (points) + 1 (50-move) = 33 channels
  constexpr int NUM_CHANNELS_TOTAL = NUM_PIECE_CHANNELS_ONLY + 4 + 4 + 4 + 1; 
  // Use magic_utils::BOARD_SIZE
  torch::Tensor tensor_cpu = torch::zeros({NUM_CHANNELS_TOTAL, magic_utils::BOARD_SIZE, magic_utils::BOARD_SIZE}, cpu_options);
  
  // Get an accessor for efficient CPU writes to piece planes
  auto piece_accessor = tensor_cpu.accessor<float, 3>();

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
              BoardLocation loc = magic_utils::from_sq_idx(sq_idx); // Use magic_utils::from_sq_idx
              if (channel >= 0 && channel < NUM_PIECE_CHANNELS_ONLY) {
                  // Use magic_utils::BOARD_SIZE for bounds checking (implicitly handled by valid sq_idx)
                  // Direct write using accessor for piece planes
                  piece_accessor[channel][loc.row][loc.col] = 1.0f;
              } else {
                  // This case should ideally not be reached if loop bounds are correct
                  std::cerr << "Warning: Invalid piece channel in board_to_tensor (from bitboard): " << channel << std::endl;
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
      // Use select().fill_() for setting entire planes on the CPU tensor
      tensor_cpu.select(0, active_status_channel_offset + i).fill_(val_to_fill);
  }

  // Current Player Channels (24-27)
  Player current_player = board.get_current_player();
  int current_player_channel_offset = active_status_channel_offset + 4; // Starts at 24
  int current_player_idx_val = static_cast<int>(current_player);
  // Initialize all player turn planes to 0
  for (int i = 0; i < 4; ++i) {
    tensor_cpu.select(0, current_player_channel_offset + i).fill_(0.0f);
  }
  // Set the current player's turn plane to 1
  if (current_player_idx_val >= 0 && current_player_idx_val < 4) {
    tensor_cpu.select(0, current_player_channel_offset + current_player_idx_val).fill_(1.0f);
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
      tensor_cpu.select(0, points_channel_offset + i).fill_(val_to_fill);
  }

  // 50-Move Rule Counter Channel (32)
  int counter_channel_idx = points_channel_offset + 4; // Starts at 32 (this is the last channel, index 32)
  int moves_since_reset = board.get_full_move_number() - board.get_move_number_of_last_reset();
  float normalized_count = std::max(0.0f, std::min(1.0f, static_cast<float>(moves_since_reset) / 50.0f));
  
  if (tensor_cpu.size(0) == NUM_CHANNELS_TOTAL && counter_channel_idx == NUM_CHANNELS_TOTAL - 1) {
       tensor_cpu.select(0, counter_channel_idx).fill_(normalized_count); 
  } else {
      throw std::runtime_error("Internal error: Tensor channel dimension mismatch for 50-move counter. Expected "
                               + std::to_string(NUM_CHANNELS_TOTAL) + " channels, offset " + std::to_string(counter_channel_idx));
  }

  // 2. Move to target device if necessary, then unsqueeze
  torch::Tensor final_tensor_on_device;
  if (device.type() != torch::kCPU) {
      final_tensor_on_device = tensor_cpu.to(device); // Default is non_blocking=false
  } else {
      final_tensor_on_device = tensor_cpu;
  }
  return final_tensor_on_device.unsqueeze(0);
}

torch::Tensor get_board_tensor_no_batch(const Board& board, torch::Device device) {
  return board_to_tensor(board, device).squeeze(0);
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