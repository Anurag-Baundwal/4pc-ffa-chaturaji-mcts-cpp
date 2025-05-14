#include "utils.h"
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
    for(int i=0; i<UTIL_PIECE_TYPE_ORDER.size(); ++i) {
        m[UTIL_PIECE_TYPE_ORDER[i]] = i;
    }
    return m;
}();


torch::Tensor board_to_tensor(const Board& board, torch::Device device) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  
  constexpr int NUM_ACTUAL_PIECE_TYPES = 5; // P, N, B, R, K
  constexpr int NUM_PIECE_CHANNELS_ONLY = 4 * NUM_ACTUAL_PIECE_TYPES; // 20
  
  // Total channels: 20 (pieces) + 4 (active status) + 4 (player turn) + 4 (points) + 1 (50-move) = 33 channels
  constexpr int NUM_CHANNELS_TOTAL = NUM_PIECE_CHANNELS_ONLY + 4 + 4 + 4 + 1; 
  torch::Tensor tensor = torch::zeros({NUM_CHANNELS_TOTAL, BOARD_SIZE, BOARD_SIZE}, options);

  const auto& grid = board.get_board_grid();
  const auto& active_players_set = board.get_active_players();
  const auto& points = board.get_player_points();
  Player current_player = board.get_current_player();

  // Piece Placement Channels (0-19)
  for (int r = 0; r < BOARD_SIZE; ++r) {
      for (int c = 0; c < BOARD_SIZE; ++c) {
          const auto& piece_opt = grid[r][c];
          if (piece_opt) {
              const Piece& piece = *piece_opt;
              int player_idx = static_cast<int>(piece.player);
              auto it = UTIL_PIECE_TYPE_TO_INDEX.find(piece.piece_type); // Use local util map
              if (it != UTIL_PIECE_TYPE_TO_INDEX.end()) {
                  int type_idx = it->second; 
                  int channel = player_idx * NUM_ACTUAL_PIECE_TYPES + type_idx;
                  if (channel >= 0 && channel < NUM_PIECE_CHANNELS_ONLY) {
                       tensor[channel][r][c] = 1.0f;
                  } else {
                       std::cerr << "Warning: Invalid piece channel in board_to_tensor: " << channel << std::endl;
                  }
              } else {
                  std::cerr << "Warning: Piece type " << static_cast<int>(piece.piece_type) 
                            << " not found in UTIL_PIECE_TYPE_TO_INDEX map." << std::endl;
              }
          }
      }
  }

  // Active Player Status Channels (20-23)
  int active_status_channel_offset = NUM_PIECE_CHANNELS_ONLY; // Starts at 20
  for (int i = 0; i < 4; ++i) {
      Player p_iter = static_cast<Player>(i);
      if (active_players_set.count(p_iter)) {
          tensor[active_status_channel_offset + i].fill_(1.0f);
      } else {
          tensor[active_status_channel_offset + i].fill_(0.0f); 
      }
  }

  // Current Player Channels (24-27)
  int current_player_channel_offset = active_status_channel_offset + 4; // Starts at 24
  int current_player_idx_val = static_cast<int>(current_player);
  if (current_player_idx_val >= 0 && current_player_idx_val < 4) {
    tensor[current_player_channel_offset + current_player_idx_val].fill_(1.0f);
  } else {
      // This case implies current_player is somehow invalid, which shouldn't happen if active_players is managed.
      // If game is over and current_player is one of the last active, this is fine.
      // If current_player is not in active_players, then no turn plane should be set.
      // The current logic is okay: if current_player_idx_val is out of 0-3, no plane is set.
  }


  // Player Points Channels (28-31)
  int points_channel_offset = current_player_channel_offset + 4; // Starts at 28
  for (int i = 0; i < 4; ++i) {
      Player p_iter = static_cast<Player>(i);
      float player_points_val = 0.0f;
      auto it = points.find(p_iter);
      if(it != points.end()){
          player_points_val = static_cast<float>(it->second);
      }
      // Normalize points (e.g. max possible points in Chaturaji might be around 50-60 without extreme scenarios)
      // A simple division by 100 is a common starting point.
      tensor[points_channel_offset + i].fill_(player_points_val / 100.0f); 
  }

  // 50-Move Rule Counter Channel (32)
  int counter_channel_idx = points_channel_offset + 4; // Starts at 32 (this is the last channel, index 32)
  int moves_since_reset = board.get_full_move_number() - board.get_move_number_of_last_reset();
  float normalized_count = std::max(0.0f, std::min(1.0f, static_cast<float>(moves_since_reset) / 50.0f));
  
  if (tensor.size(0) == NUM_CHANNELS_TOTAL && counter_channel_idx == NUM_CHANNELS_TOTAL - 1) {
       tensor[counter_channel_idx].fill_(normalized_count); 
  } else {
      throw std::runtime_error("Internal error: Tensor channel dimension mismatch for 50-move counter. Expected "
                               + std::to_string(NUM_CHANNELS_TOTAL) + " channels, offset " + std::to_string(counter_channel_idx));
  }

  return tensor.unsqueeze(0);
}

torch::Tensor get_board_tensor_no_batch(const Board& board, torch::Device device) {
  return board_to_tensor(board, device).squeeze(0);
}

int move_to_policy_index(const Move& move) {
    int fr_row = move.from_loc.row;
    int fr_col = move.from_loc.col;
    int to_row = move.to_loc.row;
    int to_col = move.to_loc.col;

    if (fr_row < 0 || fr_row >= BOARD_SIZE || fr_col < 0 || fr_col >= BOARD_SIZE ||
        to_row < 0 || to_row >= BOARD_SIZE || to_col < 0 || to_col >= BOARD_SIZE) {
        throw std::out_of_range("Move coordinates are out of board bounds for policy index.");
    }
    int from_index = fr_row * BOARD_SIZE + fr_col;
    int to_index = to_row * BOARD_SIZE + to_col;
    return from_index * (BOARD_SIZE * BOARD_SIZE) + to_index; // Range 0 to 63*64 + 63 = 4095
}


Move policy_index_to_move(int index) {
    if (index < 0 || index >= (BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE)) { // 64*64 = 4096 possibilities
         throw std::out_of_range("Policy index " + std::to_string(index) + " is out of bounds (0-4095).");
    }
    int to_index = index % (BOARD_SIZE * BOARD_SIZE);
    int from_index = index / (BOARD_SIZE * BOARD_SIZE);

    int to_row = to_index / BOARD_SIZE;
    int to_col = to_index % BOARD_SIZE;
    int from_row = from_index / BOARD_SIZE;
    int from_col = from_index % BOARD_SIZE;

    return Move(BoardLocation(from_row, from_col), BoardLocation(to_row, to_col), std::nullopt);
}

std::string get_san_string(const Move& move, const Board& board) {
     std::stringstream ss;
     const auto& grid = board.get_board_grid(); // Get grid once
     const auto& from_piece_opt = grid[move.from_loc.row][move.from_loc.col];
     const auto& to_piece_opt = grid[move.to_loc.row][move.to_loc.col]; 

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
     ss << (BOARD_SIZE - move.from_loc.row);
     if (to_piece_opt) {
         ss << 'x';
     }
     ss << static_cast<char>('a' + move.to_loc.col);
     ss << (BOARD_SIZE - move.to_loc.row);
     if (move.promotion_piece_type) {
         ss << '=';
          switch(*move.promotion_piece_type) {
            case PieceType::KNIGHT: ss << 'N'; break;
            case PieceType::BISHOP: ss << 'B'; break;
            case PieceType::ROOK:   ss << 'R'; break;
            // KING promotion is not standard, usually only Q,R,B,N in chess. Chaturaji has Rook.
            default: ss << '?'; break; 
         }
     }
     return ss.str();
}

std::string get_uci_string(const Move& move) {
    std::stringstream ss;
    ss << static_cast<char>('a' + move.from_loc.col);
    ss << (BOARD_SIZE - move.from_loc.row);
    ss << static_cast<char>('a' + move.to_loc.col);
    ss << (BOARD_SIZE - move.to_loc.row);

    if (move.promotion_piece_type) {
          switch(*move.promotion_piece_type) {
            case PieceType::KNIGHT: ss << 'n'; break;
            case PieceType::BISHOP: ss << 'b'; break;
            case PieceType::ROOK:   ss << 'r'; break;
            default: break; 
         }
    }
    return ss.str();
}

} // namespace chaturaji_cpp