#pragma once

#include <vector>
#include <torch/torch.h> // Libtorch header

#include "board.h" // Include board definition
#include "types.h" // Include move/location types

namespace chaturaji_cpp {

// --- Tensor Conversion ---

/**
 * @brief Converts the current board state into a tensor representation suitable for the NN.
 *
 * Tensor dimensions: [Batch=1, Channels=33, Height=8, Width=8] <- UPDATED Channel Count
 * Channels:
 * Piece Channels (0-19):
 *   0-4:   Player RED pieces    (P, N, B, R, K) in UTIL_PIECE_TYPE_ORDER
 *   5-9:   Player BLUE pieces   (P, N, B, R, K)
 *   10-14: Player YELLOW pieces (P, N, B, R, K)
 *   15-19: Player GREEN pieces  (P, N, B, R, K)
 * Auxiliary Channels (20-32):
 *   20-23: Active player status (RED, BLUE, YELLOW, GREEN) - plane is 1.0 if active, 0.0 otherwise.
 *   24-27: Current player turn (RED, BLUE, YELLOW, GREEN) - plane for current player is 1.0, others 0.0.
 *   28-31: Player points normalized (Player Points / 100.0 for RED, BLUE, YELLOW, GREEN).
 *   32:    50-move counter normalized (Moves since last reset / 50.0, clamped to [0,1]).              
 * Note: Order of piece types within the 6 channels per player needs to be consistent.
 *
 * @param board The board object to convert.
 * @param device The torch device (e.g., torch::kCPU, torch::kCUDA) to create the tensor on.
 * @return A torch::Tensor representing the board state.
 */
torch::Tensor board_to_tensor(const Board& board, torch::Device device);

/**
 * @brief Converts the board state to a tensor without the batch dimension [C, H, W].
 * Useful for creating batches by stacking multiple tensors.
 *
 * @param board The board object to convert.
 * @param device The torch device (e.g., torch::kCPU, torch::kCUDA).
 * @return A torch::Tensor representing the board state [Channels, Height, Width].
 */
torch::Tensor get_board_tensor_no_batch(const Board& board, torch::Device device);

std::vector<float> board_to_floats(const Board& board);

// --- Move Indexing ---

/**
 * @brief Maps a Move object to a unique integer index (0 to 4095).
 * Assumes an 8x8 board. Maps (from_row, from_col, to_row, to_col) linearly.
 * index = from_row * (8*8*8) + from_col * (8*8) + to_row * 8 + to_col
 * Promotion information is NOT encoded in this index, suitable for policy head output.
 *
 * @param move The move to encode.
 * @return The integer index representing the move's source and destination.
 */
int move_to_policy_index(const Move& move);

/**
 * @brief Maps a unique integer index (0 to 4095) back to a Move object.
 * Reverses the logic of move_to_policy_index.
 * Promotion information cannot be recovered and will be nullopt.
 *
 * @param index The integer index to decode.
 * @return The Move object (without promotion info).
 */
Move policy_index_to_move(int index);


/**
 * @brief Gets the standard algebraic notation (SAN) string for a move.
 *        Example: "Nf3", "e4", "Rxa8+", "O-O", "b1=Q"
 *        Chaturaji specifics: No castling, promotion only to Rook currently.
 *
 * @param move The move object.
 * @param board The board state *before* the move is made (needed for piece type and capture info).
 * @return std::string The SAN representation.
 */
std::string get_san_string(const Move& move, const Board& board);

/**
 * @brief Gets the Universal Chess Interface (UCI) string for a move.
 *        Example: "e2e4", "g1f3", "a7a8r" (includes promotion piece type)
 *
 * @param move The move object.
 * @return std::string The UCI representation.
 */
std::string get_uci_string(const Move& move);


} // namespace chaturaji_cpp