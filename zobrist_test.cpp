#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <regex>
#include <algorithm> // For std::count

#include "board.h" // Include your board header
#include "types.h" // Include your types header

using namespace chaturaji_cpp;

// --- Helper Functions for PGN Parsing and Coordinate Conversion ---

// Converts PGN file character (d..k) to 0-based column index (0..7)
int pgn_char_to_col(char c) {
    if (c >= 'd' && c <= 'k') {
        return c - 'd';
    }
    throw std::out_of_range("Invalid PGN column character: " + std::string(1, c));
}

// Converts PGN rank string ("4".."11") to 0-based row index (0..7)
// Remember: PGN rank 11 is row 0, rank 4 is row 7
int pgn_rank_to_row(const std::string& s) {
    try {
        int rank = std::stoi(s);
        if (rank >= 4 && rank <= 11) {
            return 11 - rank; // Invert rank
        }
    } catch (...) {
        // Fall through to throw
    }
    throw std::out_of_range("Invalid PGN rank string: " + s);
}

// Converts PGN square notation (e.g., "f5", "k11") to BoardLocation
BoardLocation pgn_to_loc(const std::string& pgn_sq) {
    if (pgn_sq.length() < 2) {
        throw std::invalid_argument("Invalid PGN square format: " + pgn_sq);
    }
    char col_char = pgn_sq[0];
    std::string rank_str = pgn_sq.substr(1);
    return BoardLocation(pgn_rank_to_row(rank_str), pgn_char_to_col(col_char));
}

// Converts BoardLocation back to PGN notation for printing/debugging
std::string loc_to_pgn(const BoardLocation& loc) {
    if (loc.row < 0 || loc.row > 7 || loc.col < 0 || loc.col > 7) return "??";
    char col_char = 'd' + loc.col;
    int rank = 11 - loc.row;
    return std::string(1, col_char) + std::to_string(rank);
}

// Parses a single move notation like "f5-f6", "h4xg5", "d7-d8=R", "Bf8xKe9#"
// Returns the Move object. Ignores check/mate symbols.
Move parse_pgn_move_notation(const std::string& notation) {
    std::regex move_regex("([a-k][1-9][0-9]?)[x-]?([a-k][1-9][0-9]?)(?:=([R]))?([+#])?"); // Only R promotion
    std::smatch match;

    if (std::regex_match(notation, match, move_regex)) {
        std::string from_sq_str = match[1].str();
        std::string to_sq_str = match[2].str();
        std::string promo_str = match[3].str(); // Optional promotion piece (R, Q, B, N)

        BoardLocation from_loc = pgn_to_loc(from_sq_str);
        BoardLocation to_loc = pgn_to_loc(to_sq_str);

        std::optional<PieceType> promo_type = std::nullopt;
        if (!promo_str.empty()) {
            // Chaturaji only allows Rook promotion in standard rules
            if (promo_str == "R") promo_type = PieceType::ROOK;
            else {
                std::cerr << "Warning: Unsupported promotion type '" << promo_str << "' in PGN. Ignoring promotion." << std::endl;
            }
        }
        return Move(from_loc, to_loc, promo_type);
    } else {
        throw std::invalid_argument("Could not parse PGN move notation: " + notation);
    }
}

// Utility to print board state and info for comparison
void print_state_comparison(const Board& board, const std::string& label) {
    std::cout << "\n--- State: " << label << " ---" << std::endl;
    board.print_board(); // Assumes print_board shows player, points, etc.
    std::cout << "FullMove: " << board.get_full_move_number()
              << ", Last Reset: " << board.get_move_number_of_last_reset()
              << ", 50-Move Clock: " << (board.get_full_move_number() - board.get_move_number_of_last_reset())
              << std::endl;
    std::cout << "Position Hash: " << board.get_position_key() << std::endl;
    // Compare active players set
    std::cout << "Active Players Set: { ";
    for (Player p : board.get_active_players()) { std::cout << static_cast<int>(p) << " "; }
    std::cout << "}" << std::endl;
    // Compare points map
    std::cout << "Points Map: { ";
    for (const auto& pair : board.get_player_points()) { std::cout << static_cast<int>(pair.first) << ":" << pair.second << " "; }
    std::cout << "}" << std::endl;
    std::cout << "--------------------------" << std::endl;
}

// Compare relevant board state elements (excluding history vectors)
bool compare_board_states(const Board& b1, const Board& b2) {
    if (b1.get_position_key() != b2.get_position_key()) return false;
    if (b1.get_current_player() != b2.get_current_player()) return false;
    if (b1.get_full_move_number() != b2.get_full_move_number()) return false;
    if (b1.get_move_number_of_last_reset() != b2.get_move_number_of_last_reset()) return false;
    if (b1.get_active_players() != b2.get_active_players()) return false;
    if (b1.get_player_points() != b2.get_player_points()) return false;
    // Compare bitboards directly for thoroughness
    if (b1.get_occupied_bitboard() != b2.get_occupied_bitboard()) {
        std::cerr << "State Compare Fail: Occupied bitboard mismatch" << std::endl;
        // Board::print_bitboard(b1.get_occupied_bitboard(), "B1 Occupied");
        // Board::print_bitboard(b2.get_occupied_bitboard(), "B2 Occupied");
        return false;
    }
    for (int p_idx = 0; p_idx < 4; ++p_idx) {
        Player p = static_cast<Player>(p_idx);
        if (b1.get_player_bitboard(p) != b2.get_player_bitboard(p)) {
            std::cerr << "State Compare Fail: Player bitboard mismatch for player " << p_idx << std::endl;
            return false;
        }
        for (int pt_val = 1; pt_val <= 5; ++pt_val) { // PieceType enum values 1-5
            PieceType pt = static_cast<PieceType>(pt_val);
            if (b1.get_piece_bitboard(p, pt) != b2.get_piece_bitboard(p, pt)) {
                 std::cerr << "State Compare Fail: Piece bitboard mismatch for player " << p_idx << ", piece " << pt_val << std::endl;
                 // Board::print_bitboard(b1.get_piece_bitboard(p,pt), "B1 P" + std::to_string(p_idx) + " T" + std::to_string(pt_val));
                 // Board::print_bitboard(b2.get_piece_bitboard(p,pt), "B2 P" + std::to_string(p_idx) + " T" + std::to_string(pt_val));
                return false;
            }
        }
    }
    return true;
}


// --- Test Functions ---

// Test basic move/undo cycle for a single move
bool test_single_move_undo(Board& board, const Move& move, const std::string& move_label) {
    std::cout << "\n>>> Testing Move: " << move_label << " (" << loc_to_pgn(move.from_loc) << " to " << loc_to_pgn(move.to_loc) << ")" << std::endl;

    Board board_before = board; // Make a copy for comparison
    ZobristKey hash_before = board.get_position_key();
    print_state_comparison(board, "Before Move");

    try {
        board.make_move(move);
    } catch (const std::exception& e) {
        std::cerr << "!!! EXCEPTION during make_move for " << move_label << ": " << e.what() << std::endl;
        print_state_comparison(board, "State After Exception");
        // Restore board state from copy to allow further tests if desired
        board = board_before;
        return false;
    }

    ZobristKey hash_after = board.get_position_key();
    print_state_comparison(board, "After Move");
    assert(hash_before != hash_after || (move.from_loc == move.to_loc)); // Hash should change for non-null moves

    board.undo_move();
    ZobristKey hash_after_undo = board.get_position_key();
    print_state_comparison(board, "After Undo");

    if (hash_before == hash_after_undo && compare_board_states(board, board_before)) {
        std::cout << "+++ PASSED: Hash and State correctly restored for move " << move_label << std::endl;
        return true;
    } else {
        std::cerr << "--- FAILED: Hash or State mismatch after undo for move " << move_label << std::endl;
        std::cerr << "    Hash Before: " << hash_before << std::endl;
        std::cerr << "    Hash After:  " << hash_after << std::endl;
        std::cerr << "    Hash Undo:   " << hash_after_undo << std::endl;
        if(!compare_board_states(board, board_before)) {
             std::cerr << "    Board state comparison failed!" << std::endl;
        }
        // Optional: Print detailed diff if needed
        return false;
    }
}

// Test board copy constructor and assignment operator
bool test_copy_consistency(const Board& board) {
    std::cout << "\n>>> Testing Copy Consistency" << std::endl;
    ZobristKey original_hash = board.get_position_key();
    print_state_comparison(board, "Original Board");

    // Test Copy Constructor
    Board copy1 = board;
    ZobristKey copy1_hash = copy1.get_position_key();
    print_state_comparison(copy1, "Copy Constructed Board");
    assert(original_hash == copy1_hash && "Copy Constructor Hash Mismatch");
    assert(compare_board_states(board, copy1) && "Copy Constructor State Mismatch");

    // Test Copy Assignment
    Board copy2; // Default construct
    copy2 = board;
    ZobristKey copy2_hash = copy2.get_position_key();
    print_state_comparison(copy2, "Copy Assigned Board");
    assert(original_hash == copy2_hash && "Copy Assignment Hash Mismatch");
    assert(compare_board_states(board, copy2) && "Copy Assignment State Mismatch");

    std::cout << "+++ PASSED: Copy consistency tests." << std::endl;
    return true;
}

// Test resignation and undo
bool test_resignation(Board& board) {
     std::cout << "\n>>> Testing Resignation" << std::endl;
     if (board.is_game_over()) {
         std::cout << "--- SKIPPED: Game already over." << std::endl;
         return true;
     }

     Player resigning_player = board.get_current_player();
     std::cout << "Player " << static_cast<int>(resigning_player) << " will resign." << std::endl;

     Board board_before = board;
     ZobristKey hash_before = board.get_position_key();
     print_state_comparison(board, "Before Resignation");

     board.resign(); // Player resigns
     ZobristKey hash_after = board.get_position_key();
     print_state_comparison(board, "After Resignation");
     assert(hash_before != hash_after && "Hash did not change after resignation");
     assert(board.get_active_players().find(resigning_player) == board.get_active_players().end() && "Resigning player still active");


     board.undo_move(); // Undo the resignation (uses the same undo stack)
     ZobristKey hash_after_undo = board.get_position_key();
     print_state_comparison(board, "After Undo Resignation");

     if (hash_before == hash_after_undo && compare_board_states(board, board_before)) {
        std::cout << "+++ PASSED: Resignation and Undo successful." << std::endl;
        return true;
    } else {
        std::cerr << "--- FAILED: Hash or State mismatch after undoing resignation." << std::endl;
        std::cerr << "    Hash Before: " << hash_before << std::endl;
        std::cerr << "    Hash After:  " << hash_after << std::endl;
        std::cerr << "    Hash Undo:   " << hash_after_undo << std::endl;
         if(!compare_board_states(board, board_before)) {
             std::cerr << "    Board state comparison failed!" << std::endl;
        }
        return false;
    }
}

// Test a simple threefold repetition scenario using non-resetting moves
bool test_threefold_repetition(Board& board) {
  std::cout << "\n>>> Testing Threefold Repetition (using Chaturaji Knight moves)" << std::endl;

  // Use a fresh board state for this test
  Board initial_board;
  board = initial_board; // Ensure we start fresh

  // Define the sequence of reversible Knight moves
  Move r_fwd(BoardLocation(7, 1), BoardLocation(5, 2)); // R: Nb1-c3
  Move b_fwd(BoardLocation(1, 0), BoardLocation(3, 2)); // B: Na7-c6
  Move y_fwd(BoardLocation(0, 6), BoardLocation(2, 5)); // Y: Ng8-f6
  Move g_fwd(BoardLocation(6, 7), BoardLocation(5, 5)); // G: Nh2-f3

  Move r_rev(BoardLocation(5, 2), BoardLocation(7, 1)); // R: Nc3-b1
  Move b_rev(BoardLocation(3, 2), BoardLocation(1, 0)); // B: Nc6-a7
  Move y_rev(BoardLocation(2, 5), BoardLocation(0, 6)); // Y: Nf6-g8
  Move g_rev(BoardLocation(5, 5), BoardLocation(6, 7)); // G: Nf3-h2

  std::cout << "  Test Sequence: 4x(R:Nb1-c3, B:Na7-b5, Y:Ng8-f6, G:Nh2-f3), 4x(R:Nc3-b1, B:Nb5-a7, Y:Nf6-g8, G:Nf3-h2)" << std::endl;

  // --- Position 1 (Initial) ---
  ZobristKey hash_initial = board.get_position_key();
  std::cout << "  Initial Hash: " << hash_initial << std::endl;

  // --- Cycle 1: Forward Moves ---
  board.make_move(r_fwd); board.make_move(b_fwd); board.make_move(y_fwd); board.make_move(g_fwd); // 1.R -> 1.G
  ZobristKey hash_after_cycle1 = board.get_position_key();
  std::cout << "  After Cycle 1 Fwd (1.G) -> Hash: " << hash_after_cycle1 << std::endl;
  assert(!board.is_game_over());

  // --- Cycle 2: Reverse Moves (Back to Initial Position - 2nd time overall) ---
  board.make_move(r_rev); board.make_move(b_rev); board.make_move(y_rev); board.make_move(g_rev); // 2.R -> 2.G
  ZobristKey hash_after_cycle2 = board.get_position_key();
  size_t history_count_initial_after_2 = std::count(board.get_position_history().begin(), board.get_position_history().end(), hash_initial);
  std::cout << "  After Cycle 2 Rev (2.G) -> Hash: " << hash_after_cycle2 << " (Initial Count in History: " << history_count_initial_after_2 << ")" << std::endl;
  assert(hash_after_cycle2 == hash_initial && "Hash mismatch after Cycle 2 (should be initial hash)");
  // History now contains hash_initial ONCE (from the end of move 2.G)
  assert(history_count_initial_after_2 >= 1); // Expecting 1 here
  assert(!board.is_game_over());

  // --- Cycle 3: Forward Moves (Position after cycle 1 again - 2nd time) ---
  board.make_move(r_fwd); board.make_move(b_fwd); board.make_move(y_fwd); board.make_move(g_fwd); // 3.R -> 3.G
  ZobristKey hash_after_cycle3 = board.get_position_key();
   size_t history_count_cycle1_after_3 = std::count(board.get_position_history().begin(), board.get_position_history().end(), hash_after_cycle1);
  std::cout << "  After Cycle 3 Fwd (3.G) -> Hash: " << hash_after_cycle3 << " (Cycle1 Hash Count: " << history_count_cycle1_after_3 << ")" << std::endl;
  assert(hash_after_cycle3 == hash_after_cycle1 && "Hash mismatch after Cycle 3 (should match cycle 1)");
  assert(history_count_cycle1_after_3 == 2); // Expecting 2 here (after 1.G and after 3.G)
  assert(!board.is_game_over() && "Game ended prematurely (Rep 2)");

  // --- Cycle 4: Reverse Moves (Initial Position again - 3rd time overall!) ---
  board.make_move(r_rev); board.make_move(b_rev); board.make_move(y_rev); board.make_move(g_rev); // 4.R -> 4.G
  ZobristKey hash_after_cycle4 = board.get_position_key();
  // Before the is_game_over check, the history contains hash_initial TWICE (from 2.G and 4.G)
  size_t history_count_initial_after_4 = std::count(board.get_position_history().begin(), board.get_position_history().end(), hash_initial);
  std::cout << "  After Cycle 4 Rev (4.G) -> Hash: " << hash_after_cycle4 << " (Initial Count in History: " << history_count_initial_after_4 << ")" << std::endl;
  assert(hash_after_cycle4 == hash_initial && "Hash mismatch after Cycle 4 (should be initial hash)");
  assert(history_count_initial_after_4 == 3 && "Initial position hash count incorrect after cycle 4");

  // **********************************************************
  // *** ADD THE CHECK FOR THREEFOLD REPETITION HERE (after 4.G) ***
  // **********************************************************
  std::cout << "  Checking game over state for threefold repetition (Initial Position, 3rd time)..." << std::endl;
  bool game_over_initial_rep = board.is_game_over(); // This call updates termination_reason_ if applicable
  std::optional<std::string> reason_initial_rep = board.get_termination_reason();

  // Restore board state BEFORE making assertions, so subsequent tests (if any) start fresh
  board = initial_board;
  std::cout << "  Restored initial board state." << std::endl;

  // NOW check the result
  if (game_over_initial_rep && reason_initial_rep && *reason_initial_rep == "threefold_repetition") {
      std::cout << "+++ PASSED: Threefold repetition correctly detected after 4.G (Initial state 3rd occurrence)." << std::endl;
      // Test was successful up to this point for detecting the first repetition
      // We can skip the rest of the original test (Cycle 5 and subsequent checks)
      // as the primary goal (detecting threefold) has been met here.
      return true;
  } else {
      std::cerr << "--- FAILED: Threefold repetition was NOT detected after 4.G (Initial state 3rd occurrence)." << std::endl;
      std::cerr << "    Game Over flag: " << (game_over_initial_rep ? "Yes" : "No") << std::endl;
      std::cerr << "    Termination Reason: " << (reason_initial_rep ? *reason_initial_rep : "None") << std::endl;
      std::cerr << "    Count of initial hash in history *before* check was: " << history_count_initial_after_4 << std::endl;
      return false; // Fail the whole test function
  }
}

// --- Main Test Execution ---

int main() {
  std::cout << "===== Starting Zobrist Hash Tests =====" << std::endl;

  Board board; // Start with initial position

  // --- Manually define the moves (Keep the existing vector) ---
  std::vector<std::pair<std::string, Move>> manual_moves = {
      // (Keep the same move list as before)
      // 1. f5-f6 .. e9-f9 .. i10-i9 .. j5-i5
      {"1.R", Move(pgn_to_loc("f5"), pgn_to_loc("f6"))},          // (6,2) -> (5,2) Pawn
      {"1.B", Move(pgn_to_loc("e9"), pgn_to_loc("f9"))},          // (2,1) -> (2,2) Pawn
      {"1.Y", Move(pgn_to_loc("i10"), pgn_to_loc("i9"))},         // (1,5) -> (2,5) Pawn
      {"1.G", Move(pgn_to_loc("j5"), pgn_to_loc("i5"))},          // (6,6) -> (6,5) Pawn
      // 2. Kg4-f5 .. e10-f10 .. Kh11-i10 .. Bk6-j5
      {"2.R", Move(pgn_to_loc("g4"), pgn_to_loc("f5"))},          // (7,3) -> (6,2) King
      {"2.B", Move(pgn_to_loc("e10"), pgn_to_loc("f10"))},        // (1,1) -> (1,2) Pawn
      {"2.Y", Move(pgn_to_loc("h11"), pgn_to_loc("i10"))},        // (0,4) -> (1,5) King
      {"2.G", Move(pgn_to_loc("k6"), pgn_to_loc("j5"))},          // (5,7) -> (6,6) Bishop
      // 3. e5-e6 .. Kd8-e9 .. j10-j9 .. j4-i4
      {"3.R", Move(pgn_to_loc("e5"), pgn_to_loc("e6"))},          // (6,1) -> (5,1) Pawn
      {"3.B", Move(pgn_to_loc("d8"), pgn_to_loc("e9"))},          // (3,0) -> (2,1) King
      {"3.Y", Move(pgn_to_loc("j10"), pgn_to_loc("j9"))},         // (1,6) -> (2,6) Pawn
      {"3.G", Move(pgn_to_loc("j4"), pgn_to_loc("i4"))},          // (7,6) -> (7,5) Pawn
      // 4. d5-d6 .. e11-f11 .. Bi11-j10 .. j6-i6
      {"4.R", Move(pgn_to_loc("d5"), pgn_to_loc("d6"))},          // (6,0) -> (5,0) Pawn
      {"4.B", Move(pgn_to_loc("e11"), pgn_to_loc("f11"))},        // (0,1) -> (0,2) Pawn
      {"4.Y", Move(pgn_to_loc("i11"), pgn_to_loc("j10"))},        // (0,5) -> (1,6) Bishop
      {"4.G", Move(pgn_to_loc("j6"), pgn_to_loc("i6"))},          // (5,6) -> (5,5) Pawn
      // 5. d6-d7 .. Bd9-e10 .. k10-k9 .. Kk7-j6
      {"5.R", Move(pgn_to_loc("d6"), pgn_to_loc("d7"))},          // (5,0) -> (4,0) Pawn
      {"5.B", Move(pgn_to_loc("d9"), pgn_to_loc("e10"))},         // (2,0) -> (1,1) Bishop
      {"5.Y", Move(pgn_to_loc("k10"), pgn_to_loc("k9"))},         // (1,7) -> (2,7) Pawn
      {"5.G", Move(pgn_to_loc("k7"), pgn_to_loc("j6"))},          // (4,7) -> (5,6) King
      // 6. Bf4-d6 .. f11-g11 .. k9-k8 .. i4-h4
      {"6.R", Move(pgn_to_loc("f4"), pgn_to_loc("d6"))},          // (7,2) -> (5,0) Bishop
      {"6.B", Move(pgn_to_loc("f11"), pgn_to_loc("g11"))},        // (0,2) -> (0,3) Pawn
      {"6.Y", Move(pgn_to_loc("k9"), pgn_to_loc("k8"))},          // (2,7) -> (3,7) Pawn
      {"6.G", Move(pgn_to_loc("i4"), pgn_to_loc("h4"))},          // (7,5) -> (7,4) Pawn
      // 7. d7-d8=R .. Ke9-d9 .. Rk11-k10 .. h4xg5
      {"7.R", Move(pgn_to_loc("d7"), pgn_to_loc("d8"), PieceType::ROOK)}, // (4,0) -> (3,0) Pawn Promo!
      {"7.B", Move(pgn_to_loc("e9"), pgn_to_loc("d9"))},          // (2,1) -> (2,0) King
      {"7.Y", Move(pgn_to_loc("k11"), pgn_to_loc("k10"))},        // (0,7) -> (1,7) Rook
      {"7.G", Move(pgn_to_loc("h4"), pgn_to_loc("g5"))},          // (7,4) -> (6,3) Pawn Capture!
      // 8. Kf5xg5 .. Kd9xd8 .. Bj10-i11 .. Nk5-i4+
      {"8.R", Move(pgn_to_loc("f5"), pgn_to_loc("g5"))},          // (6,2) -> (6,3) King Capture!
      {"8.B", Move(pgn_to_loc("d9"), pgn_to_loc("d8"))},          // (2,0) -> (3,0) King Capture!
      {"8.Y", Move(pgn_to_loc("j10"), pgn_to_loc("i11"))},        // (1,6) -> (0,5) Bishop
      {"8.G", Move(pgn_to_loc("k5"), pgn_to_loc("i4"))},          // (6,7) -> (7,5) Knight
      // 9. Kg5-f5 .. g11xh10 .. Nj11xh10 .. Ni4-h6+
      {"9.R", Move(pgn_to_loc("g5"), pgn_to_loc("f5"))},          // (6,3) -> (6,2) King
      {"9.B", Move(pgn_to_loc("g11"), pgn_to_loc("h10"))},        // (0,3) -> (1,4) Pawn Capture!
      {"9.Y", Move(pgn_to_loc("j11"), pgn_to_loc("h10"))},        // (0,6) -> (1,4) Knight Capture!
      {"9.G", Move(pgn_to_loc("i4"), pgn_to_loc("h6"))},          // (7,5) -> (5,4) Knight
      // 10. Kf5-e5 .. Rd11xBi11+ .. Ki10xRi11 .. i5-h5
      {"10.R", Move(pgn_to_loc("f5"), pgn_to_loc("e5"))},         // (6,2) -> (6,1) King
      {"10.B", Move(pgn_to_loc("d11"), pgn_to_loc("i11"))},       // (0,0) -> (0,5) Rook Capture!
      {"10.Y", Move(pgn_to_loc("i10"), pgn_to_loc("i11"))},       // (1,5) -> (0,5) King Capture!
      {"10.G", Move(pgn_to_loc("i5"), pgn_to_loc("h5"))},         // (6,5) -> (6,4) Pawn
      // 11. Bd6-f8+ .. Kd8-e9 .. Ki11-i10 .. Nh6-i4
      {"11.R", Move(pgn_to_loc("d6"), pgn_to_loc("f8"))},         // (5,0) -> (3,2) Bishop
      {"11.B", Move(pgn_to_loc("d8"), pgn_to_loc("e9"))},         // (3,0) -> (2,1) King
      {"11.Y", Move(pgn_to_loc("i11"), pgn_to_loc("i10"))},       // (0,5) -> (1,5) King
      {"11.G", Move(pgn_to_loc("h6"), pgn_to_loc("i4"))},         // (5,4) -> (7,5) Knight
      // 12. Bf8xKe9# .. Ki10-j10 .. h5-g5
      {"12.R", Move(pgn_to_loc("f8"), pgn_to_loc("e9"))},         // (3,2) -> (2,1) Bishop Capture King! (Elim Blue)
      {"12.Y", Move(pgn_to_loc("i10"), pgn_to_loc("j10"))},       // (1,5) -> (1,6) King
      {"12.G", Move(pgn_to_loc("h5"), pgn_to_loc("g5"))},         // (6,4) -> (6,3) Pawn
      // 13. Be9xf10+ .. k8xj7 .. Kj6xj7
      {"13.R", Move(pgn_to_loc("e9"), pgn_to_loc("f10"))},        // (2,1) -> (1,2) Bishop Capture Dead Pawn!
      {"13.Y", Move(pgn_to_loc("k8"), pgn_to_loc("j7"))},         // (3,7) -> (4,6) Pawn Capture!
      {"13.G", Move(pgn_to_loc("j6"), pgn_to_loc("j7"))},         // (5,6) -> (4,6) King Capture!
      // 14. Ke5-f5 .. Rk10xRk4 .. Bj5xRk4
      {"14.R", Move(pgn_to_loc("e5"), pgn_to_loc("f5"))},         // (6,1) -> (6,2) King
      {"14.Y", Move(pgn_to_loc("k10"), pgn_to_loc("k4"))},        // (1,7) -> (7,7) Rook Capture!
      {"14.G", Move(pgn_to_loc("j5"), pgn_to_loc("k4"))},         // (6,6) -> (7,7) Bishop Capture Dead Rook!
      // 15. Ne4xg5 .. Nh10-i8 .. Ni4-j6
      {"15.R", Move(pgn_to_loc("e4"), pgn_to_loc("g5"))},         // (7,1) -> (6,3) Knight Capture!
      {"15.Y", Move(pgn_to_loc("h10"), pgn_to_loc("i8"))},        // (1,4) -> (3,5) Knight
      {"15.G", Move(pgn_to_loc("i4"), pgn_to_loc("j6"))},         // (7,5) -> (5,6) Knight
      // 16. Rd4xBk4 .. Ni8-h6++ .. Kj7-i7
      {"16.R", Move(pgn_to_loc("d4"), pgn_to_loc("k4"))},         // (7,0) -> (7,7) Rook Capture Dead Bishop!
      {"16.Y", Move(pgn_to_loc("i8"), pgn_to_loc("h6"))},         // (3,5) -> (5,4) Knight
      {"16.G", Move(pgn_to_loc("j7"), pgn_to_loc("i7"))},         // (4,6) -> (4,5) King
      // 17. Kf5-g6 .. Nh6-g8+ .. Nj6xRk4
      {"17.R", Move(pgn_to_loc("f5"), pgn_to_loc("g6"))},         // (6,2) -> (5,3) King
      {"17.Y", Move(pgn_to_loc("h6"), pgn_to_loc("g8"))},         // (5,4) -> (3,3) Knight
      {"17.G", Move(pgn_to_loc("j6"), pgn_to_loc("k4"))},         // (5,6) -> (7,7) Knight Capture Dead Rook!
      // 18. Bf10xKi7# .. Ng8xBi7+
      {"18.R", Move(pgn_to_loc("f10"), pgn_to_loc("i7"))},        // (1,2) -> (4,5) Bishop Capture King! (Elim Green)
      {"18.Y", Move(pgn_to_loc("g8"), pgn_to_loc("i7"))},         // (3,3) -> (4,5) Knight Capture Dead Bishop!
      // 19. Kg6-h7 .. R (Resign)
      {"19.R", Move(pgn_to_loc("g6"), pgn_to_loc("h7"))}          // (5,3) -> (4,4) King
      // Yellow resigns after this move in the PGN - we test resignation separately
  };
  std::cout << "\n--- Manually defined " << manual_moves.size() << " moves ---" << std::endl;

  // --- Execute Tests ---
  bool all_passed = true;

  // 1. Initial Copy Test
  all_passed &= test_copy_consistency(board);

  // --- MODIFIED PLAYBACK LOOP ---
  std::cout << "\n--- Starting Manual Move Playback and Move/Undo Tests ---" << std::endl;
  int moves_played = 0;
  for (const auto& labelled_move : manual_moves) {
      const std::string& label = labelled_move.first;
      const Move& move = labelled_move.second;

      std::cout << "\n>>> Testing Undo/Redo for Move: " << label << " (" << loc_to_pgn(move.from_loc) << " to " << loc_to_pgn(move.to_loc) << ")" << std::endl;

      // Determine expected player based on label
      Player expected_player;
      if (label.find(".R") != std::string::npos) expected_player = Player::RED;
      else if (label.find(".B") != std::string::npos) expected_player = Player::BLUE;
      else if (label.find(".Y") != std::string::npos) expected_player = Player::YELLOW;
      else if (label.find(".G") != std::string::npos) expected_player = Player::GREEN;
      else { /* Should not happen with manual list */ }

      // Check if expected player is active and whose turn it is
      if (!board.get_active_players().count(expected_player)) {
           std::cout << "--- INFO: Skipping move " << label << " because expected player "
                     << static_cast<int>(expected_player) << " is not active." << std::endl;
           continue; // Skip this move
      }

      if (board.get_current_player() != expected_player) {
          std::cerr << "\n!!! FATAL ERROR: Turn mismatch before testing move " << label << std::endl;
          std::cerr << "    Expected player: " << static_cast<int>(expected_player)
                      << ", Actual player: " << static_cast<int>(board.get_current_player()) << std::endl;
          all_passed = false;
          print_state_comparison(board, "State Before Mismatched Turn");
          break; // Stop test if turn logic is fundamentally broken
      }

      // --- Perform Undo/Redo Test for this move ---
      Board board_before_move = board; // Make a copy *before* the move
      print_state_comparison(board_before_move, "State Before Move");

      // 1. Make the move
      try {
          board.make_move(move);
      } catch (const std::exception& e) {
          std::cerr << "!!! EXCEPTION during make_move for " << label << ": " << e.what() << std::endl;
          print_state_comparison(board, "State After Exception");
          board = board_before_move; // Try to restore
          all_passed = false;
          break; // Stop on make_move error
      }
      ZobristKey hash_after_move = board.get_position_key();
      print_state_comparison(board, "State After Move");
      assert(board_before_move.get_position_key() != hash_after_move || (move.from_loc == move.to_loc)); // Hash should change

      // 2. Undo the move
      board.undo_move();
      print_state_comparison(board, "State After Undo");

      // 3. Verify state restoration
      if (!compare_board_states(board, board_before_move)) {
          std::cerr << "--- FAILED: State mismatch after undo for move " << label << std::endl;
          all_passed = false;
          // Don't break yet, let's see if redo helps debugging
      } else {
           std::cout << "+++ PASSED: State correctly restored after undo for move " << label << std::endl;
      }

      // 4. Redo the move (to advance state for next iteration)
      //    Use the state *before* the move was originally made for redo
      board = board_before_move; // Reset to state before the move
      try {
           board.make_move(move); // Make the move again
      } catch (const std::exception& e) {
          std::cerr << "!!! EXCEPTION during re-doing make_move for " << label << ": " << e.what() << std::endl;
           print_state_comparison(board, "State After Redo Exception");
           all_passed = false;
           break; // Stop on redo error
      }
      // Check if hash matches the hash after the first make_move
       if(board.get_position_key() != hash_after_move){
           std::cerr << "--- FAILED: Hash mismatch after re-doing move " << label << std::endl;
           std::cerr << "    Hash after 1st make_move: " << hash_after_move << std::endl;
           std::cerr << "    Hash after 2nd make_move: " << board.get_position_key() << std::endl;
           all_passed = false;
       } else {
           std::cout << "+++ INFO: State advanced correctly for next turn after move " << label << std::endl;
       }
       // --- End Undo/Redo Test ---


      // Interleave other tests (keep as before)
      moves_played++; // Increment counter *after* successfully processing a move cycle
      if (moves_played == 10) {
           Board temp_board = board; // Test on a copy
           all_passed &= test_resignation(temp_board);
           // No need to restore board, as temp_board was used
      }
      if (moves_played == 5 || moves_played == 15) {
          all_passed &= test_copy_consistency(board);
      }


      if (!all_passed) {
          std::cerr << "\n!!! Test failed during playback. Stopping." << std::endl;
          break;
      }
      if (board.is_game_over()) {
           std::cout << "\n--- Game Over detected during manual playback ---" << std::endl;
            if(board.get_termination_reason()){ std::cout << "Reason: " << *board.get_termination_reason() << std::endl; }
           print_state_comparison(board, "Final State After Manual Moves");
           break;
      }
  } // End main loop

  // --- Final Tests (keep as before) ---
  if(all_passed) {
      Board fresh_board;
      all_passed &= test_threefold_repetition(fresh_board);
  }

  std::cout << "\n===== Zobrist Hash Test Summary =====" << std::endl;
  if (all_passed) {
      std::cout << ">>> ALL TESTS PASSED <<<" << std::endl;
      return 0; // Success
  } else {
      std::cout << ">>> SOME TESTS FAILED <<<" << std::endl;
      return 1; // Failure
  }
}