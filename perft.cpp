#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm> // For std::find

#include "board.h"
#include "utils.h"
#include "types.h"

using namespace chaturaji_cpp;

// Helper for command line args
bool cmd_option_exists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

std::string get_cmd_option(char** begin, char** end, const std::string& option) {
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return "";
}

// Recursive Perft function
uint64_t perft(Board& board, int depth) {
    // Base case: At depth 0, we count the current node as 1
    if (depth == 0) {
        return 1;
    }

    // If the game is already over, no moves can be made from here
    if (board.is_game_over()) {
        return 0;
    }

    uint64_t nodes = 0;
    Player current_player = board.get_current_player();
    
    // Get moves for the current player
    std::vector<Move> moves = board.get_pseudo_legal_moves(current_player);

    for (const auto& move : moves) {
        // Make the move
        board.make_move(move);
        
        // Recursively count nodes at the next depth
        nodes += perft(board, depth - 1);
        
        // Undo the move to restore state
        board.undo_move();
    }

    return nodes;
}

// Divide function: Prints the perft count for each root move
uint64_t divide(Board& board, int depth) {
    if (depth == 0) {
        return 1;
    }
    
    std::cout << "Divide for depth " << depth << ":" << std::endl;
    
    uint64_t total_nodes = 0;
    Player current_player = board.get_current_player();
    std::vector<Move> moves = board.get_pseudo_legal_moves(current_player);

    for (const auto& move : moves) {
        board.make_move(move);
        
        uint64_t branch_nodes = perft(board, depth - 1);
        total_nodes += branch_nodes;
        
        // Print move and node count
        // Using get_uci_string from utils.h for cleaner output, or get_san_string
        std::cout << get_uci_string(move) << ": " << branch_nodes << std::endl;

        board.undo_move();
    }

    std::cout << "\nMoves: " << moves.size() << std::endl;
    std::cout << "Total Nodes: " << total_nodes << std::endl;
    return total_nodes;
}

int main(int argc, char* argv[]) {
    int depth = 4; // Default depth
    bool do_divide = false;

    // Parse arguments
    std::string depth_str = get_cmd_option(argv, argv + argc, "--depth");
    if (!depth_str.empty()) {
        depth = std::stoi(depth_str);
    }

    if (cmd_option_exists(argv, argv + argc, "--divide")) {
        do_divide = true;
    }

    // Initialize Board
    Board board;
    // board.setup_initial_board(); // Constructor calls this automatically

    std::cout << "Starting Perft Test..." << std::endl;
    std::cout << "Depth: " << depth << std::endl;
    std::cout << "Current Player: " << static_cast<int>(board.get_current_player()) << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    uint64_t result = 0;
    if (do_divide) {
        result = divide(board, depth);
    } else {
        result = perft(board, depth);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "=============================================" << std::endl;
    std::cout << "Nodes: " << result << std::endl;
    std::cout << "Time:  " << std::fixed << std::setprecision(3) << elapsed.count() << " s" << std::endl;
    
    if (elapsed.count() > 0) {
        double nps = static_cast<double>(result) / elapsed.count();
        std::cout << "NPS:   " << static_cast<uint64_t>(nps) << std::endl;
    }
    std::cout << "=============================================" << std::endl;

    return 0;
}