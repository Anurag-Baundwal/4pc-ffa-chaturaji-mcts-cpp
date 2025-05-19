#pragma once

#include <vector>
#include <memory> // For unique_ptr
#include <optional>
#include <cmath>    // For sqrt, log
#include <limits> // For infinity
#include <array>  // For std::array

#include "board.h" // Node contains a board state
#include "types.h" // For Move, Player

namespace chaturaji_cpp {

// Virtual Loss Constant
const double VIRTUAL_LOSS_VALUE = 1.0; 

class MCTSNode {
public:
    // --- Constructor ---
    MCTSNode(Board board_state, MCTSNode* parent = nullptr, std::optional<Move> move = std::nullopt, double prior = 0.0);

    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;
    MCTSNode(MCTSNode&&) = default;
    MCTSNode& operator=(MCTSNode&&) = default;

    // --- Tree Traversal and Properties ---
    bool is_leaf() const;
    bool is_root() const;
    MCTSNode* get_parent() const;
    const std::vector<std::unique_ptr<MCTSNode>>& get_children() const; 
    const Board& get_board() const;
    const std::optional<Move>& get_move() const; 

    // --- MCTS Operations ---
    MCTSNode* select_child(double c_puct = 1.0) const;
    void expand(const std::map<Move, double>& policy_probs);

    /**
     * @brief Updates the visit count and total player values of this node.
     * @param values_for_players An array of 4 values, representing the outcome from the
     *                           simulation/evaluation for each of the 4 players (RED, BLUE, YELLOW, GREEN).
     *                           Order matches Player enum.
     */
    void update_stats(const std::array<double, 4>& values_for_players); // MODIFIED

    void increment_pending_visits();
    void decrement_pending_visits();

    // --- Accessors for Node Statistics ---
    int get_visit_count() const;
    const std::array<double, 4>& get_total_player_values() const; // MODIFIED
    double get_prior() const;
    int get_pending_visits() const; 


private:
    Board board_state_; 
    MCTSNode* parent_; 
    std::optional<Move> move_; 

    std::vector<std::unique_ptr<MCTSNode>> children_; 

    // MCTS statistics
    int visit_count_;
    std::array<double, 4> total_player_values_; // MODIFIED: Accumulated values for each player
    double prior_;       

    int pending_visits_; 

    // Helper to calculate UCT score for a child, incorporating virtual loss
    // The score is from the perspective of the current node (the parent selecting a child).
    double calculate_uct_score(const MCTSNode* child, double c_puct) const;
};

} // namespace chaturaji_cpp