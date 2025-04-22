#pragma once

#include <vector>
#include <memory> // For unique_ptr
#include <optional>
#include <cmath>    // For sqrt, log
#include <limits> // For infinity

#include "board.h" // Node contains a board state
#include "types.h" // For Move

namespace chaturaji_cpp {

class MCTSNode {
public:
    // --- Constructor ---
    // Creates a root node or a child node
    MCTSNode(Board board_state, MCTSNode* parent = nullptr, std::optional<Move> move = std::nullopt, double prior = 0.0);

    // Prevent copying (nodes manage ownership of children unique_ptrs)
    MCTSNode(const MCTSNode&) = delete;
    MCTSNode& operator=(const MCTSNode&) = delete;

    // Default move constructor/assignment should be okay if unique_ptrs are handled correctly
    MCTSNode(MCTSNode&&) = default;
    MCTSNode& operator=(MCTSNode&&) = default;


    // --- Tree Traversal and Properties ---
    bool is_leaf() const;
    bool is_root() const;
    MCTSNode* get_parent() const;
    const std::vector<std::unique_ptr<MCTSNode>>& get_children() const; // Read-only access
    const Board& get_board() const;
    const std::optional<Move>& get_move() const; // Move that led to this node

    // --- MCTS Operations ---
    /**
     * @brief Selects the best child node according to the PUCT formula.
     * @param c_puct Exploration constant.
     * @return Pointer to the selected child node, or nullptr if no children.
     */
    MCTSNode* select_child(double c_puct = 1.0) const;

    /**
     * @brief Expands this leaf node by creating children for all legal moves.
     *        Assigns prior probabilities from the policy network output.
     * @param policy_probs A map from legal moves to their prior probabilities.
     */
    void expand(const std::map<Move, double>& policy_probs);

    /**
     * @brief Updates the visit count and total value of this node and backpropagates
     *        the value up to the root.
     * @param value The value obtained from the simulation or network evaluation (from the perspective of the player *at this node*).
     */
    void update(double value);


    // --- Accessors for Node Statistics ---
    int get_visit_count() const;
    double get_total_value() const; // Value accumulated from children (relative to this node's player?) - Let's align with python: from root perspective
    double get_prior() const;


private:
    Board board_state_; // The game state this node represents
    MCTSNode* parent_; // Pointer to the parent node (nullptr for root)
    std::optional<Move> move_; // The move that led from parent to this node

    std::vector<std::unique_ptr<MCTSNode>> children_; // Child nodes owned by this node

    // MCTS statistics
    int visit_count_;
    double total_value_; // Accumulated value from simulations below this node (always from the root player's perspective)
    double prior_;       // Prior probability assigned by the policy network

    // Helper to calculate UCT score for a child
    double calculate_uct_score(const MCTSNode* child, double c_puct) const;

};

} // namespace chaturaji_cpp