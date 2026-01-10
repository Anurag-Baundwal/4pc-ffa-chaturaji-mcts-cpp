// mcts_node.h
#pragma once

#include <vector>
#include <memory> // For unique_ptr
#include <optional>
#include <cmath>    // For sqrt, log
#include <limits> // For infinity
#include <array>  // For std::array
#include <random> // For std::mt19937

#include "board.h" // Node contains a board state
#include "types.h" // For Move, Player
#include "mcts_node_pool.h" // Include the pool header

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

    // --- Overload global new/delete operators for MCTSNode ---
    static void* operator new(size_t size);
    static void operator delete(void* ptr, size_t size); 

    // --- Tree Traversal and Properties ---
    bool is_leaf() const;
    bool is_root() const;
    MCTSNode* get_parent() const;
    const std::vector<std::unique_ptr<MCTSNode>>& get_children() const; 
    std::vector<std::unique_ptr<MCTSNode>>& get_children_for_reuse(); 
    void set_parent(MCTSNode* p); 
    const Board& get_board() const;
    const std::optional<Move>& get_move() const; 

    // --- MCTS Operations ---
    /**
     * @brief Selects the best child based on UCT score.
     * @param c_puct Exploration constant.
     * @param root_player The player at the root of the search (the engine).
     * @param spite_weight Weight for Utility Mixing (0.0 = Rational, 1.0 = Paranoid).
     */
    MCTSNode* select_child(double c_puct = 1.0, Player root_player = Player::RED, double spite_weight = 0.0) const;
    
    void expand(const std::map<Move, double>& policy_probs);

    /**
     * @brief Updates the visit count and total player values of this node.
     * @param values_for_players An array of 4 values, representing the outcome.
     */
    void update_stats(const std::array<double, 4>& values_for_players); 

    void increment_pending_visits();
    void decrement_pending_visits();

    void inject_noise(double alpha, double epsilon, std::mt19937& rng);

    // --- Accessors for Node Statistics ---
    int get_visit_count() const;
    const std::array<double, 4>& get_total_player_values() const; 
    double get_prior() const;
    int get_pending_visits() const; 

private:
    Board board_state_; // State of the board at this node
    MCTSNode* parent_;  // Pointer to the parent node
    std::optional<Move> move_; // The move that led to this node from its parent
    std::vector<std::unique_ptr<MCTSNode>> children_; // Child nodes

    // MCTS statistics
    int visit_count_;           
    std::array<double, 4> total_player_values_; 
    double prior_;              
    int pending_visits_; 

    // Helper to calculate UCT score for a child, incorporating virtual loss and spite
    double calculate_uct_score(const MCTSNode* child, double c_puct, Player root_player, double spite_weight) const;

    static MCTSNodePool s_node_pool; 
};

} // namespace chaturaji_cpp