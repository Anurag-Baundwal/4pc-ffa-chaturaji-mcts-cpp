#pragma once

#include <memory> 
#include <optional>
#include <cmath>    
#include <limits> 
#include <array>  
#include <random> 

#include "board.h" 
#include "types.h" 
#include "mcts_node_pool.h" 

namespace chaturaji_cpp {

const double VIRTUAL_LOSS_VALUE = 1.0; 

class MCTSNode {
public:
    // --- Constructor ---
    MCTSNode(Board board_state, MCTSNode* parent = nullptr, std::optional<Move> move = std::nullopt, double prior = 0.0);
    
    // --- Destructor ---
    ~MCTSNode();

    MCTSNode(const MCTSNode&) = delete; 
    MCTSNode& operator=(const MCTSNode&) = delete;
    MCTSNode(MCTSNode&&) = delete; 
    MCTSNode& operator=(MCTSNode&&) = delete; 

    // --- Custom Allocators ---
    static void* operator new(size_t size);
    static void operator delete(void* ptr, size_t size); 

    // --- Tree Traversal ---
    bool is_leaf() const;
    bool is_root() const;
    MCTSNode* get_parent() const;
    void set_parent(MCTSNode* p); 
    
    // Linked List Accessors
    MCTSNode* get_first_child() const;
    MCTSNode* get_next_sibling() const;

    const Board& get_board() const;
    const std::optional<Move>& get_move() const; 

    // --- MCTS Operations ---
    MCTSNode* select_child(double c_puct = 1.0) const;
    void expand(const std::map<Move, double>& policy_probs);

    void update_stats(const std::array<double, 4>& values_for_players); 
    void increment_pending_visits();
    void decrement_pending_visits();
    void inject_noise(double alpha, double epsilon, std::mt19937& rng);

    // --- Repetition Detection (Tree Walking) ---
    // Checks if the current board state has occurred 3 times within the current branch.
    // Traverses up parent pointers.
    bool check_repetition() const;

    // --- Tree Reuse Helper ---
    // Detaches 'target_child' from this node's list, deletes all OTHER children,
    // and returns the raw pointer to target_child. 
    // This node (the parent) is left with no children.
    MCTSNode* detach_child_and_clear_others(MCTSNode* target_child);

    // --- Accessors for Node Statistics ---
    int get_visit_count() const;
    const std::array<double, 4>& get_total_player_values() const; 
    double get_prior() const;
    int get_pending_visits() const; 

private:
    Board board_state_; 
    MCTSNode* parent_;  
    std::optional<Move> move_; 

    // --- Linked List Topology ---
    MCTSNode* first_child_;
    MCTSNode* next_sibling_;

    // MCTS statistics
    int visit_count_;           
    std::array<double, 4> total_player_values_; 
    double prior_;              
    int pending_visits_; 

    double calculate_uct_score(const MCTSNode* child, double c_puct) const;

    static MCTSNodePool s_node_pool; 
};

} // namespace chaturaji_cpp