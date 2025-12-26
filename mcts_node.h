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
    // These static member functions will be called by `new MCTSNode()` and `delete node_ptr;`,
    // including when `std::make_unique<MCTSNode>()` is used.
    
    /**
     * @brief Custom allocation function that uses the MCTSNodePool.
     * @param size The size of the object to allocate (should be sizeof(MCTSNode)).
     * @return A pointer to the allocated memory.
     */
    static void* operator new(size_t size);

    /**
     * @brief Custom deallocation function that returns memory to the MCTSNodePool.
     * Uses the C++14 sized delete operator, which is more efficient as it knows the size.
     * @param ptr The pointer to the memory block to deallocate.
     * @param size The size of the object being deallocated (should be sizeof(MCTSNode)).
     */
    static void operator delete(void* ptr, size_t size); 
    // Fallback delete (for C++11 or if sized delete isn't available)
    // static void operator delete(void* ptr); // Not needed with C++14 sized delete

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
    MCTSNode* select_child(double c_puct = 1.0) const;
    void expand(const std::map<Move, double>& policy_probs);

    /**
     * @brief Updates the visit count and total player values of this node.
     * @param values_for_players An array of 4 values, representing the outcome from the
     *                           simulation/evaluation for each of the 4 players (RED, BLUE, YELLOW, GREEN).
     *                           Order corresponds to Player enum.
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
    MCTSNode* parent_;  // Pointer to the parent node (raw pointer to avoid circular unique_ptr/shared_ptr)
    std::optional<Move> move_; // The move that led to this node from its parent

    std::vector<std::unique_ptr<MCTSNode>> children_; // Child nodes, managed by unique_ptr

    // MCTS statistics
    int visit_count_;           // Number of times this node has been visited
    std::array<double, 4> total_player_values_; // Accumulated values for each player from simulations
    double prior_;              // Prior probability from the neural network policy

    int pending_visits_; // Number of active simulations traversing this node (virtual loss)

    // Helper to calculate UCT score for a child, incorporating virtual loss
    double calculate_uct_score(const MCTSNode* child, double c_puct) const;

    // Static instance of the node pool, shared by all MCTSNode objects.
    // It will be initialized once when the program starts.
    static MCTSNodePool s_node_pool; 
};

} // namespace chaturaji_cpp