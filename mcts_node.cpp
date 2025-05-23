#include "mcts_node.h"
#include <stdexcept>
#include <algorithm> // For std::max_element
#include <iostream>  // For potential debug warnings

namespace chaturaji_cpp {

// Define and initialize the static MCTSNodePool member.
// The constructor MCTSNodePool(size_t) will be called automatically once at program startup.
MCTSNodePool MCTSNode::s_node_pool(sizeof(MCTSNode), 100000); // Initialized with 100,000 node capacity

// Implementation of custom operator new
void* MCTSNode::operator new(size_t size) {
    // Ensure that only objects of exactly MCTSNode size are allocated from this pool.
    // This is crucial for fixed-size allocators.
    if (size != sizeof(MCTSNode)) {
        // If a derived class or an object of incorrect size tries to use this new,
        // it's an error for a fixed-size pool.
        throw std::logic_error("MCTSNodePool: Attempted to allocate object of wrong size. Using global operator new.");
    }
    // Delegate the allocation to the static node pool
    return s_node_pool.allocate();
}

// Implementation of custom operator delete (C++14 sized delete)
void MCTSNode::operator delete(void* ptr, size_t size) {
    // Standard behavior: ignore nullptr
    if (ptr == nullptr) return; 

    // Safety check: ensure the size matches before returning to pool
    if (size != sizeof(MCTSNode)) {
        std::cerr << "MCTSNodePool: Attempted to deallocate object of wrong size (" << size 
                  << " vs expected " << sizeof(MCTSNode) << "). Deferring to global delete." << std::endl;
        // If the size mismatches, it's safer to call the global delete operator
        // (e.g., if memory was allocated by global new or a different custom allocator).
        ::operator delete(ptr, size); 
        return;
    }
    // Delegate the deallocation to the static node pool
    s_node_pool.deallocate(ptr);
}

// --- Constructor ---
MCTSNode::MCTSNode(Board board_state, MCTSNode* parent, std::optional<Move> move, double prior) :
    board_state_(std::move(board_state)), 
    parent_(parent),
    move_(move),
    visit_count_(0),
    total_player_values_({0.0, 0.0, 0.0, 0.0}), // MODIFIED: Initialize array
    prior_(prior),
    pending_visits_(0) 
{}


// --- Tree Traversal and Properties ---
bool MCTSNode::is_leaf() const {
    return children_.empty();
}

bool MCTSNode::is_root() const {
    return parent_ == nullptr;
}

MCTSNode* MCTSNode::get_parent() const {
    return parent_;
}

const std::vector<std::unique_ptr<MCTSNode>>& MCTSNode::get_children() const {
    return children_;
}

// Added for tree reuse
std::vector<std::unique_ptr<MCTSNode>>& MCTSNode::get_children_for_reuse() {
    return children_;
}

// Added for tree reuse
void MCTSNode::set_parent(MCTSNode* p) {
    parent_ = p;
}

const Board& MCTSNode::get_board() const {
    return board_state_;
}

const std::optional<Move>& MCTSNode::get_move() const {
    return move_;
}


// --- MCTS Operations ---

MCTSNode* MCTSNode::select_child(double c_puct) const {
    if (is_leaf()) {
        return nullptr; 
    }

    MCTSNode* best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (const auto& child_ptr : children_) {
        double score = calculate_uct_score(child_ptr.get(), c_puct);
        if (score > best_score) {
            best_score = score;
            best_child = child_ptr.get();
        }
    }
    return best_child;
}

void MCTSNode::expand(const std::map<Move, double>& policy_probs) {
    if (!is_leaf()) {
         std::cerr << "Warning: Attempting to expand a non-leaf node." << std::endl;
        return;
    }
    if (board_state_.is_game_over()) {
        return;
    }

    children_.reserve(policy_probs.size()); 

    for (const auto& pair : policy_probs) {
        const Move& move = pair.first;
        double prior_prob = pair.second;
        Board next_board = Board::create_mcts_child_board(board_state_, move);
        children_.push_back(std::make_unique<MCTSNode>(std::move(next_board), this, move, prior_prob));
    }
}


void MCTSNode::update_stats(const std::array<double, 4>& values_for_players) { // MODIFIED
    visit_count_++;
    for (size_t i = 0; i < 4; ++i) {
        total_player_values_[i] += values_for_players[i];
    }
}

// --- Methods for Async Support ---
void MCTSNode::increment_pending_visits() {
    pending_visits_++;
}

void MCTSNode::decrement_pending_visits() {
    if (pending_visits_ > 0) {
        pending_visits_--;
    } else {
         std::cerr << "Warning: Decrementing pending_visits below zero for node." << std::endl;
    }
}

// --- Accessors for Node Statistics ---
int MCTSNode::get_visit_count() const {
    return visit_count_;
}

const std::array<double, 4>& MCTSNode::get_total_player_values() const { // MODIFIED
    return total_player_values_;
}

double MCTSNode::get_prior() const {
    return prior_;
}

int MCTSNode::get_pending_visits() const {
    return pending_visits_;
}

// --- Private Helper ---
double MCTSNode::calculate_uct_score(const MCTSNode* child, double c_puct) const {
    const double epsilon = 1e-8; 

    // Effective N(s,a) incorporating pending visits
    double child_visits_real = static_cast<double>(child->visit_count_);
    double child_pending_visits_val = static_cast<double>(child->pending_visits_);
    double effective_child_visits = child_visits_real + child_pending_visits_val;

    // Determine the player whose perspective matters for Q-value (the parent node's current player)
    Player parent_player_enum = this->board_state_.get_current_player();
    int parent_player_idx = static_cast<int>(parent_player_enum);

    // W'(s,a) for P_parent = W_C[P_parent] - P(s,a) * V_loss 
    // (total value for parent_player from child, minus virtual loss for pending visits)
    double child_total_value_for_parent_player = child->total_player_values_[parent_player_idx];
    double effective_value_for_parent_player = child_total_value_for_parent_player - (child_pending_visits_val * VIRTUAL_LOSS_VALUE);

    // Q'(s,a) from the perspective of the parent player
    double q_value_for_parent = 0.0;
    if (effective_child_visits > epsilon) { 
       q_value_for_parent = effective_value_for_parent_player / effective_child_visits;
    }
    // The NEGATION for opponent is REMOVED. Q-value is directly for the parent.
    
    // Parent Visits N(s) for the U-term denominator
    double parent_visits_real = static_cast<double>(this->visit_count_);
    double parent_visits_pending_val = static_cast<double>(this->pending_visits_);
    double parent_total_effective_visits = parent_visits_real + parent_visits_pending_val;

    // U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N'(s,a))
    double u_value = c_puct * child->prior_ *
                     std::sqrt(parent_total_effective_visits + epsilon) / 
                     (1.0 + effective_child_visits);

    return q_value_for_parent + u_value;
}

} // namespace chaturaji_cpp