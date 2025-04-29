#include "mcts_node.h"
#include <stdexcept>
#include <algorithm> // For std::max_element
#include <iostream>  // For potential debug warnings

namespace chaturaji_cpp {

// --- Constructor ---
MCTSNode::MCTSNode(Board board_state, MCTSNode* parent, std::optional<Move> move, double prior) :
    board_state_(std::move(board_state)), // Move the board state in
    parent_(parent),
    move_(move),
    visit_count_(0),
    total_value_(0.0),
    prior_(prior),
    pending_visits_(0) // Initialize pending visits to 0
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

const Board& MCTSNode::get_board() const {
    return board_state_;
}

const std::optional<Move>& MCTSNode::get_move() const {
    return move_;
}


// --- MCTS Operations ---

MCTSNode* MCTSNode::select_child(double c_puct) const {
    if (is_leaf()) {
        return nullptr; // Cannot select child from a leaf
    }

    MCTSNode* best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    // Ensure parent visit count is not zero before calculating sqrt (use epsilon)
    // Note: We use parent's N(s) which is `this->visit_count_ + this->pending_visits_`
    double parent_total_visits = static_cast<double>(this->visit_count_ + this->pending_visits_);

    for (const auto& child_ptr : children_) {
        // Pass parent_total_visits to avoid recalculating sqrt repeatedly
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
        // Optional: Throw an error or log a warning if trying to expand non-leaf
         std::cerr << "Warning: Attempting to expand a non-leaf node." << std::endl;
        return;
    }
    if (board_state_.is_game_over()) {
        // Cannot expand a terminal node
        return;
    }

    children_.reserve(policy_probs.size()); // Optimize allocation

    for (const auto& pair : policy_probs) {
        const Move& move = pair.first;
        double prior_prob = pair.second;

        // Create a new board state by making the move using the lightweight factory
        Board next_board = Board::create_mcts_child_board(board_state_, move);

        // Create the new child node
        children_.push_back(std::make_unique<MCTSNode>(std::move(next_board), this, move, prior_prob));
    }
}


// Renamed from update -> update_stats
void MCTSNode::update_stats(double value) {
    visit_count_++;
    total_value_ += value; // Accumulate value (root's perspective)
}

// --- Methods for Async Support ---
void MCTSNode::increment_pending_visits() {
    pending_visits_++;
}

void MCTSNode::decrement_pending_visits() {
    if (pending_visits_ > 0) {
        pending_visits_--;
    } else {
        // This indicates a potential logic error (decrementing below zero)
         std::cerr << "Warning: Decrementing pending_visits below zero for node." << std::endl;
         // Optionally, throw an exception or handle as appropriate.
    }
}

// --- Accessors for Node Statistics ---
int MCTSNode::get_visit_count() const {
    return visit_count_;
}

double MCTSNode::get_total_value() const {
    return total_value_;
}

double MCTSNode::get_prior() const {
    return prior_;
}

int MCTSNode::get_pending_visits() const {
    return pending_visits_;
}

// --- Private Helper ---
double MCTSNode::calculate_uct_score(const MCTSNode* child, double c_puct) const {
    const double epsilon = 1e-8; // Small value to prevent division by zero

    // Effective N(s,a) and W(s,a) incorporating pending visits and virtual loss
    // N'(s,a) = N(s,a) + P(s,a)  (Real visits + Pending visits)
    // W'(s,a) = W(s,a) - P(s,a) * V_loss (Real value - Virtual Losses)
    double child_visits = static_cast<double>(child->visit_count_);
    double child_pending = static_cast<double>(child->pending_visits_);
    double child_total_value = child->total_value_; // Real accumulated value

    double effective_visits = child_visits + child_pending;
    double effective_value = child_total_value - (child_pending * VIRTUAL_LOSS_VALUE);

    // Q'(s,a) = W'(s,a) / N'(s,a)
    double q_value = 0.0;
    if (effective_visits > epsilon) { // Avoid division by zero
       q_value = effective_value / effective_visits;
    }
    // If effective_visits is 0 (child unvisited and not pending), q_value remains 0.0,
    // which is appropriate as the U-term will dominate.

    // Calculate Parent Visits N(s) = sum_b N'(s,b) for the U-term denominator
    // Instead of summing children, we use the parent's recorded visits + pending visits
    // N(s) = N_parent(s) + P_parent(s)
    // Note: `this` is the parent node here.
    double parent_visits = static_cast<double>(this->visit_count_);
    double parent_pending = static_cast<double>(this->pending_visits_);
    double parent_total_effective_visits = parent_visits + parent_pending;


    // U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N'(s,a))
    // Where P(s,a) is the prior probability of the child action.
    double u_value = c_puct * child->prior_ *
                     std::sqrt(parent_total_effective_visits + epsilon) / // Add epsilon for sqrt safety
                     (1.0 + effective_visits);

    return q_value + u_value;
}


} // namespace chaturaji_cpp