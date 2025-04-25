#include "mcts_node.h"
#include <stdexcept>
#include <algorithm> // For std::max_element

namespace chaturaji_cpp {

// --- Constructor ---
MCTSNode::MCTSNode(Board board_state, MCTSNode* parent, std::optional<Move> move, double prior) :
    board_state_(std::move(board_state)), // Move the board state in
    parent_(parent),
    move_(move),
    visit_count_(0),
    total_value_(0.0),
    prior_(prior)
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
        // Optional: Throw an error or log a warning if trying to expand non-leaf
        // std::cerr << "Warning: Attempting to expand a non-leaf node." << std::endl;
        return;
    }
    if (board_state_.is_game_over()) {
        // Cannot expand a terminal node
        return;
    }

    // Get legal moves from the current board state (should match policy_probs keys)
    // std::vector<Move> legal_moves = board_state_.get_pseudo_legal_moves(board_state_.get_current_player());

    children_.reserve(policy_probs.size()); // Optimize allocation

    for (const auto& pair : policy_probs) {
        const Move& move = pair.first;
        double prior_prob = pair.second;

        // Create a new board state by making the move using the lightweight factory
        // This avoids copying the potentially large history/undo vectors from the parent.
        Board next_board = Board::create_mcts_child_board(board_state_, move);

        // Create the new child node
        children_.push_back(std::make_unique<MCTSNode>(std::move(next_board), this, move, prior_prob));
    }
}


void MCTSNode::update(double value) {
    visit_count_++;
    total_value_ += value; // Accumulate value (root's perspective)

    // Backpropagate to parent - REMOVED FOR BATCHED MCTS
    // The calling function (backpropagate_path) handles iteration up the tree.
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

// --- Private Helper ---
double MCTSNode::calculate_uct_score(const MCTSNode* child, double c_puct) const {
     // PUCT formula: Q(s,a) + U(s,a)
     // U(s,a) = c_puct * P(s,a) * sqrt(Sum_b N(s,b)) / (1 + N(s,a))
     // Q(s,a) = W(s,a) / N(s,a) (average value from child's perspective)
     // Python code used:
     // if child.visit_count == 0: score = c_puct * child.prior * math.sqrt(self.visit_count + 1e-8) / 1.0
     // else: score = (child.total_value / child.visit_count) + c_puct * child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)
     // Note: child.total_value is from the *root's* perspective in the Python code.
     // For standard PUCT, Q should be relative to the player whose turn it is at the child node.
     // Let's follow the Python implementation directly for now.

     const double epsilon = 1e-8; // Small value to prevent division by zero or log(0)

     if (child->visit_count_ == 0) {
        // Encourages exploration of unvisited nodes first based on prior
        // Using parent's visit count seems standard here.
        return c_puct * child->prior_ * std::sqrt(static_cast<double>(this->visit_count_) + epsilon);
        // Python code used sqrt(self.visit_count + 1e-8) / 1.0
        // Let's match Python's version slightly more closely:
        // return c_puct * child->prior_ * std::sqrt(static_cast<double>(this->visit_count_) + epsilon) / (1.0);
     } else {
        // Exploitation term (Q): Average value obtained from simulations passing through the child.
        // This value is from the root's perspective as implemented.
        double q_value = child->total_value_ / static_cast<double>(child->visit_count_);

        // Exploration term (U): Favors children with high prior and low visit count.
        double u_value = c_puct * child->prior_ *
                         std::sqrt(static_cast<double>(this->visit_count_)) /
                         (1.0 + static_cast<double>(child->visit_count_));

        return q_value + u_value;
     }
}


} // namespace chaturaji_cpp