#include "mcts_node.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>
namespace chaturaji_cpp {

MCTSNodePool MCTSNode::s_node_pool(sizeof(MCTSNode), 1500000);

void* MCTSNode::operator new(size_t size) {
    if (size != sizeof(MCTSNode)) {
        throw std::logic_error("MCTSNodePool: Wrong size.");
    }
    return s_node_pool.allocate();
}

void MCTSNode::operator delete(void* ptr, size_t size) {
    if (ptr == nullptr) return; 
    if (size != sizeof(MCTSNode)) {
        ::operator delete(ptr, size); 
        return;
    }
    s_node_pool.deallocate(ptr);
}

// --- Constructor ---
MCTSNode::MCTSNode(Board board_state, MCTSNode* parent, std::optional<Move> move, double prior) :
    board_state_(std::move(board_state)), 
    parent_(parent),
    move_(move),
    visit_count_(0),
    total_player_values_({0.0, 0.0, 0.0, 0.0}),
    prior_(prior),
    pending_visits_(0) 
{}

MCTSNode::~MCTSNode() {
    MCTSNode* current = first_child_;
    while (current) {
        MCTSNode* next = current->next_sibling_;
        current->next_sibling_ = nullptr;
        delete current;
        current = next;
    }
}

// --- Tree Traversal and Properties ---
bool MCTSNode::is_leaf() const {
    return first_child_ == nullptr;
}

bool MCTSNode::is_root() const {
    return parent_ == nullptr;
}

MCTSNode* MCTSNode::get_parent() const {
    return parent_;
}

MCTSNode* MCTSNode::get_first_child() const {
    return first_child_;
}

MCTSNode* MCTSNode::get_next_sibling() const {
    return next_sibling_;
}

void MCTSNode::set_next_sibling(MCTSNode* node) {
    next_sibling_ = node;
}

void MCTSNode::set_first_child(MCTSNode* node) {
    first_child_ = node;
}

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

    const double epsilon = 1e-8;
    const double cpuct_base = 6144.0;

    double parent_visits = static_cast<double>(this->visit_count_) + static_cast<double>(this->pending_visits_);
    // Compute invariant terms
    double pb_c_base_term = std::log((parent_visits + cpuct_base + 1.0) / cpuct_base) + c_puct;
    double parent_visits_sqrt = std::sqrt(parent_visits + epsilon);

    MCTSNode* best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (MCTSNode* child = first_child_; child != nullptr; child = child->next_sibling_) {
        double score = calculate_uct_score(child, pb_c_base_term, parent_visits_sqrt);
        if (score > best_score) {
            best_score = score;
            best_child = child;
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

    MCTSNode* prev_child = nullptr;

    for (const auto& pair : policy_probs) {
        const Move& move = pair.first;
        double prior_prob = pair.second;
        Board next_board = Board::create_mcts_child_board(board_state_, move);

        MCTSNode* new_node = new MCTSNode(std::move(next_board), this, move, prior_prob);

        if (prev_child) {
            prev_child->next_sibling_ = new_node;
        } else {
            first_child_ = new_node;
        }
        prev_child = new_node;
    }
}


void MCTSNode::update_stats(const std::array<double, 4>& values_for_players) {
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

void MCTSNode::inject_noise(double alpha, double epsilon, std::mt19937& rng) {
    if (is_leaf()) return;

    int child_count = 0;
    for (MCTSNode* c = first_child_; c; c = c->next_sibling_) child_count++;

    std::gamma_distribution<double> gamma_dist(alpha, 1.0);
    std::vector<double> noise_samples;
    noise_samples.reserve(child_count);
    double noise_sum = 0.0;

    for (int i = 0; i < child_count; ++i) {
        double n = gamma_dist(rng);
        noise_samples.push_back(n);
        noise_sum += n;
    }

    if (noise_sum < 1e-9) noise_sum = 1.0;

    int idx = 0;
    for (MCTSNode* c = first_child_; c; c = c->next_sibling_) {
        double normalized_noise = noise_samples[idx++] / noise_sum;
        c->prior_ = (1.0 - epsilon) * c->prior_ + epsilon * normalized_noise;
    }
}

// --- Accessors for Node Statistics ---
int MCTSNode::get_visit_count() const {
    return visit_count_;
}

const std::array<double, 4>& MCTSNode::get_total_player_values() const {
    return total_player_values_;
}

double MCTSNode::get_prior() const {
    return prior_;
}

int MCTSNode::get_pending_visits() const {
    return pending_visits_;
}

double MCTSNode::calculate_uct_score(const MCTSNode* child, double pb_c_base_term, double parent_visits_sqrt) const {
    const double epsilon = 1e-8; 
    
    double child_visits = static_cast<double>(child->visit_count_) + static_cast<double>(child->pending_visits_);

    Player parent_player_enum = this->board_state_.get_current_player();
    int parent_player_idx = static_cast<int>(parent_player_enum);

    double child_total_value_for_parent = child->total_player_values_[parent_player_idx];
    double effective_value = child_total_value_for_parent - (static_cast<double>(child->pending_visits_) * VIRTUAL_LOSS_VALUE);

    double q_value = 0.0;
    if (child_visits > epsilon) { 
       q_value = effective_value / child_visits;
    }

    // U(s,a) = pb_c * P(s,a) * sqrt(N_parent) / (1 + N_child)
    double u_value = pb_c_base_term * child->prior_ * parent_visits_sqrt / (1.0 + child_visits);

    return q_value + u_value;
}

} // namespace chaturaji_cpp