#include "mcts_node.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>
namespace chaturaji_cpp {

// Define and initialize the static MCTSNodePool member.
// The constructor MCTSNodePool(size_t) will be called automatically once at program startup.
MCTSNodePool MCTSNode::s_node_pool(sizeof(MCTSNode), 1500000); // Initialized with 1,500,000 node capacity

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

void MCTSNode::inject_noise(double alpha, double epsilon, std::mt19937& rng) {
    if (children_.empty()) return;

    std::gamma_distribution<double> gamma_dist(alpha, 1.0);
    std::vector<double> noise_samples;
    noise_samples.reserve(children_.size());
    double noise_sum = 0.0;

    // 1. Generate Gamma noise
    for (size_t i = 0; i < children_.size(); ++i) {
        double n = gamma_dist(rng);
        noise_samples.push_back(n);
        noise_sum += n;
    }

    // 2. Normalize noise
    if (noise_sum < 1e-9) noise_sum = 1.0; // Prevent div by zero

    // 3. Mix into existing priors
    for (size_t i = 0; i < children_.size(); ++i) {
        double normalized_noise = noise_samples[i] / noise_sum;
        double original_prior = children_[i]->prior_;
        
        // P(a) = (1 - eps) * P(a) + eps * Noise
        children_[i]->prior_ = (1.0 - epsilon) * original_prior + epsilon * normalized_noise;
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

    // --- Dynamic CPUCT (AlphaZero / Lc0 Formula) ---
    // User constants: base = 6144, init = c_puct (passed from search)
    const double cpuct_base = 6144.0;
    
    // Effective parent and child visits
    double parent_visits = static_cast<double>(this->visit_count_) + static_cast<double>(this->pending_visits_);
    double child_visits = static_cast<double>(child->visit_count_) + static_cast<double>(child->pending_visits_);

    // pb_c = log((parent_n + base + 1)/base) + init
    double pb_c = std::log((parent_visits + cpuct_base + 1.0) / cpuct_base) + c_puct;

    // --- Q-Value Calculation ---
    Player parent_player_enum = this->board_state_.get_current_player();
    int parent_player_idx = static_cast<int>(parent_player_enum);

    double child_total_value_for_parent = child->total_player_values_[parent_player_idx];
    double effective_value = child_total_value_for_parent - (static_cast<double>(child->pending_visits_) * VIRTUAL_LOSS_VALUE);

    double q_value = 0.0;
    if (child_visits > epsilon) { 
       q_value = effective_value / child_visits;
    }

    // --- U-Value Calculation ---
    // U(s,a) = pb_c * P(s,a) * sqrt(N_parent) / (1 + N_child)
    double u_value = pb_c * child->prior_ * std::sqrt(parent_visits + epsilon) / (1.0 + child_visits);

    return q_value + u_value;
}

} // namespace chaturaji_cpp