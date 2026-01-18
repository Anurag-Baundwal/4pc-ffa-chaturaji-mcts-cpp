#include "mcts_node.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>

namespace chaturaji_cpp {

MCTSNodePool MCTSNode::s_node_pool(sizeof(MCTSNode), 1500000); 

void* MCTSNode::operator new(size_t size) {
    if (size != sizeof(MCTSNode)) throw std::logic_error("MCTSNodePool: Allocation size mismatch.");
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
    first_child_(nullptr),   // Init
    next_sibling_(nullptr),  // Init
    visit_count_(0),
    total_player_values_({0.0, 0.0, 0.0, 0.0}),
    prior_(prior),
    pending_visits_(0) 
{}

// --- Destructor ---
MCTSNode::~MCTSNode() {
    // Iteratively delete the linked list of children to prevent recursion depth issues
    // though strict recursion on 'delete' happens here, the width is handled iteratively.
    // For deep trees, a stack-based deletion might be safer, but for MCTS depth ~200 this is okay.
    MCTSNode* current = first_child_;
    while (current) {
        MCTSNode* next = current->next_sibling_;
        delete current;
        current = next;
    }
}

// --- Tree Traversal ---
bool MCTSNode::is_leaf() const { return first_child_ == nullptr; }
bool MCTSNode::is_root() const { return parent_ == nullptr; }
MCTSNode* MCTSNode::get_parent() const { return parent_; }
void MCTSNode::set_parent(MCTSNode* p) { parent_ = p; }

MCTSNode* MCTSNode::get_first_child() const { return first_child_; }
MCTSNode* MCTSNode::get_next_sibling() const { return next_sibling_; }

const Board& MCTSNode::get_board() const { return board_state_; }
const std::optional<Move>& MCTSNode::get_move() const { return move_; }

// --- MCTS Operations ---

MCTSNode* MCTSNode::select_child(double c_puct) const {
    if (is_leaf()) return nullptr; 

    MCTSNode* best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    MCTSNode* child = first_child_;
    while (child) {
        double score = calculate_uct_score(child, c_puct);
        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
        child = child->next_sibling_;
    }
    return best_child;
}

void MCTSNode::expand(const std::map<Move, double>& policy_probs) {
    if (!is_leaf()) {
         std::cerr << "Warning: Attempting to expand a non-leaf node." << std::endl;
        return;
    }
    if (board_state_.is_game_over()) return;

    MCTSNode* tail = nullptr;

    for (const auto& pair : policy_probs) {
        const Move& move = pair.first;
        double prior_prob = pair.second;
        
        // Create new node
        Board next_board = Board::create_mcts_child_board(board_state_, move);
        MCTSNode* new_node = new MCTSNode(std::move(next_board), this, move, prior_prob);

        // Link into list
        if (!tail) {
            first_child_ = new_node;
        } else {
            tail->next_sibling_ = new_node;
        }
        tail = new_node;
    }
}

void MCTSNode::update_stats(const std::array<double, 4>& values_for_players) { 
    visit_count_++;
    for (size_t i = 0; i < 4; ++i) {
        total_player_values_[i] += values_for_players[i];
    }
}

void MCTSNode::increment_pending_visits() { pending_visits_++; }
void MCTSNode::decrement_pending_visits() {
    if (pending_visits_ > 0) pending_visits_--;
}

void MCTSNode::inject_noise(double alpha, double epsilon, std::mt19937& rng) {
    if (is_leaf()) return;

    // 1. Count children
    int child_count = 0;
    MCTSNode* curr = first_child_;
    while (curr) { child_count++; curr = curr->next_sibling_; }

    if (child_count == 0) return;

    // 2. Generate Noise
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

    // 3. Apply
    curr = first_child_;
    int idx = 0;
    while (curr) {
        double normalized_noise = noise_samples[idx] / noise_sum;
        curr->prior_ = (1.0 - epsilon) * curr->prior_ + epsilon * normalized_noise;
        curr = curr->next_sibling_;
        idx++;
    }
}

// --- Tree Reuse Logic ---
MCTSNode* MCTSNode::detach_child_and_clear_others(MCTSNode* target_child) {
    MCTSNode* curr = first_child_;
    
    // We iterate through all children.
    // If it's the target, we detach it.
    // If it's not, we delete it.
    
    while (curr) {
        MCTSNode* next = curr->next_sibling_;
        
        if (curr == target_child) {
            // Unhook this child from the list structure
            curr->next_sibling_ = nullptr;
            curr->parent_ = nullptr;
        } else {
            // Delete the unused sibling
            delete curr;
        }
        curr = next;
    }

    // Now this node (the parent) has no valid children
    first_child_ = nullptr;
    
    return target_child;
}

// --- Accessors ---
int MCTSNode::get_visit_count() const { return visit_count_; }
const std::array<double, 4>& MCTSNode::get_total_player_values() const { return total_player_values_; }
double MCTSNode::get_prior() const { return prior_; }
int MCTSNode::get_pending_visits() const { return pending_visits_; }

double MCTSNode::calculate_uct_score(const MCTSNode* child, double c_puct) const {
    const double epsilon = 1e-8; 
    const double cpuct_base = 6144.0;
    
    double parent_visits = static_cast<double>(this->visit_count_) + static_cast<double>(this->pending_visits_);
    double child_visits = static_cast<double>(child->visit_count_) + static_cast<double>(child->pending_visits_);

    double pb_c = std::log((parent_visits + cpuct_base + 1.0) / cpuct_base) + c_puct;

    Player parent_player_enum = this->board_state_.get_current_player();
    int parent_player_idx = static_cast<int>(parent_player_enum);

    double child_total_value_for_parent = child->total_player_values_[parent_player_idx];
    double effective_value = child_total_value_for_parent - (static_cast<double>(child->pending_visits_) * VIRTUAL_LOSS_VALUE);

    double q_value = 0.0;
    if (child_visits > epsilon) { 
       q_value = effective_value / child_visits;
    }

    double u_value = pb_c * child->prior_ * std::sqrt(parent_visits + epsilon) / (1.0 + child_visits);
    return q_value + u_value;
}

} // namespace chaturaji_cpp