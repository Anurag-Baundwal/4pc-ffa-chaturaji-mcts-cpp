#include "mcts_node.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

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
    first_child_(nullptr),   
    next_sibling_(nullptr),  
    visit_count_(0),
    total_player_values_({0.0, 0.0, 0.0, 0.0}),
    total_sq_values_({0.0, 0.0, 0.0, 0.0}), 
    prior_(prior),
    pending_visits_(0) 
{}

// --- Destructor (Iterative / Stack-Safe) ---
MCTSNode::~MCTSNode() {
    // Base case: If no children, nothing to do.
    // NOTE: We do NOT handle siblings here. The parent is responsible for 
    // iterating the sibling list and deleting them. This prevents double-frees.
    if (!first_child_) return;

    // Use a vector as a stack to iteratively delete the subtree.
    // This prevents Stack Overflow on deep trees.
    std::vector<MCTSNode*> stack;
    stack.reserve(64); // Heuristic reserve
    
    // 1. Collect all immediate children of *this* node
    MCTSNode* curr = first_child_;
    while (curr) {
        MCTSNode* next = curr->next_sibling_;
        
        // Critical: Break the sibling link so the nodes are isolated
        curr->next_sibling_ = nullptr; 
        
        stack.push_back(curr);
        curr = next;
    }

    // 2. Detach children from *this* node immediately
    first_child_ = nullptr;

    // 3. Process the stack
    while (!stack.empty()) {
        MCTSNode* n = stack.back();
        stack.pop_back();

        // Before we delete 'n', we must collect ITS children
        MCTSNode* child = n->first_child_;
        while (child) {
            MCTSNode* next_child = child->next_sibling_;
            
            // Isolate child
            child->next_sibling_ = nullptr; 
            
            stack.push_back(child);
            child = next_child;
        }

        // Disarm 'n' so its destructor returns immediately when called below.
        // This ensures the `delete n` call doesn't trigger recursion.
        n->first_child_ = nullptr;
        n->next_sibling_ = nullptr;

        // Delete the node (returns memory to pool)
        delete n; 
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

MCTSNode* MCTSNode::select_child(double c_puct, double risk_alpha) const {
    if (is_leaf()) return nullptr; 

    MCTSNode* best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    MCTSNode* child = first_child_;
    while (child) {
        double score = calculate_uct_score(child, c_puct, risk_alpha);
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
         // std::cerr << "Warning: Attempting to expand a non-leaf node." << std::endl;
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
        double val = values_for_players[i];
        total_player_values_[i] += val;
        total_sq_values_[i] += (val * val);
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
    
    // 1. Detach the list from *this* parent immediately.
    // The parent is now clean.
    first_child_ = nullptr; 

    while (curr) {
        MCTSNode* next = curr->next_sibling_;
        
        if (curr == target_child) {
            // Isolate the target child so we can return it safely
            curr->next_sibling_ = nullptr;
            curr->parent_ = nullptr;
        } else {
            // Delete the unused sibling.
            // Because our new destructor DOES NOT iterate siblings, 
            // we are safe to just break the link and delete.
            curr->next_sibling_ = nullptr; 
            
            // This triggers the iterative subtree deletion for 'curr'
            delete curr; 
        }
        curr = next;
    }

    return target_child;
}

// --- Accessors ---
int MCTSNode::get_visit_count() const { return visit_count_; }
const std::array<double, 4>& MCTSNode::get_total_player_values() const { return total_player_values_; }
double MCTSNode::get_prior() const { return prior_; }
int MCTSNode::get_pending_visits() const { return pending_visits_; }

double MCTSNode::calculate_uct_score(const MCTSNode* child, double c_puct, double risk_alpha) const {
    const double epsilon = 1e-8; 
    const double cpuct_base = 6144.0;
    
    // 1. Setup Visits
    // Real visits (for stats)
    double real_child_visits = static_cast<double>(child->visit_count_);
    
    // Effective visits (for UCT flow control)
    double parent_visits_total = static_cast<double>(this->visit_count_) + static_cast<double>(this->pending_visits_);
    double child_visits_total = real_child_visits + static_cast<double>(child->pending_visits_);

    // 2. Calculate PUCT Term (U)
    double pb_c = std::log((parent_visits_total + cpuct_base + 1.0) / cpuct_base) + c_puct;
    double u_value = pb_c * child->prior_ * std::sqrt(parent_visits_total + epsilon) / (1.0 + child_visits_total);

    // 3. Calculate Q Value (Mean) - INCLUDES Virtual Loss
    // This is necessary to prevent threads from swarming this node while it's being evaluated.
    Player parent_player_enum = this->board_state_.get_current_player();
    int p_idx = static_cast<int>(parent_player_enum);

    double child_total_value = child->total_player_values_[p_idx];
    
    // Apply Virtual Loss to the SUM, not the average.
    // Logic: subtract 1.0 (or VIRTUAL_LOSS_VALUE) for every pending visit.
    double effective_value_sum = child_total_value - (static_cast<double>(child->pending_visits_) * VIRTUAL_LOSS_VALUE);

    double q_value = 0.0;
    if (child_visits_total > epsilon) { 
       q_value = effective_value_sum / child_visits_total;
    }

    // 4. Calculate Risk Penalty - EXCLUDES Virtual Loss
    // We only want to penalize based on REAL statistical data.
    double risk_penalty = 0.0;

    // Only calculate risk if enabled AND we have actual data points
    if (std::abs(risk_alpha) > 1e-9 && real_child_visits > 0.0) {
        // A. Calculate Real Mean (E[x])
        double real_mean = child_total_value / real_child_visits;

        // B. Calculate Real Mean of Squares (E[x^2])
        double sum_sq = child->total_sq_values_[p_idx];
        double real_mean_sq = sum_sq / real_child_visits;

        // C. Calculate Variance: E[x^2] - (E[x])^2
        // Use max(0, ...) to handle tiny floating point errors yielding -0.000001
        double variance = std::max(0.0, real_mean_sq - (real_mean * real_mean));
        double std_dev = std::sqrt(variance);

        // D. Calculate Dynamic Beta
        double beta = std::sqrt((real_mean + 1.0) * 0.5);

        // E. Final Penalty
        risk_penalty = risk_alpha * beta * std_dev;
    }

    // Final UCT Formula: Q + U - Penalty
    return q_value + u_value - risk_penalty;
}

} // namespace chaturaji_cpp