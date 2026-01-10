// mcts_node.cpp
#include "mcts_node.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>

namespace chaturaji_cpp {

MCTSNodePool MCTSNode::s_node_pool(sizeof(MCTSNode), 1500000); 

void* MCTSNode::operator new(size_t size) {
    if (size != sizeof(MCTSNode)) throw std::logic_error("Wrong size allocation");
    return s_node_pool.allocate();
}

void MCTSNode::operator delete(void* ptr, size_t size) {
    if (ptr == nullptr) return; 
    if (size != sizeof(MCTSNode)) { ::operator delete(ptr, size); return; }
    s_node_pool.deallocate(ptr);
}

MCTSNode::MCTSNode(Board board_state, MCTSNode* parent, std::optional<Move> move, double prior) :
    board_state_(std::move(board_state)), 
    parent_(parent),
    move_(move),
    visit_count_(0),
    total_player_values_({0.0, 0.0, 0.0, 0.0}), 
    prior_(prior),
    pending_visits_(0) 
{}

bool MCTSNode::is_leaf() const { return children_.empty(); }
bool MCTSNode::is_root() const { return parent_ == nullptr; }
MCTSNode* MCTSNode::get_parent() const { return parent_; }
const std::vector<std::unique_ptr<MCTSNode>>& MCTSNode::get_children() const { return children_; }
std::vector<std::unique_ptr<MCTSNode>>& MCTSNode::get_children_for_reuse() { return children_; }
void MCTSNode::set_parent(MCTSNode* p) { parent_ = p; }
const Board& MCTSNode::get_board() const { return board_state_; }
const std::optional<Move>& MCTSNode::get_move() const { return move_; }

MCTSNode* MCTSNode::select_child(double c_puct, Player root_player, double spite_weight) const {
    if (is_leaf()) return nullptr; 

    MCTSNode* best_child = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (const auto& child_ptr : children_) {
        double score = calculate_uct_score(child_ptr.get(), c_puct, root_player, spite_weight);
        if (score > best_score) {
            best_score = score;
            best_child = child_ptr.get();
        }
    }
    return best_child;
}

void MCTSNode::expand(const std::map<Move, double>& policy_probs) {
    if (!is_leaf()) return;
    if (board_state_.is_game_over()) return;
    children_.reserve(policy_probs.size()); 
    for (const auto& pair : policy_probs) {
        Board next_board = Board::create_mcts_child_board(board_state_, pair.first);
        children_.push_back(std::make_unique<MCTSNode>(std::move(next_board), this, pair.first, pair.second));
    }
}

void MCTSNode::update_stats(const std::array<double, 4>& values_for_players) {
    visit_count_++;
    for (size_t i = 0; i < 4; ++i) total_player_values_[i] += values_for_players[i];
}

void MCTSNode::increment_pending_visits() { pending_visits_++; }
void MCTSNode::decrement_pending_visits() { if (pending_visits_ > 0) pending_visits_--; }

void MCTSNode::inject_noise(double alpha, double epsilon, std::mt19937& rng) {
    if (children_.empty()) return;
    std::gamma_distribution<double> gamma_dist(alpha, 1.0);
    std::vector<double> noise_samples;
    double noise_sum = 0.0;
    for (size_t i = 0; i < children_.size(); ++i) {
        double n = gamma_dist(rng);
        noise_samples.push_back(n);
        noise_sum += n;
    }
    if (noise_sum < 1e-9) noise_sum = 1.0;
    for (size_t i = 0; i < children_.size(); ++i) {
        children_[i]->prior_ = (1.0 - epsilon) * children_[i]->prior_ + epsilon * (noise_samples[i] / noise_sum);
    }
}

int MCTSNode::get_visit_count() const { return visit_count_; }
const std::array<double, 4>& MCTSNode::get_total_player_values() const { return total_player_values_; }
double MCTSNode::get_prior() const { return prior_; }
int MCTSNode::get_pending_visits() const { return pending_visits_; }

double MCTSNode::calculate_uct_score(const MCTSNode* child, double c_puct, Player root_player, double spite_weight) const {
    const double epsilon = 1e-8; 
    const double cpuct_base = 6144.0;
    
    double parent_visits = static_cast<double>(this->visit_count_) + static_cast<double>(this->pending_visits_);
    double child_visits = static_cast<double>(child->visit_count_) + static_cast<double>(child->pending_visits_);

    // 1. Logic Identification
    Player mover_enum = this->board_state_.get_current_player();
    int mover_idx = static_cast<int>(mover_enum);
    int root_idx = static_cast<int>(root_player);

    // 2. Base Value (Average Q)
    double q_value = 0.0;
    if (child_visits > epsilon) {
        double value_sum = child->total_player_values_[mover_idx];
        
        // --- UTILITY MIXING (SPITE) ---
        // If it's an opponent's turn, adjust their utility to prioritize lowering our score
        if (mover_idx != root_idx && spite_weight > 0.001) {
            double value_us = child->total_player_values_[root_idx];
            // U_opp = (1-w)*V_opp + w*(-V_us)
            value_sum = ((1.0 - spite_weight) * value_sum) - (spite_weight * value_us);
        }

        // Apply virtual loss to the averaged Q
        q_value = (value_sum / child_visits) - (static_cast<double>(child->pending_visits_) * VIRTUAL_LOSS_VALUE / std::max(1.0, child_visits));
    }

    // 3. U-Value (Exploration)
    double pb_c = std::log((parent_visits + cpuct_base + 1.0) / cpuct_base) + c_puct;
    double u_value = pb_c * child->prior_ * std::sqrt(parent_visits + epsilon) / (1.0 + child_visits);

    return q_value + u_value;
}

} // namespace chaturaji_cpp