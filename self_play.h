#pragma once

#include <vector>
#include <deque>
#include <map>
#include <tuple> 
#include <memory> 
#include <random> 
#include <thread> 
#include <atomic> 
#include <array> 

#include "board.h"
#include "mcts_node.h"
#include "model.h"
#include "search.h" 
#include "types.h"
#include "utils.h" 
#include "evaluator.h" 

namespace chaturaji_cpp {

class DataWriter;

using GameDataStep = std::tuple<Board, std::map<Move, double>, Player, std::array<double, 4>>;
using ReplayBuffer = std::deque<GameDataStep>;

// --- Bandit Structure for Auto-Tuning Risk Alpha ---
struct AlphaArm {
    double value;
    std::atomic<int> selections{0};
    std::atomic<double> total_score{0.0};
    
    AlphaArm(double v) : value(v) {}
};

class SelfPlay {
public:
    SelfPlay(
        Model* network, 
        int num_workers = 4,
        int simulations_per_move = 100,
        int nn_batch_size = NN_POLICY_SIZE, // (4096)
        int worker_batch_size = 16,
        double c_puct = 2.5,
        int temperature_decay_move = 4,
        double dirichlet_alpha = 0.3,
        double dirichlet_epsilon = 0.25,
        double risk_alpha = 0.0 
    );

    ~SelfPlay(); 

    size_t generate_data(int num_games);

    void save_bandit_stats(const std::string& path);
    void load_bandit_stats(const std::string& path);

private:
    void run_game_simulation(
        int worker_id,
        std::atomic<int>& games_started_counter,
        std::atomic<int>& games_completed_counter,
        int target_games,
        std::vector<GameDataStep>& local_buffer
    );

    std::map<Move, double> get_action_probs(const MCTSNode& root, double temperature) const;
    Move choose_move(const MCTSNode& root, double temperature);
    
    void process_game_result(
        std::vector<std::tuple<Board, std::map<Move, double>, Player>>& game_data_temp,
        const Board& final_board,
        std::vector<GameDataStep>& output_buffer 
    );
    
    std::map<Move, double> add_dirichlet_noise(
      const std::map<Move, double>& policy_probs,
      double alpha,
      double epsilon
    );

    void process_worker_batch(
      std::vector<SimulationState>& pending_batch,
      Player root_player, 
      bool& apply_root_noise 
    );

    // --- Bandit / Auto-Tuning Helpers ---
    int select_arm_index_ucb1();
    double calculate_ucb_score(int arm_idx, int total_plays) const;
    void update_arm_stats(int arm_idx, double reward);
    void register_arm_selection(int arm_idx);

    Model* network_handle_; // Non-owning pointer
    int num_workers_;
    int simulations_per_move_;
    double mcts_c_puct_;
    int temperature_decay_move_;
    int worker_batch_size_; 
    double dirichlet_alpha_;
    double dirichlet_epsilon_;

    std::atomic<bool> bandit_stats_updated_{false};
    
    // NOTE: risk_alpha_ member is removed/ignored in favor of alpha_arms_ below
    std::vector<std::unique_ptr<AlphaArm>> alpha_arms_; 

    std::mt19937 rng_; 

    std::unique_ptr<Evaluator> evaluator_; 
    std::vector<std::thread> worker_threads_;
    
    std::unique_ptr<DataWriter> data_writer_; 
};

} // namespace chaturaji_cpp