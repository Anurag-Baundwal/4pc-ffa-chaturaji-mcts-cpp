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

class SelfPlay {
public:
    SelfPlay(
        Model* network, 
        int num_workers = 4,
        int simulations_per_move = 100,
        size_t max_buffer_size = 1250000,
        int nn_batch_size = 4096,
        int worker_batch_size = 16,
        double c_puct = 2.5,
        int temperature_decay_move = 4,
        double dirichlet_alpha = 0.3,
        double dirichlet_epsilon = 0.25
    );

    ~SelfPlay(); 

    size_t generate_data(int num_games);
    
    const ReplayBuffer& get_buffer() const;
    void clear_buffer();

private:
    void run_game_simulation(
        int worker_id,
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

    Model* network_handle_; // Non-owning pointer
    int num_workers_;
    int simulations_per_move_;
    size_t max_buffer_size_;
    ReplayBuffer buffer_; 
    double mcts_c_puct_;
    int temperature_decay_move_;
    int worker_batch_size_; 
    double dirichlet_alpha_;
    double dirichlet_epsilon_;

    std::mt19937 rng_; 

    std::unique_ptr<Evaluator> evaluator_; 
    std::vector<std::thread> worker_threads_;
    
    std::unique_ptr<DataWriter> data_writer_; 
};

} // namespace chaturaji_cpp