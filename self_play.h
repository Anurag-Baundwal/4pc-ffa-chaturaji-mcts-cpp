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
#include <future>

#include "board.h"
#include "mcts_node.h"
#include "model.h"
#include "search.h" 
#include "types.h"
#include "utils.h" 
#include "evaluator.h" 

namespace chaturaji_cpp {

class DataWriter;

using GameDataStep = std::tuple<Board, std::map<Move, double>, Player, std::array<double, 16>>;

class SelfPlay {
public:
    SelfPlay(
        Model* network, 
        int num_workers = 4,
        int games_per_worker = 8,
        int simulations_per_move = 100,
        int nn_batch_size = NN_POLICY_SIZE, // (4096)
        int worker_batch_size = 16,
        double c_puct = 2.5,
        int temperature_decay_move = 4,
        double dirichlet_alpha = 0.3,
        double dirichlet_epsilon = 0.25
    );

    ~SelfPlay(); 

    size_t generate_data(int num_games);

private:
    void run_game_simulation(
        int worker_id,
        std::atomic<int>& games_started_counter,
        std::atomic<int>& games_completed_counter,
        int target_games,
        std::vector<GameDataStep>& local_buffer
    );

    std::map<Move, double> get_action_probs(const MCTSNode& root, double temperature, bool allow_resignation = true) const;
    Move choose_move(const MCTSNode& root, double temperature, bool allow_resignation = true);
    
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

    // --- Split Batch Processing for Async Execution ---
    
    // Submits a batch to the evaluator and populates out_futures with the returned futures
    void submit_inference_batch(
        std::vector<SimulationState>& batch,
        std::vector<std::future<EvaluationResult>>& out_futures
    );

    // Waits for futures (if necessary), processes results, and updates the tree
    void process_inference_results(
        std::vector<SimulationState>& batch,
        std::vector<std::future<EvaluationResult>>& futures,
        bool& root_noise_applicable 
    );

    Model* network_handle_; // Non-owning pointer
    int num_workers_;
    int games_per_worker_;
    int simulations_per_move_;
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