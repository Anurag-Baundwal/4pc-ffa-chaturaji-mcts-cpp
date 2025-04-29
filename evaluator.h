#pragma once

#include <vector>
#include <future>   // For std::promise, std::future
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <atomic>
#include <memory>   // For std::shared_ptr

#include <torch/torch.h>

#include "model.h" // Needs ChaturajiNN
#include "types.h" // Needs EvaluationRequest, EvaluationResult, RequestId
#include "thread_safe_queue.h" // Needs ThreadSafeQueue

namespace chaturaji_cpp {

class Evaluator {
public:
    /**
     * @param network A handle to the neural network model.
     * @param device The device to run inference on.
     * @param max_batch_size The maximum number of requests to batch together.
     */
    Evaluator(ChaturajiNN network, torch::Device device, int max_batch_size = 4096);
    ~Evaluator();

    // Rule of 5/6: Disable copy/move operations as it manages a thread
    Evaluator(const Evaluator&) = delete;
    Evaluator& operator=(const Evaluator&) = delete;
    Evaluator(Evaluator&&) = delete;
    Evaluator& operator=(Evaluator&&) = delete;

    /**
     * @brief Starts the background evaluator thread.
     */
    void start();

    /**
     * @brief Signals the evaluator thread to stop and waits for it to finish.
     */
    void stop();

    /**
     * @brief Submits an evaluation request and returns a future for the result.
     *        This method is thread-safe.
     * @param request The evaluation request (state tensor should be on CPU).
     * @return A std::future<EvaluationResult> that will eventually hold the result.
     */
    std::future<EvaluationResult> submit_request(EvaluationRequest request);


private:
    /**
     * @brief The main loop executed by the background evaluator thread.
     */
    void evaluation_loop();

    ChaturajiNN network_;
    torch::Device device_;
    int max_batch_size_;

    // Communication mechanisms
    ThreadSafeQueue<EvaluationRequest> request_queue_;
    std::map<RequestId, std::promise<EvaluationResult>> pending_results_map_;

    // Synchronization
    std::mutex map_mutex_; // Protects pending_results_map_
    std::condition_variable evaluator_cv_; // For evaluator to wait on requests
    // Note: std::promise/future handles worker waiting implicitly

    // Thread management
    std::thread evaluator_thread_;
    std::atomic<bool> stop_requested_;
    std::atomic<RequestId> next_request_id_; // Simple atomic counter for IDs
};

} // namespace chaturaji_cpp