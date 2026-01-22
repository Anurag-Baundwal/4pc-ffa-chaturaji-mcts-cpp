#pragma once

#include <vector>
#include <future>   
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <atomic>
#include <memory>   

#include "model.h" // Uses the new ONNX-based Model class
#include "types.h" 
#include "thread_safe_queue.h" 

namespace chaturaji_cpp {

class Evaluator {
public:
    /**
     * @param network Pointer to the loaded ONNX Model. The Evaluator does NOT own the model.
     * @param max_batch_size The maximum number of requests to batch together.
     */
    Evaluator(Model* network, int max_batch_size = 4096);
    ~Evaluator();

    Evaluator(const Evaluator&) = delete;
    Evaluator& operator=(const Evaluator&) = delete;
    Evaluator(Evaluator&&) = delete;
    Evaluator& operator=(Evaluator&&) = delete;

    void start();
    void stop();

    std::future<EvaluationResult> submit_request(EvaluationRequest request);

    // Optimized batch submission to reduce locking
    std::vector<std::future<EvaluationResult>> submit_batch(std::vector<EvaluationRequest>&& requests);

private:
    void evaluation_loop();

    Model* network_; // Non-owning pointer
    int max_batch_size_;

    // Sharded queue for reduced contention
    const int num_shards_ = 8;
    std::vector<std::unique_ptr<ThreadSafeQueue<std::pair<EvaluationRequest, std::promise<EvaluationResult>>>>> request_queues_;

    std::thread evaluator_thread_;
    std::atomic<bool> stop_requested_;
    std::atomic<RequestId> next_request_id_;

    // High-throughput lock-free synchronization
    std::atomic<size_t> total_pending_count_{0};
};

} // namespace chaturaji_cpp