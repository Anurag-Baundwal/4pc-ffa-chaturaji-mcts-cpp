#pragma once

#include <vector>
#include <future>   
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <atomic>
#include <memory>   

#include "model.h" 
#include "types.h" 
#include "thread_safe_queue.h" 

namespace chaturaji_cpp {

class Evaluator {
public:
    /**
     * @param network Pointer to the loaded ONNX Model.
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

private:
    void evaluation_loop();

    Model* network_; 
    int max_batch_size_;

    // --- Multi-Queue Implementation ---
    // Instead of one single queue protected by one lock (which causes a bottleneck 
    // when many threads try to push at once), we use multiple independent queues.
    using QueueItem = std::pair<EvaluationRequest, std::promise<EvaluationResult>>;
    
    static constexpr int NUM_INPUT_QUEUES = 8;
    
    // A vector of pointers to the queues. We pick one based on the request ID.
    std::vector<std::unique_ptr<ThreadSafeQueue<QueueItem>>> request_queues_;

    std::thread evaluator_thread_;
    std::atomic<bool> stop_requested_;
    std::atomic<RequestId> next_request_id_;
};

} // namespace chaturaji_cpp