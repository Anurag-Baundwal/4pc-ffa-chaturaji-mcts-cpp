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
#include "transposition_table.h"

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

private:
    void evaluation_loop();

    Model* network_; // Non-owning pointer
    int max_batch_size_;

    ThreadSafeQueue<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> request_queue_;

    std::thread evaluator_thread_;
    std::atomic<bool> stop_requested_;
    std::atomic<RequestId> next_request_id_;
};

} // namespace chaturaji_cpp