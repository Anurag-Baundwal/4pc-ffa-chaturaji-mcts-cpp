#include "utils.h"
#include "evaluator.h"
#include <iostream>
#include <chrono>   
#include <vector>
#include <stdexcept> 

namespace chaturaji_cpp {

Evaluator::Evaluator(Model* network, int max_batch_size) :
    network_(network),
    max_batch_size_(max_batch_size),
    stop_requested_(false),
    next_request_id_(0)
{
    if (!network_) {
        throw std::runtime_error("Evaluator received a null network pointer.");
    }
    
    // Initialize the input queues
    request_queues_.reserve(NUM_INPUT_QUEUES);
    for (int i = 0; i < NUM_INPUT_QUEUES; ++i) {
        request_queues_.push_back(std::make_unique<ThreadSafeQueue<QueueItem>>());
    }
}

Evaluator::~Evaluator() {
    stop(); 
}

void Evaluator::start() {
    if (evaluator_thread_.joinable()) {
        std::cerr << "Warning: Evaluator thread already started." << std::endl;
        return;
    }
    stop_requested_ = false;
    evaluator_thread_ = std::thread(&Evaluator::evaluation_loop, this);
    std::cout << "Evaluator thread started (using " << NUM_INPUT_QUEUES << " input queues)." << std::endl;
}

void Evaluator::stop() {
    if (!evaluator_thread_.joinable()) {
        return; 
    }
    stop_requested_ = true;
    signal_cv_.notify_all();
    evaluator_thread_.join();
    std::cout << "Evaluator thread stopped." << std::endl;
}

std::future<EvaluationResult> Evaluator::submit_request(EvaluationRequest request) {
    // 1. Assign Unique ID
    RequestId id = next_request_id_++;
    request.request_id = id;

    std::promise<EvaluationResult> result_promise;
    std::future<EvaluationResult> result_future = result_promise.get_future();

    // 2. Select Input Queue (Round-Robin based on ID)
    // This distributes the locking load across multiple mutexes.
    int queue_idx = id % NUM_INPUT_QUEUES;

    // 3. Push to the specific queue
    request_queues_[queue_idx]->push({std::move(request), std::move(result_promise)});
    
    // Wake up the evaluator immediately
    signal_cv_.notify_one(); 
    
    return result_future;
}

void Evaluator::evaluation_loop() {
    struct InFlightBatch {
        std::future<std::vector<EvaluationResult>> result_future;
        std::vector<QueueItem> promises;
        int context_idx; // Track which buffer this batch is using
    };

    std::deque<InFlightBatch> pipeline;
    std::deque<int> available_contexts = {0, 1, 2}; // Managed pool of indices
    
    int current_poll_queue_idx = 0;

    while (!stop_requested_ || !pipeline.empty()) {
        bool activity_in_this_loop = false;

        // --- STAGE 1: LAUNCH (CPU -> GPU) ---
        // Launch if we have room in pipeline AND a free buffer context
        if (!stop_requested_ && !available_contexts.empty()) {
            std::vector<QueueItem> batch_items;
            int consecutive_empty = 0;
            
            while (batch_items.size() < static_cast<size_t>(max_batch_size_)) {
                auto item = request_queues_[current_poll_queue_idx]->try_pop();
                if (item) {
                    batch_items.push_back(std::move(*item));
                    consecutive_empty = 0;
                } else {
                    consecutive_empty++;
                }
                current_poll_queue_idx = (current_poll_queue_idx + 1) % NUM_INPUT_QUEUES;
                if (consecutive_empty >= NUM_INPUT_QUEUES) break;
            }

            if (!batch_items.empty()) {
                int ctx_idx = available_contexts.front();
                available_contexts.pop_front();

                std::vector<EvaluationRequest> requests;
                requests.reserve(batch_items.size());
                for (auto& it : batch_items) requests.push_back(std::move(it.first));

                // Dispatch to GPU using the specific context index
                auto future = network_->evaluate_batch_async(requests, ctx_idx);
                pipeline.push_back({std::move(future), std::move(batch_items), ctx_idx});
                activity_in_this_loop = true;
            }
        }

        // --- STAGE 2: COLLECT (GPU -> CPU) ---
        if (!pipeline.empty()) {
            auto& oldest = pipeline.front();
            
            // Logic for blocking:
            // 1. If pipeline is full or we are stopping, we MUST block (backpressure).
            // 2. Otherwise, we just peek (wait_for 0) to keep the loop moving.
            bool force_block = (available_contexts.empty() || stop_requested_);
            
            auto status = force_block ? 
                oldest.result_future.wait_for(std::chrono::hours(1)) : // Effectively blocking
                oldest.result_future.wait_for(std::chrono::microseconds(0));

            if (status == std::future_status::ready) {
                try {
                    std::vector<EvaluationResult> results = oldest.result_future.get();
                    for (size_t i = 0; i < results.size(); ++i) {
                        oldest.promises[i].second.set_value(std::move(results[i]));
                    }
                } catch (...) {
                    for (auto& p : oldest.promises) p.second.set_exception(std::current_exception());
                }
                
                // CRITICAL: Return the context index to the pool for reuse
                available_contexts.push_back(oldest.context_idx);
                pipeline.pop_front();
                activity_in_this_loop = true;
            }
        }

        // --- STAGE 3: IDLE ---
        if (!activity_in_this_loop && !stop_requested_) {
            std::unique_lock<std::mutex> lock(signal_mutex_);
            // Sleep until new requests arrive or a short timeout for GPU polling
            signal_cv_.wait_for(lock, std::chrono::milliseconds(1));
        }
    }
}

} // namespace chaturaji_cpp