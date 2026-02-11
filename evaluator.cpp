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
    
    return result_future;
}

void Evaluator::evaluation_loop() {
    std::vector<QueueItem> batch_with_promises;
    batch_with_promises.reserve(max_batch_size_); 

    // Index to ensure we check all queues fairly (Round-Robin polling)
    int current_poll_queue_idx = 0;

    while (!stop_requested_) {
        batch_with_promises.clear(); 

        // --- 1. Fetch First Item (Non-Blocking Attempt) ---
        // We iterate through all queues. If all are empty, we sleep briefly.
        
        bool found_any = false;
        
        // Try to find at least one item in any queue
        for (int i = 0; i < NUM_INPUT_QUEUES; ++i) {
            int idx = (current_poll_queue_idx + i) % NUM_INPUT_QUEUES;
            std::optional<QueueItem> item = request_queues_[idx]->try_pop();
            
            if (item) {
                batch_with_promises.push_back(std::move(*item));
                found_any = true;
                // Update start index for next time to ensure fairness
                current_poll_queue_idx = (idx + 1) % NUM_INPUT_QUEUES;
                break;
            }
        }

        if (!found_any) {
            if (stop_requested_) break;
            // Sleep briefly to avoid busy waiting if all queues are empty
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // --- 2. Greedily fill the rest of the batch ---
        // Continue checking queues until batch is full or queues are drained
        int empty_queues_count = 0;
        
        while (batch_with_promises.size() < static_cast<size_t>(max_batch_size_)) {
            // Check the current queue
            std::optional<QueueItem> next_pair = request_queues_[current_poll_queue_idx]->try_pop();
            
            if (next_pair) {
                batch_with_promises.push_back(std::move(*next_pair));
                empty_queues_count = 0; // Reset consecutive empty count since we found something
            } else {
                empty_queues_count++;
            }

            // Move to next queue
            current_poll_queue_idx = (current_poll_queue_idx + 1) % NUM_INPUT_QUEUES;

            // If we've checked all queues and found nothing consecutively, stop filling
            if (empty_queues_count >= NUM_INPUT_QUEUES) {
                break;
            }
        }

        if (batch_with_promises.empty()) {
            continue;
        }

        // --- 3. Prepare requests for ONNX Model ---
        std::vector<EvaluationRequest> requests_for_nn;
        requests_for_nn.reserve(batch_with_promises.size());
        for (auto& pair : batch_with_promises) {
            requests_for_nn.push_back(std::move(pair.first));
        }

        // --- 4. Perform Batched Inference ---
        std::vector<EvaluationResult> batch_results;
        try {
            batch_results = network_->evaluate_batch(requests_for_nn);
        } catch (const std::exception& e) {
            std::cerr << "!!! EXCEPTION during ONNX batch evaluation: " << e.what() << std::endl;
            for (auto& pair : batch_with_promises) {
                try {
                    pair.second.set_exception(std::current_exception());
                } catch (...) { }
            }
            for (auto& req : requests_for_nn) {
                TensorPool::release_planes(std::move(req.input_planes));
                TensorPool::release_scalars(std::move(req.input_scalars));
            }
            continue; 
        }

        if (batch_results.size() != batch_with_promises.size()) {
            std::cerr << "Error: Model output batch size mismatch!" << std::endl;
            continue;
        }

        // --- 5. Fulfill Promises (Return Results) ---
        for (size_t i = 0; i < batch_results.size(); ++i) {
            try {
                batch_with_promises[i].second.set_value(std::move(batch_results[i]));
            } catch (const std::future_error& e) {
                 if (e.code() != std::future_errc::promise_already_satisfied && e.code() != std::future_errc::no_state) {
                    std::cerr << "Warning: std::future_error setting value: " << e.what() << std::endl;
                }
            }
        }

        // --- 6. Recycle Inputs back to pool ---
        // After fulfillng promises, the model is done with the input planes/scalars.
        for (auto& req : requests_for_nn) {
            TensorPool::release_planes(std::move(req.input_planes));
            TensorPool::release_scalars(std::move(req.input_scalars));
        }
    } 

    // Cleanup: Drain all queues on stop
    if (stop_requested_) {
        for (auto& queue_ptr : request_queues_) {
            std::optional<QueueItem> remaining_pair_opt;
            while((remaining_pair_opt = queue_ptr->try_pop())) {
                try {
                    remaining_pair_opt->second.set_exception(std::make_exception_ptr(std::runtime_error("Evaluator shutting down.")));
                } catch (...) {}
            }
        }
    }
}

} // namespace chaturaji_cpp