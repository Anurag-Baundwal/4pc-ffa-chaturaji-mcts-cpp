#include "evaluator.h"
#include <iostream>
#include <chrono>   // For timeouts
#include <vector>
#include <stdexcept> // For runtime_error

namespace chaturaji_cpp {

Evaluator::Evaluator(ChaturajiNN network, torch::Device device, int max_batch_size) :
    network_(network),
    device_(device),
    max_batch_size_(max_batch_size),
    stop_requested_(false),
    next_request_id_(0)
{
    if (!network_) {
        throw std::runtime_error("Evaluator received an invalid network module.");
    }
    network_->to(device_); // Ensure model is on the correct device
    network_->eval();      // Ensure model is in eval mode
}

Evaluator::~Evaluator() {
    stop(); // Ensure thread is stopped and joined upon destruction
}

void Evaluator::start() {
    if (evaluator_thread_.joinable()) {
        std::cerr << "Warning: Evaluator thread already started." << std::endl;
        return;
    }
    stop_requested_ = false;
    evaluator_thread_ = std::thread(&Evaluator::evaluation_loop, this);
    std::cout << "Evaluator thread started." << std::endl;
}

void Evaluator::stop() {
    if (!evaluator_thread_.joinable()) {
        return; // Nothing to stop
    }
    stop_requested_ = true;
    // Notify the evaluator_cv_ in case the try_pop_for in ThreadSafeQueue
    // is also designed to be interruptible by an external CV (though not typical for a generic queue).
    // Primarily, stop_requested_ flag will be checked after try_pop_for returns.
    evaluator_cv_.notify_one();
    evaluator_thread_.join();
    std::cout << "Evaluator thread stopped." << std::endl;
}

std::future<EvaluationResult> Evaluator::submit_request(EvaluationRequest request) {
    // Assign a unique ID
    request.request_id = next_request_id_++;

    // Create a promise to hold the result
    std::promise<EvaluationResult> result_promise;
    std::future<EvaluationResult> result_future = result_promise.get_future();

    // Store the promise in the map (protected by mutex)
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        pending_results_map_[request.request_id] = std::move(result_promise);
    }

    // Push the request onto the queue
    request_queue_.push(std::move(request));
    // The request_queue_ itself should notify its condition variable, which
    // try_pop_for will be waiting on.
    // An explicit evaluator_cv_.notify_one() here might be redundant if try_pop_for
    // is only waiting on the queue's internal CV.
    // However, it doesn't hurt and might be useful if the queue's CV logic changes.
    // evaluator_cv_.notify_one(); // This might be redundant if try_pop_for is well-implemented

    return result_future;
}


void Evaluator::evaluation_loop() {
    std::vector<EvaluationRequest> batch_requests;
    batch_requests.reserve(max_batch_size_);

    while (!stop_requested_) {
        // --- Initialize Batch for Current Iteration ---
        batch_requests.clear();

        // --- Attempt to Get First Request with Timeout ---
        // This allows the loop to wake up periodically to check stop_requested_
        // and to form batches more effectively if requests arrive in bursts.
        std::optional<EvaluationRequest> first_request_opt = request_queue_.try_pop_for(std::chrono::milliseconds(1));

        // --- Check for Stop Signal or Timeout ---
        if (stop_requested_) {
            break; // Exit loop if stop is requested
        }
        if (!first_request_opt) {
            // Timeout occurred, no request available in the given time.
            // Loop again to check stop_requested_ or wait for new requests.
            continue;
        }

        // --- Start Building Batch ---
        batch_requests.push_back(std::move(*first_request_opt));

        // --- Fill Batch (Greedy Non-Blocking) ---
        // Greedily fill the rest of the batch up to max_batch_size or until queue is empty.
        while (batch_requests.size() < static_cast<size_t>(max_batch_size_)) {
            std::optional<EvaluationRequest> next_req = request_queue_.try_pop(); // Non-blocking pop
            if (next_req) {
                batch_requests.push_back(std::move(*next_req));
            } else {
                break; // No more requests immediately available in the queue
            }
        }
        // At this point, batch_requests contains at least one request.

        // --- Perform Batched Neural Network Inference ---
        std::vector<EvaluationResult> batch_results;
        try {
            // The network_->evaluate_batch method is responsible for handling
            // tensor movement to the correct device (this->device_).
            batch_results = network_->evaluate_batch(batch_requests, device_);
        } catch (const std::exception& e) {
            std::cerr << "!!! EXCEPTION during NN batch evaluation: " << e.what() << std::endl;
            // Error handling: Fulfill promises with an error, or skip.
            // For now, skipping fulfillment for errored batches.
            // This means workers waiting on these futures might time out or block indefinitely
            // if not handled by the worker (e.g., future.wait_for with timeout).
            // TODO: Implement robust error propagation to workers.
            std::cerr << "Skipping result fulfillment for errored batch. Request IDs involved: ";
            for(const auto& req : batch_requests) { std::cerr << req.request_id << " "; }
            std::cerr << std::endl;
            continue; // Continue to the next iteration to try processing more requests
        }
        // --- End Inference ---

        // --- Fulfill Promises with Results ---
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            for (auto& result : batch_results) { // Iterate through results from NN
                auto it = pending_results_map_.find(result.request_id);
                if (it != pending_results_map_.end()) {
                    try {
                        it->second.set_value(std::move(result)); // Fulfill the promise for the worker
                    } catch (const std::exception& e) {
                        // General exception while setting value (less common).
                        // If set_value throws (other than future_error for already_satisfied),
                        // the promise is still unfulfilled or in an error state.
                        // It's usually better to try and set an exception on the promise here.
                        std::cerr << "!!! EXCEPTION setting promise value for RequestId " << result.request_id << ": " << e.what() << std::endl;
                        try {
                            it->second.set_exception(std::current_exception());
                        } catch (const std::future_error& fe) {
                             std::cerr << "Warning: std::future_error while trying to set_exception for RequestId "
                                      << result.request_id << ": " << fe.what() << std::endl;
                        }
                    }
                    // Whether successful or an error occurred trying to set the value/exception,
                    // we should remove the promise from the map as we've attempted to handle it.
                    pending_results_map_.erase(it);
                } else {
                    // This indicates a result was generated for a request ID that's no longer
                    // in the pending_results_map_. This could be due to a worker timing out
                    // and removing its own promise, or a logic error.
                    std::cerr << "Warning: No pending promise found for received ResultId: " << result.request_id << std::endl;
                }
            }
        } // Mutex unlock
        // --- End Fulfill Promises ---

    } // End while (!stop_requested_)

    std::cout << "Evaluation loop finished." << std::endl;

    // --- Cleanup on Stop (Optional but Recommended) ---
    // When stopping, there might be pending requests or promises.
    // It's good practice to signal any remaining promises with an error
    // to unblock any waiting worker threads.
    if (stop_requested_) {
        std::lock_guard<std::mutex> lock(map_mutex_);
        for (auto& pair : pending_results_map_) {
            try {
                // Signal with an error indicating the evaluator is shutting down
                pair.second.set_exception(std::make_exception_ptr(std::runtime_error("Evaluator shutting down; request cancelled.")));
            } catch (const std::future_error& e) {
                // Promise might already be satisfied or future abandoned.
                if (e.code() != std::future_errc::promise_already_satisfied && e.code() != std::future_errc::no_state) {
                    std::cerr << "Warning: std::future_error during evaluator cleanup for RequestId "
                              << pair.first << ": " << e.what() << std::endl;
                }
            }
        }
        pending_results_map_.clear();

        // Drain any remaining requests from the queue (they won't be processed)
        while(request_queue_.try_pop()) {}
    }
}

} // namespace chaturaji_cpp