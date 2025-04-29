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
    evaluator_cv_.notify_one(); // Wake up the evaluator thread if waiting
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

    // Push the request onto the queue and notify the evaluator
    request_queue_.push(std::move(request));
    evaluator_cv_.notify_one(); // Notify evaluator a request is ready

    return result_future;
}


void Evaluator::evaluation_loop() {
    std::vector<EvaluationRequest> batch_requests;
    batch_requests.reserve(max_batch_size_);

    while (!stop_requested_) {
        batch_requests.clear();

        // --- Collect Batch ---
        // Wait for the first request with a timeout to allow periodic stop checks
        std::optional<EvaluationRequest> first_request_opt = request_queue_.try_pop();

        if (!first_request_opt) {
             // If no request immediately available, wait briefly
             // This uses a simple sleep; a condition variable wait with timeout is better
             // Let's stick with simple pop/check for now
             std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Prevent busy-waiting
             // Check stop flag again after sleep
             if (stop_requested_) continue;
             // Try popping again after sleep
             first_request_opt = request_queue_.try_pop();
             if(!first_request_opt) continue; // Still no request, loop again
        }

        // Got the first request
        batch_requests.push_back(std::move(*first_request_opt));

        // Try to fill the rest of the batch without blocking excessively
        // Collect available requests up to max_batch_size
        while (batch_requests.size() < static_cast<size_t>(max_batch_size_)) {
            std::optional<EvaluationRequest> next_req = request_queue_.try_pop();
            if (next_req) {
                batch_requests.push_back(std::move(*next_req));
            } else {
                break; // No more requests immediately available
            }
        }
         // --- End Collect Batch ---


         if (batch_requests.empty()) {
             continue; // Should not happen due to logic above, but safety check
         }

        // std::cout << "Evaluator processing batch of size: " << batch_requests.size() << std::endl; // Debug

        // --- Perform Inference ---
        std::vector<EvaluationResult> batch_results;
        try {
            // evaluate_batch expects tensors on CPU or target device?
            // Let's assume evaluate_batch handles moving tensors if needed.
            batch_results = network_->evaluate_batch(batch_requests, device_);
        } catch (const std::exception& e) {
            std::cerr << "!!! EXCEPTION during NN batch evaluation: " << e.what() << std::endl;
            // How to handle? We need to fulfill promises with an error or default value.
            // For now, fulfill with a dummy error result or skip fulfillment.
            // Fulfilling with error requires changing promise type or result structure.
            // Let's skip fulfillment for errored batches for simplicity in Phase 2.
             std::cerr << "Skipping result fulfillment for errored batch." << std::endl;
            continue; // Continue to next batch attempt
        }
        // --- End Inference ---


        // --- Fulfill Promises ---
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            for (auto& result : batch_results) {
                auto it = pending_results_map_.find(result.request_id);
                if (it != pending_results_map_.end()) {
                    try {
                        it->second.set_value(std::move(result)); // Fulfill the promise
                         pending_results_map_.erase(it); // Remove fulfilled promise from map
                    } catch (const std::future_error& e) {
                         std::cerr << "Warning: std::future_error setting promise value for RequestId "
                                   << result.request_id << ". Maybe worker timed out or was cancelled? Code: "
                                   << e.code() << ", What: " << e.what() << std::endl;
                         // Still try to remove from map if setting failed
                         pending_results_map_.erase(it);
                    } catch (const std::exception& e) {
                        std::cerr << "!!! EXCEPTION setting promise value for RequestId " << result.request_id << ": " << e.what() << std::endl;
                         pending_results_map_.erase(it); // Clean up map entry
                    }
                } else {
                    // This indicates a result was generated for a request ID no longer in the map
                    // (potentially already fulfilled or timed out?)
                    std::cerr << "Warning: No pending promise found for received ResultId: " << result.request_id << std::endl;
                }
            }
        } // Mutex unlock
        // --- End Fulfill Promises ---

    } // End while loop

    std::cout << "Evaluation loop finished." << std::endl;
    // Optional: Handle any remaining requests in the queue upon stopping?
    // Optional: Signal any remaining promises with an error state?
}

} // namespace chaturaji_cpp
