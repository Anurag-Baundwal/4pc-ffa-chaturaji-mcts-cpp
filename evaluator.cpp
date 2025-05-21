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
    // The evaluation_loop will check stop_requested_ after its try_pop_for
    // or on its next iteration. No explicit notify_one() on a separate CV is needed here.
    evaluator_thread_.join();
    std::cout << "Evaluator thread stopped." << std::endl;
}

std::future<EvaluationResult> Evaluator::submit_request(EvaluationRequest request) {
    // Assign a unique ID
    request.request_id = next_request_id_++;

    // Create a promise to hold the result
    std::promise<EvaluationResult> result_promise;
    std::future<EvaluationResult> result_future = result_promise.get_future();

    // Push the request and promise together into the queue.
    // No mutex needed here as ThreadSafeQueue handles its own internal locking.
    request_queue_.push({std::move(request), std::move(result_promise)});

    return result_future;
}


void Evaluator::evaluation_loop() {
    std::vector<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> batch_with_promises;
    batch_with_promises.reserve(max_batch_size_); // Reserve capacity once

    while (!stop_requested_) {
        batch_with_promises.clear(); // Clear batch for this iteration

        // 1. Get first request with timeout
        // Use the updated queue type for try_pop_for
        std::optional<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> first_pair_opt =
            request_queue_.try_pop_for(std::chrono::milliseconds(1));

        if (stop_requested_) { break; } // Check stop after potential wait
        if (!first_pair_opt) { continue; } // Timeout, no request, try again

        batch_with_promises.push_back(std::move(*first_pair_opt));

        // 2. Greedily fill the rest of the batch
        while (batch_with_promises.size() < static_cast<size_t>(max_batch_size_)) {
            std::optional<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> next_pair = request_queue_.try_pop(); // Non-blocking pop
            if (next_pair) {
                batch_with_promises.push_back(std::move(*next_pair));
            } else {
                break; // No more requests immediately available
            }
        }

        // If no requests were collected (e.g., initial pop failed after previous batch was empty)
        if (batch_with_promises.empty()) {
            continue;
        }

        // 3. Prepare requests for NN (collect just the EvaluationRequest objects)
        std::vector<EvaluationRequest> requests_for_nn;
        requests_for_nn.reserve(batch_with_promises.size());
        for (const auto& pair : batch_with_promises) {
            requests_for_nn.push_back(pair.first); // Copy the request, promise remains in batch_with_promises
        }

        // 4. Perform Batched Neural Network Inference
        std::vector<EvaluationResult> batch_results;
        try {
            // This call will be optimized as per the next section (CPU stack before GPU transfer)
            batch_results = network_->evaluate_batch(requests_for_nn, device_);
        } catch (const std::exception& e) {
            std::cerr << "!!! EXCEPTION during NN batch evaluation: " << e.what() << std::endl;
            // On error, set exception for all promises in this batch to unblock workers
            for (auto& pair : batch_with_promises) {
                try {
                    pair.second.set_exception(std::current_exception());
                } catch (const std::future_error& fe) {
                    // This can happen if a worker timed out or cancelled its future.
                    std::cerr << "Warning: Error setting exception for promise (future error): " << fe.what() << std::endl;
                }
            }
            continue; // Continue to next batch
        }

        // Ensure results match requests (important if evaluate_batch reorders or drops requests)
        // Assuming evaluate_batch maintains order:
        if (batch_results.size() != batch_with_promises.size()) {
            std::cerr << "Error: NN output batch size mismatch with input batch size! Expected "
                      << batch_with_promises.size() << ", got " << batch_results.size() << std::endl;
            for (auto& pair : batch_with_promises) { // Unblock remaining promises with an error
                try {
                    pair.second.set_exception(std::make_exception_ptr(std::runtime_error("NN output size mismatch.")));
                } catch (const std::future_error& fe) { /* ignore */ }
            }
            continue;
        }

        // 5. Fulfill Promises with Results (NO mutex needed here anymore!)
        for (size_t i = 0; i < batch_results.size(); ++i) {
            try {
                batch_with_promises[i].second.set_value(std::move(batch_results[i]));
            } catch (const std::future_error& e) {
                // This indicates the future was already detached or set (e.g., worker timed out).
                // It's usually safe to ignore promise_already_satisfied or no_state errors.
                if (e.code() != std::future_errc::promise_already_satisfied && e.code() != std::future_errc::no_state) {
                    std::cerr << "Warning: std::future_error setting value for RequestId "
                              << batch_results[i].request_id << ": " << e.what() << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "!!! EXCEPTION setting promise value for RequestId " << batch_results[i].request_id << ": " << e.what() << std::endl;
                try {
                    batch_with_promises[i].second.set_exception(std::current_exception());
                } catch (const std::future_error& fe) { /* ignore */ }
            }
        }
    } // End while (!stop_requested_)

    // Cleanup on Stop: Unblock any remaining workers
    if (stop_requested_) {
        std::optional<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> remaining_pair_opt;
        while((remaining_pair_opt = request_queue_.try_pop())) {
            std::pair<EvaluationRequest, std::promise<EvaluationResult>> remaining_pair = std::move(*remaining_pair_opt);
            try {
                remaining_pair.second.set_exception(std::make_exception_ptr(std::runtime_error("Evaluator shutting down; request cancelled.")));
            } catch (const std::future_error& e) { /* ignore already set or no state errors */ }
        }
    }
}

} // namespace chaturaji_cpp