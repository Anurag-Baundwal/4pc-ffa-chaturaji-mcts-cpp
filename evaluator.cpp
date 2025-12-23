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
    // ONNX Runtime models are ready to run upon loading; no need for .to(device) or .eval()
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
    std::cout << "Evaluator thread started." << std::endl;
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
    request.request_id = next_request_id_++;
    std::promise<EvaluationResult> result_promise;
    std::future<EvaluationResult> result_future = result_promise.get_future();

    request_queue_.push({std::move(request), std::move(result_promise)});
    return result_future;
}

void Evaluator::evaluation_loop() {
    std::vector<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> batch_with_promises;
    batch_with_promises.reserve(max_batch_size_); 

    while (!stop_requested_) {
        batch_with_promises.clear(); 

        // 1. Get first request with timeout
        std::optional<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> first_pair_opt =
            request_queue_.try_pop_for(std::chrono::milliseconds(1));

        if (stop_requested_) { break; } 
        if (!first_pair_opt) { continue; } 

        batch_with_promises.push_back(std::move(*first_pair_opt));

        // 2. Greedily fill the rest of the batch
        while (batch_with_promises.size() < static_cast<size_t>(max_batch_size_)) {
            std::optional<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> next_pair = request_queue_.try_pop(); 
            if (next_pair) {
                batch_with_promises.push_back(std::move(*next_pair));
            } else {
                break; 
            }
        }

        if (batch_with_promises.empty()) {
            continue;
        }

        // 3. Prepare requests for ONNX Model
        std::vector<EvaluationRequest> requests_for_nn;
        requests_for_nn.reserve(batch_with_promises.size());
        for (const auto& pair : batch_with_promises) {
            requests_for_nn.push_back(pair.first); 
        }

        // 4. Perform Batched Inference
        std::vector<EvaluationResult> batch_results;
        try {
            // Call the ONNX model (Synchronous)
            batch_results = network_->evaluate_batch(requests_for_nn);
        } catch (const std::exception& e) {
            std::cerr << "!!! EXCEPTION during ONNX batch evaluation: " << e.what() << std::endl;
            for (auto& pair : batch_with_promises) {
                try {
                    pair.second.set_exception(std::current_exception());
                } catch (const std::future_error& fe) { /* ignore */ }
            }
            continue; 
        }

        if (batch_results.size() != batch_with_promises.size()) {
            std::cerr << "Error: Model output batch size mismatch!" << std::endl;
            // Handle error logic if necessary...
            continue;
        }

        // 5. Fulfill Promises with Results
        for (size_t i = 0; i < batch_results.size(); ++i) {
            try {
                batch_with_promises[i].second.set_value(std::move(batch_results[i]));
            } catch (const std::future_error& e) {
                 if (e.code() != std::future_errc::promise_already_satisfied && e.code() != std::future_errc::no_state) {
                    std::cerr << "Warning: std::future_error setting value: " << e.what() << std::endl;
                }
            }
        }
    } 

    // Cleanup
    if (stop_requested_) {
        std::optional<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> remaining_pair_opt;
        while((remaining_pair_opt = request_queue_.try_pop())) {
            try {
                remaining_pair_opt->second.set_exception(std::make_exception_ptr(std::runtime_error("Evaluator shutting down.")));
            } catch (...) {}
        }
    }
}

} // namespace chaturaji_cpp