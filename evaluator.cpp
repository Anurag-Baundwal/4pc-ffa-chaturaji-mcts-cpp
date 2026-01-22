#include "evaluator.h"
#include <iostream>
#include <chrono>   
#include <vector>
#include <stdexcept> 
#include <thread>

namespace chaturaji_cpp {

Evaluator::Evaluator(Model* network, int max_batch_size) :
    network_(network),
    max_batch_size_(max_batch_size),
    stop_requested_(false),
    next_request_id_(0),
    total_pending_count_(0)
{
    if (!network_) {
        throw std::runtime_error("Evaluator received a null network pointer.");
    }
    for (int i = 0; i < num_shards_; ++i) {
        request_queues_.push_back(std::make_unique<ThreadSafeQueue<std::pair<EvaluationRequest, std::promise<EvaluationResult>>>>());
    }
}

Evaluator::~Evaluator() {
    stop(); 
}

void Evaluator::start() {
    if (evaluator_thread_.joinable()) return;
    stop_requested_ = false;
    evaluator_thread_ = std::thread(&Evaluator::evaluation_loop, this);
    std::cout << "Evaluator thread started (Lock-Free Submission Mode)." << std::endl;
}

void Evaluator::stop() {
    if (!evaluator_thread_.joinable()) return;
    stop_requested_ = true;
    evaluator_thread_.join();
    std::cout << "Evaluator thread stopped." << std::endl;
}

std::future<EvaluationResult> Evaluator::submit_request(EvaluationRequest request) {
    uint64_t req_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    request.request_id = req_id;

    std::promise<EvaluationResult> result_promise;
    std::future<EvaluationResult> result_future = result_promise.get_future();

    int shard_idx = req_id % num_shards_;
    request_queues_[shard_idx]->push({std::move(request), std::move(result_promise)});

    // Increment global count to notify evaluator there is work
    total_pending_count_.fetch_add(1, std::memory_order_release);

    return result_future;
}

std::vector<std::future<EvaluationResult>> Evaluator::submit_batch(std::vector<EvaluationRequest>&& requests) {
    if (requests.empty()) return {};

    size_t count = requests.size();
    std::vector<std::future<EvaluationResult>> futures;
    futures.reserve(count);
    std::vector<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> queue_items;
    queue_items.reserve(count);

    uint64_t batch_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    int shard_idx = batch_id % num_shards_;

    for (size_t i = 0; i < count; ++i) {
        requests[i].request_id = (batch_id * 10000) + i;
        std::promise<EvaluationResult> p;
        futures.push_back(p.get_future());
        queue_items.emplace_back(std::move(requests[i]), std::move(p));
    }

    request_queues_[shard_idx]->push_batch(queue_items);
    total_pending_count_.fetch_add(count, std::memory_order_release);

    return futures;
}

void Evaluator::evaluation_loop() {
    std::vector<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> batch_with_promises;
    batch_with_promises.reserve(max_batch_size_);

    int current_poll_shard = 0;

    while (!stop_requested_) {
        // Fast-path: check atomic counter before even touching shards
        if (total_pending_count_.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
            continue;
        }

        batch_with_promises.clear();
        size_t total_popped = 0;

        // Drain shards greedily
        for (int i = 0; i < num_shards_; ++i) {
            int idx = (current_poll_shard + i) % num_shards_;
            size_t needed = max_batch_size_ - batch_with_promises.size();
            if (needed == 0) break;

            size_t popped = request_queues_[idx]->pop_batch(batch_with_promises, needed);
            total_popped += popped;
        }
        current_poll_shard = (current_poll_shard + 1) % num_shards_;

        if (total_popped == 0) continue;

        // Update atomic count
        total_pending_count_.fetch_sub(total_popped, std::memory_order_release);

        // Process batch
        std::vector<EvaluationRequest> requests_for_nn;
        requests_for_nn.reserve(batch_with_promises.size());
        for (const auto& pair : batch_with_promises) requests_for_nn.push_back(pair.first);

        try {
            std::vector<EvaluationResult> results = network_->evaluate_batch(requests_for_nn);
            for (size_t i = 0; i < results.size(); ++i) {
                batch_with_promises[i].second.set_value(std::move(results[i]));
            }
        } catch (const std::exception& e) {
            std::cerr << "!!! EXCEPTION: " << e.what() << std::endl;
            for (auto& pair : batch_with_promises) pair.second.set_exception(std::current_exception());
        }
    } 

    // Cleanup on shutdown
    if (stop_requested_) {
        for (auto& queue : request_queues_) {
            std::optional<std::pair<EvaluationRequest, std::promise<EvaluationResult>>> item;
            while ((item = queue->try_pop())) item->second.set_exception(std::make_exception_ptr(std::runtime_error("Shutdown")));
        }
    }
}

} // namespace chaturaji_cpp