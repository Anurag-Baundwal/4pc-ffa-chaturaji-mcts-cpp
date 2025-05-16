#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

// A basic thread-safe queue template
template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;

    // Rule of 5/6: Disable copy/move operations for simplicity
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue(ThreadSafeQueue&&) = delete;
    ThreadSafeQueue& operator=(ThreadSafeQueue&&) = delete;

    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_var_.notify_one();
    }

    // Blocking pop
    T wait_and_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this] { return !queue_.empty(); });
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    // Non-blocking pop
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    std::optional<T> try_pop_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (cond_var_.wait_for(lock, timeout, [this]{ return !queue_.empty(); })) {
            T value = std::move(queue_.front());
            queue_.pop();
            return value;
        }
        return std::nullopt; // Timeout occurred or spurious wakeup with empty queue
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_; // mutable allows locking in const methods like empty/size
    std::queue<T> queue_;
    std::condition_variable cond_var_;
};