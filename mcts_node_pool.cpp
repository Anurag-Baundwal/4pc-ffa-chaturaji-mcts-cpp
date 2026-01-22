// mcts_node_pool.cpp
#include "mcts_node_pool.h"
#include <stdexcept>
#include <algorithm> // For std::max
#include <vector>

namespace chaturaji_cpp {

// Thread-local cache for nodes
static thread_local std::vector<void*> t_free_cache;
static const size_t CACHE_BATCH_SIZE = 1000;
static const size_t MAX_CACHE_SIZE = 2000;

// Constructor: Allocates the initial chunk of memory
MCTSNodePool::MCTSNodePool(size_t node_size, size_t initial_capacity)
    : node_size_(node_size) // Initialize const member directly from argument
{
    // Ensure initial_capacity is at least 1, if 0, use GROW_CHUNK_SIZE for the first allocation
    size_t actual_initial_capacity = initial_capacity > 0 ? initial_capacity : GROW_CHUNK_SIZE;

    // Pre-reserve space in the chunks_ vector to avoid reallocations of the vector itself
    chunks_.reserve(actual_initial_capacity / GROW_CHUNK_SIZE + 5);

    grow(); // Allocate the first chunk of nodes

    std::cout << "MCTSNodePool initialized with first chunk capacity: " << GROW_CHUNK_SIZE << " nodes ("
              << (node_size_ * GROW_CHUNK_SIZE / (1024.0 * 1024.0)) << " MB) at address "
              << static_cast<void*>(chunks_[0].get()) << std::endl;
}

// Destructor: Logs statistics. Memory managed by unique_ptr in chunks_
MCTSNodePool::~MCTSNodePool() {
    std::cout << "MCTSNodePool destroyed. Total allocated (from pool): " << allocated_count_
              << ", Total freed (to pool): " << freed_count_ << ", Peak usage: " << peak_allocated_count_ << std::endl;
    // std::unique_ptr<char[]> in chunks_ handles memory deallocation automatically when chunks_ is destroyed.
}

// Allocates a new chunk of memory for nodes
void MCTSNodePool::grow() {
    std::cout << "MCTSNodePool: Growing by allocating " << GROW_CHUNK_SIZE << " more nodes." << std::endl;
    size_t bytes_to_allocate = node_size_ * GROW_CHUNK_SIZE;
    
    std::unique_ptr<char[]> new_chunk = std::make_unique<char[]>(bytes_to_allocate);
    
    for (size_t i = 0; i < GROW_CHUNK_SIZE; ++i) {
        free_list_.push_back(reinterpret_cast<MCTSNode*>(new_chunk.get() + i * node_size_));
    }
    
    chunks_.push_back(std::move(new_chunk)); 
}

// Allocates a node from the pool
void* MCTSNodePool::allocate() {
    if (t_free_cache.empty()) {
        allocate_batch(t_free_cache, CACHE_BATCH_SIZE);
    }
    
    if (t_free_cache.empty()) {
        // Fallback or OOM handling (shouldn't happen with grow())
        throw std::runtime_error("MCTSNodePool: Failed to allocate nodes.");
    }
    
    void* node = t_free_cache.back();
    t_free_cache.pop_back();
    return node;
}

// Deallocates a node, returning it to the pool's free list
void MCTSNodePool::deallocate(void* ptr) {
    if (ptr == nullptr) return;

    t_free_cache.push_back(ptr);

    if (t_free_cache.size() >= MAX_CACHE_SIZE) {
        // Move half of the cache back to the global pool
        size_t return_count = t_free_cache.size() / 2;
        std::vector<void*> batch_to_return;
        batch_to_return.reserve(return_count);

        // Move items from end of cache to batch
        for(size_t i=0; i<return_count; ++i) {
             batch_to_return.push_back(t_free_cache.back());
             t_free_cache.pop_back();
        }

        deallocate_batch(batch_to_return);
    }
}

// Batch functions (Global Lock)
void MCTSNodePool::allocate_batch(std::vector<void*>& out_nodes, size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t available = free_list_.size();
    while (available < count) {
        grow();
        available = free_list_.size();
    }
    
    // Transfer 'count' nodes
    size_t start_idx = free_list_.size() - count;
    for (size_t i = 0; i < count; ++i) {
        out_nodes.push_back(free_list_[start_idx + i]);
    }
    free_list_.resize(start_idx); // Efficiently remove from back
    
    allocated_count_ += count;
    peak_allocated_count_ = std::max(peak_allocated_count_, allocated_count_ - freed_count_);
}

void MCTSNodePool::deallocate_batch(const std::vector<void*>& in_nodes) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (void* ptr : in_nodes) {
        free_list_.push_back(reinterpret_cast<MCTSNode*>(ptr));
    }
    freed_count_ += in_nodes.size();
}

} // namespace chaturaji_cpp