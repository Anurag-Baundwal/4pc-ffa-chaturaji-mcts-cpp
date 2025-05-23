// mcts_node_pool.cpp
#include "mcts_node_pool.h"
#include <stdexcept>
#include <algorithm> // For std::max

namespace chaturaji_cpp {

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
    std::lock_guard<std::mutex> lock(mutex_); // Protect shared resources
    
    // If no nodes are available in the free list, grow the pool
    if (free_list_.empty()) {
        grow(); 
    }
    
    // Take a node from the back of the free list
    MCTSNode* node = free_list_.back();
    free_list_.pop_back();
    
    // Update statistics
    allocated_count_++;
    peak_allocated_count_ = std::max(peak_allocated_count_, allocated_count_ - freed_count_);
    
    return node;
}

// Deallocates a node, returning it to the pool's free list
void MCTSNodePool::deallocate(void* ptr) {
    // Standard behavior for delete: ignore nullptr
    if (ptr == nullptr) return; 
    
    std::lock_guard<std::mutex> lock(mutex_); // Protect shared resources
    
    // In a production-grade allocator, you'd add checks to ensure `ptr`
    // actually belongs to one of the pool's managed memory chunks to prevent
    // corruption if a foreign pointer is passed. For this context, we assume validity.
    
    // Add the pointer back to the free list
    free_list_.push_back(reinterpret_cast<MCTSNode*>(ptr));
    
    // Update statistics
    freed_count_++;
}

} // namespace chaturaji_cpp