// mcts_node_pool.h
#pragma once
#include <vector>
#include <mutex>
#include <cstddef> // For size_t
#include <iostream> // For logging
#include <memory> // For std::unique_ptr (to manage memory chunks)

#include "mcts_node_fwd.h" 

namespace chaturaji_cpp {

class MCTSNodePool {
public:
    // Initial capacity: 1.5M nodes
    // Will grow if needed
    MCTSNodePool(size_t node_size, size_t initial_capacity = 1500000);
    ~MCTSNodePool();

    // Disable copy/move operations as it manages resources
    MCTSNodePool(const MCTSNodePool&) = delete;
    MCTSNodePool& operator=(const MCTSNodePool&) = delete;
    MCTSNodePool(MCTSNodePool&&) = delete;
    MCTSNodePool& operator=(MCTSNodePool&&) = delete;

    /**
     * @brief Allocates a raw memory block suitable for an MCTSNode from the pool.
     * Thread-safe.
     * @return A void pointer to the allocated memory.
     */
    void* allocate();

    /**
     * @brief Deallocates a raw memory block, returning it to the pool for reuse.
     * Thread-safe.
     * @param ptr A void pointer to the memory block to deallocate.
     */
    void deallocate(void* ptr);

    // --- Helper methods for Thread-Local Caching ---
    // Moves a batch of nodes from the global free list to the provided vector.
    void batch_fill_from_global(std::vector<MCTSNode*>& target_cache, size_t count);

    // Moves all nodes from the provided vector back to the global free list.
    void batch_return_to_global(std::vector<MCTSNode*>& source_cache);

private:
    // std::vector of unique_ptrs to char arrays to manage dynamically allocated memory chunks.
    // This allows the pool to grow by adding new chunks without invalidating pointers
    // to nodes in previously allocated chunks.
    std::vector<std::unique_ptr<char[]>> chunks_; 
    std::vector<MCTSNode*> free_list_; // Pointers to available MCTSNode-sized blocks within chunks
    std::mutex mutex_; // Protects access to free_list_ and chunk_ management

    size_t node_size_; // The size of an MCTSNode object
    const size_t GROW_CHUNK_SIZE = 100000; // Number of nodes to allocate in each new chunk

    // Statistics for tuning and debugging
    size_t allocated_count_ = 0; // Total number of nodes allocated from the pool (ever)
    size_t freed_count_ = 0;     // Total number of nodes returned to the pool (ever)
    size_t peak_allocated_count_ = 0; // Maximum number of nodes simultaneously in use

    /**
     * @brief Allocates a new chunk of memory for nodes and adds its blocks to the free list.
     * Called automatically by allocate() if the free list is empty.
     */
    void grow(); 
};

} // namespace chaturaji_cpp