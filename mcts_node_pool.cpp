#include "mcts_node_pool.h"
#include <stdexcept>
#include <algorithm> 

namespace chaturaji_cpp {

// Configuration for caching
static constexpr size_t LOCAL_CACHE_BATCH_SIZE = 256; 
static constexpr size_t LOCAL_CACHE_MAX_SIZE = 512;   

// --- Thread Local Cache Helper ---
struct ThreadLocalNodeCache {
    std::vector<MCTSNode*> cache;
    MCTSNodePool* pool_ref = nullptr;

    ThreadLocalNodeCache() {
        cache.reserve(LOCAL_CACHE_MAX_SIZE);
    }

    ~ThreadLocalNodeCache() {
        // Safety check: Only return if we have a pool reference and items to return
        if (pool_ref && !cache.empty()) {
            pool_ref->batch_return_to_global(cache);
        }
    }
};

// INTERNAL HELPER: Ensures allocate and deallocate share the SAME thread-local instance
static ThreadLocalNodeCache& get_local_cache() {
    static thread_local ThreadLocalNodeCache t_cache;
    return t_cache;
}

// =========================================================

MCTSNodePool::MCTSNodePool(size_t node_size, size_t initial_capacity)
    : node_size_(node_size)
{
    size_t actual_initial_capacity = initial_capacity > 0 ? initial_capacity : GROW_CHUNK_SIZE;
    chunks_.reserve(actual_initial_capacity / GROW_CHUNK_SIZE + 5);
    grow(); 

    std::cout << "MCTSNodePool initialized. Chunk size: " << GROW_CHUNK_SIZE 
              << ". Thread-local batch size: " << LOCAL_CACHE_BATCH_SIZE << std::endl;
}

MCTSNodePool::~MCTSNodePool() {
    std::cout << "MCTSNodePool destroyed. Global transfers - Alloc: " << allocated_count_
              << ", Free: " << freed_count_ 
              << ", Peak Global Usage: " << peak_allocated_count_ << std::endl;
}

void MCTSNodePool::grow() {
    // Called from locked context
    size_t bytes_to_allocate = node_size_ * GROW_CHUNK_SIZE;
    std::unique_ptr<char[]> new_chunk = std::make_unique<char[]>(bytes_to_allocate);
    
    for (size_t i = 0; i < GROW_CHUNK_SIZE; ++i) {
        free_list_.push_back(reinterpret_cast<MCTSNode*>(new_chunk.get() + i * node_size_));
    }
    
    chunks_.push_back(std::move(new_chunk)); 
}

// --- Batch Operations (Locked) ---

void MCTSNodePool::batch_fill_from_global(std::vector<MCTSNode*>& target_cache, size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Ensure we have enough nodes
    while (free_list_.size() < count) {
        grow();
    }

    // Move nodes from end of free_list_ to target_cache
    size_t start_idx = free_list_.size() - count;
    target_cache.insert(target_cache.end(), 
                       free_list_.begin() + start_idx, 
                       free_list_.end());
    free_list_.resize(start_idx);

    // Update stats (Tracks transfers to threads, not exact usage)
    allocated_count_ += count;
    peak_allocated_count_ = std::max(peak_allocated_count_, allocated_count_ - freed_count_);
}

void MCTSNodePool::batch_return_to_global(std::vector<MCTSNode*>& source_cache) {
    if (source_cache.empty()) return;

    std::lock_guard<std::mutex> lock(mutex_);

    free_list_.insert(free_list_.end(), source_cache.begin(), source_cache.end());
    freed_count_ += source_cache.size();
    
    source_cache.clear();
}

// --- Public Interface (Thread-Local Optimized) ---

void* MCTSNodePool::allocate() {
    // 1. Get the shared thread-local cache
    ThreadLocalNodeCache& t_cache = get_local_cache();

    // Register the pool reference on first use (per thread)
    if (!t_cache.pool_ref) {
        t_cache.pool_ref = this;
    }

    // 2. Try to pop from local cache
    if (t_cache.cache.empty()) {
        // Cache empty? Batch fetch from global (locks mutex once)
        batch_fill_from_global(t_cache.cache, LOCAL_CACHE_BATCH_SIZE);
    }

    // 3. Return node
    MCTSNode* node = t_cache.cache.back();
    t_cache.cache.pop_back();
    return node;
}

void MCTSNodePool::deallocate(void* ptr) {
    if (ptr == nullptr) return;

    // 1. Get the shared thread-local cache
    ThreadLocalNodeCache& t_cache = get_local_cache();
    
    // Register pool ref (unlikely to occur before allocate, but safe)
    if (!t_cache.pool_ref) {
        t_cache.pool_ref = this;
    }

    // 2. Push to local cache
    t_cache.cache.push_back(static_cast<MCTSNode*>(ptr));

    // 3. Cache full? Batch return to global (locks mutex once)
    if (t_cache.cache.size() >= LOCAL_CACHE_MAX_SIZE) {
        batch_return_to_global(t_cache.cache);
    }
}

} // namespace chaturaji_cpp