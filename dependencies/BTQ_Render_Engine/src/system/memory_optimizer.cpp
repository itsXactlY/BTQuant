#include "../include/system/memory_optimizer.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace BTQuant {

struct MemoryOptimizer::Impl {
    // Memory tracking
    size_t total_allocated = 0;
    size_t total_deallocated = 0;
    size_t current_allocated = 0;
    size_t peak_usage = 0;
    size_t fragmentation = 0;
    float fragmentation_ratio = 0.0f;
    
    // Optimization parameters
    size_t compaction_threshold = 10 * 1024 * 1024; // 10MB
    size_t release_threshold = 5 * 1024 * 1024;     // 5MB
    
    // Allocation tracking
    std::vector<AllocationRecord> recent_allocations;
    std::unordered_map<void*, size_t> active_allocations;
    std::mutex allocation_mutex;
    
    // Memory pools
    struct MemoryPool {
        void* start_address;
        size_t size;
        size_t used;
        size_t capacity;
        bool is_compacted;
    };
    
    std::vector<MemoryPool> memory_pools;
    bool tracking_enabled = true;
    
    // Statistics update
    void update_statistics() {
        std::lock_guard<std::mutex> lock(allocation_mutex);
        
        current_allocated = total_allocated - total_deallocated;
        if (current_allocated > peak_usage) {
            peak_usage = current_allocated;
        }
        
        // Calculate fragmentation estimate
        fragmentation = 0;
        for (const auto& pool : memory_pools) {
            fragmentation += (pool.capacity - pool.used);
        }
        
        if (total_allocated > 0) {
            fragmentation_ratio = static_cast<float>(fragmentation) / static_cast<float>(total_allocated);
        }
    }
    
    // Find or create a memory pool
    MemoryPool* find_or_create_pool(size_t size) {
        // Look for an existing pool with enough space
        for (auto& pool : memory_pools) {
            if (pool.capacity - pool.used >= size) {
                return &pool;
            }
        }
        
        // Create a new pool
        MemoryPool new_pool;
        new_pool.size = std::max(size * 2, static_cast<size_t>(1024 * 1024)); // At least 1MB
        new_pool.start_address = std::malloc(new_pool.size);
        if (!new_pool.start_address) {
            return nullptr;
        }
        
        new_pool.used = 0;
        new_pool.capacity = new_pool.size;
        new_pool.is_compacted = true;
        
        memory_pools.push_back(new_pool);
        return &memory_pools.back();
    }
};

MemoryOptimizer::MemoryOptimizer() : impl_(std::make_unique<Impl>()) {}

MemoryOptimizer::~MemoryOptimizer() {
    // Clean up memory pools
    for (auto& pool : impl_->memory_pools) {
        if (pool.start_address) {
            std::free(pool.start_address);
        }
    }
}

void MemoryOptimizer::initialize() {
    // Initialize memory pools with default sizes
    impl_->compaction_threshold = 10 * 1024 * 1024; // 10MB
    impl_->release_threshold = 5 * 1024 * 1024;     // 5MB
}

void MemoryOptimizer::optimize_allocations() {
    // Perform various memory optimization techniques
    compact_memory_pools();
    release_unused_memory();
    impl_->update_statistics();
}

void MemoryOptimizer::compact_memory_pools() {
    std::lock_guard<std::mutex> lock(impl_->allocation_mutex);
    
    // Compact fragmented memory pools
    for (auto& pool : impl_->memory_pools) {
        if (!pool.is_compacted && pool.used < pool.capacity * 0.7f) {
            // Perform compaction by copying active allocations to a new contiguous block
            // This is a simplified implementation - in practice, this would be more complex
            pool.is_compacted = true;
        }
    }
}

void MemoryOptimizer::release_unused_memory() {
    std::lock_guard<std::mutex> lock(impl_->allocation_mutex);
    
    // Release memory pools that are below the release threshold
    auto it = impl_->memory_pools.begin();
    while (it != impl_->memory_pools.end()) {
        if (it->used < impl_->release_threshold && it->used < it->capacity * 0.2f) {
            // Only release if the pool is significantly underutilized
            if (it->start_address) {
                std::free(it->start_address);
            }
            it = impl_->memory_pools.erase(it);
        } else {
            ++it;
        }
    }
}

void MemoryOptimizer::track_allocation(size_t size, const char* tag) {
    if (!impl_->tracking_enabled) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(impl_->allocation_mutex);
    
    impl_->total_allocated += size;
    impl_->current_allocated += size;
    
    // Record the allocation
    AllocationRecord record;
    record.size = size;
    record.tag = tag;
    record.ptr = nullptr; // In a real implementation, this would be the actual pointer
    
    impl_->recent_allocations.push_back(record);
    
    // Keep only the most recent allocations
    if (impl_->recent_allocations.size() > 1000) {
        impl_->recent_allocations.erase(impl_->recent_allocations.begin());
    }
    
    impl_->update_statistics();
}

void MemoryOptimizer::track_deallocation(size_t size, const char* tag) {
    if (!impl_->tracking_enabled) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(impl_->allocation_mutex);
    
    impl_->total_deallocated += size;
    impl_->current_allocated -= size;
    
    impl_->update_statistics();
}

MemoryOptimizer::MemoryStats MemoryOptimizer::get_memory_stats() const {
    std::lock_guard<std::mutex> lock(impl_->allocation_mutex);
    
    MemoryStats stats;
    stats.total_allocated = impl_->total_allocated;
    stats.total_deallocated = impl_->total_deallocated;
    stats.current_allocated = impl_->current_allocated;
    stats.peak_usage = impl_->peak_usage;
    stats.fragmentation = impl_->fragmentation;
    stats.fragmentation_ratio = impl_->fragmentation_ratio;
    stats.pool_count = impl_->memory_pools.size();
    
    // Calculate reserved and committed memory
    for (const auto& pool : impl_->memory_pools) {
        stats.reserved_memory += pool.capacity;
        stats.committed_memory += pool.used;
    }
    
    return stats;
}

void MemoryOptimizer::set_compaction_threshold(size_t threshold_bytes) {
    impl_->compaction_threshold = threshold_bytes;
}

void MemoryOptimizer::set_release_threshold(size_t threshold_bytes) {
    impl_->release_threshold = threshold_bytes;
}

void MemoryOptimizer::force_garbage_collection() {
    std::lock_guard<std::mutex> lock(impl_->allocation_mutex);
    
    // In a real implementation, this would perform garbage collection
    // For now, just compact memory pools and release unused memory
    compact_memory_pools();
    release_unused_memory();
    
    impl_->update_statistics();
}

void MemoryOptimizer::enable_tracking(bool enabled) {
    impl_->tracking_enabled = enabled;
}

std::vector<MemoryOptimizer::AllocationRecord> MemoryOptimizer::get_recent_allocations(size_t count) const {
    std::lock_guard<std::mutex> lock(impl_->allocation_mutex);
    
    std::vector<AllocationRecord> result;
    size_t start_idx = 0;
    
    if (impl_->recent_allocations.size() > count) {
        start_idx = impl_->recent_allocations.size() - count;
    }
    
    for (size_t i = start_idx; i < impl_->recent_allocations.size(); ++i) {
        result.push_back(impl_->recent_allocations[i]);
    }
    
    return result;
}

} // namespace BTQuant