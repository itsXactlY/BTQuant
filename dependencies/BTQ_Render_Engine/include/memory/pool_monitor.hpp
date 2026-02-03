#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <map>
#include <string>

#include "memory_pool.hpp"

namespace BTQuant {

class MemoryPoolMonitor {
public:
    static MemoryPoolMonitor& getInstance() {
        static MemoryPoolMonitor instance;
        return instance;
    }

    // Record allocation statistics for a specific pool
    template<typename PoolType>
    void recordPoolStats(const std::string& pool_name, const PoolType& pool) {
        auto now = std::chrono::high_resolution_clock::now();
        
        // Store current statistics
        pool_stats_[pool_name] = {
            pool.getTotalObjects(),
            pool.getFreeObjects(),
            pool.getUsedObjects(),
            pool.getAllocationCount(),
            pool.getDeallocationCount(),
            now
        };
    }

    // Print detailed statistics for all registered pools
    void printStatistics() const {
        std::cout << "\n=== Memory Pool Statistics ===" << std::endl;
        std::cout << std::left << std::setw(30) << "Pool Name" 
                  << std::setw(12) << "Total" 
                  << std::setw(12) << "Free" 
                  << std::setw(12) << "Used" 
                  << std::setw(15) << "Alloc Count" 
                  << std::setw(15) << "Dealloc Count" 
                  << std::setw(15) << "Utilization %" << std::endl;
        std::cout << std::string(110, '-') << std::endl;

        for (const auto& [name, stats] : pool_stats_) {
            double utilization = stats.total_objects > 0 ? 
                (static_cast<double>(stats.used_objects) / stats.total_objects) * 100.0 : 0.0;
            
            std::cout << std::left << std::setw(30) << name
                      << std::setw(12) << stats.total_objects
                      << std::setw(12) << stats.free_objects
                      << std::setw(12) << stats.used_objects
                      << std::setw(15) << stats.allocation_count
                      << std::setw(15) << stats.deallocation_count
                      << std::setw(15) << std::fixed << std::setprecision(2) << utilization << std::endl;
        }
        std::cout << std::string(110, '-') << std::endl;
    }

    // Get a summary of memory savings
    std::string getMemorySavingsEstimate() const {
        size_t total_allocations = 0;
        for (const auto& [name, stats] : pool_stats_) {
            total_allocations += stats.allocation_count;
        }
        
        // Estimate based on typical heap allocation overhead (assuming ~16 bytes overhead per allocation)
        size_t estimated_saved_bytes = total_allocations * 16; // Conservative estimate
        
        std::ostringstream oss;
        oss << "Estimated memory savings: ~" << estimated_saved_bytes << " bytes ("
            << (estimated_saved_bytes / 1024.0) << " KB) in allocation overhead";
        
        return oss.str();
    }

private:
    struct PoolStats {
        size_t total_objects;
        size_t free_objects;
        size_t used_objects;
        size_t allocation_count;
        size_t deallocation_count;
        std::chrono::high_resolution_clock::time_point timestamp;
    };

    std::map<std::string, PoolStats> pool_stats_;
};

// Helper macro to easily record pool statistics
#define RECORD_POOL_STATS(pool_name_str, pool_instance) \
    BTQuant::MemoryPoolMonitor::getInstance().recordPoolStats(pool_name_str, pool_instance)

} // namespace BTQuant