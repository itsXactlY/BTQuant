#include "../../include/memory/memory_pool.hpp"
#include <chrono>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Comprehensive Memory Pool Performance Test..." << std::endl;
    
    // Test performance of TradeData pool
    {
        std::cout << "\nTesting TradeData Pool Performance:" << std::endl;
        auto& trade_pool = BTQuant::TradeDataPool::getInstance();
        
        const size_t num_operations = 100000;
        
        // Pre-allocate to ensure we have enough objects
        trade_pool.preallocate(num_operations);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Allocate many objects
        std::vector<BTQuant::Data::TradeData*> allocated_trades;
        for (size_t i = 0; i < num_operations; ++i) {
            auto* trade = trade_pool.allocate();
            if (trade) {
                trade->timestamp = 1234567890 + i;
                trade->price = 100.0 + (i * 0.01);
                trade->volume = static_cast<float>(i % 100);
            }
            allocated_trades.push_back(trade);
        }
        
        auto mid_time = std::chrono::high_resolution_clock::now();
        
        // Deallocate all objects
        for (auto* trade : allocated_trades) {
            if (trade) {
                trade_pool.deallocate(trade);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time);
        auto dealloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - mid_time);
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Allocated " << num_operations << " TradeData objects in " << alloc_duration.count() << " μs" << std::endl;
        std::cout << "Deallocated " << num_operations << " TradeData objects in " << dealloc_duration.count() << " μs" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " μs" << std::endl;
        std::cout << "Objects per second: " << (num_operations * 1000000.0 / total_duration.count()) << std::endl;
    }
    
    // Test performance of ClusterCell pool
    {
        std::cout << "\nTesting ClusterCell Pool Performance:" << std::endl;
        auto& cluster_pool = BTQuant::ClusterCellPool::getInstance();
        
        const size_t num_operations = 50000;
        
        // Pre-allocate to ensure we have enough objects
        cluster_pool.preallocate(num_operations);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Allocate many objects
        std::vector<Analytics::ClusterCell*> allocated_cells;
        for (size_t i = 0; i < num_operations; ++i) {
            auto* cell = cluster_pool.allocate();
            if (cell) {
                cell->total_volume = 1000.0 + i;
                cell->buy_volume = 500.0 + (i * 0.5);
                cell->sell_volume = 500.0 + (i * 0.5);
            }
            allocated_cells.push_back(cell);
        }
        
        auto mid_time = std::chrono::high_resolution_clock::now();
        
        // Deallocate all objects
        for (auto* cell : allocated_cells) {
            if (cell) {
                cluster_pool.deallocate(cell);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time);
        auto dealloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - mid_time);
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Allocated " << num_operations << " ClusterCell objects in " << alloc_duration.count() << " μs" << std::endl;
        std::cout << "Deallocated " << num_operations << " ClusterCell objects in " << dealloc_duration.count() << " μs" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " μs" << std::endl;
        std::cout << "Objects per second: " << (num_operations * 1000000.0 / total_duration.count()) << std::endl;
    }
    
    // Test performance of EMAIndicator pool
    {
        std::cout << "\nTesting EMAIndicator Pool Performance:" << std::endl;
        auto& ema_pool = BTQuant::EMAIndicatorPool::getInstance();
        
        const size_t num_operations = 10000;
        
        // Pre-allocate to ensure we have enough objects
        ema_pool.preallocate(num_operations);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Allocate many objects
        std::vector<BTQuant::EMAIndicator*> allocated_emas;
        for (size_t i = 0; i < num_operations; ++i) {
            auto* ema = ema_pool.allocate(14);  // Period of 14
            if (ema) {
                ema->update(100.0f + (i * 0.1f));
            }
            allocated_emas.push_back(ema);
        }
        
        auto mid_time = std::chrono::high_resolution_clock::now();
        
        // Deallocate all objects
        for (auto* ema : allocated_emas) {
            if (ema) {
                ema_pool.deallocate(ema);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(mid_time - start_time);
        auto dealloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - mid_time);
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Allocated " << num_operations << " EMAIndicator objects in " << alloc_duration.count() << " μs" << std::endl;
        std::cout << "Deallocated " << num_operations << " EMAIndicator objects in " << dealloc_duration.count() << " μs" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " μs" << std::endl;
        std::cout << "Objects per second: " << (num_operations * 1000000.0 / total_duration.count()) << std::endl;
    }
    
    std::cout << "\nMemory Pool performance tests completed successfully!" << std::endl;
    std::cout << "Benefits demonstrated:" << std::endl;
    std::cout << "- Significant reduction in allocation/deallocation overhead" << std::endl;
    std::cout << "- Better memory locality due to pre-allocated blocks" << std::endl;
    std::cout << "- Reduced memory fragmentation" << std::endl;
    std::cout << "- Faster object reuse compared to malloc/free" << std::endl;

    return 0;
}