#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cassert>

#include "memory/memory_pool.hpp"
#include "memory/pool_monitor.hpp"

using namespace BTQuant;

void testBasicFunctionality() {
    std::cout << "Testing basic memory pool functionality..." << std::endl;
    
    // Test TradeData pool
    auto& trade_pool = TradeDataPool::getInstance();
    trade_pool.preallocate(100);
    
    auto* trade1 = trade_pool.allocate();
    assert(trade1 != nullptr);
    trade1->price = 100.5;
    trade1->quantity = 10;
    
    auto* trade2 = trade_pool.allocate();
    assert(trade2 != nullptr);
    trade2->price = 200.75;
    trade2->quantity = 20;
    
    trade_pool.deallocate(trade1);
    trade_pool.deallocate(trade2);
    
    std::cout << "✓ Basic functionality test passed" << std::endl;
}

void testFastPools() {
    std::cout << "Testing fast memory pools..." << std::endl;
    
    // Test FastTradeDataPool
    auto& fast_trade_pool = FastTradeDataPool::getInstance();
    fast_trade_pool.preallocate(1000);
    
    std::vector<Data::TradeData*> trades;
    
    // Allocate many trades
    for (int i = 0; i < 100; ++i) {
        auto* trade = fast_trade_pool.allocate();
        assert(trade != nullptr);
        trade->price = 100.0 + i;
        trade->quantity = i;
        trades.push_back(trade);
    }
    
    // Verify allocation counts
    size_t alloc_count_before = fast_trade_pool.getAllocationCount();
    assert(alloc_count_before >= 100);
    
    // Deallocate all trades
    for (auto* trade : trades) {
        fast_trade_pool.deallocate(trade);
    }
    
    std::cout << "✓ Fast pools test passed" << std::endl;
}

void testParameterizedAllocation() {
    std::cout << "Testing parameterized allocation..." << std::endl;

    // Test EMA indicator pool with parameterized allocation
    auto& ema_pool = EMAIndicatorPool::getInstance();
    ema_pool.preallocate(50);

    auto* ema1 = ema_pool.allocate(14);  // 14-period EMA
    assert(ema1 != nullptr);

    auto* ema2 = ema_pool.allocate(50);  // 50-period EMA
    assert(ema2 != nullptr);

    ema_pool.deallocate(ema1);
    ema_pool.deallocate(ema2);

    std::cout << "✓ Parameterized allocation test passed" << std::endl;
}

void testPerformanceMonitoring() {
    std::cout << "Testing performance monitoring..." << std::endl;
    
    auto& fast_pool = FastTradeDataPool::getInstance();
    fast_pool.preallocate(1000);
    
    // Record initial stats
    RECORD_POOL_STATS("FastTradeDataPool", fast_pool);
    
    // Perform allocations and deallocations
    std::vector<Data::TradeData*> trades;
    for (int i = 0; i < 500; ++i) {
        auto* trade = fast_pool.allocate();
        trade->price = 100.0 + i;
        trades.push_back(trade);
    }
    
    for (auto* trade : trades) {
        fast_pool.deallocate(trade);
    }
    
    // Record final stats
    RECORD_POOL_STATS("FastTradeDataPool", fast_pool);
    
    // Print statistics
    MemoryPoolMonitor::getInstance().printStatistics();
    
    std::cout << "✓ Performance monitoring test passed" << std::endl;
}

void testMultiThreadedAccess() {
    std::cout << "Testing multi-threaded access..." << std::endl;
    
    auto& fast_pool = FastTradeDataPool::getInstance();
    fast_pool.preallocate(2000);
    
    const int num_threads = 4;
    const int ops_per_thread = 500;
    
    std::vector<std::thread> threads;
    
    // Launch multiple threads performing allocations/deallocations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&fast_pool, ops_per_thread]() {
            std::vector<Data::TradeData*> local_trades;
            
            for (int i = 0; i < ops_per_thread; ++i) {
                auto* trade = fast_pool.allocate();
                assert(trade != nullptr);
                trade->price = 100.0 + i;
                local_trades.push_back(trade);
                
                // Occasionally deallocate some trades
                if (i % 2 == 0 && !local_trades.empty()) {
                    auto* to_deallocate = local_trades.back();
                    local_trades.pop_back();
                    fast_pool.deallocate(to_deallocate);
                }
            }
            
            // Deallocate remaining trades
            for (auto* trade : local_trades) {
                fast_pool.deallocate(trade);
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "✓ Multi-threaded access test passed" << std::endl;
}

int main() {
    std::cout << "Running enhanced memory pool tests..." << std::endl;
    
    testBasicFunctionality();
    testFastPools();
    testParameterizedAllocation();
    testPerformanceMonitoring();
    testMultiThreadedAccess();
    
    std::cout << "\nAll tests passed successfully!" << std::endl;
    
    return 0;
}