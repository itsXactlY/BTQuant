#include "../../include/memory/memory_pool.hpp"
#include <iostream>
#include <cassert>
#include <vector>

#define RECORD_POOL_STATS(pool_name, pool_instance) \
    std::cout << #pool_name << " - Total: " << pool_instance.getTotalObjects() \
              << ", Free: " << pool_instance.getFreeObjects() \
              << ", Used: " << pool_instance.getUsedObjects() << std::endl;

int main() {
    std::cout << "Testing new memory pools..." << std::endl;

    // Test TradePaceDataPool
    {
        std::cout << "\nTesting TradePaceDataPool:" << std::endl;
        auto& pool = BTQuant::TradePaceDataPool::getInstance();
        
        RECORD_POOL_STATS("TradePaceDataPool", pool);
        
        // Allocate some objects
        auto* data1 = pool.allocate();
        auto* data2 = pool.allocate();
        
        RECORD_POOL_STATS("TradePaceDataPool", pool);
        
        // Initialize the objects
        if (data1) {
            data1->timestamp = 1234567890;
            data1->trades_per_minute = 100;
            data1->volume_per_minute = 5000.0;
        }
        
        if (data2) {
            data2->timestamp = 1234567891;
            data2->trades_per_minute = 200;
            data2->volume_per_minute = 10000.0;
        }
        
        // Deallocate
        if (data1) pool.deallocate(data1);
        if (data2) pool.deallocate(data2);
        
        RECORD_POOL_STATS("TradePaceDataPool", pool);
        
        std::cout << "TradePaceDataPool test completed successfully!" << std::endl;
    }

    // Test FastTradePaceDataPool
    {
        std::cout << "\nTesting FastTradePaceDataPool:" << std::endl;
        auto& pool = BTQuant::FastTradePaceDataPool::getInstance();
        
        RECORD_POOL_STATS("FastTradePaceDataPool", pool);
        std::cout << "Allocation Count: " << pool.getAllocationCount() 
                  << ", Deallocation Count: " << pool.getDeallocationCount() << std::endl;
        
        // Preallocate some objects
        pool.preallocate(100);
        
        RECORD_POOL_STATS("FastTradePaceDataPool", pool);
        
        // Allocate some objects
        auto* data1 = pool.allocate();
        auto* data2 = pool.allocate();
        
        // Initialize the objects
        if (data1) {
            data1->timestamp = 1234567892;
            data1->trades_per_minute = 300;
            data1->volume_per_minute = 15000.0;
        }
        
        if (data2) {
            data2->timestamp = 1234567893;
            data2->trades_per_minute = 400;
            data2->volume_per_minute = 20000.0;
        }
        
        // Deallocate
        if (data1) pool.deallocate(data1);
        if (data2) pool.deallocate(data2);
        
        RECORD_POOL_STATS("FastTradePaceDataPool", pool);
        std::cout << "Allocation Count: " << pool.getAllocationCount() 
                  << ", Deallocation Count: " << pool.getDeallocationCount() << std::endl;
        
        std::cout << "FastTradePaceDataPool test completed successfully!" << std::endl;
    }

    // Test TradePairPool
    {
        std::cout << "\nTesting TradePairPool:" << std::endl;
        auto& pool = BTQuant::TradePairPool::getInstance();
        
        RECORD_POOL_STATS("TradePairPool", pool);
        
        // Allocate some objects
        auto* pair1 = pool.allocate();
        auto* pair2 = pool.allocate();
        
        // Initialize the objects
        if (pair1) {
            pair1->first.timestamp = 1234567894;
            pair1->first.price = 50000.0;
            pair1->first.volume = 1.0f;
            pair1->second.timestamp = 1234567895;
            pair1->second.price = 50001.0;
            pair1->second.volume = 1.5f;
        }
        
        if (pair2) {
            pair2->first.timestamp = 1234567896;
            pair2->first.price = 50002.0;
            pair2->first.volume = 2.0f;
            pair2->second.timestamp = 1234567897;
            pair2->second.price = 50003.0;
            pair2->second.volume = 2.5f;
        }
        
        // Deallocate
        if (pair1) pool.deallocate(pair1);
        if (pair2) pool.deallocate(pair2);
        
        RECORD_POOL_STATS("TradePairPool", pool);
        
        std::cout << "TradePairPool test completed successfully!" << std::endl;
    }

    // Test FastTradePairPool
    {
        std::cout << "\nTesting FastTradePairPool:" << std::endl;
        auto& pool = BTQuant::FastTradePairPool::getInstance();
        
        RECORD_POOL_STATS("FastTradePairPool", pool);
        
        // Preallocate some objects
        pool.preallocate(50);
        
        // Allocate some objects
        auto* pair1 = pool.allocate();
        auto* pair2 = pool.allocate();
        
        // Initialize the objects
        if (pair1) {
            pair1->first.timestamp = 1234567898;
            pair1->first.price = 50004.0;
            pair1->first.volume = 3.0f;
            pair1->second.timestamp = 1234567899;
            pair1->second.price = 50005.0;
            pair1->second.volume = 3.5f;
        }
        
        if (pair2) {
            pair2->first.timestamp = 1234567900;
            pair2->first.price = 50006.0;
            pair2->first.volume = 4.0f;
            pair2->second.timestamp = 1234567901;
            pair2->second.price = 50007.0;
            pair2->second.volume = 4.5f;
        }
        
        // Deallocate
        if (pair1) pool.deallocate(pair1);
        if (pair2) pool.deallocate(pair2);
        
        RECORD_POOL_STATS("FastTradePairPool", pool);
        
        std::cout << "FastTradePairPool test completed successfully!" << std::endl;
    }

    // Test IndicatorValuePairPool
    {
        std::cout << "\nTesting IndicatorValuePairPool:" << std::endl;
        auto& pool = BTQuant::IndicatorValuePairPool::getInstance();
        
        RECORD_POOL_STATS("IndicatorValuePairPool", pool);
        
        // Allocate some objects
        auto* pair1 = pool.allocate();
        auto* pair2 = pool.allocate();
        
        // Initialize the objects
        if (pair1) {
            pair1->first = 1.5f;
            pair1->second = 2.5f;
        }
        
        if (pair2) {
            pair2->first = 3.5f;
            pair2->second = 4.5f;
        }
        
        // Deallocate
        if (pair1) pool.deallocate(pair1);
        if (pair2) pool.deallocate(pair2);
        
        RECORD_POOL_STATS("IndicatorValuePairPool", pool);
        
        std::cout << "IndicatorValuePairPool test completed successfully!" << std::endl;
    }

    // Test FastIndicatorValuePairPool
    {
        std::cout << "\nTesting FastIndicatorValuePairPool:" << std::endl;
        auto& pool = BTQuant::FastIndicatorValuePairPool::getInstance();
        
        RECORD_POOL_STATS("FastIndicatorValuePairPool", pool);
        std::cout << "Allocation Count: " << pool.getAllocationCount() 
                  << ", Deallocation Count: " << pool.getDeallocationCount() << std::endl;
        
        // Preallocate some objects
        pool.preallocate(1000);
        
        // Allocate some objects
        auto* pair1 = pool.allocate();
        auto* pair2 = pool.allocate();
        
        // Initialize the objects
        if (pair1) {
            pair1->first = 5.5f;
            pair1->second = 6.5f;
        }
        
        if (pair2) {
            pair2->first = 7.5f;
            pair2->second = 8.5f;
        }
        
        // Deallocate
        if (pair1) pool.deallocate(pair1);
        if (pair2) pool.deallocate(pair2);
        
        RECORD_POOL_STATS("FastIndicatorValuePairPool", pool);
        std::cout << "Allocation Count: " << pool.getAllocationCount() 
                  << ", Deallocation Count: " << pool.getDeallocationCount() << std::endl;
        
        std::cout << "FastIndicatorValuePairPool test completed successfully!" << std::endl;
    }

    std::cout << "\nAll memory pool tests completed successfully!" << std::endl;
    return 0;
}