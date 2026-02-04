#include <iostream>
#include <cassert>
#include "dependencies/BTQ_Render_Engine/include/memory/memory_pool.hpp"

int main() {
    std::cout << "Testing new memory pools..." << std::endl;

    // Test OHLCVPool
    {
        auto& pool = BTQuant::OHLCVPool::getInstance();
        
        // Pre-allocate some objects
        pool.preallocate(100);
        
        // Allocate an OHLCV object
        auto* ohlcv = pool.allocate(100.0f, 105.0f, 99.0f, 103.0f, 1000.0);
        assert(ohlcv != nullptr);
        std::cout << "Allocated OHLCV: O=" << ohlcv->open << ", H=" << ohlcv->high 
                  << ", L=" << ohlcv->low << ", C=" << ohlcv->close << ", V=" << ohlcv->volume << std::endl;
        
        // Check stats
        std::cout << "OHLCV Pool - Total: " << pool.getTotalObjects() 
                  << ", Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects() << std::endl;
        
        // Deallocate
        pool.deallocate(ohlcv);
        
        std::cout << "After deallocation - Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects() << std::endl;
    }

    // Test FastOHLCVPool
    {
        auto& pool = BTQuant::FastOHLCVPool::getInstance();
        
        // Pre-allocate some objects
        pool.preallocate(50);
        
        // Allocate an OHLCV object
        auto* ohlcv = pool.allocate(200.0f, 205.0f, 199.0f, 203.0f, 2000.0);
        assert(ohlcv != nullptr);
        std::cout << "Allocated Fast OHLCV: O=" << ohlcv->open << ", H=" << ohlcv->high 
                  << ", L=" << ohlcv->low << ", C=" << ohlcv->close << ", V=" << ohlcv->volume << std::endl;
        
        // Check stats
        std::cout << "Fast OHLCV Pool - Total: " << pool.getTotalObjects() 
                  << ", Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects()
                  << ", Allocs: " << pool.getAllocationCount()
                  << ", Deallocs: " << pool.getDeallocationCount() << std::endl;
        
        // Deallocate
        pool.deallocate(ohlcv);
        
        std::cout << "After deallocation - Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects() << std::endl;
    }

    // Test VolumeNodePool
    {
        auto& pool = BTQuant::VolumeNodePool::getInstance();
        
        // Pre-allocate some objects
        pool.preallocate(200);
        
        // Allocate a VolumeNode object
        auto* node = pool.allocate();
        assert(node != nullptr);
        std::cout << "Allocated VolumeNode" << std::endl;
        
        // Check stats
        std::cout << "VolumeNode Pool - Total: " << pool.getTotalObjects() 
                  << ", Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects() << std::endl;
        
        // Deallocate
        pool.deallocate(node);
        
        std::cout << "After deallocation - Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects() << std::endl;
    }

    // Test FastVolumeProfilePool
    {
        auto& pool = BTQuant::FastVolumeProfilePool::getInstance();
        
        // Pre-allocate some objects
        pool.preallocate(50);
        
        // Allocate a VolumeProfile object
        auto* profile = pool.allocate();
        assert(profile != nullptr);
        std::cout << "Allocated Fast VolumeProfile" << std::endl;
        
        // Check stats
        std::cout << "Fast VolumeProfile Pool - Total: " << pool.getTotalObjects() 
                  << ", Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects() << std::endl;
        
        // Deallocate
        pool.deallocate(profile);
        
        std::cout << "After deallocation - Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects() << std::endl;
    }

    // Test PatternPool
    {
        auto& pool = BTQuant::PatternPool::getInstance();
        
        // Pre-allocate some objects
        pool.preallocate(30);
        
        // Allocate a Pattern object
        auto* pattern = pool.allocate();
        assert(pattern != nullptr);
        std::cout << "Allocated Pattern" << std::endl;
        
        // Check stats
        std::cout << "Pattern Pool - Total: " << pool.getTotalObjects() 
                  << ", Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects() << std::endl;
        
        // Deallocate
        pool.deallocate(pattern);
        
        std::cout << "After deallocation - Used: " << pool.getUsedObjects() 
                  << ", Free: " << pool.getFreeObjects() << std::endl;
    }

    std::cout << "All tests passed!" << std::endl;
    
    return 0;
}