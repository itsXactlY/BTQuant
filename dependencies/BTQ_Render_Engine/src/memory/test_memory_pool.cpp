#include "../../include/memory/memory_pool.hpp"
#include <iostream>

int main() {
    std::cout << "Testing Memory Pool Implementation..." << std::endl;
    
    // Test TradeData pool
    {
        std::cout << "\nTesting TradeData Pool:" << std::endl;
        auto& trade_pool = BTQuant::TradeDataPool::getInstance();
        
        std::cout << "Initial stats - Total: " << trade_pool.getTotalObjects() 
                  << ", Free: " << trade_pool.getFreeObjects() 
                  << ", Used: " << trade_pool.getUsedObjects() << std::endl;
        
        // Allocate some TradeData objects
        auto* trade1 = trade_pool.allocate();
        auto* trade2 = trade_pool.allocate();
        auto* trade3 = trade_pool.allocate();
        
        if (trade1) {
            trade1->timestamp = 1234567890;
            trade1->price = 100.50;
            trade1->volume = 10.0f;
        }
        
        std::cout << "After allocation - Total: " << trade_pool.getTotalObjects() 
                  << ", Free: " << trade_pool.getFreeObjects() 
                  << ", Used: " << trade_pool.getUsedObjects() << std::endl;
        
        // Deallocate some objects
        if (trade1) trade_pool.deallocate(trade1);
        if (trade2) trade_pool.deallocate(trade2);
        
        std::cout << "After deallocation - Total: " << trade_pool.getTotalObjects() 
                  << ", Free: " << trade_pool.getFreeObjects() 
                  << ", Used: " << trade_pool.getUsedObjects() << std::endl;
    }
    
    // Test ClusterCell pool
    {
        std::cout << "\nTesting ClusterCell Pool:" << std::endl;
        auto& cluster_pool = BTQuant::ClusterCellPool::getInstance();
        
        std::cout << "Initial stats - Total: " << cluster_pool.getTotalObjects() 
                  << ", Free: " << cluster_pool.getFreeObjects() 
                  << ", Used: " << cluster_pool.getUsedObjects() << std::endl;
        
        // Allocate some ClusterCell objects
        auto* cell1 = cluster_pool.allocate();
        auto* cell2 = cluster_pool.allocate();
        
        if (cell1) {
            cell1->total_volume = 1000.0;
            cell1->buy_volume = 600.0;
            cell1->sell_volume = 400.0;
        }
        
        std::cout << "After allocation - Total: " << cluster_pool.getTotalObjects() 
                  << ", Free: " << cluster_pool.getFreeObjects() 
                  << ", Used: " << cluster_pool.getUsedObjects() << std::endl;
        
        // Deallocate
        if (cell1) cluster_pool.deallocate(cell1);
        
        std::cout << "After deallocation - Total: " << cluster_pool.getTotalObjects() 
                  << ", Free: " << cluster_pool.getFreeObjects() 
                  << ", Used: " << cluster_pool.getUsedObjects() << std::endl;
    }
    
    // Test EMAIndicator pool
    {
        std::cout << "\nTesting EMAIndicator Pool:" << std::endl;
        auto& ema_pool = BTQuant::EMAIndicatorPool::getInstance();

        std::cout << "Initial stats - Total: " << ema_pool.getTotalObjects()
                  << ", Free: " << ema_pool.getFreeObjects()
                  << ", Used: " << ema_pool.getUsedObjects() << std::endl;

        // Allocate some EMAIndicator objects
        auto* ema1 = ema_pool.allocate(14);  // Pass the period parameter directly
        auto* ema2 = ema_pool.allocate(20);

        if (ema1) {
            ema1->update(100.0f);
            ema1->update(101.0f);
            std::cout << "EMA1 value: " << ema1->get_value() << std::endl;
        }

        std::cout << "After allocation - Total: " << ema_pool.getTotalObjects()
                  << ", Free: " << ema_pool.getFreeObjects()
                  << ", Used: " << ema_pool.getUsedObjects() << std::endl;

        // Deallocate
        if (ema1) ema_pool.deallocate(ema1);

        std::cout << "After deallocation - Total: " << ema_pool.getTotalObjects()
                  << ", Free: " << ema_pool.getFreeObjects()
                  << ", Used: " << ema_pool.getUsedObjects() << std::endl;
    }

    // Test SMAIndicator pool
    {
        std::cout << "\nTesting SMAIndicator Pool:" << std::endl;
        auto& sma_pool = BTQuant::SMAIndicatorPool::getInstance();

        std::cout << "Initial stats - Total: " << sma_pool.getTotalObjects()
                  << ", Free: " << sma_pool.getFreeObjects()
                  << ", Used: " << sma_pool.getUsedObjects() << std::endl;

        // Allocate some SMAIndicator objects
        auto* sma1 = sma_pool.allocate(20);

        if (sma1) {
            sma1->update(100.0f);
            sma1->update(101.0f);
            sma1->update(102.0f);
            std::cout << "SMA1 value: " << sma1->get_value() << std::endl;
        }

        std::cout << "After allocation - Total: " << sma_pool.getTotalObjects()
                  << ", Free: " << sma_pool.getFreeObjects()
                  << ", Used: " << sma_pool.getUsedObjects() << std::endl;

        // Deallocate
        if (sma1) sma_pool.deallocate(sma1);

        std::cout << "After deallocation - Total: " << sma_pool.getTotalObjects()
                  << ", Free: " << sma_pool.getFreeObjects()
                  << ", Used: " << sma_pool.getUsedObjects() << std::endl;
    }
    
    std::cout << "\nMemory Pool tests completed successfully!" << std::endl;
    
    return 0;
}