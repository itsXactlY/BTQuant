#include "../../include/memory/memory_pool.hpp"
#include <iostream>

int main() {
    std::cout << "Testing Extended Memory Pool Implementation..." << std::endl;

    // Test OrderBookLevel pool
    {
        std::cout << "\nTesting OrderBookLevel Pool:" << std::endl;
        auto& level_pool = BTQuant::OrderBookLevelPool::getInstance();

        std::cout << "Initial stats - Total: " << level_pool.getTotalObjects()
                  << ", Free: " << level_pool.getFreeObjects()
                  << ", Used: " << level_pool.getUsedObjects() << std::endl;

        // Allocate some OrderBookLevel objects
        auto* level1 = level_pool.allocate();
        auto* level2 = level_pool.allocate();

        if (level1) {
            level1->price = 45000.0;
            level1->size = 1.5;
        }

        std::cout << "After allocation - Total: " << level_pool.getTotalObjects()
                  << ", Free: " << level_pool.getFreeObjects()
                  << ", Used: " << level_pool.getUsedObjects() << std::endl;

        // Deallocate some objects
        if (level1) level_pool.deallocate(level1);
        if (level2) level_pool.deallocate(level2);

        std::cout << "After deallocation - Total: " << level_pool.getTotalObjects()
                  << ", Free: " << level_pool.getFreeObjects()
                  << ", Used: " << level_pool.getUsedObjects() << std::endl;
    }

    // Test CompressedCandle pool
    {
        std::cout << "\nTesting CompressedCandle Pool:" << std::endl;
        auto& candle_pool = BTQuant::CompressedCandlePool::getInstance();

        std::cout << "Initial stats - Total: " << candle_pool.getTotalObjects()
                  << ", Free: " << candle_pool.getFreeObjects()
                  << ", Used: " << candle_pool.getUsedObjects() << std::endl;

        // Allocate some CompressedCandle objects
        auto* candle1 = candle_pool.allocate();
        auto* candle2 = candle_pool.allocate();

        if (candle1) {
            candle1->timestamp_delta = 1000;
            candle1->open_delta = 0.5;
            candle1->high_delta = 1.0;
            candle1->low_delta = -0.2;
            candle1->close_delta = 0.8;
            candle1->volume_delta = 100.0f;
            candle1->trade_count_delta = 5;
        }

        std::cout << "After allocation - Total: " << candle_pool.getTotalObjects()
                  << ", Free: " << candle_pool.getFreeObjects()
                  << ", Used: " << candle_pool.getUsedObjects() << std::endl;

        // Deallocate
        if (candle1) candle_pool.deallocate(candle1);

        std::cout << "After deallocation - Total: " << candle_pool.getTotalObjects()
                  << ", Free: " << candle_pool.getFreeObjects()
                  << ", Used: " << candle_pool.getUsedObjects() << std::endl;
    }

    // Test CompressedTrade pool
    {
        std::cout << "\nTesting CompressedTrade Pool:" << std::endl;
        auto& trade_pool = BTQuant::CompressedTradePool::getInstance();

        std::cout << "Initial stats - Total: " << trade_pool.getTotalObjects()
                  << ", Free: " << trade_pool.getFreeObjects()
                  << ", Used: " << trade_pool.getUsedObjects() << std::endl;

        // Allocate some CompressedTrade objects
        auto* trade1 = trade_pool.allocate();
        auto* trade2 = trade_pool.allocate();

        if (trade1) {
            trade1->timestamp_delta = 500;
            trade1->price_delta = 0.1;
            trade1->volume_delta = 0.5f;
            trade1->side = BTQuant::Data::TradeSide::BUY;
            trade1->exchange_id = 1;
            trade1->flags = 0;
        }

        std::cout << "After allocation - Total: " << trade_pool.getTotalObjects()
                  << ", Free: " << trade_pool.getFreeObjects()
                  << ", Used: " << trade_pool.getUsedObjects() << std::endl;

        // Deallocate
        if (trade1) trade_pool.deallocate(trade1);

        std::cout << "After deallocation - Total: " << trade_pool.getTotalObjects()
                  << ", Free: " << trade_pool.getFreeObjects()
                  << ", Used: " << trade_pool.getUsedObjects() << std::endl;
    }

    // Test FibonacciLevel pool
    {
        std::cout << "\nTesting FibonacciLevel Pool:" << std::endl;
        auto& fib_pool = BTQuant::FibonacciLevelPool::getInstance();

        std::cout << "Initial stats - Total: " << fib_pool.getTotalObjects()
                  << ", Free: " << fib_pool.getFreeObjects()
                  << ", Used: " << fib_pool.getUsedObjects() << std::endl;

        // Allocate some FibonacciLevel objects
        auto* fib1 = fib_pool.allocate();
        auto* fib2 = fib_pool.allocate();

        if (fib1) {
            fib1->price = 45000.0;
            fib1->ratio = 0.618;
            fib1->label = "0.618";
            fib1->color = 0xFF0000FF; // Red
        }

        std::cout << "After allocation - Total: " << fib_pool.getTotalObjects()
                  << ", Free: " << fib_pool.getFreeObjects()
                  << ", Used: " << fib_pool.getUsedObjects() << std::endl;

        // Deallocate
        if (fib1) fib_pool.deallocate(fib1);

        std::cout << "After deallocation - Total: " << fib_pool.getTotalObjects()
                  << ", Free: " << fib_pool.getFreeObjects()
                  << ", Used: " << fib_pool.getUsedObjects() << std::endl;
    }

    std::cout << "\nExtended Memory Pool tests completed successfully!" << std::endl;

    return 0;
}