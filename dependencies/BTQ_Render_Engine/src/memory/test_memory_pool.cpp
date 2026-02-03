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

    // Test Order pool
    {
        std::cout << "\nTesting Order Pool:" << std::endl;
        auto& order_pool = BTQuant::OrderPool::getInstance();

        std::cout << "Initial stats - Total: " << order_pool.getTotalObjects()
                  << ", Free: " << order_pool.getFreeObjects()
                  << ", Used: " << order_pool.getUsedObjects() << std::endl;

        // Allocate some Order objects
        auto* order1 = order_pool.allocate();
        auto* order2 = order_pool.allocate();

        if (order1) {
            order1->order_id = "ORDER_TEST_1";
            order1->symbol = "BTCUSD";
            order1->quantity = 1.5;
            order1->price = 45000.0;
        }

        std::cout << "After allocation - Total: " << order_pool.getTotalObjects()
                  << ", Free: " << order_pool.getFreeObjects()
                  << ", Used: " << order_pool.getUsedObjects() << std::endl;

        // Deallocate
        if (order1) order_pool.deallocate(order1);

        std::cout << "After deallocation - Total: " << order_pool.getTotalObjects()
                  << ", Free: " << order_pool.getFreeObjects()
                  << ", Used: " << order_pool.getUsedObjects() << std::endl;
    }

    // Test ProcessedTrade pool
    {
        std::cout << "\nTesting ProcessedTrade Pool:" << std::endl;
        auto& trade_pool = BTQuant::ProcessedTradePool::getInstance();

        std::cout << "Initial stats - Total: " << trade_pool.getTotalObjects()
                  << ", Free: " << trade_pool.getFreeObjects()
                  << ", Used: " << trade_pool.getUsedObjects() << std::endl;

        // Allocate some ProcessedTrade objects
        auto* trade1 = trade_pool.allocate();
        auto* trade2 = trade_pool.allocate();

        if (trade1) {
            trade1->symbol_id = 123;
            trade1->price = 100.50;
            trade1->size = 10.0;
            trade1->timestamp = 1234567890;
            trade1->is_buy = true;
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

    // Test CandleCluster pool
    {
        std::cout << "\nTesting CandleCluster Pool:" << std::endl;
        auto& cluster_pool = BTQuant::CandleClusterPool::getInstance();

        std::cout << "Initial stats - Total: " << cluster_pool.getTotalObjects()
                  << ", Free: " << cluster_pool.getFreeObjects()
                  << ", Used: " << cluster_pool.getUsedObjects() << std::endl;

        // Allocate some CandleCluster objects
        auto* cluster1 = cluster_pool.allocate(10.0f, 20.0f, 5.0f, 2.0f, 100, 150, 10, 101.5f, true);
        auto* cluster2 = cluster_pool.allocate();

        if (cluster1) {
            cluster1->bidVolume = 200;
            cluster1->askVolume = 180;
            cluster1->tradeCount = 15;
            cluster1->vwap = 102.0f;
        }

        std::cout << "After allocation - Total: " << cluster_pool.getTotalObjects()
                  << ", Free: " << cluster_pool.getFreeObjects()
                  << ", Used: " << cluster_pool.getUsedObjects() << std::endl;

        // Deallocate
        if (cluster1) cluster_pool.deallocate(cluster1);

        std::cout << "After deallocation - Total: " << cluster_pool.getTotalObjects()
                  << ", Free: " << cluster_pool.getFreeObjects()
                  << ", Used: " << cluster_pool.getUsedObjects() << std::endl;
    }

    // Test VolumeProfileNode pool
    {
        std::cout << "\nTesting VolumeProfileNode Pool:" << std::endl;
        auto& node_pool = BTQuant::VolumeProfileNodePool::getInstance();

        std::cout << "Initial stats - Total: " << node_pool.getTotalObjects()
                  << ", Free: " << node_pool.getFreeObjects()
                  << ", Used: " << node_pool.getUsedObjects() << std::endl;

        // Allocate some VolumeProfileNode objects
        auto* node1 = node_pool.allocate();
        auto* node2 = node_pool.allocate();

        if (node1) {
            node1->priceLevel = 100.50;
            node1->totalVolume = 1000.0;
            node1->buyVolume = 600.0;
            node1->sellVolume = 400.0;
            node1->delta = 200.0;
            node1->numTrades = 25;
        }

        std::cout << "After allocation - Total: " << node_pool.getTotalObjects()
                  << ", Free: " << node_pool.getFreeObjects()
                  << ", Used: " << node_pool.getUsedObjects() << std::endl;

        // Deallocate
        if (node1) node_pool.deallocate(node1);

        std::cout << "After deallocation - Total: " << node_pool.getTotalObjects()
                  << ", Free: " << node_pool.getFreeObjects()
                  << ", Used: " << node_pool.getUsedObjects() << std::endl;
    }

    // Test FootprintCell pool
    {
        std::cout << "\nTesting FootprintCell Pool:" << std::endl;
        auto& cell_pool = BTQuant::FootprintCellPool::getInstance();

        std::cout << "Initial stats - Total: " << cell_pool.getTotalObjects()
                  << ", Free: " << cell_pool.getFreeObjects()
                  << ", Used: " << cell_pool.getUsedObjects() << std::endl;

        // Allocate some FootprintCell objects
        auto* cell1 = cell_pool.allocate();
        auto* cell2 = cell_pool.allocate();

        if (cell1) {
            cell1->priceLevel = 100.50;
            cell1->timeBucket = 1234567890;
            cell1->buyVolume = 600.0;
            cell1->sellVolume = 400.0;
            cell1->delta = 200.0;
            cell1->numBuyTrades = 15;
            cell1->numSellTrades = 10;
            cell1->maxSingleTrade = 50.0;
        }

        std::cout << "After allocation - Total: " << cell_pool.getTotalObjects()
                  << ", Free: " << cell_pool.getFreeObjects()
                  << ", Used: " << cell_pool.getUsedObjects() << std::endl;

        // Deallocate
        if (cell1) cell_pool.deallocate(cell1);

        std::cout << "After deallocation - Total: " << cell_pool.getTotalObjects()
                  << ", Free: " << cell_pool.getFreeObjects()
                  << ", Used: " << cell_pool.getUsedObjects() << std::endl;
    }

    std::cout << "\nMemory Pool tests completed successfully!" << std::endl;

    return 0;
}