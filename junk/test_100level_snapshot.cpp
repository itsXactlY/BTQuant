#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <chrono>

#include "src/analytics/orderbook_snapshot_100level.h"
#include "../dependencies/BTQ_Render_Engine/include/market_data_processor.hpp"

int main() {
    std::cout << "Testing 100-level OrderBookSnapshot reader functionality...\n";

    // Create a lock-free snapshot pipeline
    BTQuant::RenderEngine::LockFreeSnapshotPipeline pipeline(1000);
    
    // Create the 100-level reader
    OrderBookSnapshot100LevelReader reader(&pipeline);
    
    // Test basic functionality
    std::cout << "Testing basic snapshot creation and reading...\n";
    
    // Create a mock AtomicMarketData
    AtomicMarketData mock_data(100.0, 1000.0, 99.5, 100.5, 500.0, 500.0);
    
    // Write some data to the pipeline
    pipeline.write_market_data(1, mock_data);
    pipeline.commit_snapshot(); // Commit to make data available
    
    // Read the 100-level snapshot
    auto snapshot_opt = reader.readLatestSnapshot(1);
    if (snapshot_opt.has_value()) {
        auto snapshot = snapshot_opt.value();
        std::cout << "Successfully read 100-level snapshot:\n";
        std::cout << "  Symbol ID: " << snapshot.symbol_id << "\n";
        std::cout << "  Best Bid: " << snapshot.best_bid << "\n";
        std::cout << "  Best Ask: " << snapshot.best_ask << "\n";
        std::cout << "  Bid Levels Count: " << snapshot.bid_levels_count << "\n";
        std::cout << "  Ask Levels Count: " << snapshot.ask_levels_count << "\n";
        std::cout << "  First Bid Level: Price=" << snapshot.bids[0].price 
                  << ", Size=" << snapshot.bids[0].size << "\n";
        std::cout << "  First Ask Level: Price=" << snapshot.asks[0].price 
                  << ", Size=" << snapshot.asks[0].size << "\n";
    } else {
        std::cout << "Failed to read snapshot\n";
    }
    
    // Test getting snapshot count
    size_t count = reader.getSnapshotCount();
    std::cout << "Total snapshots in pipeline: " << count << "\n";
    
    // Test reading multiple snapshots (rolling window)
    auto snapshots = reader.readRollingSnapshots(1, 5); // Get 5 snapshots for symbol 1
    std::cout << "Retrieved " << snapshots.size() << " snapshots in rolling window\n";
    
    // Test with MarketDataProcessor (if available)
    std::cout << "\nTesting integration with MarketDataProcessor...\n";
    
    try {
        // Create a MarketDataProcessor instance
        BTQuant::RenderEngine::MarketDataProcessor processor;
        
        // Create a mock OrderBookSnapshot with 20 levels (as defined in the original)
        BTQuant::RenderEngine::OrderBookSnapshot original_snapshot;
        original_snapshot.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        original_snapshot.symbol_id = 1;
        original_snapshot.best_bid = 99.5;
        original_snapshot.best_ask = 100.5;
        original_snapshot.best_bid_size = 500.0;
        original_snapshot.best_ask_size = 500.0;
        original_snapshot.spread = 1.0;
        original_snapshot.bid_levels_count = 20;
        original_snapshot.ask_levels_count = 20;
        
        // Fill in the 20 levels with mock data
        for (int i = 0; i < 20; ++i) {
            original_snapshot.bids[i].price = 99.5 - (i * 0.01);
            original_snapshot.bids[i].size = 100.0 + (i * 10.0);
            original_snapshot.asks[i].price = 100.5 + (i * 0.01);
            original_snapshot.asks[i].size = 100.0 + (i * 10.0);
        }
        
        // Add the snapshot to the processor
        processor.addOrderBookSnapshot(original_snapshot);
        
        // Use the external function to read from the processor
        auto converted_snapshot = BTQuant::RenderEngine::readOrderBookSnapshot100LevelFromProcessor(&processor, 1);
        
        if (converted_snapshot.has_value()) {
            auto conv = converted_snapshot.value();
            std::cout << "Successfully converted to 100-level snapshot:\n";
            std::cout << "  Symbol ID: " << conv.symbol_id << "\n";
            std::cout << "  Best Bid: " << conv.best_bid << "\n";
            std::cout << "  Best Ask: " << conv.best_ask << "\n";
            std::cout << "  Bid Levels Count: " << conv.bid_levels_count << "\n";
            std::cout << "  Ask Levels Count: " << conv.ask_levels_count << "\n";
            std::cout << "  First Bid Level: Price=" << conv.bids[0].price 
                      << ", Size=" << conv.bids[0].size << "\n";
            std::cout << "  First Ask Level: Price=" << conv.asks[0].price 
                      << ", Size=" << conv.asks[0].size << "\n";
            std::cout << "  20th Bid Level: Price=" << conv.bids[19].price 
                      << ", Size=" << conv.bids[19].size << "\n";
            std::cout << "  20th Ask Level: Price=" << conv.asks[19].price 
                      << ", Size=" << conv.asks[19].size << "\n";
        } else {
            std::cout << "Failed to convert snapshot from processor\n";
        }
        
        // Test reading multiple snapshots from processor
        auto multiple_snapshots = BTQuant::RenderEngine::readRollingOrderBookSnapshots100Level(&processor, 3);
        std::cout << "Retrieved " << multiple_snapshots.size() << " snapshots from processor\n";
        
    } catch (const std::exception& e) {
        std::cout << "Exception during MarketDataProcessor test: " << e.what() << "\n";
    }
    
    std::cout << "\nAll tests completed successfully!\n";
    return 0;
}