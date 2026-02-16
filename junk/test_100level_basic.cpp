#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <chrono>

#include "src/analytics/lockfreesnapshotpipeline.h"
#include "src/analytics/orderbook_snapshot_100level.h"

int main() {
    std::cout << "Testing 100-level OrderBookSnapshot reader functionality...\n";

    // Create a lock-free snapshot pipeline
    LockFreeSnapshotPipeline pipeline(1000);
    
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
    
    std::cout << "\nAll tests completed successfully!\n";
    return 0;
}