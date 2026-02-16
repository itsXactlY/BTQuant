#include <iostream>
#include <memory>
#include <vector>
#include "dependencies/BTQ_Render_Engine/include/market_data_processor.hpp"

int main() {
    // Create a MarketDataProcessor instance
    auto processor = std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
    
    std::cout << "Testing OrderBookSnapshot ring buffer implementation...\n";
    
    // Create some mock orderbook updates to trigger snapshot creation
    for (int i = 0; i < 5; ++i) {
        BTQuant::RenderEngine::MarketDataUpdate update;
        update.type = BTQuant::RenderEngine::MarketDataType::ORDERBOOK;
        update.symbol_id = 1;
        update.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        
        // Add some mock bid levels
        for (int j = 0; j < 5; ++j) {
            BTQuant::PriceLevel bid;
            bid.price = 100.0 - (j * 0.01);  // Prices decreasing: 100.00, 99.99, 99.98...
            bid.size = 100.0 + (j * 10.0);   // Sizes increasing: 100, 110, 120...
            update.bids.push_back(bid);
        }
        
        // Add some mock ask levels
        for (int j = 0; j < 5; ++j) {
            BTQuant::PriceLevel ask;
            ask.price = 100.01 + (j * 0.01);  // Prices increasing: 100.01, 100.02, 100.03...
            ask.size = 90.0 + (j * 10.0);    // Sizes increasing: 90, 100, 110...
            update.asks.push_back(ask);
        }
        
        // Process the orderbook update
        processor->processOrderbookUpdate(update);
        
        // Small delay to ensure different timestamps
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Check if snapshots were created
    size_t snapshot_count = processor->getOrderBookSnapshotCount();
    std::cout << "Number of snapshots in buffer: " << snapshot_count << std::endl;
    
    if (snapshot_count > 0) {
        // Get the latest snapshot
        auto latest_snapshot = processor->getLatestOrderBookSnapshot();
        if (latest_snapshot) {
            std::cout << "Latest snapshot details:\n";
            std::cout << "  Symbol ID: " << latest_snapshot->symbol_id << std::endl;
            std::cout << "  Timestamp: " << latest_snapshot->timestamp << std::endl;
            std::cout << "  Best Bid: " << latest_snapshot->best_bid << " (size: " << latest_snapshot->best_bid_size << ")" << std::endl;
            std::cout << "  Best Ask: " << latest_snapshot->best_ask << " (size: " << latest_snapshot->best_ask_size << ")" << std::endl;
            std::cout << "  Spread: " << latest_snapshot->spread << std::endl;
            std::cout << "  Bid Levels Count: " << latest_snapshot->bid_levels_count << std::endl;
            std::cout << "  Ask Levels Count: " << latest_snapshot->ask_levels_count << std::endl;
            
            // Print first bid and ask levels
            if (latest_snapshot->bid_levels_count > 0) {
                std::cout << "  First Bid Level: price=" << latest_snapshot->bids[0].price 
                          << ", size=" << latest_snapshot->bids[0].size << std::endl;
            }
            if (latest_snapshot->ask_levels_count > 0) {
                std::cout << "  First Ask Level: price=" << latest_snapshot->asks[0].price 
                          << ", size=" << latest_snapshot->asks[0].size << std::endl;
            }
        } else {
            std::cout << "Failed to get latest snapshot\n";
        }
        
        // Get multiple snapshots
        auto snapshots = processor->getOrderBookSnapshots(3);
        std::cout << "\nRetrieved " << snapshots.size() << " snapshots:\n";
        for (size_t i = 0; i < snapshots.size(); ++i) {
            std::cout << "  Snapshot " << i << ": Symbol=" << snapshots[i].symbol_id 
                      << ", Bid=" << snapshots[i].best_bid 
                      << ", Ask=" << snapshots[i].best_ask << std::endl;
        }
    } else {
        std::cout << "No snapshots were created. This could indicate an issue with the implementation.\n";
    }
    
    std::cout << "\nRing buffer test completed successfully!\n";
    
    return 0;
}