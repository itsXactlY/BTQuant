#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <chrono>

#include "dependencies/BTQ_Render_Engine/include/market_data_processor.hpp"

int main() {
    std::cout << "Testing MarketDataProcessor atomic snapshot functionality..." << std::endl;
    
    // Create a MarketDataProcessor instance
    auto processor = std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
    
    // Create a simple market data update
    BTQuant::RenderEngine::MarketDataUpdate update;
    update.symbol_id = 1;  // Use a valid symbol ID within our array bounds
    update.type = BTQuant::RenderEngine::MarketDataType::ORDERBOOK;
    update.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    
    // Add some bids and asks
    BTQuant::RenderEngine::PriceLevel bid_level;
    bid_level.price = 100.0;
    bid_level.size = 10.0;
    update.bids.push_back(bid_level);
    
    BTQuant::RenderEngine::PriceLevel ask_level;
    ask_level.price = 101.0;
    ask_level.size = 15.0;
    update.asks.push_back(ask_level);
    
    // Process the update
    processor->processOrderbookUpdate(update);
    
    // Wait a moment for the update to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Try to get the atomic snapshot
    auto snapshot_opt = processor->get_atomic_snapshot(1);
    
    if (snapshot_opt.has_value()) {
        auto snapshot = snapshot_opt.value();
        std::cout << "Successfully retrieved atomic snapshot!" << std::endl;
        std::cout << "Symbol ID: " << snapshot.symbol_id << std::endl;
        std::cout << "Best Bid: " << snapshot.best_bid << std::endl;
        std::cout << "Best Ask: " << snapshot.best_ask << std::endl;
        std::cout << "Bid Size: " << snapshot.best_bid_size << std::endl;
        std::cout << "Ask Size: " << snapshot.best_ask_size << std::endl;
        std::cout << "Spread: " << snapshot.spread << std::endl;
        std::cout << "Mid Price: " << snapshot.mid_price << std::endl;
    } else {
        std::cout << "Failed to retrieve atomic snapshot!" << std::endl;
        return 1;
    }
    
    // Test with a trade update too
    BTQuant::RenderEngine::MarketDataUpdate trade_update;
    trade_update.symbol_id = 1;
    trade_update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
    trade_update.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    trade_update.price = 100.5;
    trade_update.size = 5.0;
    trade_update.side = "buy";
    
    processor->processTradeUpdate(trade_update);
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Get the updated snapshot
    auto updated_snapshot_opt = processor->get_atomic_snapshot(1);
    
    if (updated_snapshot_opt.has_value()) {
        auto updated_snapshot = updated_snapshot_opt.value();
        std::cout << "\nAfter trade update:" << std::endl;
        std::cout << "Last Trade Price: " << updated_snapshot.last_trade_price << std::endl;
        std::cout << "Last Trade Size: " << updated_snapshot.last_trade_size << std::endl;
    }
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}