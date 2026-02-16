#include "orderbook_snapshot_100level.h"
#include "../../dependencies/BTQ_Render_Engine/include/market_data_processor.hpp"

// Implementation of the OrderBookSnapshot100LevelReader methods that interface with MarketDataProcessor
namespace BTQuant {
namespace RenderEngine {

// Specialized method to read from MarketDataProcessor
std::optional<OrderBookSnapshot100Level> readOrderBookSnapshot100LevelFromProcessor(
    MarketDataProcessor* processor, uint32_t symbol_id) {
    
    if (!processor) {
        return std::nullopt;
    }
    
    // Get the latest order book snapshot from the processor
    auto opt_snapshot = processor->getLatestOrderBookSnapshot();
    if (!opt_snapshot.has_value()) {
        return std::nullopt;
    }
    
    auto original_snapshot = opt_snapshot.value();
    
    // Convert the original snapshot (which has 20 levels) to 100-level
    OrderBookSnapshot100Level converted_snapshot;
    converted_snapshot.timestamp = original_snapshot.timestamp;
    converted_snapshot.symbol_id = original_snapshot.symbol_id;
    converted_snapshot.best_bid = original_snapshot.best_bid;
    converted_snapshot.best_ask = original_snapshot.best_ask;
    converted_snapshot.best_bid_size = original_snapshot.best_bid_size;
    converted_snapshot.best_ask_size = original_snapshot.best_ask_size;
    converted_snapshot.spread = original_snapshot.spread;
    converted_snapshot.total_bid_volume = original_snapshot.total_bid_volume;
    converted_snapshot.total_ask_volume = original_snapshot.total_ask_volume;
    converted_snapshot.bid_levels_count = std::min(original_snapshot.bid_levels_count, 
                                                 static_cast<uint32_t>(OrderBookSnapshot100Level::MAX_LEVELS));
    converted_snapshot.ask_levels_count = std::min(original_snapshot.ask_levels_count, 
                                                 static_cast<uint32_t>(OrderBookSnapshot100Level::MAX_LEVELS));
    
    // Copy bid levels (up to 100, but original only has 20)
    for (uint32_t i = 0; i < converted_snapshot.bid_levels_count; ++i) {
        converted_snapshot.bids[i].price = original_snapshot.bids[i].price;
        converted_snapshot.bids[i].size = original_snapshot.bids[i].size;
    }
    
    // Copy ask levels (up to 100, but original only has 20)
    for (uint32_t i = 0; i < converted_snapshot.ask_levels_count; ++i) {
        converted_snapshot.asks[i].price = original_snapshot.asks[i].price;
        converted_snapshot.asks[i].size = original_snapshot.asks[i].size;
    }
    
    // Zero out remaining levels if there are fewer than 100
    for (uint32_t i = converted_snapshot.bid_levels_count; i < OrderBookSnapshot100Level::MAX_LEVELS; ++i) {
        converted_snapshot.bids[i] = {0.0, 0.0};
    }
    for (uint32_t i = converted_snapshot.ask_levels_count; i < OrderBookSnapshot100Level::MAX_LEVELS; ++i) {
        converted_snapshot.asks[i] = {0.0, 0.0};
    }
    
    return converted_snapshot;
}

// Get multiple 100-level snapshots from the processor (rolling window)
std::vector<OrderBookSnapshot100Level> readRollingOrderBookSnapshots100Level(
    MarketDataProcessor* processor, size_t count) {
    
    if (!processor) {
        return std::vector<OrderBookSnapshot100Level>();
    }
    
    // Get snapshots from the processor
    auto original_snapshots = processor->getOrderBookSnapshots(count);
    
    std::vector<OrderBookSnapshot100Level> converted_snapshots;
    converted_snapshots.reserve(original_snapshots.size());
    
    for (const auto& original_snapshot : original_snapshots) {
        OrderBookSnapshot100Level converted_snapshot;
        converted_snapshot.timestamp = original_snapshot.timestamp;
        converted_snapshot.symbol_id = original_snapshot.symbol_id;
        converted_snapshot.best_bid = original_snapshot.best_bid;
        converted_snapshot.best_ask = original_snapshot.best_ask;
        converted_snapshot.best_bid_size = original_snapshot.best_bid_size;
        converted_snapshot.best_ask_size = original_snapshot.best_ask_size;
        converted_snapshot.spread = original_snapshot.spread;
        converted_snapshot.total_bid_volume = original_snapshot.total_bid_volume;
        converted_snapshot.total_ask_volume = original_snapshot.total_ask_volume;
        converted_snapshot.bid_levels_count = std::min(original_snapshot.bid_levels_count, 
                                                     static_cast<uint32_t>(OrderBookSnapshot100Level::MAX_LEVELS));
        converted_snapshot.ask_levels_count = std::min(original_snapshot.ask_levels_count, 
                                                     static_cast<uint32_t>(OrderBookSnapshot100Level::MAX_LEVELS));
        
        // Copy bid levels (up to 100, but original only has 20)
        for (uint32_t i = 0; i < converted_snapshot.bid_levels_count; ++i) {
            converted_snapshot.bids[i].price = original_snapshot.bids[i].price;
            converted_snapshot.bids[i].size = original_snapshot.bids[i].size;
        }
        
        // Copy ask levels (up to 100, but original only has 20)
        for (uint32_t i = 0; i < converted_snapshot.ask_levels_count; ++i) {
            converted_snapshot.asks[i].price = original_snapshot.asks[i].price;
            converted_snapshot.asks[i].size = original_snapshot.asks[i].size;
        }
        
        // Zero out remaining levels if there are fewer than 100
        for (uint32_t i = converted_snapshot.bid_levels_count; i < OrderBookSnapshot100Level::MAX_LEVELS; ++i) {
            converted_snapshot.bids[i] = {0.0, 0.0};
        }
        for (uint32_t i = converted_snapshot.ask_levels_count; i < OrderBookSnapshot100Level::MAX_LEVELS; ++i) {
            converted_snapshot.asks[i] = {0.0, 0.0};
        }
        
        converted_snapshots.push_back(converted_snapshot);
    }
    
    return converted_snapshots;
}

} // namespace RenderEngine
} // namespace BTQuant