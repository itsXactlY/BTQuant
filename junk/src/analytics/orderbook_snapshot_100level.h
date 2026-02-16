#ifndef PUBBTQUANT_ORDERBOOK_SNAPSHOT_100LEVEL_H
#define PUBBTQUANT_ORDERBOOK_SNAPSHOT_100LEVEL_H

#include <atomic>
#include <vector>
#include <memory>
#include <mutex>
#include <optional>
#include <algorithm>
#include <cstring>

#include "lockfreesnapshotpipeline.h"

// 100-level OrderBookSnapshot for high-resolution order book analysis
struct OrderBookSnapshot100Level {
    uint64_t timestamp;
    uint32_t symbol_id;
    double best_bid;
    double best_ask;
    double best_bid_size;
    double best_ask_size;
    double spread;
    double total_bid_volume;
    double total_ask_volume;
    uint32_t bid_levels_count;
    uint32_t ask_levels_count;

    // Fixed-size arrays for top 100 price levels (Standard Layout POD)
    static constexpr size_t MAX_LEVELS = 100;  // Top 100 levels per side

    struct Level {
        double price;
        double size;
    };

    Level bids[MAX_LEVELS];
    Level asks[MAX_LEVELS];

    // Constructor to initialize the struct
    OrderBookSnapshot100Level() : timestamp(0), symbol_id(0), best_bid(0.0), best_ask(0.0),
                                  best_bid_size(0.0), best_ask_size(0.0), spread(0.0),
                                  total_bid_volume(0.0), total_ask_volume(0.0),
                                  bid_levels_count(0), ask_levels_count(0) {
        // Initialize arrays to zero
        for (size_t i = 0; i < MAX_LEVELS; ++i) {
            bids[i] = Level{0.0, 0.0};
            asks[i] = Level{0.0, 0.0};
        }
    }
    
    // Copy constructor
    OrderBookSnapshot100Level(const OrderBookSnapshot100Level& other) {
        timestamp = other.timestamp;
        symbol_id = other.symbol_id;
        best_bid = other.best_bid;
        best_ask = other.best_ask;
        best_bid_size = other.best_bid_size;
        best_ask_size = other.best_ask_size;
        spread = other.spread;
        total_bid_volume = other.total_bid_volume;
        total_ask_volume = other.total_ask_volume;
        bid_levels_count = other.bid_levels_count;
        ask_levels_count = other.ask_levels_count;
        
        for (size_t i = 0; i < MAX_LEVELS; ++i) {
            bids[i] = other.bids[i];
            asks[i] = other.asks[i];
        }
    }
    
    // Assignment operator
    OrderBookSnapshot100Level& operator=(const OrderBookSnapshot100Level& other) {
        if (this != &other) {
            timestamp = other.timestamp;
            symbol_id = other.symbol_id;
            best_bid = other.best_bid;
            best_ask = other.best_ask;
            best_bid_size = other.best_bid_size;
            best_ask_size = other.best_ask_size;
            spread = other.spread;
            total_bid_volume = other.total_bid_volume;
            total_ask_volume = other.total_ask_volume;
            bid_levels_count = other.bid_levels_count;
            ask_levels_count = other.ask_levels_count;
            
            for (size_t i = 0; i < MAX_LEVELS; ++i) {
                bids[i] = other.bids[i];
                asks[i] = other.asks[i];
            }
        }
        return *this;
    }
};

// Reader class to read 100-level snapshots from the existing lock-free ring buffer
// This class interfaces with the existing LockFreeSnapshotPipeline to extract 100-level snapshots
class OrderBookSnapshot100LevelReader {
private:
    LockFreeSnapshotPipeline* pipeline_;
    
public:
    explicit OrderBookSnapshot100LevelReader(LockFreeSnapshotPipeline* pipeline) 
        : pipeline_(pipeline) {}
    
    // Read the latest 100-level snapshot for a specific symbol
    std::optional<OrderBookSnapshot100Level> readLatestSnapshot(uint32_t symbol_id) const {
        if (!pipeline_) {
            return std::nullopt;
        }
        
        AtomicMarketData data;
        if (!pipeline_->read_market_data_snapshot(symbol_id, data)) {
            return std::nullopt;
        }
        
        // Convert AtomicMarketData to 100-level snapshot
        OrderBookSnapshot100Level snapshot;
        snapshot.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            data.timestamp.load().time_since_epoch()).count();
        snapshot.symbol_id = symbol_id;
        snapshot.best_bid = data.bid_price.load();
        snapshot.best_ask = data.ask_price.load();
        snapshot.best_bid_size = data.bid_volume.load();
        snapshot.best_ask_size = data.ask_volume.load();
        snapshot.spread = snapshot.best_ask - snapshot.best_bid;
        
        // For a full 100-level snapshot, we would need more detailed order book data
        // Since AtomicMarketData only contains top-of-book data, we'll create a minimal snapshot
        // In a real implementation, this would interface with a more detailed order book structure
        snapshot.bid_levels_count = 1; // Only top level available in AtomicMarketData
        snapshot.ask_levels_count = 1; // Only top level available in AtomicMarketData
        
        snapshot.bids[0].price = data.bid_price.load();
        snapshot.bids[0].size = data.bid_volume.load();
        snapshot.asks[0].price = data.ask_price.load();
        snapshot.asks[0].size = data.ask_volume.load();
        
        return snapshot;
    }
    
    // Read multiple 100-level snapshots (rolling window) - this would need access to historical data
    std::vector<OrderBookSnapshot100Level> readRollingSnapshots(uint32_t symbol_id, size_t count) const {
        if (!pipeline_) {
            return std::vector<OrderBookSnapshot100Level>();
        }
        
        std::vector<OrderBookSnapshot100Level> snapshots;
        
        // Since the LockFreeSnapshotPipeline stores only the most recent data per symbol,
        // we can only return the latest snapshot multiple times
        // A real implementation would need a historical buffer of order book snapshots
        auto latest = readLatestSnapshot(symbol_id);
        if (latest.has_value()) {
            for (size_t i = 0; i < count; ++i) {
                snapshots.push_back(latest.value());
            }
        }
        
        return snapshots;
    }
    
    // Method to interface with the MarketDataProcessor's OrderBookSnapshot ring buffer
    // This would need to be called with access to the MarketDataProcessor instance
    template<typename MarketDataProcessorType>
    std::optional<OrderBookSnapshot100Level> readFromMarketDataProcessor(MarketDataProcessorType* processor, uint32_t symbol_id) const {
        if (!processor) {
            return std::nullopt;
        }
        
        // Try to get the latest order book snapshot from the processor
        // This assumes the processor has a method to retrieve snapshots
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
    
    // Get the total count of available snapshots in the underlying pipeline
    size_t getSnapshotCount() const {
        if (!pipeline_) {
            return 0;
        }
        
        auto stats = pipeline_->get_stats();
        return static_cast<size_t>(stats.total_updates);
    }
};

#endif // PUBBTQUANT_ORDERBOOK_SNAPSHOT_100LEVEL_H