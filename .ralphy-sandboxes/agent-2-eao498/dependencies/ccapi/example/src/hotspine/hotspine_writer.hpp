#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>

#include "hotspine_layout.hpp"
#include "market_data_types.h"

// Include SymbolRegistry for proper symbol ID mapping
#include "../../../tests/new/include/symbol_registry.hpp"

namespace HotSpine {

// HotSpine shared memory writer interface
class HotSpineWriter {
public:
    explicit HotSpineWriter(const std::string& shm_name = "/btquant_hotspine");
    ~HotSpineWriter();

    // Load symbol mappings from file
    bool loadSymbolMappings(const std::string& filepath);

    // Write trade data to HotSpine
    bool writeTrade(const MarketData::Trade& trade);
    
    // Write batch of trades to HotSpine (more efficient)
    bool writeTrades(const std::vector<MarketData::Trade>& trades);
    
    // Write orderbook data to HotSpine
    bool writeOrderbook(const MarketData::OrderbookSnapshot& ob, const std::vector<HotSpine::HotOrderbookLevel>& bids, const std::vector<HotSpine::HotOrderbookLevel>& asks);
    
    // Get statistics
    uint64_t getTradesWritten() const { return trades_written_; }
    uint64_t getWriteErrors() const { return write_errors_; }
    
    // Check if writer is healthy
    bool isHealthy() const;
     
    // Get detailed statistics
    std::string getDetailedStats() const;
     
    // Batching control
    void setBatchingEnabled(bool enabled) { batching_enabled_ = enabled; }
    void setBatchSize(size_t size) { batch_size_ = size; }
    void flushBatch();

private:
    std::string shm_name_;
    int shm_fd_{-1};
    void* shm_ptr_{nullptr};
    SharedMemoryHeader* header_{nullptr};
    HotTrade* trades_buffer_{nullptr};
    HotOrderbookSnapshot* orderbooks_buffer_{nullptr};
    
    mutable std::atomic<uint64_t> trades_written_{0};
    mutable std::atomic<uint64_t> write_errors_{0};
    mutable std::mutex stats_mutex_;
    
    // For batching writes
    std::vector<MarketData::Trade> batch_buffer_;
    std::mutex batch_mutex_;
    std::atomic<bool> batching_enabled_{true};
    std::atomic<size_t> batch_size_{100};
    
    bool attachToSharedMemory();
    bool detachFromSharedMemory();
    uint32_t getSymbolId(const std::string& exchange, 
                        const std::string& symbol,
                        const std::string& market_type) const;
    
    static uint64_t getCurrentTimestampMicros();
};

} // namespace HotSpine
