#pragma once

#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <deque>
#include <unordered_map>

// HotSpine includes
// #include "../../tests/new/include/hotspine_reader.hpp"
// #include "../../tests/new/include/hotspine_layout.hpp"
// #include "../../tests/new/include/symbol_registry.hpp"
#include "stubs/hotspine_reader.hpp"
#include "stubs/hotspine_layout.hpp"
#include "stubs/symbol_registry.hpp"
namespace BTQuant {
namespace RenderEngine {

// Market data types for visualization
enum class MarketDataType {
    TRADE,
    ORDERBOOK,
    TICKER
};

struct PriceLevel {
    double price;
    double size;
};

struct MarketDataUpdate {
    MarketDataType type;
    uint32_t symbol_id;
    std::string exchange;
    std::string symbol;
    uint64_t timestamp_us;
    uint64_t local_timestamp_us;
    
    // Trade data
    double price = 0.0;
    double size = 0.0;
    std::string side;
    
    // Orderbook data
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
};

struct SymbolData {
    uint32_t symbol_id;
    std::string exchange;
    std::string symbol;
    std::string full_symbol;
    
    // Market data
    double last_price = 0.0;
    double price_change = 0.0;
    double price_change_percent = 0.0;
    double volume_24h = 0.0;
    double high_24h = 0.0;
    double low_24h = 0.0;
    double bid_price = 0.0;
    double ask_price = 0.0;
    double spread = 0.0;
    double spread_percent = 0.0;
    double momentum = 0.0;  // For heatmap visualization
    uint64_t last_update_time = 0;
};

struct PerformanceMetrics {
    double trades_per_second = 0.0;
    double orderbooks_per_second = 0.0;
    uint64_t total_trades_processed = 0;
    uint64_t total_orderbooks_processed = 0;
    double avg_processing_latency_us = 0.0;
    double buffer_utilization_percent = 0.0;
    bool connection_healthy = false;
};

// Internal symbol market data for tracking
struct SymbolMarketData {
    double last_price = 0.0;
    double price_change = 0.0;
    double price_change_percent = 0.0;
    double volume_24h = 0.0;
    double high_24h = 0.0;
    double low_24h = 0.0;
    double bid_price = 0.0;
    double ask_price = 0.0;
    double spread = 0.0;
    double spread_percent = 0.0;
    double momentum = 0.0;
    uint64_t last_update_time = 0;
    std::deque<double> momentum_history;  // For momentum calculation
};

/**
 * HotSpineDataBridge - Real-time data integration bridge
 * 
 * This class provides the main interface between HotSpine shared memory
 * and the Vulkan dashboard visualization system. It handles:
 * - Real-time data streaming from HotSpine
 * - Symbol registry integration
 * - Market data processing and calculations
 * - Performance monitoring
 * - Thread-safe data access
 */
class HotSpineDataBridge {
public:
    /**
     * Constructor
     * @param shm_name HotSpine shared memory name (e.g., "/btquant_hotspine")
     * @param symbols_file Path to symbol mappings file (e.g., "/dev/shm/btquant_symbols.json")
     */
    explicit HotSpineDataBridge(const std::string& shm_name = "/btquant_hotspine", 
                               const std::string& symbols_file = "/dev/shm/btquant_symbols.json");
    
    ~HotSpineDataBridge();
    
    // Non-copyable, non-movable
    HotSpineDataBridge(const HotSpineDataBridge&) = delete;
    HotSpineDataBridge& operator=(const HotSpineDataBridge&) = delete;
    HotSpineDataBridge(HotSpineDataBridge&&) = delete;
    HotSpineDataBridge& operator=(HotSpineDataBridge&&) = delete;
    
    /**
     * Start real-time data processing
     * @return true if started successfully
     */
    bool start();
    
    /**
     * Stop data processing
     */
    void stop();
    
    /**
     * Check if connected to HotSpine
     * @return true if connected and healthy
     */
    bool isConnected() const;
    
    /**
     * Get latest market data updates for visualization
     * @return vector of market data updates (trades, orderbooks)
     */
    std::vector<MarketDataUpdate> getLatestUpdates();
    
    /**
     * Get all available symbols with current market data
     * @return vector of symbol data for grid display
     */
    std::vector<SymbolData> getAllSymbols() const;
    
    /**
     * Get symbols for a specific exchange
     * @param exchange Exchange name (e.g., "binance", "okx")
     * @return vector of symbol data for the exchange
     */
    std::vector<SymbolData> getExchangeSymbols(const std::string& exchange) const;
    
    /**
     * Get list of available exchanges
     * @return vector of exchange names
     */
    std::vector<std::string> getAvailableExchanges() const;
    
    /**
     * Get performance metrics for monitoring
     * @return current performance metrics
     */
    PerformanceMetrics getPerformanceMetrics() const;
    
    /**
     * Reconnect to HotSpine (for recovery)
     * @return true if reconnected successfully
     */
    bool reconnect();
    
    /**
     * Reload symbol mappings from file
     */
    void reloadSymbolMappings();

private:
    // Configuration
    std::string shm_name_;
    std::string symbols_file_;
    static constexpr size_t MAX_BUFFER_SIZE = 10000;
    
    // HotSpine integration
    std::unique_ptr<HotSpine::HotSpineReader> hotspine_reader_;
    BTQuant::SymbolRegistry* symbol_registry_;
    
    // Threading
    std::atomic<bool> running_;
    std::thread data_thread_;
    std::thread perf_thread_;
    
    // Data buffers (thread-safe)
    mutable std::mutex data_mutex_;
    std::vector<HotSpine::HotTrade> trade_buffer_;
    std::vector<HotSpine::HotOrderbookSnapshot> orderbook_buffer_;
    
    // Symbol market data tracking
    mutable std::mutex symbol_data_mutex_;
    std::unordered_map<uint32_t, SymbolMarketData> symbol_market_data_;
    
    // Performance monitoring
    mutable std::mutex perf_mutex_;
    PerformanceMetrics performance_metrics_;
    std::atomic<uint64_t> last_trade_count_;
    std::atomic<uint64_t> last_orderbook_count_;
    std::atomic<uint64_t> total_trades_processed_;
    std::atomic<uint64_t> total_orderbooks_processed_;
    std::atomic<int64_t> data_latency_us_;
    std::atomic<double> update_frequency_hz_;
    std::chrono::high_resolution_clock::time_point last_update_time_;
    
    // Private methods
    void dataProcessingLoop();
    void performanceMonitoringLoop();
    void processTrade(const HotSpine::HotTrade& trade);
    void processOrderbook(const HotSpine::HotOrderbookSnapshot& orderbook);
    void updateSymbolMarketData(uint32_t symbol_id, double price, double size, uint64_t timestamp);
    void updateSymbolOrderbookData(uint32_t symbol_id, double bid_price, double ask_price, uint64_t timestamp);
    void printStatistics();
};

} // namespace RenderEngine
} // namespace BTQuant