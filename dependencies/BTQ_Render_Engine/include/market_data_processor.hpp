#pragma once

#include <immintrin.h>  // For SIMD intrinsics

#include <atomic>
#include <chrono>
#include <coroutine>
#include <cstdint>
#include <deque>
#include <execution>
#include <experimental/simd>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hotspine_data_bridge.hpp"
#include "data/data_types.hpp"
#include "cache_manager.hpp"
// Lock-free queue (header-only, fetched by CMake)
#include "concurrentqueue.h"
// Lock-free hash map (assuming available or use std::unordered_map with atomic
// ops) #include <folly/AtomicHashMap.h> // Example, or implement custom
// lock-free map

namespace BTQuant {
namespace RenderEngine {

// Market data update type
enum class MarketDataType { TRADE, ORDERBOOK };

// Market data update structure
struct MarketDataUpdate {
  MarketDataType type;
  uint32_t symbol_id;
  uint64_t timestamp;
  double price;
  double size;
  std::string side;
  std::vector<PriceLevel> bids;
  std::vector<PriceLevel> asks;
};

// Trade data for analytics
struct TradeData {
  std::string symbol;
  uint32_t symbol_id = 0;
  uint64_t timestamp;
  double price;
  double size;
  bool is_buy;
};

// Orderbook data for analytics
struct OrderbookData {
  std::string symbol;
  uint32_t symbol_id = 0;
  uint64_t timestamp;
  std::vector<PriceLevel> bids;
  std::vector<PriceLevel> asks;
  double spread;
  double spread_percent;
  double bid_depth;
  double ask_depth;
  double total_depth;
  double imbalance;  // (bid_depth - ask_depth) / total_depth
};


// Indicator cache entry
struct IndicatorCacheEntry {
  uint64_t timestamp;
  std::vector<double> values;
};

// Caching structure for indicators
struct IndicatorCache {
  std::unordered_map<std::string, IndicatorCacheEntry> cache;
  std::mutex mutex;
};

// Comprehensive symbol analytics
struct SymbolAnalytics {
  // Basic data
  uint32_t symbol_id = 0;
  uint64_t last_update_time = 0;

  // OHLCV candle data for multiple time frames
  std::unordered_map<TimeFrame, std::vector<OHLCVCandle>> candles;
  std::unordered_map<TimeFrame, OHLCVCandle> current_candles;  // In-progress candles

  // Trade analytics
  std::vector<TradeData> recent_trades;
  uint64_t trade_count = 0;
  double last_trade_price = 0.0;
  double last_trade_size = 0.0;
  uint64_t last_trade_time = 0;

  // Volume analytics
  double volume_1m = 0.0;
  double volume_5m = 0.0;
  double volume_15m = 0.0;
  double buy_volume = 0.0;
  double sell_volume = 0.0;
  uint64_t buy_count = 0;
  uint64_t sell_count = 0;
  double buy_sell_ratio = 0.5;  // 0.5 = balanced, >0.5 = more buying

  // Price analytics
  double vwap = 0.0;               // Volume Weighted Average Price
  double vwap_deviation = 0.0;     // Current price deviation from VWAP (%)
  double momentum = 0.0;           // Price momentum (%)
  double momentum_strength = 0.0;  // Volatility of momentum
  double price_min = 0.0;
  double price_max = 0.0;
  double price_position = 0.0;  // Position between min/max (0-100%)

  // Volatility analytics
  double volatility = 0.0;    // Annualized volatility
  double sharpe_ratio = 0.0;  // Return/volatility ratio

  // Trade size analytics
  double avg_trade_size = 0.0;
  uint64_t large_trade_count = 0;  // Trades > 2x average size

  // L2 Orderbook Aggregation (Price -> Size)
  // We use functional comparators: std::greater for Bids (Desc), std::less for
  // Asks (Asc)
  std::map<double, double, std::greater<double>> consolidated_bids;
  std::map<double, double, std::less<double>> consolidated_asks;

  // Orderbook analytics
  std::deque<OrderbookData> recent_orderbooks;
  double current_spread = 0.0;
  double current_spread_percent = 0.0;
  double avg_spread = 0.0;
  double avg_spread_percent = 0.0;
  double current_imbalance = 0.0;
  double avg_imbalance = 0.0;
  double market_depth = 0.0;

  // VWAP incremental calculation fields
  double running_total_price_volume = 0.0;
  double running_total_volume = 0.0;

  // Momentum calculation fields
  std::deque<double> momentum_prices;
  size_t momentum_window_size = 20;  // Default window size

  // Volatility calculation fields
  std::deque<double> log_returns;
  size_t volatility_window_size = 30;  // Default window size

  // Volume Profile (Session)
  std::map<double, VolumeProfileLevel> session_volume_profile;
};

// Performance metrics for the processor
struct ProcessorPerformanceMetrics {
  uint64_t total_trades_processed = 0;
  uint64_t total_orderbooks_processed = 0;
  std::chrono::high_resolution_clock::time_point last_update_time;
  double trades_per_second = 0.0;
  double orderbooks_per_second = 0.0;
  double avg_latency_ms = 0.0;
  double processing_latency_us = 0.0;
};

// Ranking criteria for symbol sorting
enum class RankingCriteria { VOLUME, MOMENTUM, VOLATILITY, SPREAD, IMBALANCE };

// Symbol ranking result
struct SymbolRanking {
  uint32_t symbol_id;
  double value;
  std::string label;
};

// Market summary statistics
struct MarketSummary {
  size_t total_symbols = 0;
  size_t active_symbols = 0;
  size_t trending_up = 0;
  size_t trending_down = 0;
  double avg_volume = 0.0;
  double avg_momentum = 0.0;
  double avg_volatility = 0.0;
  std::chrono::high_resolution_clock::time_point last_update;
};

// C++26 Push Notification Types
enum class NotificationType { TRADE, ORDERBOOK, CANDLE, ANALYTICS };

// Callback signature for push notifications
using SymbolCallback = std::function<void(uint32_t symbol_id, NotificationType type)>;

// Subscription entry for reactive updates
struct Subscription {
  uint64_t id;
  uint32_t symbol_id;       // 0 = all symbols
  NotificationType filter;  // Which events to receive
  SymbolCallback callback;
};

/**
 * MarketDataProcessor - Advanced market data analytics engine
 *
 * This class processes real-time market data updates and calculates
 * comprehensive analytics including:
 * - VWAP (Volume Weighted Average Price)
 * - Price momentum and volatility
 * - Spread analysis and market depth
 * - Trading volume patterns
 * - Market microstructure metrics
 * - OHLCV candle aggregation for multiple time frames
 *
 * Optimizations:
 * - Caching of indicator calculations
 * - Multithreaded processing
 * - Reduced data copying
 * - Support for sub-second time frames
 */
class MarketDataProcessor {
 public:
  MarketDataProcessor();
  ~MarketDataProcessor();

  // Non-copyable, non-movable
  MarketDataProcessor(const MarketDataProcessor&) = delete;
  MarketDataProcessor& operator=(const MarketDataProcessor&) = delete;
  MarketDataProcessor(MarketDataProcessor&&) = delete;
  MarketDataProcessor& operator=(MarketDataProcessor&&) = delete;

  /**
   * Process a trade update (asynchronous)
   * @param update Market data update containing trade information
   */
  void processTradeUpdate(const MarketDataUpdate& update);

  /**
   * Process multiple trade updates in a single batch (synchronous)
   * Designed for high-performance initial loading and HFT bursts.
   * @param updates Vector of market data updates
   */
  void processTradeUpdates(const std::vector<MarketDataUpdate>& updates);

  /**
   * Process an orderbook update (asynchronous)
   * @param update Market data update containing orderbook information
   */
  void processOrderbookUpdate(const MarketDataUpdate& update);

  /**
   * Get comprehensive analytics for a symbol (thread-safe)
   * @param symbol_id Symbol ID to get analytics for
   * @return Symbol analytics data
   */
  SymbolAnalytics getSymbolAnalytics(uint32_t symbol_id) const;

  /**
   * Get list of all active symbols (thread-safe)
   * @return Vector of symbol IDs that have recent data
   */
  std::vector<uint32_t> getActiveSymbols() const;

  /**
   * Clear all historical data (e.g. for resets)
   */
  void clearHistory();

  /**
   * Get performance metrics for the processor (thread-safe)
   * @return Current performance metrics
   */
  ProcessorPerformanceMetrics getPerformanceMetrics() const;

  /**
   * Get symbol rankings based on criteria (thread-safe)
   * @param criteria Ranking criteria (volume, momentum, etc.)
   * @param limit Maximum number of results (0 = no limit)
   * @return Vector of ranked symbols
   */
  std::vector<SymbolRanking> getRankings(RankingCriteria criteria, size_t limit = 0) const;

  /**
   * Get OHLCV candles for a symbol and time frame (cached, thread-safe)
   * @param symbol_id Symbol ID to get candles for
   * @param timeframe Time frame of the candles
   * @return Vector of OHLCV candles
   */
  std::vector<OHLCVCandle> getCandles(uint32_t symbol_id, TimeFrame timeframe) const;

  /**
   * Get current (in-progress) candle for a symbol and time frame (thread-safe)
   * @param symbol_id Symbol ID to get current candle for
   * @param timeframe Time frame of the candle
   * @return Current OHLCV candle if available, empty optional otherwise
   */
  std::optional<OHLCVCandle> getCurrentCandle(uint32_t symbol_id, TimeFrame timeframe) const;

  /**
   * Get latest orderbook data for a symbol (thread-safe)
   * @param symbol_id Symbol ID to get orderbook for
   * @return Latest OrderbookData if available, empty optional otherwise
   */
  std::optional<OrderbookData> getOrderbookData(uint32_t symbol_id) const;

  std::vector<OrderbookData> getHistoricalOrderbooks(uint32_t symbol_id, size_t count) const;
  std::vector<VolumeProfileLevel> getVolumeProfile(uint32_t symbol_id, TimeFrame timeframe) const;

  // Methods required by multi_vwap_panel
  bool hasData() const {
    // Check if we have any active symbols with data
    return !getActiveSymbols().empty();
  }

  std::vector<OHLCVCandle> getChartData() const {
    // Return chart data for the first active symbol, or empty vector if none
    auto active_symbols = getActiveSymbols();
    if (!active_symbols.empty()) {
      // Return candles for the first symbol using the smallest timeframe
      return getCandles(active_symbols[0], TimeFrame::TF_1MIN);
    }
    return std::vector<OHLCVCandle>();
  }

  /**
   * Get market summary statistics (thread-safe)
   * @return Market-wide summary data
   */
  MarketSummary getMarketSummary() const;

  /**
   * Helper method to convert time frame to microseconds
   * @param timeframe Time frame to convert
   * @return Duration in microseconds
   */
  static uint64_t getTimeFrameDuration(TimeFrame timeframe);

  /**
   * Clear analytics data for a specific symbol
   * @param symbol_id Symbol ID to clear
   */
  void clearSymbolData(uint32_t symbol_id);

  /**
   * Clear all analytics data
   */
  void clearAllData();

  /**
   * Configuration methods
   */
  void setVWAPWindow(size_t window_size);
  void setMomentumWindow(size_t window_size);
  void setVolatilityWindow(size_t window_size);

  /**
   * Indicator cache methods
   */
  void clearIndicatorCache(uint32_t symbol_id, const std::string& indicator_name);
  void clearAllIndicatorCaches();

  /**
   * Parallel processing configuration
   */
  void setParallelProcessingEnabled(bool enabled);
  bool isParallelProcessingEnabled() const;

  /**
   * C++26 Push Notification Subscription System
   * Panels subscribe to receive immediate callbacks when data updates
   */

  /**
   * Subscribe to symbol updates (lock-free)
   * @param symbol_id Symbol to subscribe to (0 = all symbols)
   * @param filter Which notification types to receive
   * @param callback Function to call when data updates
   * @return Subscription ID for later unsubscription
   */
  uint64_t subscribe(uint32_t symbol_id, NotificationType filter, SymbolCallback callback);

  /**
   * Unsubscribe from updates (lock-free)
   * @param subscription_id ID returned from subscribe()
   */
  void unsubscribe(uint64_t subscription_id);

 private:
  // Configuration parameters
  size_t vwap_window_size_;
  size_t momentum_window_size_;
  size_t volatility_window_size_;
  size_t spread_analysis_window_;
  bool parallel_processing_enabled_;

  // Indicator caching (Global for now, protected by its own internal mutexes
  // per entry or global lock in methods)
  mutable std::unordered_map<uint32_t, IndicatorCache> indicator_caches_;
  mutable std::mutex indicator_cache_mutex_;

  // Data storage (Sharded)
  struct Shard {
    mutable std::shared_mutex mutex;
    std::unordered_map<uint32_t, SymbolAnalytics> data;
    // Padding to prevent false sharing cache line contention (64 bytes)
    char padding[64];
  };

  // 16 Shards should be sufficient for thousands of symbols
  // SymbolID % 16 -> Shard Index
  static constexpr size_t NUM_SHARDS = 16;
  std::vector<std::unique_ptr<Shard>> shards_;

  // Lock-free Ingestion Queue
  // Using moodycamel::ConcurrentQueue for high-throughput non-blocking
  // ingestion
  moodycamel::ConcurrentQueue<MarketDataUpdate> update_queue_;

  // Cache manager for storing processed data
  std::shared_ptr<RenderEngine::CacheManager> cache_manager_;

  // Worker threads (C++20 jthread automatically joins on destruction)
  std::vector<std::jthread> workers_;
  std::atomic<bool> running_{true};

  // Performance metrics (Atomic is fine)
  struct AtomicPerformanceMetrics {
    std::atomic<uint64_t> total_trades_processed{0};
    std::atomic<uint64_t> total_orderbooks_processed{0};
    std::atomic<std::chrono::high_resolution_clock::time_point::rep> last_update_time{0};
    std::atomic<double> trades_per_second{0.0};
    std::atomic<double> orderbooks_per_second{0.0};
    std::atomic<double> avg_latency_ms{0.0};
    std::atomic<double> processing_latency_us{0.0};

    ProcessorPerformanceMetrics toNonAtomic() const {
      ProcessorPerformanceMetrics result;
      result.total_trades_processed = total_trades_processed.load();
      result.total_orderbooks_processed = total_orderbooks_processed.load();
      result.last_update_time = std::chrono::high_resolution_clock::time_point(
          std::chrono::high_resolution_clock::duration(last_update_time.load()));
      result.trades_per_second = trades_per_second.load();
      result.orderbooks_per_second = orderbooks_per_second.load();
      result.avg_latency_ms = avg_latency_ms.load();
      result.processing_latency_us = processing_latency_us.load();
      return result;
    }

    void fromNonAtomic(const ProcessorPerformanceMetrics& other) {
      total_trades_processed.store(other.total_trades_processed);
      total_orderbooks_processed.store(other.total_orderbooks_processed);
      last_update_time.store(other.last_update_time.time_since_epoch().count());
      trades_per_second.store(other.trades_per_second);
      orderbooks_per_second.store(other.orderbooks_per_second);
      avg_latency_ms.store(other.avg_latency_ms);
      processing_latency_us.store(other.processing_latency_us);
    }
  };

  mutable AtomicPerformanceMetrics performance_metrics_;

  // Delta trackers for rate calculation
  mutable std::atomic<uint64_t> trade_count_delta_{0};
  mutable std::atomic<uint64_t> book_count_delta_{0};
  mutable uint64_t last_performance_update_us_ = 0;

  // Private calculation methods
  void updateVWAP(SymbolAnalytics& symbol_data);
  void updateMomentum(SymbolAnalytics& symbol_data);
  void updateVolatility(SymbolAnalytics& symbol_data);
  void updateTradingMetrics(SymbolAnalytics& symbol_data, const TradeData& trade);
  void updateSpreadAnalysis(SymbolAnalytics& symbol_data);
  void processTradeIncrementally(SymbolAnalytics& symbol_data, const TradeData& trade);

  // OHLCV aggregation methods
  void updateCandles(SymbolAnalytics& symbol_data, const TradeData& trade);
  void updateCandleForTimeframe(SymbolAnalytics& symbol_data, const TradeData& trade,
                                TimeFrame timeframe);
  OHLCVCandle createNewCandle(uint64_t timestamp, double price, double size) const;
  bool isTradeInCurrentCandle(const OHLCVCandle& candle, uint64_t trade_timestamp,
                              TimeFrame timeframe) const;
  void updateCandle(OHLCVCandle& candle, double price, double size) const;

  // Helper methods
  double calculateMarketDepth(const std::vector<PriceLevel>& levels) const;
  double calculateVolumeInWindow(const std::vector<TradeData>& trades, uint64_t window_us) const;

  // Worker Loop
  void processQueueLoop();
  void processUpdate(const MarketDataUpdate& update);

  // Helper to get shard for a symbol
  Shard& getShard(uint32_t symbol_id) const { return *shards_[symbol_id % NUM_SHARDS]; }

  // C++26 Lock-free subscriber access using copy-on-read pattern
  // Writes (subscribe/unsubscribe) are rare, reads (notify) are frequent
  // Take mutex only for write, copy shared_ptr for lock-free iteration
  using SubscriberList = std::vector<Subscription>;
  mutable std::mutex subscribers_mutex_;
  std::shared_ptr<SubscriberList> subscribers_{std::make_shared<SubscriberList>()};
  std::atomic<uint64_t> next_subscription_id_{1};

  // Notify all relevant subscribers (called from worker threads)
  void notifySubscribers(uint32_t symbol_id, NotificationType type) const;
};

}  // namespace RenderEngine
}  // namespace BTQuant
