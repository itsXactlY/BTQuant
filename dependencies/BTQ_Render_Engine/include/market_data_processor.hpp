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
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cache_manager.hpp"
#include "data/data_types.hpp"
#include "hotspine_data_bridge.hpp"
#include "render_snapshot.hpp"
#include "threading/double_buffered_state.hpp"
#include "triple_buffer.hpp"
#include "data/orderbook_snapshot_manager.hpp"
// Lock-free queue (header-only, fetched by CMake)
#include "concurrentqueue.h"

// Forward declaration to avoid including heavy headers
namespace Analytics {
class ClusterEngine;
}

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


// Atomic structure for indicator values that can be written by background threads
// and read by the UI thread atomically
struct AtomicIndicatorValues {
  std::atomic<double> sma_9{0.0};
  std::atomic<double> sma_10{0.0};
  std::atomic<double> sma_20{0.0};
  std::atomic<double> sma_50{0.0};
  std::atomic<double> sma_200{0.0};
  
  std::atomic<double> ema_9{0.0};
  std::atomic<double> ema_10{0.0};
  std::atomic<double> ema_21{0.0};
  std::atomic<double> ema_50{0.0};
  std::atomic<double> ema_200{0.0};
  
  std::atomic<double> rsi{0.0};
  std::atomic<double> macd_line{0.0};
  std::atomic<double> macd_signal{0.0};
  std::atomic<double> macd_histogram{0.0};
  std::atomic<double> bollinger_upper{0.0};
  std::atomic<double> bollinger_middle{0.0};
  std::atomic<double> bollinger_lower{0.0};
  std::atomic<double> stochastic_k{0.0};
  std::atomic<double> stochastic_d{0.0};
  std::atomic<double> atr{0.0};
  
  std::atomic<uint64_t> last_updated{0};  // Timestamp of last update
  
  // Constructor
  AtomicIndicatorValues() = default;
  
  // Copy constructor (loads values atomically)
  AtomicIndicatorValues(const AtomicIndicatorValues& other) {
    sma_9.store(other.sma_9.load(std::memory_order_relaxed), std::memory_order_relaxed);
    sma_10.store(other.sma_10.load(std::memory_order_relaxed), std::memory_order_relaxed);
    sma_20.store(other.sma_20.load(std::memory_order_relaxed), std::memory_order_relaxed);
    sma_50.store(other.sma_50.load(std::memory_order_relaxed), std::memory_order_relaxed);
    sma_200.store(other.sma_200.load(std::memory_order_relaxed), std::memory_order_relaxed);
    
    ema_9.store(other.ema_9.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ema_10.store(other.ema_10.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ema_21.store(other.ema_21.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ema_50.store(other.ema_50.load(std::memory_order_relaxed), std::memory_order_relaxed);
    ema_200.store(other.ema_200.load(std::memory_order_relaxed), std::memory_order_relaxed);
    
    rsi.store(other.rsi.load(std::memory_order_relaxed), std::memory_order_relaxed);
    macd_line.store(other.macd_line.load(std::memory_order_relaxed), std::memory_order_relaxed);
    macd_signal.store(other.macd_signal.load(std::memory_order_relaxed), std::memory_order_relaxed);
    macd_histogram.store(other.macd_histogram.load(std::memory_order_relaxed), std::memory_order_relaxed);
    bollinger_upper.store(other.bollinger_upper.load(std::memory_order_relaxed), std::memory_order_relaxed);
    bollinger_middle.store(other.bollinger_middle.load(std::memory_order_relaxed), std::memory_order_relaxed);
    bollinger_lower.store(other.bollinger_lower.load(std::memory_order_relaxed), std::memory_order_relaxed);
    stochastic_k.store(other.stochastic_k.load(std::memory_order_relaxed), std::memory_order_relaxed);
    stochastic_d.store(other.stochastic_d.load(std::memory_order_relaxed), std::memory_order_relaxed);
    atr.store(other.atr.load(std::memory_order_relaxed), std::memory_order_relaxed);
    
    last_updated.store(other.last_updated.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }
  
  // Assignment operator
  AtomicIndicatorValues& operator=(const AtomicIndicatorValues& other) {
    if (this != &other) {
      sma_9.store(other.sma_9.load(std::memory_order_relaxed), std::memory_order_relaxed);
      sma_10.store(other.sma_10.load(std::memory_order_relaxed), std::memory_order_relaxed);
      sma_20.store(other.sma_20.load(std::memory_order_relaxed), std::memory_order_relaxed);
      sma_50.store(other.sma_50.load(std::memory_order_relaxed), std::memory_order_relaxed);
      sma_200.store(other.sma_200.load(std::memory_order_relaxed), std::memory_order_relaxed);
      
      ema_9.store(other.ema_9.load(std::memory_order_relaxed), std::memory_order_relaxed);
      ema_10.store(other.ema_10.load(std::memory_order_relaxed), std::memory_order_relaxed);
      ema_21.store(other.ema_21.load(std::memory_order_relaxed), std::memory_order_relaxed);
      ema_50.store(other.ema_50.load(std::memory_order_relaxed), std::memory_order_relaxed);
      ema_200.store(other.ema_200.load(std::memory_order_relaxed), std::memory_order_relaxed);
      
      rsi.store(other.rsi.load(std::memory_order_relaxed), std::memory_order_relaxed);
      macd_line.store(other.macd_line.load(std::memory_order_relaxed), std::memory_order_relaxed);
      macd_signal.store(other.macd_signal.load(std::memory_order_relaxed), std::memory_order_relaxed);
      macd_histogram.store(other.macd_histogram.load(std::memory_order_relaxed), std::memory_order_relaxed);
      bollinger_upper.store(other.bollinger_upper.load(std::memory_order_relaxed), std::memory_order_relaxed);
      bollinger_middle.store(other.bollinger_middle.load(std::memory_order_relaxed), std::memory_order_relaxed);
      bollinger_lower.store(other.bollinger_lower.load(std::memory_order_relaxed), std::memory_order_relaxed);
      stochastic_k.store(other.stochastic_k.load(std::memory_order_relaxed), std::memory_order_relaxed);
      stochastic_d.store(other.stochastic_d.load(std::memory_order_relaxed), std::memory_order_relaxed);
      atr.store(other.atr.load(std::memory_order_relaxed), std::memory_order_relaxed);
      
      last_updated.store(other.last_updated.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
  }
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

  // Trade analytics - using double buffered state for thread safety
  mutable btq::threading::DoubleBufferedState<std::vector<TradeData>> recent_trades_db;
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

  // Constructor to initialize double buffered state
  SymbolAnalytics() : recent_trades_db(std::vector<TradeData>{}) {}
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
   * Get lock-free snapshot buffer for a symbol (render thread).
   * Returns nullptr if the symbol hasn't been seen yet.
   * Once obtained, call consume() then read() — zero copies.
   */
  TripleBuffer<RenderSnapshot>* getSnapshotBuffer(uint32_t symbol_id) {
    auto& shard = getShard(symbol_id);
    std::shared_lock lock(shard.mutex);  // Brief lookup only, no data copy
    auto it = shard.snapshot_buffers.find(symbol_id);
    if (it != shard.snapshot_buffers.end()) return it->second.get();
    return nullptr;
  }

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

  // Public accessor for atomic snapshots (zero-lock access for UI/rendering)
  const HotSpine::V3::AtomicSymbolInfo* get_atomic_snapshot(uint32_t id) const {
    return atomic_registry_.get_atomic_snapshot(id);
  }

  // Public accessor for orderbook snapshots via the dedicated manager (atomic access for UI/rendering)
  const std::shared_ptr<const OrderbookData> get_orderbook_snapshot(uint32_t symbol_id) const {
    if (orderbook_snapshot_manager_) {
      return orderbook_snapshot_manager_->getSnapshot(symbol_id);
    }
    return nullptr;
  }

  // Public accessor for atomic indicator values (for fast UI polling)
  const AtomicIndicatorValues* get_atomic_indicator_values(uint32_t symbol_id) const {
    auto& shard = getShard(symbol_id);
    std::shared_lock lock(shard.mutex);
    auto it = shard.atomic_indicator_values.find(symbol_id);
    if (it != shard.atomic_indicator_values.end()) {
      return &(it->second);
    }
    return nullptr;
  }

  // Method to update atomic indicator values from background threads
  void update_atomic_indicator_values(uint32_t symbol_id, const AtomicIndicatorValues& values) {
    auto& shard = getShard(symbol_id);
    std::unique_lock lock(shard.mutex);
    shard.atomic_indicator_values[symbol_id] = values;
  }

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
    // Per-symbol atomic indicator values for fast UI access
    std::unordered_map<uint32_t, AtomicIndicatorValues> atomic_indicator_values;
    // Per-symbol lock-free snapshot buffers for render thread
    std::unordered_map<uint32_t, std::unique_ptr<TripleBuffer<RenderSnapshot>>> snapshot_buffers;
    // Per-symbol orderbook snapshots for atomic access by renderer
    std::unordered_map<uint32_t, std::shared_ptr<const OrderbookData>> orderbook_snapshots;
    // Per-symbol publish throttle timestamps (accessed under unique_lock)
    std::unordered_map<uint32_t, std::chrono::steady_clock::time_point> last_publish_time;
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

  // Ring buffer polling thread
  std::thread polling_thread_;
  uint64_t local_read_tail_{0};
  std::shared_ptr<BTQuant::HotSpineDataBridge> hotspine_bridge_;  // Reference to the HotSpineDataBridge

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
  
  // Indicator calculation methods
  std::vector<double> calculate_sma(const std::vector<double>& prices, int period) const;
  std::vector<double> calculate_ema(const std::vector<double>& prices, int period) const;
  std::vector<double> calculate_rsi(const std::vector<double>& prices, int period) const;
  std::vector<double> calculate_macd_line(const std::vector<double>& prices, int fast_period, int slow_period) const;
  std::vector<double> calculate_macd_signal(const std::vector<double>& macd_line, int signal_period) const;
  std::vector<double> calculate_macd_histogram(const std::vector<double>& macd_line, const std::vector<double>& signal_line) const;
  std::vector<double> calculate_bollinger_bands(const std::vector<double>& prices, int period, double std_dev, 
                                               std::vector<double>& upper_band, std::vector<double>& middle_band, std::vector<double>& lower_band) const;
  std::vector<double> calculate_stochastic_k(const std::vector<double>& highs, const std::vector<double>& lows, 
                                            const std::vector<double>& closes, int k_period) const;
  std::vector<double> calculate_atr(const std::vector<double>& highs, const std::vector<double>& lows, 
                                   const std::vector<double>& closes, int period) const;
  
  // Method to update atomic indicator values based on current analytics
  void update_atomic_indicators(uint32_t symbol_id, const SymbolAnalytics& analytics);

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

  // Build and publish a RenderSnapshot from SymbolAnalytics (worker thread)
  void publishSnapshot(uint32_t symbol_id, const SymbolAnalytics& analytics);

  // Ring buffer polling methods
  void pollingLoop();
  void pollSharedMemoryRingBuffer();
  
  // Poll hotspine function that updates atomic storage directly
  void poll_hotspine();
  
  // Method to set the HotSpineDataBridge for direct SHM access
  void setHotSpineBridge(std::shared_ptr<BTQuant::HotSpineDataBridge> bridge) {
    hotspine_bridge_ = bridge;
  }

  // Atomic registry for zero-lock access to market data
  mutable HotSpine::V3::AtomicRegistry atomic_registry_;

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

  // Cluster Engine for advanced analytics
  std::unique_ptr<Analytics::ClusterEngine> cluster_engine_;
  
  // Orderbook snapshot manager for atomic access by renderer
  std::unique_ptr<OrderbookSnapshotManager> orderbook_snapshot_manager_;
};

}  // namespace RenderEngine
}  // namespace BTQuant
