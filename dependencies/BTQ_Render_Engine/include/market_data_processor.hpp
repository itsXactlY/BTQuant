#pragma once
#include "rendering/candlestick_instancing.hpp"
using namespace BTQuant::Rendering;

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "data/core_types.hpp"
#include "rendering/candlestick_instancing.hpp"
#include "rendering/vulkan_chart_pipeline.hpp"
#include "threading/lockfree_queue.hpp"

namespace BTQuant {

/**
 * MarketDataProcessor - Central hub for market data processing
 *
 * Features:
 * - Lock-free trade queue for sub-microsecond ingestion
 * - Atomic orderbook pointer swaps for zero-copy updates
 * - Subscription system for reactive UI updates
 * - Symbol analytics aggregation
 */
class MarketDataProcessor {
 public:
  static constexpr size_t MAX_SYMBOLS = 100;

  MarketDataProcessor() {
    for (size_t i = 0; i < MAX_SYMBOLS; ++i) {
      latest_prices_[i].store(0.0, std::memory_order_relaxed);
      active_books_[i].store(nullptr, std::memory_order_relaxed);
    }
  }

  virtual ~MarketDataProcessor() = default;

  // ==========================================
  // INGESTION (Aufgerufen vom Netzwerk-Thread)
  // ==========================================

  // Pusht einen Trade in die lock-free Queue (Sub-Mikrosekunde)
  bool enqueue_trade(const TradeData& trade) { return trade_queue_.push(trade); }

  // Setzt das aktive Orderbuch über einen atomaren Pointer-Swap (Double-Buffering)
  void update_orderbook(uint32_t symbol_id, OrderBookSnapshot* new_snapshot) {
    if (symbol_id < MAX_SYMBOLS) {
      active_books_[symbol_id].store(new_snapshot, std::memory_order_release);
    }
  }

  // ==========================================
  // GPU ZERO-COPY BINDING
  // ==========================================

  // Bind a host-visible GPU buffer for a specific symbol to receive direct lock-free candlestick
  // updates
  void bind_gpu_candlestick_buffer(uint32_t symbol_id, CandlestickInstance* mapped_buffer,
                                   size_t max_capacity) {
    std::lock_guard<std::mutex> lock(gpu_bindings_mutex_);
    if (symbol_id >= MAX_SYMBOLS || !mapped_buffer) return;

    gpu_buffers_[symbol_id].buffer = mapped_buffer;
    gpu_buffers_[symbol_id].capacity = max_capacity;
    gpu_buffers_[symbol_id].current_count.store(0, std::memory_order_relaxed);
  }

  // Retrieve the active buffer state (count) for rendering
  uint32_t get_gpu_candlestick_count(uint32_t symbol_id) const {
    if (symbol_id >= MAX_SYMBOLS) return 0;
    return gpu_buffers_[symbol_id].current_count.load(std::memory_order_acquire);
  }

  // ==========================================
  // PROCESSING (Aufgerufen vom Engine-Thread)
  // ==========================================

  // Leert die Queue und aktualisiert die globalen States
  void process_queues() {
    TradeData trade;
    // Solange Daten da sind, auslesen
    while (trade_queue_.pop(trade)) {
      if (trade.symbol_id < MAX_SYMBOLS) {
        // UI-Pointer aktualisieren
        latest_prices_[trade.symbol_id].store(trade.price, std::memory_order_release);

        // Store in recent trades for analytics
        {
          std::lock_guard<std::mutex> lock(analytics_mutex_);
          auto& trades = symbol_trades_[trade.symbol_id];
          trades.push_back(trade);
          // Keep only last 1000 trades per symbol
          if (trades.size() > 1000) {
            trades.erase(trades.begin(), trades.begin() + (trades.size() - 1000));
          }
        }

        // Zero-Copy GPU Buffer Ingestion:
        // If a GPU buffer is bound for this symbol, write the data directly into CPU-mapped GPU
        // memory lock-free
        {
          std::lock_guard<std::mutex> lock(
              gpu_bindings_mutex_);  // Only needed for bounds check protection against rebinding
          auto& binding = gpu_buffers_[trade.symbol_id];
          if (binding.buffer && binding.capacity > 0) {
            uint32_t current_idx = binding.current_count.load(std::memory_order_relaxed);
            if (current_idx < binding.capacity) {
              // Convert trade to a basic Candlestick for visualization
              // In a real scenario, this would aggregate OHLCV data over time windows,
              // but for this phase we map single trades to visual instances instantly.
              binding.buffer[current_idx].open = trade.price;
              binding.buffer[current_idx].high = trade.price;
              binding.buffer[current_idx].low = trade.price;
              binding.buffer[current_idx].close = trade.price;
              binding.buffer[current_idx].time =
                  static_cast<float>(trade.timestamp_us) / 1000000.0f;
              binding.buffer[current_idx].width = 1.0f;
              binding.buffer[current_idx].flags = trade.is_buy() ? 1 : 0;  // Bullish flag

              binding.current_count.store(current_idx + 1, std::memory_order_release);
            }
          }
        }

        // Notify subscribers
        notify_subscribers(trade.symbol_id, RenderEngine::NotificationType::TRADE);
      }
    }
  }

  // ==========================================
  // UI READS (Aufgerufen vom ImGui Render-Thread)
  // ==========================================

  // O(1) Lock-free read. Keine Mutexe. UI blockiert niemals.
  double get_latest_price(uint32_t symbol_id) const {
    if (symbol_id >= MAX_SYMBOLS) return 0.0;
    return latest_prices_[symbol_id].load(std::memory_order_acquire);
  }

  // Hole den Pointer auf das aktuellste Buch für den DOM
  OrderBookSnapshot* get_active_orderbook(uint32_t symbol_id) const {
    if (symbol_id >= MAX_SYMBOLS) return nullptr;
    return active_books_[symbol_id].load(std::memory_order_acquire);
  }

  // ==========================================
  // SUBSCRIPTION SYSTEM (C++26 Reactive Pattern)
  // ==========================================

  /**
   * Subscribe to notifications for a specific symbol and data type
   * @param symbol_id The symbol to subscribe to
   * @param type The type of notification (TRADE, ORDERBOOK, etc.)
   * @param callback The callback to invoke when data arrives
   * @return Subscription ID for later unsubscription
   */
  uint64_t subscribe(uint32_t symbol_id, RenderEngine::NotificationType type,
                     RenderEngine::NotificationCallback callback) {
    std::lock_guard<std::mutex> lock(subscription_mutex_);
    uint64_t id = next_subscription_id_++;
    subscriptions_[id] = {id, symbol_id, type, callback};
    active_symbols_.insert(symbol_id);
    return id;
  }

  /**
   * Unsubscribe from notifications
   * @param subscription_id The ID returned from subscribe()
   */
  void unsubscribe(uint64_t subscription_id) {
    std::lock_guard<std::mutex> lock(subscription_mutex_);
    subscriptions_.erase(subscription_id);
  }

  // ==========================================
  // SYMBOL MANAGEMENT
  // ==========================================

  /**
   * Get list of active symbols (those with subscriptions or data)
   */
  std::vector<uint32_t> getActiveSymbols() const {
    std::lock_guard<std::mutex> lock(subscription_mutex_);
    return std::vector<uint32_t>(active_symbols_.begin(), active_symbols_.end());
  }

  /**
   * Get symbol name by ID
   */
  std::string getSymbolName(uint32_t symbol_id) const {
    std::lock_guard<std::mutex> lock(analytics_mutex_);
    auto it = symbol_names_.find(symbol_id);
    return (it != symbol_names_.end()) ? it->second : "Unknown";
  }

  /**
   * Register a symbol name
   */
  void registerSymbol(uint32_t symbol_id, const std::string& name) {
    std::lock_guard<std::mutex> lock(analytics_mutex_);
    symbol_names_[symbol_id] = name;
  }

  // ==========================================
  // ANALYTICS & DATA ACCESS
  // ==========================================

  /**
   * Get analytics for a specific symbol
   */
  RenderEngine::SymbolAnalytics getSymbolAnalytics(uint32_t symbol_id) const {
    std::lock_guard<std::mutex> lock(analytics_mutex_);

    RenderEngine::SymbolAnalytics analytics;
    analytics.symbol_id = symbol_id;
    analytics.symbol_name = getSymbolName(symbol_id);
    analytics.latest_price = latest_prices_[symbol_id].load(std::memory_order_acquire);

    auto trades_it = symbol_trades_.find(symbol_id);
    if (trades_it != symbol_trades_.end()) {
      analytics.recent_trades = trades_it->second;

      // Calculate VWAP from recent trades
      double total_volume = 0.0;
      double volume_weighted_price = 0.0;
      for (const auto& trade : trades_it->second) {
        volume_weighted_price += trade.price * trade.volume;
        total_volume += trade.volume;
      }
      if (total_volume > 0) {
        analytics.vwap = volume_weighted_price / total_volume;
      }
      analytics.volume_24h = total_volume;

      if (!trades_it->second.empty()) {
        analytics.last_update_ts = trades_it->second.back().timestamp_us;
      }
    }

    return analytics;
  }

  /**
   * Get orderbook data for a symbol
   */
  std::optional<OrderBookSnapshot> getOrderbookData(uint32_t symbol_id) const {
    OrderBookSnapshot* snapshot = get_active_orderbook(symbol_id);
    if (snapshot) {
      return *snapshot;
    }
    return std::nullopt;
  }

  /**
   * Get candles for a symbol and timeframe
   */
  std::vector<RenderEngine::OHLCVCandle> getCandles(uint32_t symbol_id,
                                                    RenderEngine::TimeFrame timeframe) const {
    // TODO: Implement candle aggregation from trades
    // For now, return empty vector
    (void)symbol_id;
    (void)timeframe;
    return {};
  }

  // ==========================================
  // PERFORMANCE METRICS
  // ==========================================

  /**
   * Get performance metrics for monitoring
   */
  RenderEngine::PerformanceMetrics getPerformanceMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return performance_metrics_;
  }

  /**
   * Get dirty flag for heatmap updates
   */
  std::atomic<bool>& getHeatmapDirtyFlag() { return heatmap_dirty_; }

  /**
   * Get the duration in microseconds for a given timeframe
   */
  static uint64_t getTimeFrameDuration(RenderEngine::TimeFrame tf) {
    switch (tf) {
      case RenderEngine::TimeFrame::TF_1MS:
        return 1000ULL;
      case RenderEngine::TimeFrame::TF_10MS:
        return 10000ULL;
      case RenderEngine::TimeFrame::TF_100MS:
        return 100000ULL;
      case RenderEngine::TimeFrame::TF_500MS:
        return 500000ULL;
      case RenderEngine::TimeFrame::TF_1SEC:
        return 1000000ULL;
      case RenderEngine::TimeFrame::TF_3SEC:
        return 3000000ULL;
      case RenderEngine::TimeFrame::TF_5SEC:
        return 5000000ULL;
      case RenderEngine::TimeFrame::TF_15SEC:
        return 15000000ULL;
      case RenderEngine::TimeFrame::TF_30SEC:
        return 30000000ULL;
      case RenderEngine::TimeFrame::TF_1MIN:
        return 60000000ULL;
      case RenderEngine::TimeFrame::TF_2MIN:
        return 120000000ULL;
      case RenderEngine::TimeFrame::TF_5MIN:
        return 300000000ULL;
      case RenderEngine::TimeFrame::TF_15MIN:
        return 900000000ULL;
      case RenderEngine::TimeFrame::TF_30MIN:
        return 1800000000ULL;
      case RenderEngine::TimeFrame::TF_1HOUR:
        return 3600000000ULL;
      case RenderEngine::TimeFrame::TF_2HOUR:
        return 7200000000ULL;
      case RenderEngine::TimeFrame::TF_4HOUR:
        return 14400000000ULL;
      case RenderEngine::TimeFrame::TF_6HOUR:
        return 21600000000ULL;
      case RenderEngine::TimeFrame::TF_12HOUR:
        return 43200000000ULL;
      case RenderEngine::TimeFrame::TF_1DAY:
        return 86400000000ULL;
      case RenderEngine::TimeFrame::TF_1WEEK:
        return 604800000000ULL;
      default:
        return 1000000ULL;  // Default to 1 second
    }
  }

 protected:
  // Notify all subscribers for a symbol/type
  void notify_subscribers(uint32_t symbol_id, RenderEngine::NotificationType type) {
    std::lock_guard<std::mutex> lock(subscription_mutex_);
    for (const auto& [id, sub] : subscriptions_) {
      if ((sub.symbol_id == symbol_id || sub.symbol_id == 0) &&
          (sub.type == type || sub.type == RenderEngine::NotificationType::ALL)) {
        sub.callback(symbol_id, type);
      }
    }
  }

 private:
  // Die reine SPSC Pipeline
  SpscRingBuffer<TradeData, 65536> trade_queue_;

  // ==========================================
  // ATOMARER SPEICHER (Getrennt durch Cache-Lines gegen False Sharing)
  // ==========================================
  alignas(64) std::atomic<double> latest_prices_[MAX_SYMBOLS];
  alignas(64) std::atomic<OrderBookSnapshot*> active_books_[MAX_SYMBOLS];

  // ==========================================
  // SUBSCRIPTION SYSTEM
  // ==========================================
  struct Subscription {
    uint64_t id;
    uint32_t symbol_id;
    RenderEngine::NotificationType type;
    RenderEngine::NotificationCallback callback;
  };

  mutable std::mutex subscription_mutex_;
  std::unordered_map<uint64_t, Subscription> subscriptions_;
  std::unordered_set<uint32_t> active_symbols_;
  uint64_t next_subscription_id_ = 1;

  // ==========================================
  // ANALYTICS STORAGE
  // ==========================================
  mutable std::mutex analytics_mutex_;
  std::unordered_map<uint32_t, std::vector<TradeData>> symbol_trades_;
  std::unordered_map<uint32_t, std::string> symbol_names_;

  // ==========================================
  // PERFORMANCE METRICS
  // ==========================================
  mutable std::mutex metrics_mutex_;
  RenderEngine::PerformanceMetrics performance_metrics_;
  std::atomic<bool> heatmap_dirty_{false};

  // ==========================================
  // GPU ZERO-COPY BINDINGS
  // ==========================================
  struct GpuBufferBinding {
    CandlestickInstance* buffer = nullptr;
    size_t capacity = 0;
    std::atomic<uint32_t> current_count{0};
  };
  mutable std::mutex gpu_bindings_mutex_;
  GpuBufferBinding gpu_buffers_[MAX_SYMBOLS];
};

}  // namespace BTQuant
