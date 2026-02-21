#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace BTQuant {

// Forward declarations
struct VolumeProfileLevel {
  double price = 0.0;
  double total_volume = 0.0;
  double buy_volume = 0.0;
  double sell_volume = 0.0;
};

enum class MarketDataType : uint8_t { TRADE = 0, ORDERBOOK = 1 };
enum class TradeSide : uint8_t { BUY = 0, SELL = 1 };

// 16 Bytes
struct PriceLevel {
  double price;
  double size;
};
static_assert(std::is_trivial_v<PriceLevel>, "PriceLevel MUST be trivial");

// 64 Bytes - Cache-line aligned for false-sharing elimination in SpscRingBuffer
struct alignas(64) TradeData {
  uint64_t timestamp_us;  // 8
  double price;           // 8
  float volume;           // 4
  uint32_t symbol_id;     // 4
  TradeSide side;         // 1
  uint8_t exchange_id;    // 1
  uint8_t flags;          // 1
  uint8_t _pad[33];       // 33 → total = 8+8+4+4+1+1+1+33 = 60 + 4(alignment) = 64

  // ==============================================================
  // LEGACY COMPATIBILITY METHODS (no reference members!)
  // ==============================================================

  // Helper for is_buy (old code used boolean, new uses TradeSide enum)
  bool is_buy() const { return side == TradeSide::BUY; }
  void set_is_buy(bool buy) { side = buy ? TradeSide::BUY : TradeSide::SELL; }
};
static_assert(sizeof(TradeData) == 64, "TradeData must be exactly 64 bytes (one cache line)");
static_assert(std::is_trivial_v<TradeData>, "TradeData MUST be trivial");
static_assert(std::is_standard_layout_v<TradeData>, "TradeData MUST be standard layout");

// Contiguous Orderbook Snapshot - MUST remain trivial
struct alignas(64) OrderBookSnapshot {
  static constexpr size_t MAX_LEVELS = 100;

  uint64_t timestamp_us;
  uint32_t symbol_id;
  uint32_t bid_levels_count;
  uint32_t ask_levels_count;

  double best_bid;
  double best_ask;
  double best_bid_size;
  double best_ask_size;

  // Computed fields for convenience
  double spread;          // best_ask - best_bid
  double spread_percent;  // spread / mid_price * 100
  double imbalance;       // (bid_volume - ask_volume) / total_volume

  PriceLevel bids[MAX_LEVELS];
  PriceLevel asks[MAX_LEVELS];

  // ==============================================================
  // LEGACY COMPATIBILITY LAYER - Vector-like interface
  // ==============================================================

  // Check if bids/asks are empty
  bool bids_empty() const { return bid_levels_count == 0; }
  bool asks_empty() const { return ask_levels_count == 0; }

  // Get number of valid levels
  size_t bids_size() const { return bid_levels_count; }
  size_t asks_size() const { return ask_levels_count; }

  // Get first valid bid/ask
  const PriceLevel& bids_front() const { return bids[0]; }
  const PriceLevel& asks_front() const { return asks[0]; }

  // Iterator support for range-based construction
  const PriceLevel* bids_begin() const { return bids; }
  const PriceLevel* bids_end() const { return bids + bid_levels_count; }
  const PriceLevel* asks_begin() const { return asks; }
  const PriceLevel* asks_end() const { return asks + ask_levels_count; }
};
static_assert(std::is_trivial_v<OrderBookSnapshot>, "OrderBookSnapshot MUST be trivial");
static_assert(std::is_standard_layout_v<OrderBookSnapshot>,
              "OrderBookSnapshot MUST be standard layout");

// ==============================================================
// LEGACY UI COMPATIBILITY LAYER
// ==============================================================
class MarketDataProcessor;

namespace RenderEngine {
enum class TimeFrame : uint8_t {
  TF_1MS,
  TF_10MS,
  TF_100MS,
  TF_500MS,
  TF_1SEC,
  TF_3SEC,
  TF_5SEC,
  TF_15SEC,
  TF_30SEC,
  TF_1MIN,
  TF_2MIN,
  TF_5MIN,
  TF_15MIN,
  TF_30MIN,
  TF_1HOUR,
  TF_2HOUR,
  TF_4HOUR,
  TF_6HOUR,
  TF_12HOUR,
  TF_1DAY,
  TF_1WEEK
};

struct OHLCVCandle {
  uint64_t timestamp;
  double open, high, low, close, volume;
  uint64_t trade_count = 0;  // Number of trades in this candle
};

// Notification type for subscribe/unsubscribe callbacks
enum class NotificationType : uint8_t { TRADE = 0, ORDERBOOK = 1, CANDLE = 2, ALL = 255 };

// Symbol analytics structure for panel displays
struct SymbolAnalytics {
  uint32_t symbol_id = 0;
  std::string symbol_name;

  // Price data
  double latest_price = 0.0;
  double open_price = 0.0;
  double high_24h = 0.0;
  double low_24h = 0.0;
  double close_price = 0.0;

  // Last trade data (legacy compatibility)
  double last_trade_price = 0.0;
  double last_trade_size = 0.0;
  uint64_t last_trade_time = 0;

  // Volume data
  double volume_24h = 0.0;
  double volume_1m = 0.0;
  double volume_5m = 0.0;
  double volume_15m = 0.0;
  double vwap = 0.0;
  double vwap_deviation = 0.0;

  // Buy/Sell metrics
  double buy_volume = 0.0;
  double sell_volume = 0.0;
  uint64_t buy_count = 0;
  uint64_t sell_count = 0;
  double buy_sell_ratio = 0.5;

  // Trade statistics
  uint64_t trade_count = 0;
  double avg_trade_size = 0.0;
  uint64_t large_trade_count = 0;

  // Change calculations
  double change_pct = 0.0;
  double change_dollar = 0.0;

  // Momentum and volatility
  double momentum = 0.0;
  double momentum_strength = 0.0;
  double volatility = 0.0;
  double sharpe_ratio = 0.0;

  // Price range
  double price_min = 0.0;
  double price_max = 0.0;
  double price_position = 0.0;

  // Spread analysis
  double current_spread = 0.0;
  double current_spread_percent = 0.0;
  double avg_spread = 0.0;
  double avg_spread_percent = 0.0;

  // Order book imbalance
  double current_imbalance = 0.0;
  double avg_imbalance = 0.0;
  double market_depth = 0.0;

  // Recent trades for tape display - use BTQuant::TradeData directly
  std::vector<BTQuant::TradeData> recent_trades;

  // Candle data by timeframe
  std::unordered_map<TimeFrame, std::vector<OHLCVCandle>> candles;
  std::unordered_map<TimeFrame, OHLCVCandle> current_candles;

  // Volume profile
  std::map<double, VolumeProfileLevel> session_volume_profile;

  // Timestamps
  uint64_t last_update_ts = 0;
  uint64_t last_update_time = 0;  // Legacy compatibility

  // Incremental update state (for processTradeIncrementally)
  std::deque<double> momentum_prices;
  size_t momentum_window_size = 50;
  std::deque<double> log_returns;
  size_t volatility_window_size = 100;
  double running_total_price_volume = 0.0;
  double running_total_volume = 0.0;
};

// Performance metrics structure
struct PerformanceMetrics {
  uint64_t trades_processed = 0;
  uint64_t orderbook_updates = 0;
  double avg_latency_us = 0.0;
  double max_latency_us = 0.0;
  uint64_t queue_depth = 0;
  double throughput_per_sec = 0.0;
};

// Leitet alte Pointer auf unseren neuen Processor um!
using MarketDataProcessor = BTQuant::MarketDataProcessor;

// Callback type for notifications
using NotificationCallback = std::function<void(uint32_t symbol_id, NotificationType type)>;
}  // namespace RenderEngine

struct IndicatorItem {
  std::string name;
  std::map<std::string, float> parameters;
  RenderEngine::TimeFrame source_timeframe;
};

// Subscription handle for unsubscribe
struct SubscriptionHandle {
  uint64_t id = 0;
  uint32_t symbol_id = 0;
  RenderEngine::NotificationType type = RenderEngine::NotificationType::ALL;
};

}  // namespace BTQuant
