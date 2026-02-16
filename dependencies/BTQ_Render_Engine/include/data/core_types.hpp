#pragma once

/**
 * @file core_types.hpp
 * @brief Single source of truth for core data types in PubBTQuant
 * 
 * This header defines fundamental types used throughout the trading system:
 * - Market data structures (trades, orderbook, OHLCV)
 * - Time frame enumerations
 * - Trading primitives (price levels, volume profiles)
 * - Shared memory layout for lock-free data exchange
 */

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// ============================================================================
// Namespace: BTQuant
// ============================================================================
namespace BTQuant {

// ============================================================================
// 1. Core Enumerations
// ============================================================================

/**
 * @brief Time frame definitions for OHLCV aggregation
 * 
 * Extended to include higher timeframes for multi-timeframe analysis.
 * Values are ordered from smallest to largest timeframe.
 */
enum class TimeFrame : uint8_t {
  TF_1MS = 0,    // 1 millisecond
  TF_10MS = 1,   // 10 milliseconds
  TF_100MS = 2,  // 100 milliseconds
  TF_500MS = 3,  // 500 milliseconds
  TF_1SEC = 4,   // 1 second
  TF_3SEC = 5,   // 3 seconds
  TF_5SEC = 6,   // 5 seconds
  TF_15SEC = 7,  // 15 seconds
  TF_30SEC = 8,  // 30 seconds
  TF_1MIN = 9,   // 1 minute
  TF_2MIN = 10,  // 2 minutes
  TF_5MIN = 11,  // 5 minutes
  TF_15MIN = 12, // 15 minutes
  TF_30MIN = 13, // 30 minutes
  TF_1HOUR = 14, // 1 hour
  TF_2HOUR = 15, // 2 hours
  TF_4HOUR = 16, // 4 hours
  TF_6HOUR = 17, // 6 hours
  TF_12HOUR = 18,// 12 hours
  TF_1DAY = 19,  // 1 day
  TF_1WEEK = 20  // 1 week
};

/**
 * @brief Market data update type
 */
enum class MarketDataType : uint8_t { 
  TRADE = 0, 
  ORDERBOOK = 1 
};

/**
 * @brief Trade side enumeration
 */
enum class TradeSide : uint8_t { 
  BUY = 0, 
  SELL = 1 
};

/**
 * @brief Trade flags bitmask
 */
enum class TradeFlags : uint8_t {
  NONE = 0x00,
  LIQUIDITY_ADDED = 0x01,
  LIQUIDITY_REMOVED = 0x02,
  AGGRESSIVE_ORDER = 0x04,
  PASSIVE_ORDER = 0x08,
  MARKET_ORDER = 0x10,
  LIMIT_ORDER = 0x20
};

// ============================================================================
// 2. Market Data Structures
// ============================================================================

/**
 * @brief Price level for order book data
 */
struct PriceLevel {
  double price;
  double size;

  PriceLevel() : price(0.0), size(0.0) {}
  PriceLevel(double p, double s) : price(p), size(s) {}
};

/**
 * @brief Volume profile level
 */
struct VolumeProfileLevel {
  double price;
  double total_volume;
  double buy_volume;
  double sell_volume;

  VolumeProfileLevel() 
    : price(0.0), total_volume(0.0), buy_volume(0.0), sell_volume(0.0) {}
  VolumeProfileLevel(double p, double tv, double bv, double sv)
    : price(p), total_volume(tv), buy_volume(bv), sell_volume(sv) {}
};

/**
 * @brief Trade data structure (packed for memory efficiency)
 */
#pragma pack(push, 1)
struct TradeData {
  uint64_t timestamp;   // Unix timestamp in milliseconds
  double price;         // Price of the trade
  float volume;         // Volume of the trade
  TradeSide side;       // Side of the trade (BUY/SELL)
  uint8_t exchange_id;  // Exchange identifier
  uint8_t flags;        // Bitmask of trade flags

  TradeData()
      : timestamp(0), price(0.0), volume(0.0f), side(TradeSide::BUY), 
        exchange_id(0), flags(0) {}

  TradeData(uint64_t ts, double p, float v, TradeSide s, uint8_t ex_id, uint8_t f)
      : timestamp(ts), price(p), volume(v), side(s), exchange_id(ex_id), flags(f) {}
};
#pragma pack(pop)

/**
 * @brief Orderbook data for analytics
 */
struct OrderbookData {
  std::string symbol;
  uint32_t symbol_id;
  uint64_t timestamp;
  std::vector<PriceLevel> bids;
  std::vector<PriceLevel> asks;
  double spread;
  double spread_percent;
  double bid_depth;
  double ask_depth;
  double total_depth;
  double imbalance;  // (bid_depth - ask_depth) / total_depth

  OrderbookData() 
    : symbol_id(0), timestamp(0), spread(0.0), spread_percent(0.0),
      bid_depth(0.0), ask_depth(0.0), total_depth(0.0), imbalance(0.0) {}
};

/**
 * @brief OHLCV Candle Data for Charting
 */
struct OHLCVCandle {
  uint64_t timestamp;  // Start time of the candle in microseconds
  double open;
  double high;
  double low;
  double close;
  double volume;
  uint64_t trade_count;

  OHLCVCandle()
    : timestamp(0), open(0.0), high(0.0), low(0.0), close(0.0), 
      volume(0.0), trade_count(0) {}
};

/**
 * @brief Market data update structure
 */
struct MarketDataUpdate {
  MarketDataType type;
  uint32_t symbol_id;
  uint64_t timestamp;
  double price;
  double size;
  std::string side;
  std::vector<PriceLevel> bids;
  std::vector<PriceLevel> asks;

  MarketDataUpdate() 
    : type(MarketDataType::TRADE), symbol_id(0), timestamp(0), 
      price(0.0), size(0.0) {}
};

// ============================================================================
// 3. OrderBook Snapshot (Standard Layout POD for ring buffer)
// ============================================================================

/**
 * @brief Order book snapshot for lock-free ring buffer
 */
struct OrderBookSnapshot {
  static constexpr size_t MAX_LEVELS = 20;  // Top 20 levels per side

  struct Level {
    double price;
    double size;

    Level() : price(0.0), size(0.0) {}
    Level(double p, double s) : price(p), size(s) {}
  };

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
  Level bid_levels[MAX_LEVELS];
  Level ask_levels[MAX_LEVELS];

  OrderBookSnapshot()
    : timestamp(0), symbol_id(0), best_bid(0.0), best_ask(0.0),
      best_bid_size(0.0), best_ask_size(0.0), spread(0.0),
      total_bid_volume(0.0), total_ask_volume(0.0),
      bid_levels_count(0), ask_levels_count(0) {}
};

// ============================================================================
// 4. HotSpine Shared Memory Types (Lock-Free Architecture)
// ============================================================================
namespace HotSpine::V3 {

/**
 * @brief Sequence lock for lock-free synchronization
 */
struct SeqLock {
  std::atomic<uint64_t> seq{0};

  void write_begin() {
    seq.fetch_add(1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_acquire);
  }

  void write_end() {
    std::atomic_thread_fence(std::memory_order_release);
    seq.fetch_add(1, std::memory_order_release);
  }

  uint64_t read_begin() const { 
    return seq.load(std::memory_order_acquire); 
  }

  bool read_retry(uint64_t start_seq) const {
    std::atomic_thread_fence(std::memory_order_acquire);
    return (start_seq % 2 != 0) || 
           (seq.load(std::memory_order_relaxed) != start_seq);
  }
};

/**
 * @brief Volume node (16 bytes, cache-line aligned)
 */
struct alignas(16) VolumeNode {
  float buy_vol;         // 4B
  float sell_vol;        // 4B
  uint16_t trade_count;  // 2B
  uint16_t tpo_bits;     // 2B - Bitmask for 30min brackets (0-15)
  uint8_t padding[4];    // 4B

  VolumeNode() 
    : buy_vol(0.0f), sell_vol(0.0f), trade_count(0), tpo_bits(0) {
    padding[0] = padding[1] = padding[2] = padding[3] = 0;
  }
};
static_assert(sizeof(VolumeNode) == 16, "VolumeNode size mismatch");

/**
 * @brief Viewport rows constant
 */
constexpr std::size_t VIEWPORT_ROWS = 256;

/**
 * @brief Cluster column for heatmap visualization (64-byte aligned)
 */
struct alignas(64) ClusterColumn {
  int64_t timestamp_us;
  double open;
  double high;
  double low;
  double close;
  int64_t base_tick_index;  // The absolute price index of row 0
  double tick_size;
  VolumeNode rows[VIEWPORT_ROWS];  // The visual rows

  ClusterColumn() 
    : timestamp_us(0), open(0.0), high(0.0), low(0.0), close(0.0),
      base_tick_index(0), tick_size(0.0) {}
};

/**
 * @brief Heatmap bin for DOM aggregation
 */
struct alignas(64) HeatmapBin {
  int64_t price_tick_index;
  double total_volume;
  uint32_t order_count;
  uint32_t padding;

  HeatmapBin() 
    : price_tick_index(0), total_volume(0.0), order_count(0), padding(0) {}
};

/**
 * @brief Shared memory layout V3 header
 */
struct SharedMemoryLayoutV3 {
  struct Header {
    uint32_t magic;  // 0x42545133 "BTQ3"
    uint32_t padding;
    SeqLock global_lock;
    std::atomic<uint64_t> head_index;
    uint8_t reserved[32];  // Padding to align body

    Header() : magic(0x42545133), padding(0), head_index(0) {}
  };

  Header header;
  ClusterColumn history[1024];  // Ring buffer
  HeatmapBin dom[512];          // Aggregated DOM
};

static_assert(sizeof(VolumeNode) == 16);
static_assert(alignof(ClusterColumn) == 64);

} // namespace HotSpine::V3

// ============================================================================
// 5. Helper Functions
// ============================================================================

/**
 * @brief Set a trade flag
 */
inline void set_flag(uint8_t& flags, TradeFlags flag) { 
  flags |= static_cast<uint8_t>(flag); 
}

/**
 * @brief Clear a trade flag
 */
inline void clear_flag(uint8_t& flags, TradeFlags flag) { 
  flags &= ~static_cast<uint8_t>(flag); 
}

/**
 * @brief Check if a trade flag is set
 */
inline bool has_flag(const uint8_t& flags, TradeFlags flag) {
  return (flags & static_cast<uint8_t>(flag)) != 0;
}

} // namespace BTQuant
