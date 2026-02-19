#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

// ============================================================================
// MMT GENESIS - Bare Metal Spine Data Types
// Zero-allocation, lock-free, GPU-compatible POD structures
// ============================================================================

// Price level for order book data
// NOTE: No alignas(64) here - PriceLevel is used in std::vector which handles
// its own memory allocation. Cache-line alignment is only needed for
// lock-free cross-thread structures (TradeData, OrderBookSnapshot).
struct PriceLevel {
  double price;
  double size;
};

// Validate PriceLevel is trivial POD
static_assert(std::is_trivial_v<PriceLevel>, "PriceLevel must be trivial");
static_assert(std::is_standard_layout_v<PriceLevel>, "PriceLevel must be standard layout");
static_assert(sizeof(PriceLevel) == 16, "PriceLevel must be 16 bytes (2 doubles)");

// ============================================================================
// TradeData - MMT Genesis Core Trade Structure
// NOTE: NO alignas(64) here - TradeData is used in std::vector which handles
// its own memory allocation. Cache-line alignment causes heap corruption
// when used with standard containers. Alignment is only needed for
// lock-free SPSC ring buffer structures (use aligned_allocator there).
// Trivially copyable for SPSC ring buffer and GPU transfer
// ============================================================================
struct TradeData {
  // --- Timestamp & Ordering (16 bytes) ---
  uint64_t timestamp;     // Unix timestamp in microseconds
  uint64_t sequence;      // Monotonic sequence number for ordering

  // --- Symbol & Side (8 bytes) ---
  uint32_t symbol_id;     // Symbol identifier (lookup in SymbolRegistry)
  uint8_t  side;          // 0 = Buy (taker buy), 1 = Sell (taker sell)
  uint8_t  flags;         // Trade flags (aggressor, implied, etc.)
  uint8_t  padding1[2];   // Padding for alignment

  // --- Price & Size (16 bytes) ---
  double   price;         // Trade price
  double   size;          // Trade size/volume

  // --- Extended Data (16 bytes) ---
  uint64_t trade_id;      // Unique trade identifier
  uint32_t exchange_id;   // Exchange identifier
  uint8_t  condition;     // Trade condition flags
  uint8_t  padding2[3];   // Padding for 64-byte alignment

  // --- Reserved (8 bytes) ---
  uint64_t reserved;      // Reserved for future use
};

// Validate TradeData is trivial POD for SPSC ring buffer
static_assert(std::is_trivial_v<TradeData>, "TradeData must be trivial");
static_assert(std::is_standard_layout_v<TradeData>, "TradeData must be standard layout");
static_assert(sizeof(TradeData) == 64, "TradeData must be exactly 64 bytes");

// ============================================================================
// OrderBookSnapshot - MMT Genesis GPU-Compatible Order Book
// STD430 aligned for Vulkan SSBO transfer
// Uses C-arrays instead of std::array for trivial copyability
// ============================================================================
struct alignas(64) OrderBookSnapshot {
  // --- Header (32 bytes) ---
  uint64_t timestamp;       // Unix timestamp in microseconds
  uint64_t sequence;        // Monotonic sequence number for ordering
  uint32_t symbol_id;       // Symbol identifier
  uint16_t bids_count;      // Number of active bid levels
  uint16_t asks_count;      // Number of active ask levels
  uint32_t padding;         // Padding for 64-byte alignment

  // --- Price Levels - STD430 Compatible (C-arrays for trivial copyability) ---
  // Each array is 50 * 8 = 400 bytes
  // Total price level data: 4 * 400 = 1600 bytes
  // STD430: 16-byte stride for vec4 compatibility
  double bid_prices[50];    // Bid price levels (descending)
  double bid_volumes[50];   // Bid volume at each level
  double ask_prices[50];    // Ask price levels (ascending)
  double ask_volumes[50];   // Ask volume at each level
};

// Validate OrderBookSnapshot is trivial POD for GPU transfer
static_assert(std::is_trivial_v<OrderBookSnapshot>, "OrderBookSnapshot must be trivial");
static_assert(std::is_standard_layout_v<OrderBookSnapshot>, "OrderBookSnapshot must be standard layout");

// ============================================================================
// TradeDataConverter - Conversion layer for backward compatibility
// Converts between legacy std::string-based TradeData and new POD TradeData
// ============================================================================
namespace BTQuant {
namespace Data {

// Legacy TradeData structure for backward compatibility
// Used by existing code that depends on std::string symbol
struct LegacyTradeData {
  std::string symbol;
  uint32_t symbol_id = 0;
  uint64_t timestamp = 0;
  double price = 0.0;
  double size = 0.0;
  bool is_buy = true;
  
  // Convert to new POD TradeData
  TradeData toPod() const {
    TradeData pod{};
    pod.timestamp = timestamp;
    pod.sequence = 0; // Caller must set
    pod.symbol_id = symbol_id;
    pod.side = is_buy ? 0 : 1;
    pod.flags = 0;
    pod.price = price;
    pod.size = size;
    pod.trade_id = 0;
    pod.exchange_id = 0;
    pod.condition = 0;
    pod.reserved = 0;
    return pod;
  }
  
  // Convert from new POD TradeData
  static LegacyTradeData fromPod(const TradeData& pod, const std::string& symbol_name = "") {
    LegacyTradeData legacy;
    legacy.symbol = symbol_name;
    legacy.symbol_id = pod.symbol_id;
    legacy.timestamp = pod.timestamp;
    legacy.price = pod.price;
    legacy.size = pod.size;
    legacy.is_buy = (pod.side == 0);
    return legacy;
  }
};

} // namespace Data
} // namespace BTQuant

// Volume profile level for market profile analysis
struct VolumeProfileLevel {
  double price;
  double total_volume;
  double buy_volume;
  double sell_volume;
};

namespace BTQuant {
namespace RenderEngine {

// Time frame definitions for OHLCV aggregation
// Extended to include higher timeframes for multi-timeframe analysis
enum class TimeFrame {
  TF_1MS,    // 1 millisecond
  TF_10MS,   // 10 milliseconds
  TF_100MS,  // 100 milliseconds
  TF_500MS,  // 500 milliseconds
  TF_1SEC,   // 1 second
  TF_3SEC,   // 3 seconds
  TF_5SEC,   // 5 seconds
  TF_15SEC,  // 15 seconds
  TF_30SEC,  // 30 seconds
  TF_1MIN,   // 1 minute
  TF_2MIN,   // 2 minutes
  TF_5MIN,   // 5 minutes
  TF_15MIN,  // 15 minutes
  TF_30MIN,  // 30 minutes
  TF_1HOUR,  // 1 hour
  TF_2HOUR,  // 2 hours
  TF_4HOUR,  // 4 hours
  TF_6HOUR,  // 6 hours
  TF_12HOUR, // 12 hours
  TF_1DAY,   // 1 day
  TF_1WEEK   // 1 week
};

// OHLCV Candle Data for Charting
struct OHLCVCandle {
  uint64_t timestamp;  // Start time of the candle in microseconds
  double open;
  double high;
  double low;
  double close;
  double volume;
  uint64_t trade_count;
};

} // namespace RenderEngine
} // namespace BTQuant