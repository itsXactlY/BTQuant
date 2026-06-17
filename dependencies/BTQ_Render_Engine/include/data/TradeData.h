#pragma once

#include <cstdint>
#include <type_traits>

namespace BTQuant {
namespace Data {

// ============================================================================
// Trade Data Structure
// ============================================================================

enum class TradeSide : uint8_t { BUY = 0, SELL = 1 };

enum class TradeFlags : uint8_t {
  NONE = 0x00,
  LIQUIDITY_ADDED = 0x01,
  LIQUIDITY_REMOVED = 0x02,
  AGGRESSIVE_ORDER = 0x04,
  PASSIVE_ORDER = 0x08,
  MARKET_ORDER = 0x10,
  LIMIT_ORDER = 0x20
};

// ---------------------------------------------------------------------------
// Backwards-compat: legacy `Data::TradeData` had a `timestamp` field at
// offset 16. The spec-compliant name is `timestamp_us`. To avoid breaking
// every existing call-site that reads `trade.timestamp` (without growing
// the struct past 64 bytes), the two names share storage via a union of
// `uint64_t`. The union-of-trivial-types is itself trivial + standard-
// layout, so the outer struct still satisfies the PHASE 0 contract.
// ---------------------------------------------------------------------------
union TimestampAlias {
  uint64_t timestamp_us;  // canonical (spec-compliant) name
  uint64_t timestamp;     // deprecated alias — read/write same storage
};

// ============================================================================
// Spec-compliant TradeData (PHASE 0 / TUGW section 0.1)
// ---------------------------------------------------------------------------
// 64-byte aligned, padded to one cache line. Designed to be passed by value
// through the SPSC ring buffer without false-sharing and without touching
// adjacent slots on push/pop.
//
// Layout:
//   double   price;        // offset 0   (8 bytes)
//   double   volume;       // offset 8   (8 bytes)
//   uint64_t timestamp_us; // offset 16  (8 bytes; `timestamp` aliases it)
//   uint32_t symbol_id;    // offset 24  (4 bytes)
//   TradeSide side;        // offset 28  (1 byte,  0 = Buy, 1 = Sell)
//   uint8_t  _pad[31];     // offset 29  (31 bytes,  -> 60)
//   [implicit]             // alignas(64) tail pad -> 64
//
// Trivial + standard-layout is REQUIRED for zero-copy SPSC use. All fields
// are public; the user can value-initialise with `TradeData{}`.
// ============================================================================
struct alignas(64) TradeData {
  double          price;        // 0..7
  double          volume;       // 8..15
  TimestampAlias  ts;           // 16..23  (ts.timestamp_us OR ts.timestamp)
  uint32_t        symbol_id;    // 24..27
  TradeSide       side;         // 28
  TradeFlags      flags;        // 29  (added for quality_monitor back-compat)
  uint8_t         _pad[30];     // 30..59
};

// Compile-time size guarantees — these are the Phase-0 acceptance contract.
static_assert(sizeof(TradeData) == 64,
              "TradeData must be exactly 64 bytes (alignas(64))");
static_assert(alignof(TradeData) == 64,
              "TradeData must be aligned to 64 bytes");
static_assert(std::is_trivial_v<TradeData>,
              "TradeData must be trivial for hot-path memcpy safety");
static_assert(std::is_standard_layout_v<TradeData>,
              "TradeData must be standard-layout for ABI stability");
static_assert(std::is_trivially_copyable_v<TradeData>,
              "TradeData must be trivially copyable for zero-copy SPSC use");

// Helper functions for flag manipulation
inline void set_flag(uint8_t& flags, TradeFlags flag) { flags |= static_cast<uint8_t>(flag); }
inline void set_flag(TradeFlags& flags, TradeFlags flag) {
    flags = static_cast<TradeFlags>(static_cast<uint8_t>(flags) | static_cast<uint8_t>(flag));
}

inline void clear_flag(uint8_t& flags, TradeFlags flag) { flags &= ~static_cast<uint8_t>(flag); }
inline void clear_flag(TradeFlags& flags, TradeFlags flag) {
    flags = static_cast<TradeFlags>(static_cast<uint8_t>(flags) & ~static_cast<uint8_t>(flag));
}

inline bool has_flag(const uint8_t& flags, TradeFlags flag) {
    return (flags & static_cast<uint8_t>(flag)) != 0;
}
inline bool has_flag(TradeFlags flags, TradeFlags flag) {
    return (static_cast<uint8_t>(flags) & static_cast<uint8_t>(flag)) != 0;
}

}  // namespace Data
}  // namespace BTQuant

// Hash function for TradeData to enable use in unordered containers
namespace std {
template <>
struct hash<BTQuant::Data::TradeData> {
  size_t operator()(const BTQuant::Data::TradeData& trade) const {
    size_t h1 = hash<uint64_t>{}(trade.ts.timestamp_us);
    size_t h2 = hash<double>{}(trade.price);
    size_t h3 = hash<double>{}(trade.volume);
    size_t h4 = hash<uint32_t>{}(trade.symbol_id);
    size_t h5 = hash<uint8_t>{}(static_cast<uint8_t>(trade.side));

    size_t seed = h1;
    seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= h5 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};
} // namespace std
