#pragma once

#include <cstdint>

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

// Optimized TradeData struct with proper packing for memory efficiency
// alignas(64) prevents cache-line false sharing across CPU cores
// Note: No default member initializers to maintain std::is_trivial_v == true
#pragma pack(push, 1)
struct alignas(64) TradeData {
  uint64_t timestamp;    // Unix timestamp in milliseconds
  double price;          // Price of the trade
  float volume;          // Volume of the trade
  TradeSide side;        // Side of the trade (BUY/SELL)
  uint8_t exchange_id;   // Exchange identifier
  uint8_t flags;         // Bitmask of trade flags
};
#pragma pack(pop)

static_assert(sizeof(TradeData) == 64, "TradeData must be exactly 64 bytes");

// Helper functions for flag manipulation
inline void set_flag(uint8_t& flags, TradeFlags flag) { flags |= static_cast<uint8_t>(flag); }

inline void clear_flag(uint8_t& flags, TradeFlags flag) { flags &= ~static_cast<uint8_t>(flag); }

inline bool has_flag(const uint8_t& flags, TradeFlags flag) {
  return (flags & static_cast<uint8_t>(flag)) != 0;
}

}  // namespace Data
}  // namespace BTQuant

// Hash function for TradeData to enable use in unordered containers
namespace std {
template <>
struct hash<BTQuant::Data::TradeData> {
  size_t operator()(const BTQuant::Data::TradeData& trade) const {
    size_t h1 = hash<uint64_t>{}(trade.timestamp);
    size_t h2 = hash<double>{}(trade.price);
    size_t h3 = hash<float>{}(trade.volume);
    size_t h4 = hash<uint8_t>{}(static_cast<uint8_t>(trade.side));
    size_t h5 = hash<uint8_t>{}(trade.exchange_id);
    size_t h6 = hash<uint8_t>{}(trade.flags);

    // Combine hashes using boost-style hash combination
    size_t seed = h1;
    seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= h5 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= h6 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};
} // namespace std