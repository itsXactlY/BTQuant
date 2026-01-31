#pragma once

#include <cstdint>
#include <string>

namespace BTQuant {
namespace Data {

// ============================================================================
// Trade Data Structure
// ============================================================================

enum class TradeSide {
    BUY,
    SELL
};

enum class TradeFlags : uint8_t {
    NONE = 0x00,
    LIQUIDITY_ADDED = 0x01,
    LIQUIDITY_REMOVED = 0x02,
    AGGRESSIVE_ORDER = 0x04,
    PASSIVE_ORDER = 0x08,
    MARKET_ORDER = 0x10,
    LIMIT_ORDER = 0x20
};

struct TradeData {
    uint64_t timestamp;      // Unix timestamp in milliseconds
    double price;            // Price of the trade
    float volume;            // Volume of the trade (changed from double to float for optimization)
    TradeSide side;          // Side of the trade (BUY/SELL)
    uint8_t exchange_id;     // Exchange identifier (changed from string to uint8_t for optimization)
    uint8_t flags;           // Bitmask of trade flags

    // Default constructor
    TradeData() : timestamp(0), price(0.0), volume(0.0f), side(TradeSide::BUY), exchange_id(0), flags(0) {}

    // Parameterized constructor
    TradeData(uint64_t ts, double p, float v, TradeSide s, uint8_t ex_id, uint8_t f)
        : timestamp(ts), price(p), volume(v), side(s), exchange_id(ex_id), flags(f) {}
};

// Helper functions for flag manipulation
inline void set_flag(uint8_t& flags, TradeFlags flag) {
    flags |= static_cast<uint8_t>(flag);
}

inline void clear_flag(uint8_t& flags, TradeFlags flag) {
    flags &= ~static_cast<uint8_t>(flag);
}

inline bool has_flag(const uint8_t& flags, TradeFlags flag) {
    return (flags & static_cast<uint8_t>(flag)) != 0;
}

}  // namespace Data
}  // namespace BTQuant