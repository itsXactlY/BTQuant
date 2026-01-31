#pragma once

#include <cstdint>

namespace BTQuant {
namespace Data {

// ============================================================================
// Optimized Trade Data Structure
// ============================================================================

enum class TradeSide : uint8_t {
    BUY = 0,
    SELL = 1
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

// Optimized TradeData struct with memory layout considerations
// Total size: 23 bytes (packed efficiently with padding added by compiler if needed)
struct __attribute__((packed)) TradeData {
    uint64_t timestamp;      // Unix timestamp in milliseconds (8 bytes)
    double price;            // Price of the trade (8 bytes)
    float volume;            // Volume of the trade (4 bytes)
    TradeSide side;          // Side of the trade (BUY/SELL) (1 byte)
    uint8_t exchange_id;     // Exchange identifier (1 byte)
    uint8_t flags;           // Bitmask of trade flags (1 byte)

    // Default constructor
    TradeData() : timestamp(0), price(0.0), volume(0.0f), side(TradeSide::BUY),
                  exchange_id(0), flags(0) {}

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