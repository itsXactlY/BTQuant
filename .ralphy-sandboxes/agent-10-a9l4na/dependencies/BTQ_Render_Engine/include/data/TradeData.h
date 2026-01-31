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

// Optimized TradeData struct with careful member ordering for memory alignment
struct TradeData {
    uint64_t timestamp;      // Unix timestamp in milliseconds (8 bytes)
    double price;            // Price of the trade (8 bytes)
    float volume;            // Volume of the trade (4 bytes)
    uint8_t exchange_id;     // Exchange identifier (1 byte)
    uint8_t flags;           // Bitmask of trade flags (1 byte)
    TradeSide side;          // Side of the trade (BUY/SELL) (1 byte)
    uint8_t _reserved;       // Reserved for future use/fills padding (1 byte)

    // Default constructor
    TradeData() : timestamp(0), price(0.0), volume(0.0f), exchange_id(0), flags(0), side(TradeSide::BUY), _reserved(0) {}

    // Parameterized constructor
    TradeData(uint64_t ts, double p, float v, TradeSide s, uint8_t ex_id, uint8_t f)
        : timestamp(ts), price(p), volume(v), exchange_id(ex_id), flags(f), side(s), _reserved(0) {}

    // Convenience methods
    inline bool is_buy() const { return side == TradeSide::BUY; }
    inline bool is_sell() const { return side == TradeSide::SELL; }
    inline bool is_aggressive() const { return has_flag(flags, TradeFlags::AGGRESSIVE_ORDER); }
    inline bool is_passive() const { return has_flag(flags, TradeFlags::PASSIVE_ORDER); }
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