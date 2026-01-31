#pragma once

#include <cstdint>

namespace BTQuant {
namespace Data {

// ============================================================================
// Trade Data Structure
// ============================================================================

enum class TradeSide {
    BUY,
    SELL
};

struct TradeData {
    uint64_t timestamp;      // Unix timestamp in milliseconds
    double price;            // Price of the trade
    float volume;            // Volume of the trade
    TradeSide side;          // Side of the trade (BUY/SELL)
    uint8_t exchange_id;     // ID of the exchange where the trade occurred
    uint8_t flags;           // Bitmask for additional trade flags

    // Default constructor
    TradeData() : timestamp(0), price(0.0), volume(0.0f), side(TradeSide::BUY), exchange_id(0), flags(0) {}

    // Parameterized constructor
    TradeData(uint64_t ts, double p, float v, TradeSide s, uint8_t ex_id, uint8_t f = 0)
        : timestamp(ts), price(p), volume(v), side(s), exchange_id(ex_id), flags(f) {}
};

}  // namespace Data
}  // namespace BTQuant