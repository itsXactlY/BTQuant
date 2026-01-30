#pragma once

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

struct TradeData {
    uint64_t timestamp;      // Unix timestamp in milliseconds
    double price;            // Price of the trade
    double volume;           // Volume of the trade
    TradeSide side;          // Side of the trade (BUY/SELL)
    std::string exchange;    // Exchange where the trade occurred
    
    // Default constructor
    TradeData() : timestamp(0), price(0.0), volume(0.0), side(TradeSide::BUY) {}
    
    // Parameterized constructor
    TradeData(uint64_t ts, double p, double v, TradeSide s, const std::string& ex) 
        : timestamp(ts), price(p), volume(v), side(s), exchange(ex) {}
};

}  // namespace Data
}  // namespace BTQuant