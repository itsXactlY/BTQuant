#pragma once

#include <cstdint>
#include <cstddef>

#ifndef HOTSPINE_LAYOUT_HPP
#define HOTSPINE_LAYOUT_HPP

namespace HotSpine {

// Magic number for HotSpine shared memory (0x42545155 = "BTQU" in little endian)
constexpr uint32_t SHM_MAGIC = 0x42545155;

// Price level for orderbook entries
struct HotPriceLevel {
    double price;
    double size;
};

// Trade data structure (matches shared memory layout)
struct HotTrade {
    uint64_t ts_exchange;   // Exchange timestamp in microseconds
    uint64_t ts_local;      // Local receive timestamp in microseconds
    double price;
    double size;
    uint32_t symbol_id;
    uint8_t side;           // 0 = buy, 1 = sell
    uint8_t padding[3];     // Padding for alignment
};

// Orderbook snapshot structure (matches shared memory layout)
struct HotOrderbookSnapshot {
    uint64_t ts_exchange;   // Exchange timestamp in microseconds
    uint64_t ts_local;      // Local receive timestamp in microseconds
    uint32_t symbol_id;
    uint8_t bids_count;     // Number of bid levels (max 10)
    uint8_t asks_count;     // Number of ask levels (max 10)
    uint8_t padding[2];     // Padding for alignment
    HotPriceLevel bids[10]; // Bid price levels
    HotPriceLevel asks[10]; // Ask price levels
};

// Constants for orderbook limits
constexpr size_t MAX_BID_LEVELS = 10;
constexpr size_t MAX_ASK_LEVELS = 10;

} // namespace HotSpine

#endif // HOTSPINE_LAYOUT_HPP
