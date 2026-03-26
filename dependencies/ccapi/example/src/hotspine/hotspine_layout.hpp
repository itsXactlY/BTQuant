#pragma once

#include <cstdint>
#include <cstddef>

namespace HotSpine {

// Shared memory layout constants
constexpr uint64_t HOTSPINE_VERSION = 1;
constexpr uint64_t DEFAULT_CAPACITY = 1000000; // 1 million trades
constexpr size_t HEADER_SIZE = 4096; // 4KB for header

// Shared memory header structure
struct SharedMemoryHeader {
    uint64_t version;
    uint64_t capacity;
    uint64_t write_index;
    uint64_t read_index;
    uint64_t lost_count;
    uint8_t padding[40];   // for future use and alignment
};

// Trade data structure (must match between writer and reader)
struct HotTrade {
    uint64_t ts_exchange;  // exchange timestamp in microseconds
    uint64_t ts_local;     // local receive timestamp in microseconds  
    double price;
    double size;
    uint32_t symbol_id;    // symbol ID (hash or mapping)
    uint8_t side;         // 0=buy, 1=sell
};

// Calculate total shared memory size needed
static inline size_t calculateSharedMemorySize(uint64_t capacity) {
    return HEADER_SIZE + (capacity * sizeof(HotTrade));
}

} // namespace HotSpine