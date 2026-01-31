#pragma once

#include <cstddef>
#include <cstdint>

// Check if HotSpine types are already defined (by tests/new version)
#ifndef HOTSPINE_LAYOUT_HPP
#define HOTSPINE_LAYOUT_HPP

namespace HotSpine {

// Magic number for HotSpine shared memory (0x42545155 = "BTQU" in little endian)
constexpr uint32_t HOTSPINE_MAGIC = 0x42545155;

// Shared memory layout constants
constexpr uint64_t HOTSPINE_VERSION = 2;
constexpr uint64_t DEFAULT_CAPACITY = 1000000;           // 1 million trades
constexpr uint64_t DEFAULT_ORDERBOOK_CAPACITY = 100000;  // 100k orderbooks
constexpr size_t HEADER_SIZE = 4096;                     // 4KB for header

// Shared memory header structure
struct SharedMemoryHeader {
  uint32_t magic;        // Magic number for validation (0x42545155)
  uint32_t version;      // Version number
  uint64_t capacity;     // Number of trade entries in buffer
  uint64_t write_index;  // Write position (next slot to write)
  uint64_t read_index;   // Read position (next slot to read)
  uint64_t lost_count;   // Number of lost trades due to buffer overflow

  // Orderbook fields (replaces previous padding to match Python side)
  uint64_t orderbook_write_index;
  uint64_t orderbook_read_index;
  uint64_t orderbook_lost_count;
  uint64_t orderbook_capacity;  // Number of orderbook entries in buffer

  uint8_t padding[8];  // remaining padding for alignment
};

// Trade data structure (must match between writer and reader)
struct HotTrade {
  uint64_t ts_exchange;  // exchange timestamp in microseconds
  uint64_t ts_local;     // local receive timestamp in microseconds
  double price;
  double size;
  uint32_t symbol_id;  // symbol ID (hash or mapping)
  uint8_t side;        // 0=buy, 1=sell
  uint8_t padding[3];  // Explicit padding for 8-byte alignment
};

// Orderbook level structure
struct HotOrderbookLevel {
  double price;
  double size;
};

// Orderbook snapshot structure (must match between writer and reader)
struct HotOrderbookSnapshot {
  uint64_t ts_exchange;         // exchange timestamp in microseconds
  uint64_t ts_local;            // local receive timestamp in microseconds
  uint32_t symbol_id;           // symbol ID (hash or mapping)
  uint8_t bids_count;           // number of bid levels
  uint8_t asks_count;           // number of ask levels
  uint8_t padding[2];           // Explicit padding for 8-byte alignment of levels array
  HotOrderbookLevel bids[200];  // max 200 bid levels
  HotOrderbookLevel asks[200];  // max 200 ask levels
};

// Calculate total shared memory size needed
static inline size_t calculateSharedMemorySize(uint64_t capacity, uint64_t orderbook_capacity = DEFAULT_ORDERBOOK_CAPACITY) {
  return HEADER_SIZE + (capacity * sizeof(HotTrade)) + (orderbook_capacity * sizeof(HotOrderbookSnapshot));
}

}  // namespace HotSpine

#endif  // HOTSPINE_LAYOUT_HPP
