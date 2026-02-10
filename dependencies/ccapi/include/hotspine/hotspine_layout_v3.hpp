#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>

// Check if HotSpine V3 types are already defined
#ifndef HOTSPINE_LAYOUT_V3_HPP
#define HOTSPINE_LAYOUT_V3_HPP

namespace HotSpine {
namespace V3 {

// Ring buffer size definition (2^20 = 1048576)
constexpr uint64_t RING_BUFFER_SIZE = 1048576;

// Magic number for HotSpine shared memory (0x42545155 = "BTQU" in little endian)
constexpr uint32_t HOTSPINE_MAGIC = 0x42545155;

// Shared memory layout constants
constexpr uint64_t HOTSPINE_VERSION = 3;
constexpr size_t HEADER_SIZE = 4096;                     // 4KB for header

// SeqLock for synchronization
struct SeqLock {
  std::atomic<uint64_t> sequence{0};
};

// Shared memory header structure
struct SharedMemoryHeaderV3 {
  uint32_t magic;                                    // Magic number for validation (0x42545155)
  uint32_t version;                                  // Version number
  uint64_t capacity;                                 // Number of entries in buffer (should be RING_BUFFER_SIZE)
  std::atomic<uint64_t> write_head{0};               // Write position (next slot to write)
  std::atomic<uint64_t> read_tail{0};                // Read position (next slot to read)
  std::atomic<uint64_t> lost_count{0};               // Number of lost entries due to buffer overflow

  uint8_t padding[4016];                             // remaining space for alignment (4096 - 80 bytes used)
};

// Shared memory layout structure
struct SharedMemoryLayoutV3 {
  SeqLock seqlock;                    // Sequence lock for synchronization
  SharedMemoryHeaderV3 header;        // Header with metadata
  uint8_t padding[0];                 // Flexible array member equivalent for data
};

// Data entry structure (to be defined based on actual data type)
struct HotSpineData {
  // Placeholder for actual data structure
  // This would typically be HotTrade, HotOrderbookSnapshot, or similar
  uint64_t timestamp;
  uint32_t symbol_id;
  double price;
  double size;
  uint8_t data_type;     // 0=trade, 1=orderbook, etc.
  uint8_t side;          // 0=buy, 1=sell (for trades)
  uint8_t padding[2];    // Explicit padding for alignment
};

// Calculate index using mask (efficient modulo operation)
static inline uint64_t getIndex(uint64_t head) {
  return head & (RING_BUFFER_SIZE - 1);
}

// Calculate total shared memory size needed
static inline size_t calculateSharedMemorySize(uint64_t capacity = RING_BUFFER_SIZE) {
  return HEADER_SIZE + (capacity * sizeof(HotSpineData));
}

}  // namespace V3
}  // namespace HotSpine

#endif  // HOTSPINE_LAYOUT_V3_HPP