#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>

// Constants
constexpr uint32_t HOTSPINE_MAGIC = 0x42545155;  // "BTQU"
constexpr size_t HEADER_SIZE = sizeof(RingBufferHeader);

namespace HotSpine::V3 {

// Helper functions
inline constexpr uint64_t getIndex(uint64_t counter) {
    return counter & RING_BUFFER_MASK;
}

// =========================================================================================
// 1.1 Atomic Primitives (The SeqLock)
// =========================================================================================
struct alignas(64) SeqLock {
  std::atomic<uint64_t> seq{0};

  void write_begin() {
    // Increment to odd
    seq.fetch_add(1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_acquire);
  }

  void write_end() {
    std::atomic_thread_fence(std::memory_order_release);
    // Increment to even
    seq.fetch_add(1, std::memory_order_release);
  }

  uint64_t read_begin() const { return seq.load(std::memory_order_acquire); }

  bool read_retry(uint64_t start_seq) const {
    std::atomic_thread_fence(std::memory_order_acquire);
    return (start_seq % 2 != 0) || (seq.load(std::memory_order_relaxed) != start_seq);
  }
};

// =========================================================================================
// 1.2 Data Atoms (The "Pixel")
// =========================================================================================
struct alignas(16) VolumeNode {
  float buy_vol;         // 4B
  float sell_vol;        // 4B
  uint16_t trade_count;  // 2B
  uint16_t tpo_bits;     // 2B - Bitmask for 30min brackets (0-15)
  uint8_t padding[4];    // 4B
};
static_assert(sizeof(VolumeNode) == 16, "VolumeNode size mismatch");

// =========================================================================================
// 1.3 The Viewport (The Render Window)
// =========================================================================================
constexpr std::size_t VIEWPORT_ROWS = 256;

struct alignas(64) ClusterColumn {
  int64_t timestamp_us;
  double open;
  double high;
  double low;
  double close;
  int64_t base_tick_index;  // The absolute price index of row 0
  double tick_size;

  VolumeNode rows[VIEWPORT_ROWS];  // The visual rows
};
// Size check: 8 + 8*4 + 8 + 8 + 16*256 = 56 + 4096 = 4152 bytes.
// alignas(64) pads it to multiple of 64. 4152 / 64 = 64.875 -> 4160 bytes.

struct alignas(64) HeatmapBin {
  int64_t price_tick_index;
  double total_volume;
  uint32_t order_count;
  uint32_t padding;
};

// =========================================================================================
// 1.4 Ring Buffer Layout (Raw Ring Buffer Header)
// =========================================================================================
constexpr size_t RING_BUFFER_SIZE = 8192; // Power of 2 for efficient masking
constexpr size_t RING_BUFFER_MASK = RING_BUFFER_SIZE - 1; // For indexing: idx = counter & MASK

struct alignas(64) RingBufferHeader {
  uint32_t magic;  // 0x42545155 "BTQ3"
  uint32_t version; // Version identifier
  alignas(64) std::atomic<uint64_t> write_head{0};  // Index of next write slot
  alignas(64) std::atomic<uint64_t> read_tail{0};   // Index of next read slot
  std::atomic<uint64_t> dropped_count{0};  // Count of dropped events due to overflow
  uint8_t reserved[24];  // Padding to align to 64-byte boundary
  
  // Inline helper methods
  inline uint64_t get_next_write_slot() const {
    return write_head.load(std::memory_order_acquire) & RING_BUFFER_MASK;
  }
  
  inline void commit_write() {
    write_head.fetch_add(1, std::memory_order_release);
  }
  
  inline uint64_t get_available_count() const {
    uint64_t write_idx = write_head.load(std::memory_order_acquire);
    uint64_t read_idx = read_tail.load(std::memory_order_acquire);
    return write_idx - read_idx;
  }
  
  inline bool is_full() const {
    return get_available_count() >= RING_BUFFER_SIZE;
  }
  
  inline bool is_empty() const {
    return get_available_count() == 0;
  }
};

// =========================================================================================
// 1.5 HotspineData — the core data struct for shared memory
// =========================================================================================
struct alignas(64) HotspineData {
  uint64_t timestamp;            // Nanosecond timestamp of the event
  uint32_t symbol_id;            // Symbol identifier
  uint32_t event_type;           // Type of event (trade, quote, etc.)
  double price;                  // Price value
  double volume;                 // Volume value
  uint8_t flags;                 // Flags: Bit 0: IS_WARMUP, Bit 1: IS_SNAPSHOT
  uint8_t reserved_flags[3];     // Reserved for future flags
  uint32_t sequence_number;      // Sequence number for ordering
  uint32_t payload_size;         // Size of additional payload data
  uint8_t padding[20];           // Explicit padding to reach 64 bytes total

  // Flag bit positions
  static constexpr uint8_t IS_WARMUP = 0x01;    // Bit 0: Warm-up event
  static constexpr uint8_t IS_SNAPSHOT = 0x02;  // Bit 1: Snapshot event
};

// =========================================================================================
// 1.6 The Global Layout
// =========================================================================================
struct SharedMemoryLayoutV3 {
  SeqLock seqlock;             // SeqLock for atomic reads
  RingBufferHeader header;  // Ring buffer header with write_head and read_tail

  // Flexible array member for ring buffer data (C++ equivalent using byte array)
  // Using a fixed size for now - individual events will be written at calculated offsets
  alignas(64) uint8_t ring_buffer_data[RING_BUFFER_SIZE * 64]; // Assuming max 64 bytes per event

  // Legacy fields preserved for compatibility (may be removed later)
  ClusterColumn history[1024];  // Ring buffer (kept for backward compatibility)
  HeatmapBin dom[512];          // Aggregated DOM
};

// Constants
constexpr size_t HEADER_SIZE = sizeof(RingBufferHeader);

static_assert(sizeof(VolumeNode) == 16);
static_assert(alignof(ClusterColumn) == 64);
static_assert(alignof(RingBufferHeader) == 64);
static_assert(alignof(SeqLock) == 64, "SeqLock must be 64-byte aligned");
static_assert(sizeof(HotspineData) == 64, "HotspineData must be exactly 64 bytes for cache alignment");
static_assert(alignof(HotspineData) == 64, "HotspineData must be 64-byte aligned");

// Helper: calculate total shared memory size
static inline constexpr size_t calculateSharedMemorySize(size_t ring_buffer_size) {
  return sizeof(SharedMemoryLayoutV3) + (ring_buffer_size * 64);
}

// Helper functions
inline constexpr uint64_t getIndex(uint64_t counter) {
    return counter & RING_BUFFER_MASK;
}

}  // namespace HotSpine::V3
