#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>

namespace HotSpine::V3 {

// =========================================================================================
// 1.1 Atomic Primitives (The SeqLock)
// =========================================================================================
struct SeqLock {
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
// 1.4 The Global Layout
// =========================================================================================
struct SharedMemoryLayoutV3 {
  struct Header {
    uint32_t magic;  // 0x42545133 "BTQ3"
    uint32_t padding;
    SeqLock global_lock;
    std::atomic<uint64_t> head_index;
    uint8_t reserved[32];  // Padding to align body
  };

  Header header;

  // Body
  ClusterColumn history[1024];  // Ring buffer
  HeatmapBin dom[512];          // Aggregated DOM
};

static_assert(sizeof(VolumeNode) == 16);
static_assert(alignof(ClusterColumn) == 64);

}  // namespace HotSpine::V3
