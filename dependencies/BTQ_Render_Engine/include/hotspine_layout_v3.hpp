#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <cstddef> // fix size_t
#include <cstdint>
#include <new>
#include <span>

// ============================================================================
// AGENT 1: PROTOCOL DESIGNER
// MISSION: Zero-Latency, Lock-Free Shared Memory Layout (V3)
// ============================================================================

namespace HotSpine::V3 {

// Alignment Constants to prevent False Sharing (Cache Coherence)
constexpr std::size_t CACHE_LINE_SIZE = 64;

// ------------------------------------------------------------------------
// atomic_seq_lock: Zero-Mutex Concurrency
// Writer: Increment to ODD (locking), Update, Increment to EVEN (release).
// Reader: Read seq (must be even), Read Data, Re-read seq. Retry if diff.
// ------------------------------------------------------------------------
struct alignas(CACHE_LINE_SIZE) AtomicSeqLock {
  std::atomic<uint64_t> seq_{0};

  // Writer Side
  void begin_write() {
    uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_release); // Make ODD
    std::atomic_signal_fence(std::memory_order_acq_rel);
  }

  void end_write() {
    std::atomic_signal_fence(std::memory_order_acq_rel);
    uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_release); // Make EVEN
  }

  // Reader Side helper (Non-blocking)
  // returns true if snapshot is consistent
  template <typename Func> bool read_optimistic(Func &&read_op) const {
    uint64_t s1 = seq_.load(std::memory_order_acquire);
    if (s1 & 1)
      return false; // Locked by writer

    read_op(); // Perform copy/read

    std::atomic_thread_fence(std::memory_order_acquire);
    uint64_t s2 = seq_.load(std::memory_order_relaxed);
    return s1 == s2;
  }
};

// ------------------------------------------------------------------------
// VolumeNode: 16 Bytes Packed
// Represents a single price level's volume in the footprint
// ------------------------------------------------------------------------
struct alignas(16) VolumeNode {
  double price;  // 8 bytes
  double volume; // 8 bytes
};
static_assert(sizeof(VolumeNode) == 16, "VolumeNode must be 16 bytes");
static_assert(std::is_standard_layout_v<VolumeNode>);

// ------------------------------------------------------------------------
// ClusterColumn: The Viewport (Render Window)
// Fixed-size array representing the visible price ladder or TPO profile
// This is valid POD (Plain Old Data) for direct GPU upload or ImGui render
// ------------------------------------------------------------------------
constexpr std::size_t VIEWPORT_ROWS = 256;

struct alignas(CACHE_LINE_SIZE) ClusterColumn {
  uint64_t timestamp_us;
  uint32_t symbol_id;
  uint32_t active_rows; // How many rows are actually populated

  // High/Low range for this column
  double high_price;
  double low_price;
  double total_volume;
  double delta; // Buy Vol - Sell Vol

  // The Render Data
  std::array<VolumeNode, VIEWPORT_ROWS> rows;
};

// ------------------------------------------------------------------------
// SharedMemoryLayoutV3: The Ring Buffer
// 1024 Columns. Reader chases Head.
// ------------------------------------------------------------------------
constexpr std::size_t RING_BUFFER_SIZE = 1024;

struct alignas(CACHE_LINE_SIZE) SharedMemoryLayoutV3 {
  // Control Block
  static constexpr uint64_t MAGIC = 0x484F5433; // "HOT3" in ASCII
  uint64_t magic;
  uint64_t version;

  alignas(CACHE_LINE_SIZE) AtomicSeqLock header_lock;
  std::atomic<uint64_t> head_index{0}; // Monotonically increasing

  // Data Block
  // We do not lock individual slots; we rely on the head_index and
  // the fact that we won't wrap around fast enough to corrupt the
  // reader's specific slot before they are done (or they detect tear).
  // For strict correctness, each slot can have its own SeqLock if needed,
  // but for high-throughput ring buffers, a single head update is often
  // sufficient IF the reader is fast. However, to satisfy directives:
  // "Implement the atomic SeqLock mechanism", we will embed a SeqLock PER SLOT
  // to allow random access reading of history without fearing overwritten data
  // during the read.

  struct Slot {
    alignas(CACHE_LINE_SIZE) AtomicSeqLock seq_lock;
    ClusterColumn data;
  };

  std::array<Slot, RING_BUFFER_SIZE> buffer;

  // Helper to get slot
  Slot &get_slot(uint64_t index) { return buffer[index % RING_BUFFER_SIZE]; }

  const Slot &get_slot(uint64_t index) const {
    return buffer[index % RING_BUFFER_SIZE];
  }
};

// Safety Checks
static_assert(std::is_standard_layout_v<SharedMemoryLayoutV3>);

} // namespace HotSpine::V3
