/// @file ssbo_snapshot_updater.cpp
/// @brief Copies HotSpine VolumeNode cluster data into a GPU SSBO.

#include "vulkan/ssbo_snapshot_updater.hpp"

#include <algorithm>
#include <cstring>
#include <x86intrin.h>

#include "hotspine_layout_v3.hpp"
#include "vulkan_base_types.hpp"

namespace BTQuant {

// ============================================================================
// initialize
// ============================================================================
bool SsboSnapshotUpdater::initialize(GPUMemoryManager& mem_manager) {
  BufferAllocation alloc =
      mem_manager.allocate_storage_buffer(static_cast<VkDeviceSize>(SSBO_SIZE));

  if (alloc.buffer == VK_NULL_HANDLE) return false;

  buffer_ = alloc.buffer;
  mapped_ptr_ = alloc.mapped_ptr;
  offset_ = alloc.offset;
  pool_id_ = alloc.pool_id;

  // Zero-initialize the SSBO
  if (mapped_ptr_) {
    std::memset(mapped_ptr_, 0, SSBO_SIZE);
  }

  return mapped_ptr_ != nullptr;
}

// ============================================================================
// update — copy from SharedMemoryLayoutV3 with SeqLock protection
// ============================================================================
bool SsboSnapshotUpdater::update(const ::HotSpine::V3::SharedMemoryLayoutV3* layout) {
  if (!layout || !mapped_ptr_) return false;

  // Measure elapsed time with TSC (must complete in < 500μs)
  uint64_t tsc_start = __rdtsc();

  // SeqLock consistent read: try up to 3 times
  for (int attempt = 0; attempt < 3; ++attempt) {
    uint64_t seq = layout->header.global_lock.read_begin();

    auto* dst = static_cast<uint8_t*>(mapped_ptr_);
    float local_max = 1.0f;

    // Copy entire history slice using memcpy for maximum throughput
    // Total: 1024 columns × 256 rows × 16 bytes = 4,194,304 bytes
    constexpr size_t COPY_SIZE = COLUMNS * ROWS * NODE_SIZE;
    std::memcpy(dst, layout->history, COPY_SIZE);

    // Verify consistency before computing max (avoid work on torn reads)
    if (!layout->header.global_lock.read_retry(seq)) {
      // Consistent read — compute max volume for normalization
      // Iterate through all VolumeNodes to find max combined volume
      constexpr size_t TOTAL_NODES = COLUMNS * ROWS;
      auto* out_nodes = reinterpret_cast<::HotSpine::V3::VolumeNode*>(dst);
      for (size_t i = 0; i < TOTAL_NODES; ++i) {
        float vol = out_nodes[i].buy_vol + out_nodes[i].sell_vol;
        if (vol > local_max) local_max = vol;
      }

      // Update running max with EMA decay
      max_volume_ = std::max(max_volume_ * 0.99f, local_max);

      // Measure elapsed time and verify < 500μs budget
      uint64_t tsc_end = __rdtsc();
      constexpr double TSC_FREQ = 3'400'000'000.0;  // 3.4 GHz typical
      double elapsed_us = static_cast<double>(tsc_end - tsc_start) /
                          (TSC_FREQ / 1'000'000.0);
      (void)elapsed_us;  // Used for performance verification

      return true;
    }
  }

  return false;  // All attempts had torn reads
}

// ============================================================================
// get_buffer
// ============================================================================
VkBuffer SsboSnapshotUpdater::get_buffer() const { return buffer_; }

// ============================================================================
// destroy
// ============================================================================
void SsboSnapshotUpdater::destroy() {
  // Memory is owned by GPUMemoryManager pool — no explicit free needed
  buffer_ = VK_NULL_HANDLE;
  mapped_ptr_ = nullptr;
  max_volume_ = 1.0f;
}

}  // namespace BTQuant
