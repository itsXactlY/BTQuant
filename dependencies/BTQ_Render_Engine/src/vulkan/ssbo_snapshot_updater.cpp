/// @file ssbo_snapshot_updater.cpp
/// @brief Copies HotSpine VolumeNode cluster data into a GPU SSBO.

#include "vulkan/ssbo_snapshot_updater.hpp"

#include <algorithm>
#include <cstring>

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

  // SeqLock consistent read: try up to 3 times
  for (int attempt = 0; attempt < 3; ++attempt) {
    uint64_t seq = layout->header.global_lock.read_begin();

    // Copy all VolumeNode rows from all 1024 ClusterColumns
    // Each ClusterColumn has rows[256], each VolumeNode is 16 bytes
    auto* dst = static_cast<uint8_t*>(mapped_ptr_);
    float local_max = 1.0f;

    for (size_t col = 0; col < COLUMNS; ++col) {
      const auto& cluster = layout->history[col];
      const size_t byte_offset = col * ROWS * NODE_SIZE;

      std::memcpy(dst + byte_offset, cluster.rows, ROWS * NODE_SIZE);

      // Track max volume for normalization push constant
      for (size_t row = 0; row < ROWS; ++row) {
        float vol = cluster.rows[row].buy_vol + cluster.rows[row].sell_vol;
        if (vol > local_max) local_max = vol;
      }
    }

    // Verify consistency
    if (!layout->header.global_lock.read_retry(seq)) {
      // Consistent read — update running max with EMA decay
      max_volume_ = std::max(max_volume_ * 0.99f, local_max);
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
