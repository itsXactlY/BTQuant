#include "../../include/vulkan/ssbo_snapshot_updater.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace BTQuant {

void SsboSnapshotUpdater::initialize(GPUMemoryManager& mem) {
  // SSBO size: 1024 columns x 256 rows x sizeof(VolumeNode) = 4MB
  constexpr VkDeviceSize ssbo_size =
      static_cast<VkDeviceSize>(HotSpine::V3::VIEWPORT_ROWS) * 1024 *
      sizeof(HotSpine::V3::VolumeNode);

  allocation_ = mem.allocate_storage_buffer(ssbo_size);

  if (allocation_.buffer == VK_NULL_HANDLE) {
    std::cerr << "[SsboSnapshotUpdater] Failed to allocate SSBO (" << ssbo_size << " bytes)"
              << std::endl;
    return;
  }

  // Allocate a staging buffer (host-visible) for CPU->GPU copy
  staging_ = mem.allocate_staging_buffer(ssbo_size);
  if (staging_.buffer == VK_NULL_HANDLE) {
    std::cerr << "[SsboSnapshotUpdater] Failed to allocate staging buffer" << std::endl;
    return;
  }

  device_ = mem.get_device();

  std::cout << "[SsboSnapshotUpdater] Initialized: SSBO=" << ssbo_size
            << " bytes, offset=" << allocation_.offset << std::endl;
}

void SsboSnapshotUpdater::update(const HotSpine::V3::ClusterColumn* history, uint32_t count) {
  if (!staging_.mapped_ptr || count == 0) return;

  // Clamp count
  uint32_t cols = std::min(count, 1024u);

  // Copy each column's VolumeNode rows into the flat staging buffer
  // Layout: nodes[col * 256 + row] = history[col].rows[row]
  char* dst = static_cast<char*>(staging_.mapped_ptr);
  constexpr size_t rows = HotSpine::V3::VIEWPORT_ROWS;
  constexpr size_t row_bytes = sizeof(HotSpine::V3::VolumeNode);

  for (uint32_t col = 0; col < cols; ++col) {
    std::memcpy(dst + (static_cast<size_t>(col) * rows * row_bytes),
                history[col].rows,
                rows * row_bytes);
  }

  // Zero remaining columns if count < 1024
  if (cols < 1024) {
    size_t remaining = (1024 - cols) * rows * row_bytes;
    std::memset(dst + (static_cast<size_t>(cols) * rows * row_bytes), 0, remaining);
  }
}

void SsboSnapshotUpdater::destroy(VkDevice device, GPUMemoryManager& mem) {
  if (staging_.buffer != VK_NULL_HANDLE) {
    mem.deallocate_buffer(staging_);
    staging_ = {};
  }
  if (allocation_.buffer != VK_NULL_HANDLE) {
    mem.deallocate_buffer(allocation_);
    allocation_ = {};
  }
  device_ = VK_NULL_HANDLE;
}

}  // namespace BTQuant
