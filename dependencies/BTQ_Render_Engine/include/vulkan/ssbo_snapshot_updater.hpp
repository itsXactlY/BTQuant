#pragma once

/// @file ssbo_snapshot_updater.hpp
/// @brief Copies HotSpine VolumeNode cluster data into a GPU SSBO for compute shader consumption.

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>

// Forward declaration in the correct (global) namespace
namespace HotSpine::V3 {
struct SharedMemoryLayoutV3;
}

namespace BTQuant {

class GPUMemoryManager;
struct BufferAllocation;

class SsboSnapshotUpdater {
 public:
  /// Total SSBO size: 1024 columns × 256 rows × 16 bytes/VolumeNode = 4MB
  static constexpr size_t COLUMNS = 1024;
  static constexpr size_t ROWS = 256;
  static constexpr size_t NODE_SIZE = 16;                          // sizeof(VolumeNode)
  static constexpr size_t SSBO_SIZE = COLUMNS * ROWS * NODE_SIZE;  // 4,194,304 bytes

  SsboSnapshotUpdater() = default;

  /// Allocate the SSBO via GPUMemoryManager's storage pool (HOST_VISIBLE | HOST_COHERENT).
  bool initialize(GPUMemoryManager& mem_manager);

  /// Copy one full history snapshot into the SSBO mapped pointer.
  /// Respects SeqLock for consistent reads. Updates running max_volume.
  /// @param layout Pointer to shared memory layout (may be in /dev/shm)
  /// @return true if copy succeeded (SeqLock consistent)
  bool update(const ::HotSpine::V3::SharedMemoryLayoutV3* layout);

  /// Get the Vulkan buffer handle for descriptor binding.
  VkBuffer get_buffer() const;

  /// Get the SSBO size for descriptor binding.
  VkDeviceSize get_buffer_size() const { return SSBO_SIZE; }

  /// Get the running max volume for push constant normalization.
  float get_max_volume() const { return max_volume_; }

  /// Clean up (deallocation is handled by GPUMemoryManager pool).
  void destroy();

 private:
  VkBuffer buffer_ = VK_NULL_HANDLE;
  void* mapped_ptr_ = nullptr;
  VkDeviceSize offset_ = 0;
  uint32_t pool_id_ = 0;
  float max_volume_ = 1.0f;  ///< Running max for normalization
};

}  // namespace BTQuant
