#pragma once

#include <cstdint>
#include <cstring>

#include "../vulkan_base_types.hpp"
#include "../hotspine_layout_v3.hpp"

namespace BTQuant {

// SSBO Snapshot Updater
// Allocates a 4MB storage buffer (1024 columns x 256 rows x 16 bytes/VolumeNode)
// and copies ClusterColumn history data to it for GPU compute consumption.
class SsboSnapshotUpdater {
 public:
  SsboSnapshotUpdater() = default;
  ~SsboSnapshotUpdater() = default;

  // Initialize: allocate SSBO via GPUMemoryManager (device-local storage buffer)
  void initialize(GPUMemoryManager& mem);

  // Copy cluster column data to mapped SSBO
  // history: pointer to array of ClusterColumn
  // count:   number of columns to copy (max 1024)
  void update(const HotSpine::V3::ClusterColumn* history, uint32_t count);

  // Get the buffer allocation for descriptor set binding
  BufferAllocation& get_allocation() { return allocation_; }
  const BufferAllocation& get_allocation() const { return allocation_; }

  bool is_initialized() const { return allocation_.buffer != VK_NULL_HANDLE; }

  // Cleanup
  void destroy(VkDevice device, GPUMemoryManager& mem);

 private:
  BufferAllocation allocation_{};

  // Staging buffer for GPU upload (host-visible)
  BufferAllocation staging_{};
  VkDevice device_ = VK_NULL_HANDLE;
};

}  // namespace BTQuant
