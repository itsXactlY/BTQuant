// Phase 1 SSBO Snapshot Updater — STUB
// Real implementation deferred. The class declaration is part of the Phase 1
// interface contract; this file exists only so the link succeeds when the
// dashboard is built without GPU compute wired up.

#include "vulkan/ssbo_snapshot_updater.hpp"

#include <cstring>
#include <iostream>

namespace BTQuant {

void SsboSnapshotUpdater::initialize(GPUMemoryManager& mem) {
    // Allocate 4MB host-visible storage: 1024 cols * 256 rows * 16 bytes/VolumeNode
    constexpr VkDeviceSize kSsboBytes =
        1024ULL * 256ULL * sizeof(HotSpine::V3::VolumeNode);
    allocation_ = mem.allocate_storage_buffer(kSsboBytes);
    // staging_ would be used for device-local transfer; not needed for stub
}

void SsboSnapshotUpdater::update(const HotSpine::V3::ClusterColumn* history,
                                 uint32_t count) {
    if (!allocation_.buffer || !history || count == 0) return;
    // STUB: in production this would memcpy(count * 256 rows * 16 bytes) to the
    // mapped SSBO. No consumers in this build, so the operation is a no-op.
}

void SsboSnapshotUpdater::destroy(VkDevice device, GPUMemoryManager& mem) {
    if (allocation_.buffer != VK_NULL_HANDLE) {
        mem.deallocate_buffer(allocation_);
        allocation_ = {};
    }
    if (staging_.buffer != VK_NULL_HANDLE) {
        mem.deallocate_buffer(staging_);
        staging_ = {};
    }
}

}  // namespace BTQuant
