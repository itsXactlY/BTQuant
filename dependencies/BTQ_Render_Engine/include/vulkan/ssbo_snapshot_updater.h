#ifndef PUBBTQUANT_SSBO_SNAPSHOT_UPDATER_H
#define PUBBTQUANT_SSBO_SNAPSHOT_UPDATER_H

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

#include "../vulkan_base_types.hpp"
#include "../market_data_processor.hpp"

namespace BTQuant {

// GPU-compatible version of OrderBookSnapshot with explicit alignment
struct OrderBookSnapshotGPU {
    uint64_t timestamp;
    uint32_t symbol_id;
    double best_bid;
    double best_ask;
    double best_bid_size;
    double best_ask_size;
    double spread;
    double total_bid_volume;
    double total_ask_volume;
    uint32_t bid_levels_count;
    uint32_t ask_levels_count;

    // Fixed-size arrays for top N price levels (GPU-compatible)
    static constexpr size_t MAX_LEVELS = 20;  // Same as CPU version

    struct Level {
        double price;
        double size;
    };

    Level bids[MAX_LEVELS];
    Level asks[MAX_LEVELS];
};

// SSBO structure for GPU access
struct OrderBookSnapshotSSBO {
    uint32_t numSnapshots;
    OrderBookSnapshotGPU snapshots[1];  // Flexible array member simulation
};

class SSBOSnapshotUpdater {
public:
    explicit SSBOSnapshotUpdater(VulkanCore* vulkan_core);
    ~SSBOSnapshotUpdater();

    // Initialize the SSBO with the maximum number of snapshots it can hold
    void initialize(size_t max_snapshots);

    // Update the SSBO with the latest snapshots from the market data processor
    void updateSSBO(const std::vector<RenderEngine::OrderBookSnapshot>& snapshots);

    // Get the descriptor set for binding to compute shaders
    VkDescriptorSet get_descriptor_set() const;

private:
    VulkanCore* vulkan_core_;
    size_t max_snapshots_{0};

    // SSBO resources
    VkBuffer ssbo_buffer_;
    VkDeviceMemory ssbo_memory_;
    BufferAllocation storage_buffer_allocation_;
    VkDescriptorSet descriptor_set_;

    void create_descriptor_set();
    void copy_via_staging_buffer(const void* data, VkDeviceSize size);
    void cleanup();
};

} // namespace BTQuant

#endif // PUBBTQUANT_SSBO_SNAPSHOT_UPDATER_H