#pragma once

#include "vulkan_base_types.hpp"
#include "VulkanSynchronization.h"
#include "trading/HotspineData.h"
#include "hotspine_data_bridge.hpp"
#include <memory>
#include <span>

namespace BTQuant {
namespace RenderEngine {

// Forward declarations
class MarketDataProcessor;

// ============================================================================
// Market Microstructure Renderer - Main Component
// ============================================================================

class MarketMicrostructureRenderer {
public:
    MarketMicrostructureRenderer(VulkanCore* vulkanCore,
                                std::shared_ptr<HotSpineDataBridge> hotspineBridge,
                                std::shared_ptr<MarketDataProcessor> marketDataProcessor,
                                const RendererConfig& config = RendererConfig());
    
    ~MarketMicrostructureRenderer();

    // Initialize Vulkan resources
    void initialize();

    // Cleanup Vulkan resources
    void cleanup();

    // Render frame - must be called from within a command buffer
    void render(VkCommandBuffer cmdBuffer, uint32_t currentFrame,
               VulkanSyncContext& syncContext, TimelineSemaphore& timelineSemaphore);

    // Update order book data
    void updateLOBData(const HotspineOrderBookSnapshot& snapshot);

    // Update trade data
    void updateTradeData(std::span<const HotspineTradeTick> trades);

    // Update footprint chart clusters
    void updateFootprintClusters(std::span<const CandleCluster> clusters);

    // Get renderer configuration
    const RendererConfig& getConfig() const { return config_; }

    // Update renderer configuration
    void updateConfig(const RendererConfig& config);

    // Get renderer statistics
    RendererStats getStats() const;

    // Reset all statistics
    void resetStats();

    // Check if renderer is initialized
    bool isInitialized() const { return initialized_; }

private:
    // Vulkan resource creation
    void createComputePipelines();
    void createGraphicsPipelines();
    void createDescriptorSets();
    void createStorageBuffers();
    void createTextureResources();

    // Compute shader execution
    void executeLOBHeatmapCompute(VkCommandBuffer cmdBuffer);
    void executeTPOProfileCompute(VkCommandBuffer cmdBuffer);

    // Graphics rendering
    void renderFootprintChart(VkCommandBuffer cmdBuffer);
    void renderHeatmapTexture(VkCommandBuffer cmdBuffer);
    void renderTPOProfile(VkCommandBuffer cmdBuffer);

    // Resource management
    void updateUniformBuffers(uint32_t currentFrame);
    void updateStorageBuffers();

    // Performance tracking
    void recordFrameStats();

private:
    VulkanCore* vulkanCore_ = nullptr;
    std::shared_ptr<HotSpineDataBridge> hotspineBridge_;
    std::shared_ptr<MarketDataProcessor> marketDataProcessor_;
    RendererConfig config_;
    bool initialized_ = false;

    // Renderer statistics
    mutable std::mutex statsMutex_;
    RendererStats stats_;
    std::chrono::high_resolution_clock::time_point lastFrameTime_;

    // LOB Heatmap resources
    VkPipeline lobHeatmapPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout lobHeatmapPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout lobHeatmapDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet lobHeatmapDescriptorSet_ = VK_NULL_HANDLE;
    BufferAllocation lobHeatmapSSBO_;
    BufferAllocation lobHeatmapUBO_;
    VkImage lobHeatmapImage_ = VK_NULL_HANDLE;
    VkImageView lobHeatmapImageView_ = VK_NULL_HANDLE;
    VkDeviceMemory lobHeatmapImageMemory_ = VK_NULL_HANDLE;
    VkSampler lobHeatmapSampler_ = VK_NULL_HANDLE;
    uint32_t currentHeatmapTimeIndex_ = 0;

    // Footprint Chart resources
    VkPipeline footprintPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout footprintPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout footprintDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet footprintDescriptorSet_ = VK_NULL_HANDLE;
    BufferAllocation footprintSSBO_;
    BufferAllocation footprintUBO_;
    uint32_t currentClusterCount_ = 0;

    // TPO Profile resources
    VkPipeline tpoProfilePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout tpoProfilePipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout tpoProfileDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet tpoProfileDescriptorSet_ = VK_NULL_HANDLE;
    BufferAllocation tpoProfileSSBO_;
    BufferAllocation tpoProfileUBO_;
    BufferAllocation tpoProfileHistogram_;
    bool tpoProfileNeedsReset_ = true;

    // Ring buffer for data updates
    std::unique_ptr<RingBufferManager> ringBufferManager_;

    // Current frame data
    std::vector<HotspineOrderBookSnapshot> currentOrderBookData_;
    std::vector<HotspineTradeTick> currentTradeData_;
    std::vector<CandleCluster> currentFootprintClusters_;
    mutable std::mutex dataMutex_;
};

// ============================================================================
// Helper Functions
// ============================================================================

// Create default renderer configuration
inline RendererConfig createDefaultRendererConfig() {
    RendererConfig config;
    config.lobHeatmap.width = 1024;
    config.lobHeatmap.height = 512;
    config.lobHeatmap.maxLiquidity = 100000.0f;
    config.lobHeatmap.invertYAxis = true;
    
    config.footprintChart.maxClusters = 4096;
    config.footprintChart.cellMinSize = 2.0f;
    config.footprintChart.cellMaxSize = 20.0f;
    config.footprintChart.showLabels = true;
    
    config.tpoProfile.bucketCount = 256;
    config.tpoProfile.priceResolution = 0.1f;
    config.tpoProfile.timeWindowMs = 30000;
    config.tpoProfile.resetOnUpdate = true;
    
    return config;
}

// Calculate heatmap dimensions based on config
inline VkExtent2D getHeatmapExtent(const LOBHeatmapConfig& config) {
    return { config.width, config.height };
}

// Calculate texture size for heatmap
inline VkDeviceSize getHeatmapTextureSize(const LOBHeatmapConfig& config) {
    return config.width * config.height * 4;  // RGBA 8-bit per channel
}

} // namespace RenderEngine
} // namespace BTQuant
