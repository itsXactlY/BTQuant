/**
 * @file MarketMicrostructureRenderer.h
 * @brief Main Market Microstructure Renderer Component
 * 
 * High-frequency trading visualization system that implements:
 * - LOB Surface (Limit Order Book Heatmap)
 * - Volumetric Footprint Charts (Cluster Chart)
 * - TPO (Time Price Opportunity) Profile
 * 
 * Uses Vulkan 1.3 dynamic rendering and compute shaders for zero-latency visualization.
 * 
 * @author Market Microstructure Renderer Team
 * @version 1.0.0
 */

#ifndef MARKET_MICROSTRUCTURE_RENDERER_H
#define MARKET_MICROSTRUCTURE_RENDERER_H

#include <vulkan/vulkan.h>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "VulkanSynchronization.h"
#include "../trading/HotspineData.h"

namespace components {

// ============================================
// Forward Declarations
// ============================================

class VulkanDevice;
class VulkanSwapChain;
class VulkanRenderPass;
class VulkanPipeline;
class VulkanBuffer;
class VulkanImage;
class VulkanDescriptorSet;

// ============================================
// Configuration Structures
// ============================================

struct LOBHeatmapConfig {
    uint32_t width = 1024;          ///< Heatmap texture width (time steps)
    uint32_t height = 512;          ///< Heatmap texture height (price levels)
    float maxLiquidity = 100000.0f; ///< Maximum liquidity for normalization
    bool invertYAxis = true;        ///< Invert Y-axis (price increasing downward)
};

struct FootprintChartConfig {
    uint32_t maxClusters = 4096;    ///< Maximum number of candle clusters
    float cellMinSize = 2.0f;       ///< Minimum cell size in pixels
    float cellMaxSize = 20.0f;      ///< Maximum cell size in pixels
    bool showLabels = true;         ///< Enable text labels on cells
};

struct TPOProfileConfig {
    uint32_t bucketCount = 256;     ///< Number of histogram buckets
    float priceResolution = 0.1f;   ///< Price resolution per bucket
    uint32_t timeWindowMs = 30000;  ///< Time window in milliseconds
    bool resetOnUpdate = true;      ///< Reset histogram on each update
};

struct RendererConfig {
    LOBHeatmapConfig lobHeatmap;
    FootprintChartConfig footprintChart;
    TPOProfileConfig tpoProfile;
};

// ============================================
// Renderer Statistics
// ============================================

struct RendererStats {
    uint32_t framesRendered = 0;
    uint32_t lobUpdates = 0;
    uint32_t tradeUpdates = 0;
    uint32_t footprintCellsRendered = 0;
    double averageFrameTimeMs = 0.0;
    uint64_t lastUpdateTimeNs = 0;
};

// ============================================
// Market Microstructure Renderer
// ============================================

class MarketMicrostructureRenderer {
public:
    /**
     * @brief Create and initialize the renderer
     * @param device Vulkan device pointer
     * @param config Renderer configuration
     */
    static std::unique_ptr<MarketMicrostructureRenderer> create(VkDevice device,
                                                               const RendererConfig& config);
    
    ~MarketMicrostructureRenderer();
    
    MarketMicrostructureRenderer(const MarketMicrostructureRenderer&) = delete;
    MarketMicrostructureRenderer& operator=(const MarketMicrostructureRenderer&) = delete;
    MarketMicrostructureRenderer(MarketMicrostructureRenderer&&) noexcept;
    MarketMicrostructureRenderer& operator=(MarketMicrostructureRenderer&&) noexcept;
    
    /**
     * @brief Initialize the renderer
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Render the market microstructure visualization
     * @param cmdBuffer Command buffer to record into
     * @param currentFrame Current frame index
     * @param syncContext Synchronization context
     * @param timelineSemaphore Timeline semaphore for synchronization
     */
    bool render(VkCommandBuffer cmdBuffer,
               uint32_t currentFrame,
               vk::VulkanSyncContext& syncContext,
               vk::TimelineSemaphore& timelineSemaphore);
    
    /**
     * @brief Update LOB data from Hotspine
     * @param snapshot Order book snapshot
     * @return true if update scheduled successfully
     */
    bool updateLOBData(const trading::HotspineOrderBookSnapshot& snapshot);
    
    /**
     * @brief Update trade data from Hotspine
     * @param trades Trade ticks data
     * @return true if update scheduled successfully
     */
    bool updateTradeData(const trading::HotspineTradeTicks& trades);
    
    /**
     * @brief Update footprint chart clusters
     * @param clusters Candle clusters
     * @return true if update scheduled successfully
     */
    bool updateFootprintClusters(std::span<const trading::CandleCluster> clusters);
    
    /**
     * @brief Get current renderer statistics
     */
    RendererStats getStats() const;
    
    /**
     * @brief Get current configuration
     */
    const RendererConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set new heatmap configuration
     */
    void setHeatmapConfig(const LOBHeatmapConfig& config);
    
    /**
     * @brief Set new footprint chart configuration
     */
    void setFootprintConfig(const FootprintChartConfig& config);
    
    /**
     * @brief Set new TPO profile configuration
     */
    void setTPOConfig(const TPOProfileConfig& config);
    
private:
    MarketMicrostructureRenderer(VkDevice device, const RendererConfig& config);
    
    bool createComputePipelines();
    bool createRenderingPipelines();
    bool createDescriptorSets();
    bool createBuffers();
    bool createImages();
    
    void updateHeatmapCompute(VkCommandBuffer cmdBuffer, uint32_t currentFrame);
    void updateTPOCompute(VkCommandBuffer cmdBuffer, uint32_t currentFrame);
    void renderHeatmap(VkCommandBuffer cmdBuffer, uint32_t currentFrame);
    void renderFootprintChart(VkCommandBuffer cmdBuffer, uint32_t currentFrame);
    void renderTPOProfile(VkCommandBuffer cmdBuffer, uint32_t currentFrame);
    
    void computeHeatmapParams(const trading::HotspineOrderBookSnapshot& snapshot);
    
    // Core Vulkan resources
    VkDevice device_;
    RendererConfig config_;
    std::atomic<bool> initialized_{false};
    
    // Compute pipelines
    struct ComputePipelines {
        VkPipeline lobHeatmap;
        VkPipelineLayout lobHeatmapLayout;
        VkPipeline tpoProfile;
        VkPipelineLayout tpoProfileLayout;
    } computePipelines_;
    
    // Rendering pipelines
    struct RenderPipelines {
        VkPipeline heatmapSampler;
        VkPipelineLayout heatmapSamplerLayout;
        VkPipeline footprintChart;
        VkPipelineLayout footprintChartLayout;
        VkPipeline tpoProfile;
        VkPipelineLayout tpoProfileLayout;
    } renderPipelines_;
    
    // Descriptor sets
    struct DescriptorSets {
        VkDescriptorPool pool;
        VkDescriptorSetLayout lobHeatmap;
        VkDescriptorSet lobHeatmapSet;
        VkDescriptorSetLayout tpoProfile;
        VkDescriptorSet tpoProfileSet;
        VkDescriptorSetLayout footprintChart;
        VkDescriptorSet footprintChartSet;
    } descriptorSets_;
    
    // LOB Heatmap resources
    struct LOBResources {
        std::unique_ptr<VulkanBuffer> orderBookSSBO;
        std::unique_ptr<VulkanImage> heatmapImage;
        std::unique_ptr<VulkanBuffer> heatmapParamsUBO;
        uint32_t currentTimeIndex = 0;
        float basePrice = 0.0f;
        float priceRange = 0.0f;
        bool needsUpdate = false;
        std::mutex updateMutex;
    } lobResources_;
    
    // TPO Profile resources
    struct TPOResources {
        std::unique_ptr<VulkanBuffer> tradeTicksSSBO;
        std::unique_ptr<VulkanBuffer> tpoHistogramSSBO;
        std::unique_ptr<VulkanBuffer> tpoParamsUBO;
        bool needsUpdate = false;
        std::mutex updateMutex;
    } tpoResources_;
    
    // Footprint Chart resources
    struct FootprintResources {
        std::unique_ptr<VulkanBuffer> clustersSSBO;
        std::unique_ptr<VulkanBuffer> viewParamsUBO;
        std::unique_ptr<VulkanBuffer> textParamsUBO;
        std::unique_ptr<VulkanImage> fontAtlasImage;
        std::unique_ptr<VulkanBuffer> vertexBuffer;
        std::unique_ptr<VulkanBuffer> indexBuffer;
        uint32_t indexCount = 0;
        uint32_t clusterCount = 0;
        bool needsUpdate = false;
        std::mutex updateMutex;
    } footprintResources_;
    
    // Timeline semaphore for zero-latency synchronization
    vk::TimelineSemaphore timelineSemaphore_;
    
    // Synchronization context
    std::unique_ptr<vk::HotspineBarrierManager> barrierManager_;
    std::unique_ptr<vk::RingBufferSyncManager> ringBufferSync_;
    
    // Statistics tracking
    RendererStats stats_;
    mutable std::mutex statsMutex_;
    uint64_t lastFrameTimeNs_ = 0;
    
    // Threading synchronization
    std::condition_variable updateCondition_;
    std::mutex updateMutex_;
    std::atomic<bool> shouldExit_{false};
};

} // namespace components

#endif // MARKET_MICROSTRUCTURE_RENDERER_H