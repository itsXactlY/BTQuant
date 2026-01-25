#pragma once

#include "../vulkan_base_types.hpp"
#include "VulkanSynchronization.h"
#include "hotspine_data_bridge.hpp"
#include "trading/HotspineData.h"
#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace BTQuant {
namespace RenderEngine {

class MarketDataProcessor;

// C++26 Error Types
enum class RendererError {
  NotInitialized,
  NullVulkanCore,
  PipelineCreationFailed,
  BufferAllocationFailed,
  TooManyClusters,
  ShaderLoadFailed,
  ShaderCompilationFailed
};

[[nodiscard]] constexpr std::string_view
to_string(RendererError error) noexcept {
  switch (error) {
  case RendererError::NotInitialized:
    return "Renderer not initialized";
  case RendererError::NullVulkanCore:
    return "VulkanCore is null";
  case RendererError::PipelineCreationFailed:
    return "Pipeline creation failed";
  case RendererError::BufferAllocationFailed:
    return "Buffer allocation failed";
  case RendererError::TooManyClusters:
    return "Too many clusters provided";
  case RendererError::ShaderLoadFailed:
    return "Shader load failed";
  case RendererError::ShaderCompilationFailed:
    return "Shader compilation failed";
  }
  return "Unknown error";
}

// ============================================================================
// Market Microstructure Renderer - Main Component
// ============================================================================

class MarketMicrostructureRenderer {
public:
  MarketMicrostructureRenderer(
      VulkanCore *vulkanCore,
      std::shared_ptr<HotSpineDataBridge> hotspineBridge,
      std::shared_ptr<MarketDataProcessor> marketDataProcessor,
      const RendererConfig &config = RendererConfig());

  ~MarketMicrostructureRenderer();

  // Initialize Vulkan resources (C++26 style)
  [[nodiscard]] std::expected<void, RendererError> initialize();

  // Cleanup Vulkan resources
  void cleanup() noexcept;

  // Prepare data (updates storage buffers, etc.)
  [[nodiscard]] std::expected<void, RendererError> prepare();

  // Execute compute shaders - must be outside render pass
  void executeCompute(VkCommandBuffer cmdBuffer);

  // Execute graphics commands - must be inside render pass
  void executeGraphics(VkCommandBuffer cmdBuffer);

  // Update order book data
  void updateLOBData(const HotspineOrderBookSnapshot &snapshot);

  // Update trade data
  void updateTradeData(std::span<const HotspineTradeTick> trades);

  // Update footprint chart clusters
  void updateFootprintClusters(std::span<const CandleCluster> clusters);

  // Get renderer configuration
  const RendererConfig &getConfig() const { return config_; }

  // Update renderer configuration
  void updateConfig(const RendererConfig &config);

  // Get footprint clusters for UI labels
  std::vector<CandleCluster> getFootprintClusters() const {
    std::lock_guard lock(dataMutex_);
    return currentFootprintClusters_;
  }

  // Get renderer statistics
  RendererStats getStats() const;

  // Reset all statistics
  void resetStats();

  // Check if renderer is initialized
  bool isInitialized() const { return initialized_; }

  // Get ImGui Texture ID for the LOB Heatmap
  // This utilizes ImGui_ImplVulkan_AddTexture manually
  // Get current price bounds for UI synchronization
  std::pair<float, float> getLOBPriceBounds() const {
    std::lock_guard lock(dataMutex_);
    return {lastBasePrice_, lastPriceRange_};
  }

private:
  // Vulkan resource creation
  // Vulkan resource creation
  [[nodiscard]] std::expected<void, RendererError> createComputePipelines();
  [[nodiscard]] std::expected<void, RendererError> createGraphicsPipelines();
  [[nodiscard]] std::expected<void, RendererError> createDescriptorSets();
  [[nodiscard]] std::expected<void, RendererError> createStorageBuffers();
  [[nodiscard]] std::expected<void, RendererError> createTextureResources();

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
  VulkanCore *vulkanCore_ = nullptr;
  std::shared_ptr<HotSpineDataBridge> hotspineBridge_;
  std::shared_ptr<MarketDataProcessor> marketDataProcessor_;
  RendererConfig config_;
  bool initialized_ = false;

  // Renderer statistics
  mutable std::mutex statsMutex_;
  RendererStats stats_;
  std::chrono::high_resolution_clock::time_point lastFrameTime_;

  // Ring buffer for data updates
  std::unique_ptr<RingBufferManager> ringBufferManager_;

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

  // Current frame data (Storage for GPU-friendly formats)
  std::vector<uint8_t> lobSnapshotBuffer_;
  std::vector<HotspineTradeTick> currentTradeData_;
  std::vector<CandleCluster> currentFootprintClusters_;
  mutable std::mutex dataMutex_;

  // ImGui Texture state
  void *heatmapTextureID_ = nullptr;
  VkDescriptorSet heatmapDescriptorSet_ = VK_NULL_HANDLE;

  // Scaling state for UI sync
  float lastBasePrice_ = 0;
  float lastPriceRange_ = 100;
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
inline VkExtent2D getHeatmapExtent(const LOBHeatmapConfig &config) {
  return {config.width, config.height};
}

// Calculate texture size for heatmap
inline VkDeviceSize getHeatmapTextureSize(const LOBHeatmapConfig &config) {
  return config.width * config.height * 4; // RGBA 8-bit per channel
}

} // namespace RenderEngine
} // namespace BTQuant
