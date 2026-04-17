#pragma once

#include <expected>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "../../include/analytics/cluster_engine.hpp"
#include "../../include/market_data_processor.hpp"
#include "../data/VolumeDataTypes.h"
#include "../vulkan_base_types.hpp"
#include "VulkanSynchronization.h"

// Forward declarations and missing type definitions
struct LOBHeatmapConfig {
    uint32_t width = 1024;
    uint32_t height = 512;
    float maxLiquidity = 100000.0f;
    bool invertYAxis = false;
};

struct FootprintChartConfig {
    uint32_t maxClusters = 4096;
    float cellMinSize = 2.0f;
    float cellMaxSize = 20.0f;
    bool showLabels = true;
};

struct TPOProfileConfig {
    uint32_t bucketCount = 256;
    float priceResolution = 0.25f;
    uint32_t timeWindowMs = 30000; // 30 seconds
    bool resetOnUpdate = false;
};

struct RendererConfig {
    LOBHeatmapConfig lobHeatmap;
    FootprintChartConfig footprintChart;
    TPOProfileConfig tpoProfile;
};

namespace BTQuant {
namespace RenderEngine {

struct CandleCluster {
    double centerX;                   // Center X coordinate (time)
    double centerY;                   // Center Y coordinate (price)
    float width;                      // Cluster width (time duration)
    float height;                     // Cluster height (price range)
    uint32_t bidVolume;               // Total bid volume
    uint32_t askVolume;               // Total ask volume
    uint32_t tradeCount;              // Number of trades
    float vwap;                       // Volume-weighted average price
    uint32_t hasTrades;               // Trade activity indicator (0 or 1, was bool)
    uint32_t buyTradeCount;           // Number of buy trades
    uint32_t sellTradeCount;          // Number of sell trades
    float maxSingleTradeVolume;       // Maximum single trade volume in cluster
    uint32_t startTimeLow;            // Start time lower 32 bits (was uint64_t)
    uint32_t startTimeHigh;           // Start time upper 32 bits (was uint64_t)
    uint32_t endTimeLow;              // End time lower 32 bits (was uint64_t)
    uint32_t endTimeHigh;             // End time upper 32 bits (was uint64_t)

    // Constructor
    CandleCluster(double x = 0.0, double y = 0.0, float w = 0.0f, float h = 0.0f,
                 uint32_t bidVol = 0, uint32_t askVol = 0, uint32_t count = 0, float v = 0.0f,
                 bool has = true, uint32_t buyCount = 0, uint32_t sellCount = 0,
                 float maxVol = 0.0f, uint64_t startNs = 0, uint64_t endNs = 0)
        : centerX(x), centerY(y), width(w), height(h), bidVolume(bidVol), askVolume(askVol),
          tradeCount(count), vwap(v), hasTrades(has ? 1u : 0u), buyTradeCount(buyCount),
          sellTradeCount(sellCount), maxSingleTradeVolume(maxVol),
          startTimeLow(static_cast<uint32_t>(startNs)),
          startTimeHigh(static_cast<uint32_t>(startNs >> 32)),
          endTimeLow(static_cast<uint32_t>(endNs)),
          endTimeHigh(static_cast<uint32_t>(endNs >> 32)) {}

    // Helper methods to get/set the full 64-bit timestamps
    uint64_t getStartTimeNs() const {
        return (static_cast<uint64_t>(startTimeHigh) << 32) | startTimeLow;
    }
    
    uint64_t getEndTimeNs() const {
        return (static_cast<uint64_t>(endTimeHigh) << 32) | endTimeLow;
    }
    
    void setStartTimeNs(uint64_t timeNs) {
        startTimeLow = static_cast<uint32_t>(timeNs);
        startTimeHigh = static_cast<uint32_t>(timeNs >> 32);
    }
    
    void setEndTimeNs(uint64_t timeNs) {
        endTimeLow = static_cast<uint32_t>(timeNs);
        endTimeHigh = static_cast<uint32_t>(timeNs >> 32);
    }
};

struct RendererStats {
    uint32_t framesRendered = 0;
    uint32_t lobUpdates = 0;
    uint32_t tradeUpdates = 0;
    uint32_t footprintCellsRendered = 0;
    double averageFrameTimeMs = 0.0;
    uint64_t lastUpdateTimeNs = 0;
    uint32_t totalClusters = 0;
    uint32_t activeClusters = 0;
};

// Forward declaration removed as we iterate include
// class MarketDataProcessor; // Included now

// C++26 Error Types

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

[[nodiscard]] constexpr std::string_view to_string(RendererError error) noexcept {
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
  MarketMicrostructureRenderer(VulkanCore* vulkanCore,
                               std::shared_ptr<MarketDataProcessor> marketDataProcessor,
                               const RendererConfig& config = RendererConfig());

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
  void updateLOBData(const OrderbookData& orderbookData);

  // Update trade data
  void updateTradeData(std::span<const TradeData> trades);

  // Update footprint chart clusters
  void updateFootprintClusters(std::span<const CandleCluster> clusters);

  // Get renderer configuration
  const RendererConfig& getConfig() const { return config_; }

  // Update renderer configuration
  void updateConfig(const RendererConfig& config);

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

  // Set the active symbol for visualization
  // This will subscribe to market data for the given symbol
  void setSymbol(uint32_t symbol_id);

  // Set price aggregation type for footprint charts
  void setPriceAggregationType(Data::PriceAggregationType type) { price_aggregation_type_ = type; }

  // Get price aggregation type
  Data::PriceAggregationType getPriceAggregationType() const { return price_aggregation_type_; }

  // Set custom price aggregation value
  void setCustomPriceAggregationValue(double value) { custom_price_aggregation_value_ = value; }

  // Get custom price aggregation value
  double getCustomPriceAggregationValue() const { return custom_price_aggregation_value_; }

  // Notify that price aggregation settings have changed and clusters need to be recalculated
  void notifyPriceAggregationChanged();

  // Set time aggregation type for footprint charts
  void setTimeAggregationType(Data::TimeAggregationType type) { time_aggregation_type_ = type; }

  // Get time aggregation type
  Data::TimeAggregationType getTimeAggregationType() const { return time_aggregation_type_; }

  // Set volume-based aggregation parameters
  void setVolumeBasedNContracts(int n) { volume_based_n_contracts_ = std::max(1, n); }

  // Get volume-based aggregation parameters
  int getVolumeBasedNContracts() const { return volume_based_n_contracts_; }

  // Set tick-based aggregation parameters
  void setTickBasedNTicks(int n) { tick_based_n_ticks_ = std::max(1, n); }

  // Get tick-based aggregation parameters
  int getTickBasedNTicks() const { return tick_based_n_ticks_; }

  // Get ImGui Texture ID for the LOB Heatmap
  // This utilizes ImGui_ImplVulkan_AddTexture manually
  void* getHeatmapTextureID();

  // Get ClusterCell data for footprint analysis
  // This provides access to the underlying ClusterCell data for advanced analysis
  std::vector<std::vector<Analytics::ClusterCell>> getClusterCells() const;

  // Set a callback function to be called when the cluster engine processes a trade
  // This is used to mark panels as dirty when new trade data arrives
  void set_on_cluster_engine_trade_callback(std::function<void()> callback);

  // Set the panel manager that will be marked as dirty when trades arrive in the cluster engine
  void set_cluster_engine_panel_manager(BTQuant::PanelManager* panel_manager);

 private:
  // Data Update Callback
  void onMarketDataUpdate(uint32_t symbol_id, NotificationType type);

  uint32_t current_symbol_id_ = 0;
  uint64_t subscription_id_ = 0;
  uint64_t subscription_id_lob_ = 0;  // Separate subscription for LOB if needed

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
  VulkanCore* vulkanCore_ = nullptr;
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
  std::vector<TradeData> currentTradeData_;
  std::vector<CandleCluster> currentFootprintClusters_;
  mutable std::mutex dataMutex_;

  // ImGui Texture state
  void* heatmapTextureID_ = nullptr;
  VkDescriptorSet heatmapDescriptorSet_ = VK_NULL_HANDLE;

  // Price aggregation settings
  Data::PriceAggregationType price_aggregation_type_ = Data::PriceAggregationType::P_1TICK;
  double custom_price_aggregation_value_ = 0.1;

  // Time aggregation settings
  Data::TimeAggregationType time_aggregation_type_ = Data::TimeAggregationType::T_1MIN;
  int volume_based_n_contracts_ = 1000;  // Default: every 1000 contracts
  int tick_based_n_ticks_ = 100;         // Default: every 100 ticks

  // Cluster Engine for advanced analytics
  std::unique_ptr<Analytics::ClusterEngine> cluster_engine_;
  
  // Callback to notify when cluster engine processes a trade (for marking panels dirty)
  std::function<void()> on_cluster_engine_trade_callback_;
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
  return {config.width, config.height};
}

// Calculate texture size for heatmap
inline VkDeviceSize getHeatmapTextureSize(const LOBHeatmapConfig& config) {
  return config.width * config.height * 4;  // RGBA 8-bit per channel
}

}  // namespace RenderEngine
}  // namespace BTQuant
