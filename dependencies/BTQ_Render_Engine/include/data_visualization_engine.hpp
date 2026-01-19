#pragma once

#include "vulkan_base_types.hpp"
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace BTQuant {
namespace RenderEngine {

struct PriceLevel {
  double price;
  double size;
};

struct SymbolData {
  uint32_t symbol_id;
  double last_price;
  double price_change;
  double price_change_percent;
  double volume_24h;
  double bid_price;
  double ask_price;
  double spread;
  double momentum;
  uint64_t last_update_time;
};

// GPU data structures for efficient rendering
struct ColorRGBA {
  float r, g, b, a;
};

// Symbol flags for GPU rendering
constexpr uint32_t SYMBOL_FLAG_ACTIVE = 1 << 0;
constexpr uint32_t SYMBOL_FLAG_HIGH_MOMENTUM = 1 << 1;
constexpr uint32_t SYMBOL_FLAG_HIGH_VOLUME = 1 << 2;
constexpr uint32_t SYMBOL_FLAG_TRENDING = 1 << 3;

// Grid data structure for GPU (symbol grid display)
struct GridDataGPU {
  uint32_t symbol_id;
  float last_price;
  float price_change;
  float price_change_percent;
  float volume_24h;
  float bid_price;
  float ask_price;
  float spread;
  float momentum;
  ColorRGBA color;
  float intensity;
  uint32_t flags;
  float padding[2]; // Align to 16 bytes
};

// Heatmap data structure for GPU
struct HeatmapDataGPU {
  uint32_t symbol_id;
  float momentum;
  float volume;
  float volatility;
  float intensity;
  ColorRGBA color;
  float grid_x;
  float grid_y;
  float padding[2]; // Align to 16 bytes
};

// Candle data structure for GPU instanced rendering
struct CandleDataGPU {
  float x;
  float open;
  float high;
  float low;
  float close;
  uint32_t color;
  float padding[2]; // Align to 16 bytes
};

// Chart point data structure for GPU
struct ChartPointGPU {
  uint64_t timestamp;
  float price;
  float volume;
  float x; // Normalized X coordinate (0-1)
  float y; // Normalized Y coordinate (0-1)
  ColorRGBA color;
  float padding[3]; // Align to 16 bytes
};

// Orderbook level data structure for GPU
struct OrderbookLevelGPU {
  float price;
  float size;
  float normalized_size; // Size normalized to 0-1 range
  uint32_t is_bid;       // 1 for bid, 0 for ask
  uint32_t level_index;
  ColorRGBA color;
  float padding[3]; // Align to 16 bytes
};

// Chart point for input data
struct ChartPoint {
  uint64_t timestamp;
  double price;
  double volume;
};

// Performance metrics for visualization engine
struct VisualizationPerformanceMetrics {
  uint64_t grid_update_count = 0;
  uint64_t heatmap_update_count = 0;
  uint64_t chart_update_count = 0;
  uint64_t orderbook_update_count = 0;

  double grid_update_latency_us = 0.0;
  double heatmap_update_latency_us = 0.0;
  double chart_update_latency_us = 0.0;
  double orderbook_update_latency_us = 0.0;

  size_t symbols_processed = 0;
  size_t total_gpu_memory_mb = 0;

  std::chrono::high_resolution_clock::time_point last_update_time;
};

/**
 * DataVisualizationEngine - GPU-accelerated data visualization pipeline
 *
 * This class handles the efficient transfer of market data to GPU buffers
 * for high-performance real-time visualization. It manages:
 * - GPU buffer allocation and management
 * - Data transformation for GPU-optimized formats
 * - Color calculation and visual effects
 * - Memory-efficient data streaming
 * - Performance monitoring and optimization
 */
class DataVisualizationEngine {
public:
  /**
   * Constructor
   * @param device Vulkan logical device
   * @param physical_device Vulkan physical device
   */
  DataVisualizationEngine(VkDevice device, VkPhysicalDevice physical_device);

  ~DataVisualizationEngine();

  // Check if initialization was successful
  bool isValid() const { return initialized_; }

  // Non-copyable, non-movable
  DataVisualizationEngine(const DataVisualizationEngine &) = delete;
  DataVisualizationEngine &operator=(const DataVisualizationEngine &) = delete;
  DataVisualizationEngine(DataVisualizationEngine &&) = delete;
  DataVisualizationEngine &operator=(DataVisualizationEngine &&) = delete;

  /**
   * Update grid data for symbol display
   * @param symbols Vector of symbol data to display in grid
   */
  void updateGridData(const std::vector<SymbolData> &symbols);

  /**
   * Update heatmap data for momentum visualization
   * @param symbols Vector of symbol data for heatmap
   */
  void updateHeatmapData(const std::vector<SymbolData> &symbols);

  /**
   * Update chart data for a specific chart chart_id using candle data
   * @param chart_id Unique chart instance ID
   * @param candles Vector of OHLCV candles
   */
  void updateChartData(uint32_t chart_id,
                       const std::vector<OHLCVCandle> &candles,
                       double base_x = 0.0);

  /**
   * Update orderbook data for a specific symbol
   * @param symbol_id Symbol ID for the orderbook
   * @param bids Vector of bid levels
   * @param asks Vector of ask levels
   */
  void updateOrderbookData(uint32_t symbol_id,
                           const std::vector<PriceLevel> &bids,
                           const std::vector<PriceLevel> &asks);

  /**
   * Get performance metrics
   * @return Current visualization performance metrics
   */
  VisualizationPerformanceMetrics getPerformanceMetrics() const;

  /**
   * Get GPU buffer handles for rendering
   */
  VkBuffer getGridBuffer() const;
  VkBuffer getHeatmapBuffer() const;
  VkBuffer getChartBuffer() const;
  VkBuffer getOrderbookBuffer() const;
  VkBuffer getIndicatorBuffer() const;

  /**
   * Configuration
   */
  void setMaxSymbols(size_t max_symbols) { max_symbols_ = max_symbols; }
  void setMaxChartPoints(size_t max_points) { max_chart_points_ = max_points; }
  void setMaxOrderbookLevels(size_t max_levels) {
    max_orderbook_levels_ = max_levels;
  }

  size_t getSymbolIndex(uint32_t symbol_id);

private:
  // Vulkan objects
  VkDevice device_;
  VkPhysicalDevice physical_device_;
  VkCommandPool command_pool_;

  // GPU buffers
  VkBuffer staging_buffer_;
  VkDeviceMemory staging_memory_;

  VkBuffer grid_buffer_;
  VkDeviceMemory grid_memory_;

  VkBuffer heatmap_buffer_;
  VkDeviceMemory heatmap_memory_;

  VkBuffer chart_buffer_;
  VkDeviceMemory chart_memory_;

  VkBuffer orderbook_buffer_;
  VkDeviceMemory orderbook_memory_;

  VkBuffer indicator_buffer_;
  VkDeviceMemory indicator_memory_;

  // Configuration
  size_t max_symbols_;
  size_t max_chart_points_;
  size_t max_orderbook_levels_;
  bool initialized_ = false;

  // Chart mapping (chart_id to GPU buffer slot)
  mutable std::mutex mapping_mutex_;
  std::unordered_map<uint32_t, size_t> chart_id_to_index_;
  size_t next_chart_index_ = 0;

  // Performance tracking
  mutable std::mutex perf_mutex_;
  VisualizationPerformanceMetrics performance_metrics_;

private:
  // Private methods
  bool initializeBuffers();
  bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &memory);
  uint32_t findMemoryType(uint32_t type_filter,
                          VkMemoryPropertyFlags properties);
  void transferDataToGPU(const void *data, size_t size, VkBuffer dst_buffer,
                         size_t offset = 0);

  // Color calculation methods
  ColorRGBA calculatePriceChangeColor(float change_percent);
  ColorRGBA calculateHeatmapColor(float intensity);

  void cleanup();
};

} // namespace RenderEngine
} // namespace BTQuant