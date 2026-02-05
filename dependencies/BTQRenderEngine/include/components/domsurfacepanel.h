#pragma once

#include "imgui.h"
#include "implot.h"

#include <chrono>
#include <deque>
#include <memory>
#include <vector>

#include "market_data_processor.hpp"
#include "panel_base.hpp"

// Forward declaration for Vulkan
struct VkImage_T;
struct VkImageView_T;
struct VkSampler_T;
struct VkDeviceMemory_T;

typedef VkImage_T* VkImage;
typedef VkImageView_T* VkImageView;
typedef VkSampler_T* VkSampler;
typedef VkDeviceMemory_T* VkDeviceMemory;

namespace BTQuant {
namespace RenderEngine {

// Forward declarations
class VulkanCore;

// Large Order Marker Structure
struct LargeOrderMarker {
  double x;            // Time position (X-axis)
  double y;            // Price position (Y-axis)
  double size;         // Order size
  double price;        // Exact price
  bool is_bid;         // true = Bid, false = Ask
  uint64_t timestamp;  // Timestamp for fade-out
  float radius;        // Calculated radius for rendering

  // Constructor
  LargeOrderMarker(double x_pos, double y_pos, double order_size, double order_price, bool bid,
                   uint64_t ts)
      : x(x_pos),
        y(y_pos),
        size(order_size),
        price(order_price),
        is_bid(bid),
        timestamp(ts),
        radius(8.0f) {}
};

class DomSurfacePanel : public PanelBase {
 public:
  explicit DomSurfacePanel(std::shared_ptr<MarketDataProcessor> processor);
  ~DomSurfacePanel() override;

  void render() override;
  void setSymbol(uint32_t symbol_id);

  // Configuration
  void setHistoryDepth(int depth) { history_depth_ = depth; }
  void setPriceRange(double range) { price_range_ = range; }

  // Large Order Marker Configuration
  void setLargeOrderThreshold(double threshold) { large_order_threshold_ = threshold; }
  void setMaxLargeOrderMarkers(int max) { max_large_order_markers_ = max; }
  void setLargeOrderFadeOut(bool enable) { enable_fade_out_ = enable; }

  // Vulkan resource initialization
  void initializeVulkanResources(VulkanCore* core);

 private:
  std::shared_ptr<MarketDataProcessor> processor_;
  uint32_t current_symbol_id_ = 0;

  // Visualization parameters
  int history_depth_ = 300;    // Number of snapshots to show (X-axis time)
  int price_bins_ = 100;       // Number of vertical price buckets (Y-axis price)
  double price_range_ = 0.02;  // +/- 2% from mid price

  // Data storage for heatmap
  // ImPlot PlotHeatmap data size = rows * cols
  // Rows = Price Levels, Cols = Time
  std::vector<double> heatmap_data_;
  double bounds_min_[2] = {0, 0};  // X min, Y min
  double bounds_max_[2] = {1, 1};  // X max, Y max
  double scale_min_ = 0;
  double scale_max_ = 100;

  // History tracking for alignment
  uint64_t history_start_timestamp_ = 0;
  uint64_t history_end_timestamp_ = 0;

  // Auto-scaling configuration
  bool auto_scale_price_ = true;  // Automatically determine min/max price from history

  // Large Order Marker System
  std::vector<LargeOrderMarker> large_order_markers_;
  double median_order_size_ = 0.0;
  std::deque<double> recent_order_sizes_;  // For median calculation
  static constexpr size_t MEDIAN_WINDOW_SIZE = 1000;

  // Large Order Marker Configuration
  double large_order_threshold_ = 10.0;  // Threshold: order_size > threshold * median_size
  int max_large_order_markers_ = 100;    // Max active markers
  bool enable_fade_out_ = false;         // Enable fade-out after 60 seconds
  static constexpr uint64_t FADE_OUT_DURATION_US = 60'000'000;  // 60 seconds in microseconds

  // Marker Rendering Configuration
  static constexpr float BASE_RADIUS = 8.0f;  // Base radius in pixels
  static constexpr float MIN_RADIUS = 6.0f;   // Minimum radius
  static constexpr float MAX_RADIUS = 40.0f;  // Maximum radius

  // Vulkan resources for accelerated rendering
  VulkanCore* vulkan_core_ = nullptr;
  VkImage heatmap_image_ = nullptr;
  VkImageView heatmap_image_view_ = nullptr;
  VkSampler heatmap_sampler_ = nullptr;
  VkDeviceMemory heatmap_image_memory_ = nullptr;
  void* vulkan_texture_id_ = nullptr;
  
  // Track texture dimensions
  int current_texture_width_ = 0;
  int current_texture_height_ = 0;

  // Helper to refresh data buffer
  void updateHeatmapData();

  // Large Order Marker Methods
  void updateLargeOrderMarkers();
  void detectLargeOrders(const OrderbookData& orderbook);
  void calculateMedianOrderSize();
  void renderLargeOrderMarkers();
  void cleanupOldMarkers();
  float calculateMarkerRadius(double order_size) const;
  ImU32 getMarkerColor(const LargeOrderMarker& marker) const;
  std::string getMarkerTooltip(const LargeOrderMarker& marker) const;

  // Callback for reactive updates
  void onDataUpdate(uint32_t symbol_id, NotificationType type);

  // Vulkan texture management
  void createVulkanTexture();
  void recreateVulkanTexture(int new_width, int new_height);
  void updateVulkanTexture();
  void cleanupVulkanResources();
};

}  // namespace RenderEngine
}  // namespace BTQuant