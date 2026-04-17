#pragma once

#include <imgui.h>
#include <implot.h>

#include <chrono>
#include <deque>
#include <memory>
#include <vector>

#include "market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

// Order Book Level Structure for heatmap data
struct OrderBookLevel {
    float price;           // Price level
    uint32_t askQuantity;  // Ask volume at this price
    uint32_t bidQuantity;  // Bid volume at this price
    uint32_t numOrders;    // Number of orders at this price
};

// Trade Bubble Structure
struct TradeBubble {
  double x;            // Time position (X-axis)
  double y;            // Price position (Y-axis)
  double volume;       // Trade volume
  double price;        // Exact price
  bool is_buy;         // true = Buy, false = Sell
  uint64_t timestamp;  // Timestamp for positioning
  float radius;        // Calculated radius for rendering

  // Constructor
  TradeBubble(double x_pos, double y_pos, double vol, double trade_price, bool buy, uint64_t ts)
      : x(x_pos),
        y(y_pos),
        volume(vol),
        price(trade_price),
        is_buy(buy),
        timestamp(ts),
        radius(5.0f) {}  // Default radius
};

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
  explicit DomSurfacePanel(std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  ~DomSurfacePanel() override;

  void render() override;
  void setSymbol(uint32_t symbol_id);

  // Override panel header to add heatmap intensity control
  void render_panel_header();

  // Configuration
  void setHistoryDepth(int depth) { history_depth_ = depth; }
  void setPriceRange(double range) { price_range_ = range; }

  // Getters for serialization
  uint32_t get_symbol_id() const { return current_symbol_id_; }
  double get_price_range() const { return price_range_; }
  int get_price_bins() const { return price_bins_; }
  bool get_auto_scale_price() const { return auto_scale_price_; }
  float get_heatmap_intensity() const { return heatmap_intensity_; }

  // Large Order Marker Configuration
  void setLargeOrderThreshold(double threshold) { large_order_threshold_ = threshold; }
  void setMaxLargeOrderMarkers(int max) { max_large_order_markers_ = max; }
  void setLargeOrderFadeOut(bool enable) { enable_fade_out_ = enable; }
  float get_large_order_threshold() const { return large_order_threshold_; }
  bool get_enable_fade_out() const { return enable_fade_out_; }

  // Setters for deserialization (snake_case aliases)
  void set_price_range(double range) { price_range_ = range; }
  void set_price_bins(int bins) { price_bins_ = bins; }
  void set_auto_scale_price(bool auto_scale) { auto_scale_price_ = auto_scale; }
  void set_large_order_threshold(float threshold) { large_order_threshold_ = threshold; }
  void set_enable_fade_out(bool enable) { enable_fade_out_ = enable; }
  void set_heatmap_intensity(float intensity) { heatmap_intensity_ = intensity; }

  // Persistent Level Configuration
  void setPersistenceThresholdMs(uint64_t ms) { persistence_threshold_ms_ = ms; }
  void setPersistenceTimeoutMs(double ms) { persistence_timeout_ms_ = ms; }
  void setShowPersistentLines(bool show) { show_persistent_lines_ = show; }
  uint64_t getPersistenceThresholdMs() const { return persistence_threshold_ms_; }
  double getPersistenceTimeoutMs() const { return persistence_timeout_ms_; }
  bool getShowPersistentLines() const { return show_persistent_lines_; }

 private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  uint32_t current_symbol_id_ = 0;

  // Visualization parameters
  int history_depth_ = 300;         // Number of snapshots to show (X-axis time)
  int price_bins_ = 100;            // Number of vertical price buckets (Y-axis price)
  double price_range_ = 0.02;       // +/- 2% from mid price
  float heatmap_intensity_ = 1.0f;  // Intensity/sensitivity of heatmap color mapping

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

  // Trade Bubbles System
  std::vector<TradeBubble> trade_bubbles_;
  double max_trade_volume_ = 1.0;                      // For scaling bubble sizes
  static constexpr size_t TRADE_HISTORY_SIZE = 10000;  // Number of recent trades to track

  // Large Order Marker System
  std::vector<LargeOrderMarker> large_order_markers_;
  double median_order_size_ = 0.0;
  std::deque<double> recent_order_sizes_;  // For median calculation
  static constexpr size_t MEDIAN_WINDOW_SIZE = 1000;

  // Large Order Marker Configuration
  float large_order_threshold_ = 10.0f;  // Threshold: order_size > threshold * median_size
  int max_large_order_markers_ = 100;    // Max active markers
  bool enable_fade_out_ = false;         // Enable fade-out after 60 seconds
  static constexpr uint64_t FADE_OUT_DURATION_US = 60'000'000;  // 60 seconds in microseconds

  // Marker Rendering Configuration
  static constexpr float BASE_RADIUS = 8.0f;  // Base radius in pixels
  static constexpr float MIN_RADIUS = 6.0f;   // Minimum radius
  static constexpr float MAX_RADIUS = 40.0f;  // Maximum radius

  // Persistent Large Order Tracker (Horizontal Lines/Rectangles)
  struct PersistentLevel {
    double price;                  // Price level where large order persists
    bool is_bid;                   // true = Bid, false = Ask
    double size;                   // Size of the large order
    uint64_t first_detected_time;  // When first detected at this level
    uint64_t last_updated_time;    // Last time order was seen at this level
    bool is_active;                // Whether the level is currently active

    PersistentLevel(double p, bool b, double s, uint64_t time)
        : price(p),
          is_bid(b),
          size(s),
          first_detected_time(time),
          last_updated_time(time),
          is_active(true) {}
  };

  std::vector<PersistentLevel> persistent_levels_;
  uint64_t persistence_threshold_ms_ = 5000;  // 5 seconds persistence threshold
  double persistence_timeout_ms_ = 30000;     // 30 seconds timeout for inactive levels
  bool show_persistent_lines_ = true;         // Toggle for persistent line display

  // Helper to refresh data buffer
  void updateHeatmapData();

  // Trade Bubbles Methods
  void updateTradeBubbles();
  void processRecentTrades();
  void renderTradeBubbles();
  void cleanupOldTradeBubbles();
  float calculateBubbleRadius(double volume) const;
  ImU32 getBubbleColor(const TradeBubble& bubble) const;

  // Large Order Marker Methods
  void updateLargeOrderMarkers();
  void detectLargeOrders(const RenderEngine::OrderbookData& orderbook);
  void calculateMedianOrderSize();
  void renderLargeOrderMarkers();
  void cleanupOldMarkers();
  float calculateMarkerRadius(double order_size) const;
  ImU32 getMarkerColor(const LargeOrderMarker& marker) const;
  std::string getMarkerTooltip(const LargeOrderMarker& marker) const;

  // Persistent Level Tracking Methods
  void updatePersistentLevels(const RenderEngine::OrderbookData& orderbook);
  void addOrUpdatePersistentLevel(double price, bool is_bid, double size);
  void cleanupInactivePersistentLevels();
  void renderPersistentLevels();
  ImU32 getPersistentLevelColor(const PersistentLevel& level) const;

  // Callback for reactive updates
  void onDataUpdate(uint32_t symbol_id, RenderEngine::NotificationType type);

  // Flush DOM Ruler functionality
  void renderFlushDOMRuler();
  void updateFlushDOMRulerData();
  bool show_flush_dom_ruler_ = true;  // Toggle for flush DOM ruler display
  float flush_dom_ruler_width_ = 0.05f;  // Width as fraction of plot (5%)

  // DOM Ladder (5-column price ladder)
  void renderDOMLadder();
  bool show_dom_ladder_ = true;          // Toggle: ladder vs heatmap-only
  bool show_heatmap_overlay_ = false;    // Keep heatmap as optional background
  int ladder_visible_rows_ = 30;         // Price levels above/below center
  double running_cvd_ = 0.0;             // Cumulative Volume Delta
};

}  // namespace BTQuant
