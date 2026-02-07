#pragma once

#include <imgui.h>

#include <memory>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include "panel_manager.hpp"  // Include for PanelManager methods

namespace BTQuant {

enum class DepthChartVisualizationMode {
  CUMULATIVE_AREA,  // Current mode: cumulative depth as area chart
  SEPARATE_SIDES,   // Alternative mode: bid area on left, ask area on right (negative values)
  BID_ASK_SPLIT      // New mode: bid area on left, ask area on right as separate areas
};

/**
 * DepthChartPanel - Cumulative bid/ask depth visualization
 *
 * C++26 Reactive Architecture:
 * - Subscribes to ORDERBOOK notifications from MarketDataProcessor
 * - markDirty() in callback, consumeDirty() in render()
 * - No polling timer - event-driven updates
 *
 * Uses ImPlot for professional financial charting:
 * - Shaded area for bid depth (green)
 * - Shaded area for ask depth (red)
 * - Gradient fills with alpha
 */
class DepthChartPanel : public PanelBase {
 public:
  DepthChartPanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                  std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  ~DepthChartPanel() override;

  void render() override;
  void set_symbol(uint32_t symbol_id, const std::string& symbol_name);

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  uint32_t symbol_id_ = 0;
  std::string symbol_name_ = "BTC-USDT";
  DepthChartVisualizationMode visualization_mode_ = DepthChartVisualizationMode::CUMULATIVE_AREA;

  // Cached orderbook data - pre-computed for rendering
  RenderEngine::OrderbookData cached_orderbook_;

  // Pre-computed plot data (avoids per-frame allocation)
  std::vector<double> bid_prices_;
  std::vector<double> bid_cumulative_;
  std::vector<double> ask_prices_;
  std::vector<double> ask_cumulative_;
  double mid_price_ = 0.0;
  double max_depth_ = 1.0;

  void compute_depth_data();
  void render_depth_chart_implot();
  void render_depth_chart_separate_sides();
  void render_depth_chart_bid_ask_split();
  void render_visualization_mode_selector();
  void render_stats();
  void subscribe_to_updates();
};

}  // namespace BTQuant
