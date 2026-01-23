#pragma once

#include "imgui.h"
#include "market_data_processor.hpp"
#include "panel_base.hpp"
#include <deque>
#include <implot.h>
#include <memory>
#include <vector>

namespace BTQuant {

class DomSurfacePanel : public PanelBase {
public:
  explicit DomSurfacePanel(
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  ~DomSurfacePanel() override;

  void render() override;
  void setSymbol(uint32_t symbol_id);

  // Configuration
  void setHistoryDepth(int depth) { history_depth_ = depth; }
  void setPriceRange(double range) { price_range_ = range; }

private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  uint32_t current_symbol_id_ = 0;

  // Visualization parameters
  int history_depth_ = 50;    // Number of snapshots to show (X-axis time)
  int price_bins_ = 100;      // Number of vertical price buckets (Y-axis price)
  double price_range_ = 0.02; // +/- 2% from mid price

  // Data storage for heatmap
  // ImPlot PlotHeatmap data size = rows * cols
  // Rows = Price Levels, Cols = Time
  std::vector<double> heatmap_data_;
  double bounds_min_[2] = {0, 0}; // X min, Y min
  double bounds_max_[2] = {1, 1}; // X max, Y max
  double scale_min_ = 0;
  double scale_max_ = 100;

  // Helper to refresh data buffer
  void updateHeatmapData();

  // Callback for reactive updates
  void onDataUpdate(uint32_t symbol_id, RenderEngine::NotificationType type);
};

} // namespace BTQuant
