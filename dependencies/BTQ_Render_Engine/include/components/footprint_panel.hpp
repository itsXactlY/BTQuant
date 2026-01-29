#pragma once

#include "MarketMicrostructureRenderer.h"
#include "panel_base.hpp"
#include <memory>
#include <vector>

namespace BTQuant {

// Footprint Cell Structure for Exocharts-style visualization
struct FootprintCell {
  double x;             // Time position (X-axis)
  double y;             // Price position (Y-axis)
  double width;         // Cell width (time duration)
  double height;        // Cell height (price range)
  double bid_volume;    // Total bid volume
  double ask_volume;    // Total ask volume
  double delta;         // Delta (bid_volume - ask_volume)
  uint32_t trade_count; // Number of trades
  double vwap;          // Volume-weighted average price

  // Constructor
  FootprintCell(double x_pos, double y_pos, double w, double h, double bid_vol,
                double ask_vol, uint32_t count, double vwap_price)
      : x(x_pos), y(y_pos), width(w), height(h), bid_volume(bid_vol),
        ask_volume(ask_vol), delta(bid_vol - ask_vol), trade_count(count),
        vwap(vwap_price) {}
};

class FootprintPanel : public PanelBase {
public:
  FootprintPanel(const PanelConfig &config,
                 RenderEngine::MarketMicrostructureRenderer *renderer);

  void update(float dt) override;
  void render() override;

  uint32_t get_symbol_id() const { return symbol_id_; }
  void set_symbol_id(uint32_t id) {
    symbol_id_ = id;
    if (renderer_)
      renderer_->setSymbol(id);
  }

  // Configuration
  void setGridSize(int cols, int rows) {
    grid_cols_ = cols;
    grid_rows_ = rows;
  }
  void setShowVolumeLabels(bool show) { show_volume_labels_ = show; }
  void setShowDeltaIndicator(bool show) { show_delta_indicator_ = show; }
  void setDeltaThreshold(double threshold) { delta_threshold_ = threshold; }

private:
  RenderEngine::MarketMicrostructureRenderer *renderer_;
  uint32_t symbol_id_ = 0;

  // Grid Configuration (Exocharts-style: 60 columns × 100 rows)
  int grid_cols_ = 60;  // Number of time columns (minutes)
  int grid_rows_ = 100; // Number of price rows (ticks)

  // Visualization Options
  bool show_volume_labels_ = true;
  bool show_delta_indicator_ = true;
  double delta_threshold_ = 0.0; // Threshold for delta coloring

  // Cell Data (CPU-side aggregation)
  std::vector<FootprintCell> cells_;

  // Rendering Helpers
  ImU32 getCellColor(const FootprintCell &cell) const;
  std::string getCellLabel(const FootprintCell &cell) const;
  void renderCell(const FootprintCell &cell, ImDrawList *draw_list);
};

} // namespace BTQuant
