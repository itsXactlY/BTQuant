#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include <imgui.h>
#include <memory>

namespace BTQuant {

/**
 * DepthChartPanel - Cumulative bid/ask depth visualization
 *
 * Shows an area chart with:
 * - Bid depth (green) accumulating from mid price leftward
 * - Ask depth (red) accumulating from mid price rightward
 * - X-axis: price, Y-axis: cumulative size
 */
class DepthChartPanel : public PanelBase {
public:
  DepthChartPanel(const PanelConfig &config,
                  std::shared_ptr<HotSpineDataBridge> bridge,
                  std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void update(float dt) override;
  void render() override;

  void set_symbol(uint32_t symbol_id, const std::string &symbol_name);

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  uint32_t symbol_id_ = 0;
  std::string symbol_name_ = "BTC-USDT";

  // Cached orderbook data
  RenderEngine::OrderbookData cached_orderbook_;
  float update_timer_ = 0.0f;
  static constexpr float UPDATE_INTERVAL = 0.05f; // 50ms refresh

  void render_depth_chart();
  void render_stats();
};

} // namespace BTQuant
