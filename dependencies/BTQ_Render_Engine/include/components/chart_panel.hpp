#pragma once

#include "panel_base.hpp"
#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "chart_manager.hpp"
#include "indicator_renderer.hpp"
#include <memory>

namespace BTQuant {

class ChartPanel : public PanelBase {
public:
  ChartPanel(const PanelConfig& config,
             std::shared_ptr<HotSpineDataBridge> bridge,
             std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
             ChartManager* chart_manager);

  void update(float dt) override;
  void render() override;
  void initialize() override;

  // Chart-specific methods
  void set_symbol(const std::string& symbol, const std::string& exchange = "Binance");
  void set_timeframe(RenderEngine::TimeFrame timeframe);
  uint32_t get_chart_id() const { return chart_id_; }

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  ChartManager* chart_manager_;
  IndicatorRenderer* indicator_renderer_;

  std::string symbol_ = "BTC-USDT";
  std::string exchange_ = "Binance";
  RenderEngine::TimeFrame timeframe_ = RenderEngine::TimeFrame::MINUTE_1;
  uint32_t chart_id_ = 0;

  IndicatorConfig indicator_config_;

  void render_chart_controls();
  void render_indicator_selector();
  void render_instrument_chart(const ChartInstance& chart);
};

} // namespace BTQuant