#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "chart_manager.hpp"
#include "indicator_renderer.hpp"
#include "panel_base.hpp"
#include <memory>

namespace BTQuant {

// Indicator configuration for the chart panel
struct IndicatorConfig {
  bool show_sma_10 = false;
  bool show_sma_20 = false;
  bool show_sma_50 = false;
  bool show_ema_10 = false;
  bool show_ema_20 = false;
  bool show_ema_50 = false;
  bool show_rsi = false;
  bool show_macd = false;
  bool show_bollinger = false;
};

class ChartPanel : public PanelBase {
public:
  ChartPanel(const PanelConfig &config,
             std::shared_ptr<HotSpineDataBridge> bridge,
             std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
             ChartManager *chart_manager);

  void update(float dt) override;
  void render() override;
  void initialize() override;

  // Chart-specific methods
  void set_symbol(const std::string &symbol,
                  const std::string &exchange = "Binance");
  void set_timeframe(RenderEngine::TimeFrame timeframe);
  uint32_t get_chart_id() const { return chart_id_; }

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  ChartManager *chart_manager_;
  IndicatorRenderer *indicator_renderer_;

  std::string symbol_ = "BTC-USDT";
  std::string exchange_ = "Binance";
  RenderEngine::TimeFrame timeframe_ = RenderEngine::TimeFrame::TF_1MIN;
  uint32_t chart_id_ = 0;

  IndicatorConfig indicator_config_;

  void render_chart_controls();
  void render_indicator_selector();
  void render_instrument_chart(const ChartInstance &chart);
  void render_candlestick(const ChartInstance &chart);
};

} // namespace BTQuant