#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../vulkan_dashboard_advanced.hpp"
#include "chart_manager.hpp"
#include "imgui.h"
#include "implot.h"
#include "indicator_renderer.hpp"
#include <memory>
#include <unordered_map>

namespace BTQuant {

struct IndicatorConfig {
  bool show_sma_10 = true;
  bool show_sma_20 = true;
  bool show_sma_50 = false;
  bool show_ema_10 = false;
  bool show_ema_20 = false;
  bool show_ema_50 = false;
  bool show_rsi = true;
  bool show_macd = true;
  bool show_bollinger = false;
  bool show_stochastic = false;
  bool show_waddah_explosion = false;
};

class QuantWorkspaceComponent : public UIComponent {
public:
  explicit QuantWorkspaceComponent(
      std::shared_ptr<HotSpineDataBridge> bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  virtual ~QuantWorkspaceComponent() = default;

  void update(float dt) override;
  void render_gui() override;

  void initialize_vulkan_resources(VulkanCore *core) override;
  void clear_data() override;

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::unique_ptr<ChartManager> chart_manager_;
  std::unique_ptr<IndicatorRenderer> indicator_renderer_;
  std::map<uint32_t, IndicatorConfig> indicator_configs_;

  void render_instrument_chart(uint32_t chart_id, const std::string &symbol,
                               const InstrumentStore &inst,
                               RenderEngine::TimeFrame timeframe);
  void render_timeframe_selector();
  void render_indicator_selector();
  void render_chart_controls();

  RenderEngine::TimeFrame selected_timeframe_ =
      RenderEngine::TimeFrame::TF_1MIN;
  bool show_chart_controls_ = true;
  bool show_indicator_selector_ = true;
};

} // namespace BTQuant
