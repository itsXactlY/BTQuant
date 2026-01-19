#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../vulkan_dashboard_advanced.hpp"
#include "../dashboard_config.hpp"
#include "theme_customization_component.hpp"
#include "imgui.h"
#include "implot.h"

namespace BTQuant {

class QuantWorkspaceComponent : public UIComponent {
public:
  explicit QuantWorkspaceComponent(std::shared_ptr<HotSpineDataBridge> bridge);
  virtual ~QuantWorkspaceComponent() = default;

  void update(float dt) override;
  void render_gui() override;

  void initialize_vulkan_resources(VulkanCore *core) override {}
  void clear_data() override {}

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::unique_ptr<RenderEngine::MarketDataProcessor> market_data_processor_;
  RenderEngine::TimeFrame current_timeframe_ = RenderEngine::TimeFrame::TF_1MIN;
  RenderEngine::DashboardConfig config_;
  std::unique_ptr<ThemeCustomizationComponent> theme_customization_;
  
  void render_instrument_chart(const std::string &id,
                               const MarketInstrument &instrument);
  void render_order_book_mini(const MarketInstrument &instrument);
  void render_stats_panel(const MarketInstrument &instrument);
  void render_timeframe_selector();
};

} // namespace BTQuant
