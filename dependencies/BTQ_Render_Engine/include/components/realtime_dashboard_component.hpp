#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../trading/order_manager.hpp"
#include "../trading/position_manager.hpp"
#include "../trading/risk_assessment.hpp"
#include "../vulkan_dashboard_advanced.hpp"
#include "dashboard_controls.hpp"
#include "hierarchical_selector.hpp"
#include "panel_manager.hpp"

namespace BTQuant {

class RealtimeDashboardComponent : public UIComponent {
 public:
  explicit RealtimeDashboardComponent(
      std::shared_ptr<HotSpineDataBridge> bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
      RenderEngine::MarketMicrostructureRenderer* renderer = nullptr);
  virtual ~RealtimeDashboardComponent() = default;

  void update(float dt) override;
  void render_gui() override;

  void initialize_vulkan_resources(VulkanCore* core) override;
  void clear_data() override;

  // Dashboard Layout
  void setup_modern_layout();

 private:
  // Data sources
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  // Trading Subsystems
  std::shared_ptr<OrderManager> order_manager_;
  std::shared_ptr<PositionManager> position_manager_;
  std::shared_ptr<RiskAssessment> risk_assessment_;

  // Microstructure Renderer (Shared or Unique)
  // Microstructure Renderer (Reference)
  RenderEngine::MarketMicrostructureRenderer* microstructure_renderer_;

  // Panel Manager (The Core)
  std::unique_ptr<PanelManager> panel_manager_;

  // Symbol Selection
  HierarchicalSelector hierarchical_selector_;
  HierarchicalSelectorState selector_state_;

  // UI state
  bool show_dashboard_controls_ = true;

  // Dashboard Controls component
  std::unique_ptr<DashboardControls> dashboard_controls_;
};

}  // namespace BTQuant