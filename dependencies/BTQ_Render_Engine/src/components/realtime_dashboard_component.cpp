#include "../../include/components/realtime_dashboard_component.hpp"

#include <imgui.h>

#include <iostream>

#include "../../include/symbol_registry.hpp"

namespace BTQuant {

RealtimeDashboardComponent::RealtimeDashboardComponent(
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    RenderEngine::MarketMicrostructureRenderer* renderer)
    : UIComponent({0, 0}, {0, 0}),
      processor_(std::move(processor)),
      microstructure_renderer_(renderer) {
  // Initialize Trading Subsystems
  order_manager_ = std::make_shared<OrderManager>();
  position_manager_ = std::make_shared<PositionManager>();
  risk_assessment_ = std::make_shared<RiskAssessment>();

  // Ensure Renderer is valid
  if (!microstructure_renderer_) {
    std::cerr << "[RealtimeDashboard] Warning: Microstructure Renderer is null" << std::endl;
  }

  // Initialize Panel Manager
  panel_manager_ =
      std::make_unique<PanelManager>(processor_, order_manager_, position_manager_,
                                     risk_assessment_, microstructure_renderer_);

  panel_manager_->initialize();

  // Load symbols (legacy requirement for some components)
  SymbolRegistry::instance().load_from_file("/dev/shm/btquant_symbols.json");

  // Initialize Dashboard Controls
  dashboard_controls_ = std::make_unique<DashboardControls>(panel_manager_.get());

  setup_modern_layout();
}

void RealtimeDashboardComponent::setup_modern_layout() {
  panel_manager_->clear_panels();

  // Layout Grid: 6 columns x 5 rows
  panel_manager_->set_grid_layout(6, 5);

  // 1. Chart (Top Left, Large) - 4x3
  panel_manager_->add_panel(PanelType::CHART, "BTCUSDT Chart", 0, 0, 4, 3);

  // 2. DOM Surface (Top Right) - 2x2
  // Visualizes full depth liquidity
  panel_manager_->add_panel(PanelType::HEATMAP, "DOM Surface", 4, 0, 2, 2);

  // 3. Orderbook (Middle Right) - 2x2
  // Standard LOB view
  panel_manager_->add_panel(PanelType::ORDERBOOK, "Orderbook", 4, 2, 2, 2);

  // 4. Time & Sales (Bottom Left 1) - 2x1
  panel_manager_->add_panel(PanelType::TAPE, "Time & Sales", 0, 3, 2, 1);

  // 5. Watchlist (Bottom Left 2) - 2x1
  panel_manager_->add_panel(PanelType::WATCHLIST, "Watchlist", 2, 3, 2, 1);

  // 6. Positions / Risk (Bottom Row) - 6x1
  panel_manager_->add_panel(PanelType::TRADING_POSITIONS, "Positions", 0, 4, 6, 1);

  // Auto arrange to ensure everything snaps correctly
  // panel_manager_->auto_arrange_panels();
}

void RealtimeDashboardComponent::update(float dt) {
  panel_manager_->update(dt);
  if (dashboard_controls_) {
    dashboard_controls_->update(dt);
  }
}

void RealtimeDashboardComponent::render_gui() {
  // Render all panels first
  panel_manager_->render();
  
  // Render dashboard controls last to ensure they stay on top
  if (show_dashboard_controls_ && dashboard_controls_) {
    dashboard_controls_->render_gui();
  }
}


void RealtimeDashboardComponent::initialize_vulkan_resources(VulkanCore* core) {
  // Shared renderer is initiated by parent (VulkanDashboard)
  (void)core;
}

void RealtimeDashboardComponent::clear_data() {
  panel_manager_->clear_panels();
  if (dashboard_controls_) {
    dashboard_controls_->clear_data();
  }
}

}  // namespace BTQuant