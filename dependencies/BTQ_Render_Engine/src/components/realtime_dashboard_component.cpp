#include "components/realtime_dashboard_component.hpp"
#include "components/panel_manager.hpp"
#include "components/dashboard_controls.hpp"
#include "components/hierarchical_selector.hpp"

#include <imgui.h>
#include <iostream>

namespace BTQuant {

RealtimeDashboardComponent::RealtimeDashboardComponent(
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    RenderEngine::MarketMicrostructureRenderer* renderer)
    : UIComponent(glm::vec2(0.0f, 0.0f), glm::vec2(1920.0f, 1080.0f)),
      processor_(std::move(processor)),
      microstructure_renderer_(renderer) {

    order_manager_ = std::make_shared<OrderManager>();
    position_manager_ = std::make_shared<PositionManager>();
    risk_assessment_ = std::make_shared<RiskAssessment>();

    // Create PanelManager
    panel_manager_ = std::make_unique<PanelManager>(
        processor_, order_manager_, position_manager_, risk_assessment_,
        microstructure_renderer_);

    dashboard_controls_ = std::make_unique<DashboardControls>(panel_manager_.get());
}

void RealtimeDashboardComponent::update(float dt) {
    if (panel_manager_) {
        panel_manager_->update(dt);
    }
    if (dashboard_controls_) {
        dashboard_controls_->update(dt);
    }
}

void RealtimeDashboardComponent::render_gui() {
    if (show_dashboard_controls_ && dashboard_controls_) {
        dashboard_controls_->render_gui();
    }

    if (panel_manager_) {
        panel_manager_->render();
    }
}

void RealtimeDashboardComponent::initialize_vulkan_resources(VulkanCore* core) {
    (void)core;
    // Vulkan resources initialization placeholder
}

void RealtimeDashboardComponent::clear_data() {
    if (panel_manager_) {
        panel_manager_->clear_panels();
    }
}

void RealtimeDashboardComponent::setup_modern_layout() {
    if (!panel_manager_) return;

    panel_manager_->clear_panels();
    panel_manager_->set_grid_layout(6, 10);

    panel_manager_->add_panel(PanelType::CHART, "Main Chart", 0, 0, 4, 3);
    panel_manager_->add_panel(PanelType::ORDERBOOK, "Orderbook", 4, 0, 2, 2);
    panel_manager_->add_panel(PanelType::TAPE, "Time & Sales", 0, 3, 2, 1);
    panel_manager_->add_panel(PanelType::TRADING_ORDERS, "Orders", 2, 3, 2, 1);
    panel_manager_->add_panel(PanelType::TRADING_POSITIONS, "Positions", 0, 4, 2, 1);
    panel_manager_->add_panel(PanelType::VOLUME_PROFILE, "Volume Profile", 2, 4, 2, 1);
    panel_manager_->add_panel(PanelType::WATCHLIST, "Watchlist", 4, 4, 2, 1);

    panel_manager_->auto_arrange_panels();
}

}  // namespace BTQuant
