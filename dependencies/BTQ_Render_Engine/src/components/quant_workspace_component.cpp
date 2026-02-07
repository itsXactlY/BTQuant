#include "../../include/components/quant_workspace_component.hpp"

#include <glm/glm.hpp>
#include <iostream>

#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"

namespace BTQuant {

QuantWorkspaceComponent::QuantWorkspaceComponent(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    RenderEngine::MarketMicrostructureRenderer* micro_renderer)
    : UIComponent(::glm::vec2(0, 0), ::glm::vec2(0, 0)), bridge_(bridge), processor_(processor) {
  // Initialize Trading Systems
  order_manager_ = std::make_shared<OrderManager>();
  position_manager_ = std::make_shared<PositionManager>();
  risk_assessment_ = std::make_shared<RiskAssessment>();

  // Set up callbacks for order execution -> position updates
  order_manager_->set_execution_callback([this](const OrderManager::OrderExecution& execution) {
    position_manager_->update_position(execution);
  });

  // Initialize the new panel-based system
  panel_manager_ = std::make_unique<PanelManager>(
      bridge_, processor_, order_manager_, position_manager_, risk_assessment_, micro_renderer);
  panel_manager_->initialize();

  // Load symbols from shared memory for hierarchical selector
  SymbolRegistry::instance().load_from_file("/dev/shm/btquant_symbols.json");

  // Initialize hierarchical selector state
  hierarchical_selector_.refresh_data(selector_state_, panel_manager_->get_chart_manager());
}

void QuantWorkspaceComponent::initialize_vulkan_resources(VulkanCore* core) {
  (void)core;  // Suppress unused parameter warning
  // Panel system handles its own Vulkan resources
}

void QuantWorkspaceComponent::update(float dt) {
  // Sync data from shared memory on every frame
  if (bridge_) {
    bridge_->sync();
  }
  panel_manager_->update(dt);
}

void QuantWorkspaceComponent::render_gui() {
  // Docking not supported in this branch of ImGui.
  // We'll just render the panels normally.

  // Render all panels through the panel manager first
  panel_manager_->render();

  // Render dashboard controls after all other panels so it appears visually on top
  if (show_dashboard_controls_) {
    render_dashboard_controls();
  }
}

void QuantWorkspaceComponent::render_dashboard_controls() {
  // Position in top-right corner and ensure it stays on top
  ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 310, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(300, 250), ImGuiCond_FirstUseEver);

  // Ensure the Dashboard Controls window stays on top
  ImGui::SetNextWindowFocus();

  if (ImGui::Begin("Dashboard Controls", &show_dashboard_controls_, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse)) {
    ImGui::Text("Ultra-Quantitative Dashboard");
    ImGui::Separator();

    // Panel management
    if (ImGui::CollapsingHeader("Add Panels", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::Button("Add Chart Panel")) {
        panel_manager_->add_panel(PanelType::CHART);
      }
      ImGui::SameLine();
      if (ImGui::Button("Add Metrics Panel")) {
        panel_manager_->add_panel(PanelType::METRICS);
      }

      if (ImGui::Button("Add Heatmap Panel")) {
        panel_manager_->add_panel(PanelType::HEATMAP);
      }
      ImGui::SameLine();
      if (ImGui::Button("Add Orderbook")) {
        panel_manager_->add_panel(PanelType::ORDERBOOK);
      }
      if (ImGui::Button("Add Footprint")) {
        panel_manager_->add_panel(PanelType::FOOTPRINT_CHART);
      }
      ImGui::SameLine();
      if (ImGui::Button("Add TPO Profile")) {
        panel_manager_->add_panel(PanelType::TPO_PROFILE);
      }
    }

    // Hierarchical Selector (Exchange -> Symbol -> Chart)
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Symbol Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Render the hierarchical selector
      bool selection_changed = hierarchical_selector_.render(selector_state_);

      if (selection_changed && !selector_state_.selected_symbol.empty()) {
        // CRITICAL: Propagate symbol to all panels (orderbook, etc.)
        panel_manager_->set_active_symbol(selector_state_.selected_symbol_id,
                                          selector_state_.selected_symbol);

        // Create or switch to chart for selected symbol/timeframe
        auto* chart_manager = panel_manager_->get_chart_manager();
        if (chart_manager) {
          // Check if chart already exists for this symbol/timeframe
          auto charts = chart_manager->get_charts_for_symbol(selector_state_.selected_symbol);

          bool found = false;
          for (const auto& chart : charts) {
            if (chart.timeframe == selector_state_.selected_timeframe) {
              selector_state_.selected_chart_id = chart.chart_id;
              found = true;
              break;
            }
          }

          // Create new chart if doesn't exist
          if (!found) {
            selector_state_.selected_chart_id = chart_manager->create_chart(
                selector_state_.selected_symbol, selector_state_.selected_exchange,
                selector_state_.selected_symbol_id, selector_state_.selected_timeframe);

            // Add chart panel to display it
            panel_manager_->add_panel(PanelType::CHART);
          }
        }
      }

      // Refresh button
      if (ImGui::Button("Refresh Symbols")) {
        SymbolRegistry::instance().load_from_file("/dev/shm/btquant_symbols.json");
        hierarchical_selector_.refresh_data(selector_state_, panel_manager_->get_chart_manager());
      }
    }

    // Layout controls
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Layout")) {
      if (ImGui::Button("Auto Arrange")) {
        panel_manager_->auto_arrange_panels();
      }
    }

    // Performance info
    ImGui::Separator();
    ImGui::Text("Real-time Dashboard Active");
    ImGui::Text("Data Pipeline: Connected");
  }
  ImGui::End();
}

void QuantWorkspaceComponent::clear_data() { panel_manager_.reset(); }

void QuantWorkspaceComponent::refresh_hierarchical_selector() {
  hierarchical_selector_.refresh_data(selector_state_, panel_manager_->get_chart_manager());
}

}  // namespace BTQuant
