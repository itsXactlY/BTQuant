#include "../../include/components/realtime_dashboard_component.hpp"

#include <imgui.h>

#include <iostream>

#include "../../include/symbol_registry.hpp"

namespace BTQuant {

RealtimeDashboardComponent::RealtimeDashboardComponent(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    RenderEngine::MarketMicrostructureRenderer* renderer)
    : UIComponent({0, 0}, {0, 0}),
      bridge_(std::move(bridge)),
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
      std::make_unique<PanelManager>(bridge_, processor_, order_manager_, position_manager_,
                                     risk_assessment_, microstructure_renderer_);

  panel_manager_->initialize();

  // Load symbols (legacy requirement for some components)
  SymbolRegistry::instance().load_from_file("/dev/shm/btquant_symbols.json");

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

void RealtimeDashboardComponent::update(float dt) { panel_manager_->update(dt); }

void RealtimeDashboardComponent::render_gui() {
  if (show_dashboard_controls_) {
    render_dashboard_controls();
  }
  panel_manager_->render();
}

void RealtimeDashboardComponent::render_dashboard_controls() {
  // Floating Dashboard Controls Panel (similar to old QuantWorkspaceComponent)
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(320, 350), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Dashboard Controls", &show_dashboard_controls_)) {
    ImGui::Text("Modern Dashboard");
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

      // Extra panels available in modern dashboard
      if (ImGui::Button("Add Tape")) {
        panel_manager_->add_panel(PanelType::TAPE);
      }
      ImGui::SameLine();
      if (ImGui::Button("Add Depth Chart")) {
        panel_manager_->add_panel(PanelType::DEPTH_CHART);
      }

      if (ImGui::Button("Add Watchlist")) {
        panel_manager_->add_panel(PanelType::WATCHLIST);
      }
      ImGui::SameLine();
      if (ImGui::Button("Add Positions")) {
        panel_manager_->add_panel(PanelType::TRADING_POSITIONS);
      }
    }

    // Hierarchical Selector (Exchange -> Symbol -> Chart)
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Symbol Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Render the hierarchical selector
      bool selection_changed = hierarchical_selector_.render(selector_state_);

      if (selection_changed && !selector_state_.selected_symbol.empty()) {
        // Propagate symbol to all panels (orderbook, etc.)
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
      if (ImGui::Button("Reset to Default")) {
        setup_modern_layout();
      }
      ImGui::SameLine();
      if (ImGui::Button("Auto Arrange")) {
        panel_manager_->auto_arrange_panels();
      }

      ImGui::Separator();
      if (ImGui::Button("Save Layout")) {
        panel_manager_->save_layout("user_layout.json");
      }
      ImGui::SameLine();
      if (ImGui::Button("Load Layout")) {
        panel_manager_->load_layout("user_layout.json");
      }
    }

    // Status info
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Real-time Dashboard Active");
  }
  ImGui::End();
}

void RealtimeDashboardComponent::initialize_vulkan_resources(VulkanCore* core) {
  // Shared renderer is initiated by parent (VulkanDashboard)
  (void)core;
}

void RealtimeDashboardComponent::clear_data() { panel_manager_->clear_panels(); }

}  // namespace BTQuant