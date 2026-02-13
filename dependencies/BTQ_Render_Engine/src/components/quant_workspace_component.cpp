#include "../../include/components/quant_workspace_component.hpp"

#include <glm/glm.hpp>
#include <iostream>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

QuantWorkspaceComponent::QuantWorkspaceComponent(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
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
      bridge_, processor_, order_manager_, position_manager_, risk_assessment_);
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
  // NOTE: Data sync is handled in main loop (main_trading_terminal.cpp)
  // to avoid double-sync per frame
  panel_manager_->update(dt);
}

void QuantWorkspaceComponent::render_gui() {
  // Docking not supported in this branch of ImGui.
  // We'll just render the panels normally.

  // Render dashboard controls
  if (show_dashboard_controls_) {
    render_dashboard_controls();
  }

  // Render all panels through the panel manager
  panel_manager_->render();
}

void QuantWorkspaceComponent::render_dashboard_controls() {
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(300, 250), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Dashboard Controls", &show_dashboard_controls_)) {
    ImGui::Text("Ultra-Quantitative Dashboard");
    ImGui::Separator();

    // Panel management - Charts
    if (ImGui::CollapsingHeader("Charts", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::Button("Chart")) {
        panel_manager_->add_panel(PanelType::CHART);
      }
      ImGui::SameLine();
      if (ImGui::Button("Footprint")) {
        panel_manager_->add_panel(PanelType::FOOTPRINT_CHART);
      }
      ImGui::SameLine();
      if (ImGui::Button("TPO Profile")) {
        panel_manager_->add_panel(PanelType::TPO_PROFILE);
      }
      
      if (ImGui::Button("Volume Profile")) {
        panel_manager_->add_panel(PanelType::VOLUME_PROFILE);
      }
      ImGui::SameLine();
      if (ImGui::Button("Depth Chart")) {
        panel_manager_->add_panel(PanelType::DEPTH_CHART);
      }
      ImGui::SameLine();
      if (ImGui::Button("Chart Replay")) {
        panel_manager_->add_panel(PanelType::CHART_REPLAY);
      }
    }

    // Panel management - Market Data
    if (ImGui::CollapsingHeader("Market Data")) {
      if (ImGui::Button("Order Book")) {
        panel_manager_->add_panel(PanelType::ORDERBOOK);
      }
      ImGui::SameLine();
      if (ImGui::Button("Time & Sales")) {
        panel_manager_->add_panel(PanelType::TIME_AND_SALES);
      }
      ImGui::SameLine();
      if (ImGui::Button("Tape")) {
        panel_manager_->add_panel(PanelType::TAPE);
      }
      
      if (ImGui::Button("Watchlist")) {
        panel_manager_->add_panel(PanelType::WATCHLIST);
      }
      ImGui::SameLine();
      if (ImGui::Button("Screener")) {
        panel_manager_->add_panel(PanelType::SCREENER);
      }
      ImGui::SameLine();
      if (ImGui::Button("Heatmap")) {
        panel_manager_->add_panel(PanelType::HEATMAP);
      }
      
      if (ImGui::Button("Hist. T&S")) {
        panel_manager_->add_panel(PanelType::HISTORICAL_TIME_SALES);
      }
    }

    // Panel management - Trading
    if (ImGui::CollapsingHeader("Trading")) {
      if (ImGui::Button("Orders")) {
        panel_manager_->add_panel(PanelType::TRADING_ORDERS);
      }
      ImGui::SameLine();
      if (ImGui::Button("Positions")) {
        panel_manager_->add_panel(PanelType::TRADING_POSITIONS);
      }
      ImGui::SameLine();
      if (ImGui::Button("Alerts")) {
        panel_manager_->add_panel(PanelType::ALERTS);
      }
    }

    // Panel management - Analysis
    if (ImGui::CollapsingHeader("Analysis")) {
      if (ImGui::Button("Metrics")) {
        panel_manager_->add_panel(PanelType::METRICS);
      }
      ImGui::SameLine();
      if (ImGui::Button("Risk Metrics")) {
        panel_manager_->add_panel(PanelType::RISK_METRICS);
      }
      ImGui::SameLine();
      if (ImGui::Button("Risk Analyzer")) {
        panel_manager_->add_panel(PanelType::RISK_ANALYZER);
      }
      
      if (ImGui::Button("Histogram")) {
        panel_manager_->add_panel(PanelType::HISTOGRAM);
      }
      ImGui::SameLine();
      if (ImGui::Button("Scatter Plot")) {
        panel_manager_->add_panel(PanelType::SCATTER_PLOT);
      }
      ImGui::SameLine();
      if (ImGui::Button("Time Series")) {
        panel_manager_->add_panel(PanelType::TIME_SERIES);
      }
      
      if (ImGui::Button("Time Stats")) {
        panel_manager_->add_panel(PanelType::TIME_STATISTICS);
      }
      ImGui::SameLine();
      if (ImGui::Button("Time Histogram")) {
        panel_manager_->add_panel(PanelType::TIME_HISTOGRAM);
      }
      
      if (ImGui::Button("Strategy Builder")) {
        panel_manager_->add_panel(PanelType::STRATEGY_BUILDER);
      }
      ImGui::SameLine();
      if (ImGui::Button("Option Analytics")) {
        panel_manager_->add_panel(PanelType::OPTION_ANALYTICS);
      }
    }

    // Panel management - System
    if (ImGui::CollapsingHeader("System")) {
      if (ImGui::Button("Log Panel")) {
        panel_manager_->add_panel(PanelType::LOG_PANEL);
      }
      ImGui::SameLine();
      if (ImGui::Button("Perf Monitor")) {
        panel_manager_->add_panel(PanelType::PERFORMANCE_MONITOR);
      }
      ImGui::SameLine();
      if (ImGui::Button("Status Bar")) {
        panel_manager_->add_panel(PanelType::STATUS_BAR);
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
