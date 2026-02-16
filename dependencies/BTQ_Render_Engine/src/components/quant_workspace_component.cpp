#include "../../include/components/quant_workspace_component.hpp"
#include "../../include/components/chart_panel.hpp"  // Required for complete type in dynamic_cast

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
  
  // Update crosshair synchronization if enabled
  if (global_crosshair_enabled_) {
    // Check if any chart panel is currently showing crosshair info
    // This would be handled by the individual chart panels, but we can coordinate them here
    // For now, we'll just track the mouse position for potential synchronization
    ImVec2 current_mouse_pos = ImGui::GetMousePos();
    
    // Only update if mouse has moved significantly
    float mouse_move_threshold = 1.0f; // Minimum movement to trigger update
    float distance = sqrt(pow(current_mouse_pos.x - last_crosshair_position_.x, 2) + 
                          pow(current_mouse_pos.y - last_crosshair_position_.y, 2));
    
    if (distance > mouse_move_threshold) {
      last_crosshair_position_ = current_mouse_pos;
      crosshair_active_ = true;
    } else {
      crosshair_active_ = false;
    }
  }
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
  
  // Handle global crosshair synchronization after all panels are rendered
  if (global_crosshair_enabled_) {
    handle_global_crosshair_sync();
  }
}

void QuantWorkspaceComponent::handle_global_crosshair_sync() {
  // This method will coordinate crosshair positions across all chart panels
  // For true global crosshair sync, we need to share crosshair position data between panels
  
  // Get all panel IDs
  auto panel_ids = panel_manager_->get_all_panel_ids();
  
  // Find all chart panels
  std::vector<ChartPanel*> chart_panels;
  for (uint32_t panel_id : panel_ids) {
    auto* panel = panel_manager_->get_panel_by_id(panel_id);
    if (!panel) continue;
    
    // Check if this is a chart panel
    if (panel->get_config().type == PanelType::CHART) {
      auto* chart_panel = dynamic_cast<ChartPanel*>(panel);
      if (chart_panel) {
        chart_panels.push_back(chart_panel);
      }
    }
  }
  
  // If we have multiple chart panels, implement crosshair synchronization
  if (chart_panels.size() > 1) {
    // Find the chart panel that currently has the mouse cursor
    ChartPanel* active_chart = get_chart_panel_under_cursor();
    
    if (active_chart) {
      // Get the mouse position in screen coordinates
      ImVec2 mouse_pos = ImGui::GetMousePos();
      
      // Find the panel ID for the active chart to get its position
      auto all_panel_ids = panel_manager_->get_all_panel_ids();
      uint32_t active_panel_id = 0;
      for (uint32_t id : all_panel_ids) {
        auto* panel = panel_manager_->get_panel_by_id(id);
        if (panel == static_cast<PanelBase*>(active_chart)) {
          active_panel_id = id;
          break;
        }
      }
      
      if (active_panel_id != 0) {
        // Get the chart panel's position and size to calculate relative mouse position
        ImVec2 panel_pos = panel_manager_->get_panel_position(active_panel_id);
        ImVec2 panel_size = panel_manager_->get_panel_size(active_panel_id);
        
        // Calculate the relative X position within the active chart panel (0.0 to 1.0)
        float rel_x = (mouse_pos.x - panel_pos.x) / panel_size.x;
        
        // Synchronize this relative position to all other chart panels
        for (auto* chart_panel : chart_panels) {
          if (chart_panel != active_chart) {
            // Calculate the absolute screen X position for this chart panel
            uint32_t target_panel_id = 0;
            for (uint32_t id : all_panel_ids) {
              auto* panel = panel_manager_->get_panel_by_id(id);
              if (panel == static_cast<PanelBase*>(chart_panel)) {
                target_panel_id = id;
                break;
              }
            }
            
            if (target_panel_id != 0) {
              ImVec2 target_panel_pos = panel_manager_->get_panel_position(target_panel_id);
              ImVec2 target_panel_size = panel_manager_->get_panel_size(target_panel_id);
              
              // Calculate the absolute X position in the target panel based on relative position
              float target_x = target_panel_pos.x + rel_x * target_panel_size.x;
              
              // Set the global crosshair position for this chart
              chart_panel->set_global_crosshair_position(target_x, true);
            }
          } else {
            // For the active chart, we still enable the global crosshair state
            chart_panel->set_global_crosshair_position(mouse_pos.x, true);
          }
        }
      }
    } else {
      // If no chart has the mouse, disable global sync on all charts
      for (auto* chart_panel : chart_panels) {
        chart_panel->set_global_crosshair_position(0.0, false);
      }
    }
  }
}

ChartPanel* QuantWorkspaceComponent::get_chart_panel_under_cursor() const {
  ImVec2 mouse_pos = ImGui::GetMousePos();
  auto panel_ids = panel_manager_->get_all_panel_ids();
  
  for (uint32_t panel_id : panel_ids) {
    auto* panel = panel_manager_->get_panel_by_id(panel_id);
    if (!panel) continue;
    
    // Check if this is a chart panel
    if (panel->get_config().type == PanelType::CHART) {
      auto* chart_panel = dynamic_cast<ChartPanel*>(panel);
      if (chart_panel) {
        // Get the panel's position and size
        ImVec2 panel_pos = panel_manager_->get_panel_position(panel_id);
        ImVec2 panel_size = panel_manager_->get_panel_size(panel_id);
        
        // Check if mouse is within the panel bounds
        if (mouse_pos.x >= panel_pos.x && mouse_pos.x <= panel_pos.x + panel_size.x &&
            mouse_pos.y >= panel_pos.y && mouse_pos.y <= panel_pos.y + panel_size.y) {
          return chart_panel;
        }
      }
    }
  }
  
  return nullptr; // No chart panel found under cursor
}

void QuantWorkspaceComponent::render_dashboard_controls() {
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(320, 400), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Dashboard Controls", &show_dashboard_controls_)) {
    ImGui::Text("Ultra-Quantitative Dashboard");
    ImGui::Separator();

    // Layout Presets Menu
    if (ImGui::CollapsingHeader("Layout Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::Button("Default")) {
        panel_manager_->apply_layout_preset(LayoutPreset::DEFAULT);
      }
      ImGui::SameLine();
      if (ImGui::Button("Modern Trading")) {
        panel_manager_->apply_layout_preset(LayoutPreset::MODERN_TRADING);
      }
      ImGui::SameLine();
      if (ImGui::Button("Dashboard")) {
        panel_manager_->apply_layout_preset(LayoutPreset::DASHBOARD_ONLY);
      }
      
      if (ImGui::Button("Chart Focus")) {
        panel_manager_->apply_layout_preset(LayoutPreset::CHART_FOCUS);
      }
      ImGui::SameLine();
      if (ImGui::Button("Risk Monitoring")) {
        panel_manager_->apply_layout_preset(LayoutPreset::RISK_MONITORING);
      }
    }

    // Panel management - Charts (reorganized per requirements)
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
    }

    // Panel management - Market Data (reorganized per requirements)
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
      if (ImGui::Button("Heatmap")) {
        panel_manager_->add_panel(PanelType::HEATMAP);
      }
      ImGui::SameLine();
      if (ImGui::Button("DOM Surface")) {
        panel_manager_->add_panel(PanelType::DOM_SURFACE);
      }
      
      if (ImGui::Button("Hist. T&S")) {
        panel_manager_->add_panel(PanelType::HISTORICAL_TIME_SALES);
      }
      ImGui::SameLine();
      if (ImGui::Button("Screener")) {
        panel_manager_->add_panel(PanelType::SCREENER);
      }
    }

    // Panel management - Trading (reorganized per requirements)
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
      
      if (ImGui::Button("Strategy Builder")) {
        panel_manager_->add_panel(PanelType::STRATEGY_BUILDER);
      }
    }

    // Panel management - Analysis (reorganized per requirements)
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
      
      if (ImGui::Button("Option Analytics")) {
        panel_manager_->add_panel(PanelType::OPTION_ANALYTICS);
      }
      ImGui::SameLine();
      if (ImGui::Button("Multi VWAP")) {
        panel_manager_->add_panel(PanelType::MULTI_VWAP);
      }
      
      if (ImGui::Button("Correlation")) {
        panel_manager_->add_panel(PanelType::CORRELATION_HEATMAP);
      }
      ImGui::SameLine();
      if (ImGui::Button("Tech Indicators")) {
        panel_manager_->add_panel(PanelType::TECHNICAL_INDICATORS);
      }
      
      if (ImGui::Button("Time Stats")) {
        panel_manager_->add_panel(PanelType::TIME_STATISTICS);
      }
      ImGui::SameLine();
      if (ImGui::Button("Time Histogram")) {
        panel_manager_->add_panel(PanelType::TIME_HISTOGRAM);
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
    }

    // Panel management - Tools (new category)
    if (ImGui::CollapsingHeader("Tools")) {
      if (ImGui::Button("Drawing Tools")) {
        panel_manager_->add_panel(PanelType::DRAWING_TOOLS);
      }
      ImGui::SameLine();
      if (ImGui::Button("Chart Replay")) {
        panel_manager_->add_panel(PanelType::CHART_REPLAY);
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

      if (ImGui::Button("Shortcuts")) {
        panel_manager_->add_panel(PanelType::KEYBOARD_SHORTCUTS);
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
