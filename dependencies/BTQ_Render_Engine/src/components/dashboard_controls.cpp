#include "../../include/components/dashboard_controls.hpp"

#include <imgui.h>

#include <iostream>

#include "../../include/components/panel_manager.hpp"
#include "../../include/ui/ui_base.hpp"

namespace BTQuant {

DashboardControls::DashboardControls(PanelManager* panel_manager)
    : UIComponent({0, 0}, {0, 0}), panel_manager_(panel_manager) {
  if (!panel_manager_) {
    std::cerr << "[DashboardControls] Error: PanelManager is null" << std::endl;
  }
}

void DashboardControls::render_gui() {
  render_dashboard_controls();
}

void DashboardControls::render_dashboard_controls() {
  // Floating Dashboard Controls Panel
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(350, 400), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Dashboard Controls", nullptr)) {
    ImGui::Text("Trading Dashboard Controls");
    ImGui::Separator();

    // Panel management section with all requested panel types
    if (ImGui::CollapsingHeader("Add Panels", ImGuiTreeNodeFlags_DefaultOpen)) {
      
      // Add Chart Button
      if (ImGui::Button("Add Chart", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::CHART);
        }
      }
      ImGui::Spacing();

      // Add Footprint Button
      if (ImGui::Button("Add Footprint", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::FOOTPRINT_CHART);
        }
      }
      ImGui::Spacing();

      // Add Volume Profile Button
      if (ImGui::Button("Add Volume Profile", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::VOLUME_PROFILE);
        }
      }
      ImGui::Spacing();

      // Add Order Book Button
      if (ImGui::Button("Add Order Book", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ORDERBOOK);
        }
      }
      ImGui::Spacing();

      // Add Time&Sales Button
      if (ImGui::Button("Add Time&Sales", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::TIME_AND_SALES);
        }
      }
      ImGui::Spacing();

      // Add Watchlist Button
      if (ImGui::Button("Add Watchlist", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::WATCHLIST);
        }
      }
      ImGui::Spacing();

      // Add News Button (using Alerts panel as news feed)
      if (ImGui::Button("Add News", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ALERTS);
        }
      }
      ImGui::Spacing();

      // Additional useful panels
      if (ImGui::Button("Add Metrics", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::METRICS);
        }
      }
      ImGui::Spacing();

      if (ImGui::Button("Add Screener", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::SCREENER);
        }
      }
      ImGui::Spacing();
    }

    // Layout controls
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Layout Management")) {
      if (ImGui::Button("Reset Layout")) {
        if (panel_manager_) {
          // Clear all panels and set up default layout
          panel_manager_->clear_panels();
          
          // Add default panels
          panel_manager_->add_panel(PanelType::CHART, "Default Chart", 0, 0, 4, 3);
          panel_manager_->add_panel(PanelType::ORDERBOOK, "Order Book", 4, 0, 2, 2);
          panel_manager_->add_panel(PanelType::WATCHLIST, "Watchlist", 0, 3, 2, 1);
          panel_manager_->add_panel(PanelType::TIME_AND_SALES, "Time & Sales", 2, 3, 2, 1);
        }
      }
      
      ImGui::Spacing();
      
      if (ImGui::Button("Auto Arrange")) {
        if (panel_manager_) {
          panel_manager_->auto_arrange_panels();
        }
      }
      
      ImGui::Spacing();
      
      if (ImGui::Button("Clear All Panels")) {
        if (panel_manager_) {
          panel_manager_->clear_panels();
        }
      }
    }

    // Status information
    ImGui::Separator();
    if (panel_manager_) {
      auto panel_count = panel_manager_->get_panel_count();
      ImGui::Text("Active Panels: %zu", panel_count);
    }
    ImGui::Text("Ready to add panels");
  }
  ImGui::End();
}

void DashboardControls::update(float dt) {
  // No update logic needed for controls panel
  (void)dt;
}

void DashboardControls::initialize_vulkan_resources(VulkanCore* core) {
  // No Vulkan resources needed for controls panel
  (void)core;
}

void DashboardControls::clear_data() {
  // No data to clear for controls panel
}

}  // namespace BTQuant