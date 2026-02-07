#include "../../include/components/tabbed_panel.hpp"

#include <algorithm>
#include <iostream>

namespace BTQuant {

TabbedPanel::TabbedPanel(const PanelConfig& config, PanelManager* panel_manager)
    : PanelBase(config), panel_manager_(panel_manager) {
  tab_bar_id_ = "tab_bar_" + std::to_string(reinterpret_cast<uintptr_t>(this));
}

TabbedPanel::~TabbedPanel() {
  // When this tabbed panel is destroyed, we need to ensure the contained panels
  // are properly handled - they might need to be made visible again or moved elsewhere
  // For now, we'll just clear the tabbed panel list
  tabbed_panel_ids_.clear();
}

void TabbedPanel::update(float dt) {
  // Update any internal state if needed
  PanelBase::update(dt);
}

void TabbedPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  // Render the panel header
  render_panel_header();

  // Render the tab bar and content
  if (!tabbed_panel_ids_.empty()) {
    render_tab_bar();
    render_active_tab_content();
  } else {
    // If no tabs, show a placeholder message
    ImGui::Text("Drag a panel here to create a tabbed group");
    ImGui::Spacing();
    ImGui::Text("Supported panels:");
    ImGui::BulletText("Chart Panel");
    ImGui::BulletText("Time & Sales Panel");
    ImGui::BulletText("Other panels...");
  }

  end_panel_window();
}

void TabbedPanel::render_tab_bar() {
  if (ImGui::BeginTabBar(tab_bar_id_.c_str(), ImGuiTabBarFlags_None)) {
    for (uint32_t panel_id : tabbed_panel_ids_) {
      if (PanelBase* panel = panel_manager_->get_panel_by_id(panel_id)) {
        std::string tab_label = panel->get_title();
        
        // Add close button if this isn't the only tab
        ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
        if (tabbed_panel_ids_.size() > 1) {
          tab_flags |= ImGuiTabItemFlags_UnsavedDocument; // Shows close button
        }
        
        bool tab_open = true;
        if (ImGui::BeginTabItem(tab_label.c_str(), &tab_open, tab_flags)) {
          // Set this as the active tab
          active_tab_id_ = panel_id;
          
          // Render the panel content directly within the tab
          // We need to temporarily adjust the panel's visibility
          bool was_visible = panel->is_visible();
          panel->set_visible(true);
          
          // Temporarily hide the panel's window decorations since it's inside a tab
          // We'll just render the core content
          panel->render();
          
          ImGui::EndTabItem();
        }
        
        // Handle tab closing
        if (!tab_open && tabbed_panel_ids_.size() > 1) {
          // Remove this panel from the tabbed panel and restore its visibility
          remove_panel_from_tab(panel_id);
          panel->set_visible(true);
        }
      }
    }
    
    ImGui::EndTabBar();
  }
}

void TabbedPanel::render_active_tab_content() {
  if (active_tab_id_ != 0) {
    if (PanelBase* panel = panel_manager_->get_panel_by_id(active_tab_id_)) {
      // The content is rendered within the tab bar, so we don't need to render anything here
      // The active tab's content is already rendered in render_tab_bar()
    }
  }
}

bool TabbedPanel::add_panel_to_tab(uint32_t panel_id) {
  // Check if panel exists and isn't already in this tabbed panel
  if (!panel_manager_->get_panel_by_id(panel_id)) {
    return false;
  }
  
  // Check if panel is already in this tabbed panel
  if (std::find(tabbed_panel_ids_.begin(), tabbed_panel_ids_.end(), panel_id) != tabbed_panel_ids_.end()) {
    return false;
  }
  
  // Add to tabbed panels
  tabbed_panel_ids_.push_back(panel_id);
  
  // Hide the original panel since it's now displayed in a tab
  if (PanelBase* panel = panel_manager_->get_panel_by_id(panel_id)) {
    panel->set_visible(false);
  }
  
  // Set as active tab if it's the first one
  if (tabbed_panel_ids_.size() == 1) {
    active_tab_id_ = panel_id;
  }
  
  return true;
}

bool TabbedPanel::remove_panel_from_tab(uint32_t panel_id) {
  auto it = std::find(tabbed_panel_ids_.begin(), tabbed_panel_ids_.end(), panel_id);
  if (it == tabbed_panel_ids_.end()) {
    return false;
  }
  
  // Remove from tabbed panels
  tabbed_panel_ids_.erase(it);
  
  // Restore visibility of the panel
  if (PanelBase* panel = panel_manager_->get_panel_by_id(panel_id)) {
    panel->set_visible(true);
  }
  
  // If we removed the active tab, set a new active tab
  if (panel_id == active_tab_id_) {
    if (!tabbed_panel_ids_.empty()) {
      active_tab_id_ = tabbed_panel_ids_[0];  // Set first tab as active
    } else {
      active_tab_id_ = 0;  // No active tab
    }
  }
  
  return true;
}

bool TabbedPanel::set_active_tab(uint32_t panel_id) {
  auto it = std::find(tabbed_panel_ids_.begin(), tabbed_panel_ids_.end(), panel_id);
  if (it == tabbed_panel_ids_.end()) {
    return false;
  }
  
  active_tab_id_ = panel_id;
  return true;
}

}  // namespace BTQuant