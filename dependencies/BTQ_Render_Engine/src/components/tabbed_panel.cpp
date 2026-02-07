#include "../../include/components/tabbed_panel.hpp"

#include <algorithm>
#include <iostream>

// Include the panel manager header for implementation
#include "../../include/components/panel_manager.hpp"

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
    // The active tab content is rendered within the tab bar, so no need to call render_active_tab_content() separately
  } else {
    // If no tabs, show a placeholder message with enhanced visual feedback
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f)); // Slightly dimmed text
    ImGui::TextWrapped("Drag a panel here to create a tabbed group");
    ImGui::PopStyleColor();
    
    ImGui::Spacing();
    
    // Show supported panel types with better formatting
    ImGui::Text("Supported panels:");
    ImGui::Indent(20.0f);
    ImGui::Text("• Chart Panel");
    ImGui::Text("• Time & Sales Panel");
    ImGui::Text("• Other panels...");
    ImGui::Unindent(20.0f);
    
    // Visual indicator showing where to drag
    ImVec2 cursor_pos = ImGui::GetCursorPos();
    ImVec2 window_size = ImVec2(ImGui::GetWindowWidth() - 20, 40);
    ImGui::SetCursorPos(ImVec2(cursor_pos.x + 10, cursor_pos.y + 10));
    
    // Draw a dashed border to indicate the drop zone
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 drop_zone_min = ImGui::GetCursorScreenPos();
    ImVec2 drop_zone_max = ImVec2(drop_zone_min.x + window_size.x, drop_zone_min.y + window_size.y);
    
    // Draw dashed rectangle
    ImU32 borderColor = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
    float dash_length = 6.0f;
    float gap_length = 4.0f;
    
    // Draw horizontal lines (top and bottom)
    for (float x = drop_zone_min.x; x < drop_zone_max.x; x += dash_length + gap_length) {
        float end_x = (x + dash_length < drop_zone_max.x) ? x + dash_length : drop_zone_max.x;
        draw_list->AddLine(ImVec2(x, drop_zone_min.y), ImVec2(end_x, drop_zone_min.y), borderColor, 2.0f);
        draw_list->AddLine(ImVec2(x, drop_zone_max.y), ImVec2(end_x, drop_zone_max.y), borderColor, 2.0f);
    }
    
    // Draw vertical lines (left and right)
    for (float y = drop_zone_min.y; y < drop_zone_max.y; y += dash_length + gap_length) {
        float end_y = (y + dash_length < drop_zone_max.y) ? y + dash_length : drop_zone_max.y;
        draw_list->AddLine(ImVec2(drop_zone_min.x, y), ImVec2(drop_zone_min.x, end_y), borderColor, 2.0f);
        draw_list->AddLine(ImVec2(drop_zone_max.x, y), ImVec2(drop_zone_max.x, end_y), borderColor, 2.0f);
    }
    
    ImGui::Dummy(window_size);
    ImGui::SetCursorPos(cursor_pos);
  }

  end_panel_window();
}

void TabbedPanel::render_tab_bar() {
  if (ImGui::BeginTabBar(tab_bar_id_.c_str(), ImGuiTabBarFlags_FittingPolicyScroll)) {
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

          // Calculate the available space for the panel content within the tab
          ImVec2 content_size = ImGui::GetContentRegionAvail();

          // Temporarily adjust the panel's size to fit within the tab content area
          ImVec2 original_size = panel->get_config().size;
          ImVec2 adjusted_size = ImVec2(content_size.x, content_size.y);

          // Create a child window to contain the panel content
          std::string child_window_id = "panel_content_" + std::to_string(panel_id);
          ImGui::BeginChild(child_window_id.c_str(), adjusted_size, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

          // Temporarily hide the panel's window decorations since it's inside a tab
          // We'll just render the core content
          panel->render();

          ImGui::EndChild();

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