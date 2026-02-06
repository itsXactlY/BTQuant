#include "../../include/components/tabbed_panel.hpp"

#include <iostream>

namespace BTQuant {

TabbedPanel::TabbedPanel(const PanelConfig& config, PanelManager* panel_manager)
    : PanelBase(config), panel_manager_(panel_manager) {
    // Initialize with the first panel if provided
    if (panel_manager_) {
        // Initially empty - panels will be added via drag-and-drop or API
    }
}

void TabbedPanel::render() {
    begin_panel_window();

    if (!is_visible()) {
        end_panel_window();
        return;
    }

    // Render the tab bar
    render_tab_bar();

    // Render the active tab content
    render_active_tab_content();

    end_panel_window();
}

void TabbedPanel::render_tab_bar() {
    if (tabbed_panels_.empty()) {
        // If no panels in tabbed panel, show a placeholder
        ImGui::Text("Drag a panel here to create tabs");
        return;
    }

    // Create a child window for the tab bar to handle scrolling if needed
    ImGui::BeginChild("TabBarContainer", ImVec2(0, ImGui::GetFrameHeight()), false, 
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    
    // Use ImGui's tab bar functionality
    if (ImGui::BeginTabBar("TabBar", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_FittingPolicyScroll)) {
        
        for (int i = 0; i < static_cast<int>(tabbed_panels_.size()); ++i) {
            uint32_t panel_id = tabbed_panels_[i];
            
            // Get the panel from panel manager to get its title
            PanelBase* panel = panel_manager_ ? panel_manager_->get_panel_by_id(panel_id) : nullptr;
            std::string tab_label = panel ? panel->get_title() : "Panel " + std::to_string(panel_id);
            
            // Add close button to the tab
            bool tab_closed = false;
            ImGuiTabItemFlags flags = (i == active_tab_index_) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
            
            if (ImGui::BeginTabItem(tab_label.c_str(), &tab_closed, flags)) {
                // Tab is selected, update active index
                active_tab_index_ = i;
                
                ImGui::EndTabItem();
            }
            
            // Handle tab closing
            if (tab_closed) {
                remove_panel(panel_id);
            }
            
            // Handle right-click context menu for the tab
            if (ImGui::BeginPopupContextItem(("TabContextMenu" + std::to_string(i)).c_str())) {
                handle_tab_context_menu(i);
                ImGui::EndPopup();
            }
        }
        
        ImGui::EndTabBar();
    }
    
    ImGui::EndChild();
}

void TabbedPanel::render_active_tab_content() {
    if (tabbed_panels_.empty() || active_tab_index_ < 0 || 
        active_tab_index_ >= static_cast<int>(tabbed_panels_.size())) {
        // No active tab or invalid index, show placeholder
        ImGui::Text("No active tab. Add panels to create tabs.");
        return;
    }
    
    uint32_t active_panel_id = tabbed_panels_[active_tab_index_];
    
    // Get the active panel from the panel manager and render it
    if (panel_manager_) {
        PanelBase* active_panel = panel_manager_->get_panel_by_id(active_panel_id);
        if (active_panel) {
            // Temporarily adjust the panel's visibility to ensure it renders within our tab
            bool original_visibility = active_panel->is_visible();
            active_panel->set_visible(true);
            
            // Render the panel content directly within our tabbed panel
            // We need to adjust the panel's size to fit within our tab area
            ImVec2 available_size = ImGui::GetContentRegionAvail();
            
            // Create a child window to contain the panel content
            ImGui::BeginChild("TabContent", available_size, true);
            
            // Temporarily adjust the panel's size to fit the tab content area
            ImVec2 original_size = active_panel->get_config().size;
            const_cast<PanelConfig&>(active_panel->get_config()).size = available_size;
            
            // Call the panel's render method directly
            active_panel->render();
            
            // Restore original size
            const_cast<PanelConfig&>(active_panel->get_config()).size = original_size;
            
            ImGui::EndChild();
            
            // Restore original visibility
            active_panel->set_visible(original_visibility);
        } else {
            ImGui::Text("Panel not found: %d", active_panel_id);
        }
    }
}

void TabbedPanel::add_panel(uint32_t panel_id) {
    // Check if panel is already in this tabbed panel
    for (uint32_t existing_id : tabbed_panels_) {
        if (existing_id == panel_id) {
            return; // Already exists
        }
    }
    
    tabbed_panels_.push_back(panel_id);
    
    // Set the newly added panel as the active tab
    active_tab_index_ = static_cast<int>(tabbed_panels_.size()) - 1;
}

void TabbedPanel::remove_panel(uint32_t panel_id) {
    for (auto it = tabbed_panels_.begin(); it != tabbed_panels_.end(); ++it) {
        if (*it == panel_id) {
            int index = std::distance(tabbed_panels_.begin(), it);
            tabbed_panels_.erase(it);
            
            // Adjust active tab index if needed
            if (active_tab_index_ >= static_cast<int>(tabbed_panels_.size()) && !tabbed_panels_.empty()) {
                active_tab_index_ = static_cast<int>(tabbed_panels_.size()) - 1;
            } else if (tabbed_panels_.empty()) {
                active_tab_index_ = 0;
            } else if (index < active_tab_index_) {
                active_tab_index_--;
            }
            
            // Show the panel again since it's no longer in the tabbed panel
            if (panel_manager_) {
                PanelBase* panel = panel_manager_->get_panel_by_id(panel_id);
                if (panel) {
                    panel->set_visible(true);
                }
            }
            
            break;
        }
    }
}

void TabbedPanel::set_active_tab(int tab_index) {
    if (tab_index >= 0 && tab_index < static_cast<int>(tabbed_panels_.size())) {
        active_tab_index_ = tab_index;
    }
}

void TabbedPanel::handle_tab_context_menu(int tab_index) {
    if (ImGui::MenuItem("Close Tab")) {
        if (tab_index >= 0 && tab_index < static_cast<int>(tabbed_panels_.size())) {
            uint32_t panel_id_to_remove = tabbed_panels_[tab_index];
            remove_panel(panel_id_to_remove);
        }
    }
    
    if (ImGui::MenuItem("Close Other Tabs")) {
        if (tab_index >= 0 && tab_index < static_cast<int>(tabbed_panels_.size())) {
            uint32_t panel_to_keep = tabbed_panels_[tab_index];
            // Remove all other panels and keep only the selected one
            std::vector<uint32_t> panels_to_remove;
            for (int i = 0; i < static_cast<int>(tabbed_panels_.size()); ++i) {
                if (i != tab_index) {
                    panels_to_remove.push_back(tabbed_panels_[i]);
                }
            }
            
            // Remove all panels except the selected one
            for (uint32_t panel_id : panels_to_remove) {
                remove_panel(panel_id);
            }
            
            // Set the remaining panel as the only one in the tabbed panel
            tabbed_panels_ = {panel_to_keep};
            active_tab_index_ = 0;
        }
    }
    
    if (ImGui::MenuItem("Close Tabs to the Right")) {
        // Close all tabs to the right of the current tab
        if (tab_index < static_cast<int>(tabbed_panels_.size()) - 1) {
            // Remove panels from the right side
            std::vector<uint32_t> panels_to_remove;
            for (int i = tab_index + 1; i < static_cast<int>(tabbed_panels_.size()); ++i) {
                panels_to_remove.push_back(tabbed_panels_[i]);
            }
            
            // Remove panels in reverse order to maintain indices
            for (int i = static_cast<int>(panels_to_remove.size()) - 1; i >= 0; --i) {
                remove_panel(panels_to_remove[i]);
            }
            
            if (active_tab_index_ >= static_cast<int>(tabbed_panels_.size())) {
                active_tab_index_ = static_cast<int>(tabbed_panels_.size()) - 1;
            }
        }
    }
}

bool TabbedPanel::handle_drop(uint32_t source_panel_id) {
    // Check if the source panel is already in this tabbed panel
    for (uint32_t existing_id : tabbed_panels_) {
        if (existing_id == source_panel_id) {
            return false; // Already exists in this tabbed panel
        }
    }
    
    // Add the source panel to this tabbed panel
    add_panel(source_panel_id);
    
    // If the panel manager exists, we need to properly manage the panel
    if (panel_manager_) {
        // Check if the source panel is part of another group and remove it
        auto group_ids = panel_manager_->get_panel_groups_for_panel(source_panel_id);
        for (uint32_t group_id : group_ids) {
            panel_manager_->remove_panel_from_group(group_id, source_panel_id);
        }
        
        // We don't remove the panel from the main panel manager, but we hide it
        // since it's now managed by this tabbed panel
        PanelBase* source_panel = panel_manager_->get_panel_by_id(source_panel_id);
        if (source_panel) {
            // Hide the source panel since it's now shown inside this tabbed panel
            source_panel->set_visible(false);
        }
    }
    
    return true;
}

} // namespace BTQuant