#include "ui/quick_actions.hpp"
#include "imgui.h"
#include "components/panel_manager.hpp"
#include <algorithm>

namespace BTQuant {

QuickActionsToolbar::QuickActionsToolbar() {
    initialize_default_actions();
}

void QuickActionsToolbar::initialize_default_actions() {
    // Add default quick actions for the trading terminal
    actions_.push_back(QuickAction("New Chart", "📈", []() {
        // Create a new chart panel
        // This would typically interact with the PanelManager to create a new panel
        // For now, we'll just log that the action was triggered
        printf("New Chart action triggered\n");
    }, "Create a new chart panel"));
    
    actions_.push_back(QuickAction("New Watchlist", "👁️", []() {
        printf("New Watchlist action triggered\n");
    }, "Create a new watchlist panel"));
    
    actions_.push_back(QuickAction("New Order Book", "📊", []() {
        printf("New Order Book action triggered\n");
    }, "Create a new order book panel"));
    
    actions_.push_back(QuickAction("Screenshot", "📷", []() {
        printf("Screenshot action triggered\n");
    }, "Take screenshot of active panel"));
    
    actions_.push_back(QuickAction("Settings", "⚙️", []() {
        printf("Settings action triggered\n");
    }, "Open application settings"));
    
    actions_.push_back(QuickAction("Export Data", "📤", []() {
        printf("Export Data action triggered\n");
    }, "Export current data"));
    
    actions_.push_back(QuickAction("Refresh", "🔄", []() {
        printf("Refresh action triggered\n");
    }, "Refresh current view"));
    
    actions_.push_back(QuickAction("Zoom Reset", "🔍", []() {
        printf("Zoom Reset action triggered\n");
    }, "Reset zoom level"));
}

void QuickActionsToolbar::render() {
    if (!visible_) return;
    
    render_toolbar_window();
}

void QuickActionsToolbar::render_toolbar_window() {
    // Create a floating window for the toolbar
    ImGui::SetNextWindowPos(position_, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(button_size_.x * actions_.size() + 40, button_size_.y + 20), ImGuiCond_FirstUseEver);
    
    // Use window management functions to keep it on top since ImGuiWindowFlags_TopMost doesn't exist
    ImGui::SetNextWindowFocus(); // Emulate "always on top" by focusing the window each frame
    
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoScrollbar |
                                   ImGuiWindowFlags_NoScrollWithMouse |
                                   ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus; // Prevent other windows from stealing focus

    // Make it look like a floating toolbar
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5.0f, 5.0f));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.12f, 0.9f));

    if (ImGui::Begin("Quick Actions Toolbar", nullptr, window_flags)) {
        // Handle dragging if enabled
        if (draggable_) {
            handle_dragging();
        }
        
        // Render action buttons horizontally
        for (auto& action : actions_) {
            if (!action.enabled) continue;
            
            // Button styling
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.22f, 0.8f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.32f, 0.9f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.42f, 1.0f));
            
            if (ImGui::Button(action.icon.c_str(), button_size_)) {
                if (action.callback) {
                    action.callback();
                }
            }
            
            // Show tooltip on hover
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                ImGui::Text("%s", action.name.c_str());
                if (!action.tooltip.empty()) {
                    ImGui::Separator();
                    ImGui::TextDisabled("%s", action.tooltip.c_str());
                }
                ImGui::EndTooltip();
            }
            
            ImGui::PopStyleColor(3);
            
            // Add spacing between buttons
            ImGui::SameLine();
        }
        
        // Remove the extra SameLine from the last button
        ImGui::Dummy(ImVec2(0, 0));
    }
    ImGui::End();
    
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void QuickActionsToolbar::handle_dragging() {
    // Handle dragging of the toolbar window
    if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 mouse_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        if (mouse_delta.x != 0.0f || mouse_delta.y != 0.0f) {
            position_ = ImVec2(position_.x + mouse_delta.x, position_.y + mouse_delta.y);
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
        }
    }
}

void QuickActionsToolbar::add_action(const QuickAction& action) {
    // Check if action with same name already exists
    auto it = std::find_if(actions_.begin(), actions_.end(),
                          [&action](const QuickAction& existing) {
                              return existing.name == action.name;
                          });
    
    if (it != actions_.end()) {
        // Replace existing action
        *it = action;
    } else {
        // Add new action
        actions_.push_back(action);
    }
}

void QuickActionsToolbar::remove_action(const std::string& name) {
    actions_.erase(std::remove_if(actions_.begin(), actions_.end(),
                                 [&name](const QuickAction& action) {
                                     return action.name == name;
                                 }), actions_.end());
}

void QuickActionsToolbar::set_position(const ImVec2& pos) {
    position_ = pos;
}

std::vector<QuickAction> QuickActionsToolbar::get_default_actions() {
    QuickActionsToolbar temp_toolbar;
    return temp_toolbar.actions_;
}

} // namespace BTQuant