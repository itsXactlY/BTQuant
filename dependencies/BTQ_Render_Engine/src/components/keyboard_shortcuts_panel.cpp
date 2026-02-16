#include "../../include/components/keyboard_shortcuts_panel.hpp"
#include "imgui.h"
#include <algorithm>

namespace BTQuant {

KeyboardShortcutsPanel::KeyboardShortcutsPanel(const PanelConfig& config)
    : PanelBase(config) {
    // Initialize with default shortcuts
    shortcuts_ = {
        {"New Chart", "Ctrl+N", 0, false},
        {"Open Chart", "Ctrl+O", 0, false},
        {"Save Layout", "Ctrl+S", 0, false},
        {"Close Panel", "Ctrl+W", 0, false},
        {"Toggle Fullscreen", "F11", 0, false},
        {"Switch Symbol", "Ctrl+Shift+S", 0, false},
        {"Timeframe Up", "Ctrl+Up", 0, false},
        {"Timeframe Down", "Ctrl+Down", 0, false},
        {"Buy Market", "F1", 0, false},
        {"Sell Market", "F2", 0, false},
        {"Cancel Orders", "Ctrl+Delete", 0, false},
        {"Toggle Crosshair", "C", 0, false},
        {"Zoom In", "Ctrl++", 0, false},
        {"Zoom Out", "Ctrl+-", 0, false},
        {"Reset View", "Ctrl+R", 0, false}
    };
}

void KeyboardShortcutsPanel::initialize() {
    // Stub implementation
}

void KeyboardShortcutsPanel::render_content() {
    begin_panel_window();
    
    if (!is_visible()) {
        end_panel_window();
        return;
    }
    
    render_search_filter();
    ImGui::Separator();
    render_shortcut_list();
    ImGui::Separator();
    render_actions();
    
    end_panel_window();
}

void KeyboardShortcutsPanel::render_search_filter() {
    ImGui::Text("Keyboard Shortcuts");
    
    // Search filter
    ImGui::SetNextItemWidth(200);
    ImGui::InputText("Search", search_filter_, sizeof(search_filter_));
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        search_filter_[0] = '\0';
    }
    
    // Category filter
    const char* categories[] = {"All", "General", "Trading", "Chart", "Navigation"};
    ImGui::SameLine();
    ImGui::Combo("Category", &selected_category_, categories, IM_ARRAYSIZE(categories));
}

void KeyboardShortcutsPanel::render_shortcut_list() {
    // Column headers
    if (ImGui::BeginTable("ShortcutsTable", 3, 
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | 
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable,
                          ImVec2(0, 300))) {
        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Shortcut", ImGuiTableColumnFlags_WidthFixed, 150);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 100);
        ImGui::TableHeadersRow();
        
        std::string search = search_filter_;
        std::transform(search.begin(), search.end(), search.begin(), ::tolower);
        
        for (size_t i = 0; i < shortcuts_.size(); ++i) {
            auto& shortcut = shortcuts_[i];
            
            // Filter by search
            std::string action_lower = shortcut.action;
            std::transform(action_lower.begin(), action_lower.end(), 
                          action_lower.begin(), ::tolower);
            
            if (!search.empty() && action_lower.find(search) == std::string::npos) {
                continue;
            }
            
            ImGui::TableNextRow();
            
            // Action name
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", shortcut.action.c_str());
            
            // Shortcut key
            ImGui::TableSetColumnIndex(1);
            if (shortcut.is_recording) {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Press key...");
            } else {
                // Display shortcut with a button-like appearance
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.4f, 0.4f, 1.0f));
                ImGui::SmallButton(shortcut.key_combo.c_str());
                ImGui::PopStyleColor(2);
            }
            
            // Actions
            ImGui::TableSetColumnIndex(2);
            ImGui::PushID(static_cast<int>(i));
            
            if (shortcut.is_recording) {
                if (ImGui::SmallButton("Stop")) {
                    shortcut.is_recording = false;
                }
            } else {
                if (ImGui::SmallButton("Edit")) {
                    start_recording(static_cast<int>(i));
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Reset")) {
                    // Reset to default (stub)
                }
            }
            
            ImGui::PopID();
        }
        
        ImGui::EndTable();
    }
}

void KeyboardShortcutsPanel::render_actions() {
    // Add custom shortcut
    if (ImGui::Button("Add Custom Shortcut")) {
        ImGui::OpenPopup("AddShortcutPopup");
    }
    
    if (ImGui::BeginPopup("AddShortcutPopup")) {
        static char action_name[64] = "";
        static char key_combo[32] = "";
        
        ImGui::InputText("Action", action_name, sizeof(action_name));
        ImGui::InputText("Key Combo", key_combo, sizeof(key_combo));
        
        if (ImGui::Button("Add") && action_name[0] != '\0') {
            shortcuts_.push_back({std::string(action_name), 
                                  std::string(key_combo), 0, false});
            action_name[0] = '\0';
            key_combo[0] = '\0';
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }
        
        ImGui::EndPopup();
    }
    
    ImGui::SameLine();
    
    // Reset all to defaults
    if (ImGui::Button("Reset All to Defaults")) {
        ImGui::OpenPopup("ConfirmReset");
    }
    
    if (ImGui::BeginPopup("ConfirmReset")) {
        ImGui::Text("Reset all shortcuts to defaults?");
        if (ImGui::Button("Yes")) {
            reset_to_defaults();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("No")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    
    ImGui::SameLine();
    
    // Export/Import
    if (ImGui::Button("Export...")) {
        export_shortcuts("shortcuts.json");
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button("Import...")) {
        import_shortcuts("shortcuts.json");
    }
    
    // Help text
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1), 
                       "Click 'Edit' to record a new key combination");
}

void KeyboardShortcutsPanel::start_recording(int index) {
    // Stop any other recording
    for (auto& s : shortcuts_) {
        s.is_recording = false;
    }
    
    // Start recording this one
    if (index >= 0 && index < static_cast<int>(shortcuts_.size())) {
        shortcuts_[index].is_recording = true;
        recording_index_ = index;
    }
}

void KeyboardShortcutsPanel::stop_recording() {
    if (recording_index_ >= 0 && recording_index_ < static_cast<int>(shortcuts_.size())) {
        shortcuts_[recording_index_].is_recording = false;
    }
    recording_index_ = -1;
}

void KeyboardShortcutsPanel::reset_to_defaults() {
    shortcuts_ = {
        {"New Chart", "Ctrl+N", 0, false},
        {"Open Chart", "Ctrl+O", 0, false},
        {"Save Layout", "Ctrl+S", 0, false},
        {"Close Panel", "Ctrl+W", 0, false},
        {"Toggle Fullscreen", "F11", 0, false},
        {"Switch Symbol", "Ctrl+Shift+S", 0, false},
        {"Timeframe Up", "Ctrl+Up", 0, false},
        {"Timeframe Down", "Ctrl+Down", 0, false},
        {"Buy Market", "F1", 0, false},
        {"Sell Market", "F2", 0, false},
        {"Cancel Orders", "Ctrl+Delete", 0, false},
        {"Toggle Crosshair", "C", 0, false},
        {"Zoom In", "Ctrl++", 0, false},
        {"Zoom Out", "Ctrl+-", 0, false},
        {"Reset View", "Ctrl+R", 0, false}
    };
}

bool KeyboardShortcutsPanel::export_shortcuts(const std::string& filename) {
    // Stub: Would export to JSON
    return true;
}

bool KeyboardShortcutsPanel::import_shortcuts(const std::string& filename) {
    // Stub: Would import from JSON
    return true;
}

}  // namespace BTQuant
