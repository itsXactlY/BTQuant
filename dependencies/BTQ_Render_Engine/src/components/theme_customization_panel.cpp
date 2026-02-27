#include "../../include/components/theme_customization_panel.hpp"
#include "imgui.h"
#include <fstream>
#include <sstream>

namespace BTQuant {

ThemeCustomizationPanel::ThemeCustomizationPanel(const PanelConfig& config)
    : PanelBase(config) {
    // Initialize with default theme colors
    current_theme_ = "Dark";
    themes_["Dark"] = {
        ImVec4(0.09f, 0.09f, 0.09f, 1.00f),  // Background
        ImVec4(0.15f, 0.15f, 0.15f, 1.00f),  // WindowBg
        ImVec4(0.20f, 0.20f, 0.20f, 1.00f),  // ChildBg
        ImVec4(0.26f, 0.59f, 0.98f, 1.00f),  // Accent
        ImVec4(0.40f, 0.40f, 0.40f, 1.00f),  // Border
        ImVec4(1.00f, 1.00f, 1.00f, 1.00f),  // Text
        ImVec4(0.00f, 0.80f, 0.00f, 1.00f),  // BullColor
        ImVec4(1.00f, 0.00f, 0.00f, 1.00f),  // BearColor
        ImVec4(1.00f, 0.85f, 0.00f, 1.00f)   // VolumeColor
    };
    
    themes_["Light"] = {
        ImVec4(0.94f, 0.94f, 0.94f, 1.00f),  // Background
        ImVec4(1.00f, 1.00f, 1.00f, 1.00f),  // WindowBg
        ImVec4(0.98f, 0.98f, 0.98f, 1.00f),  // ChildBg
        ImVec4(0.26f, 0.59f, 0.98f, 1.00f),  // Accent
        ImVec4(0.70f, 0.70f, 0.70f, 1.00f),  // Border
        ImVec4(0.00f, 0.00f, 0.00f, 1.00f),  // Text
        ImVec4(0.00f, 0.60f, 0.00f, 1.00f),  // BullColor
        ImVec4(0.80f, 0.00f, 0.00f, 1.00f),  // BearColor
        ImVec4(0.60f, 0.40f, 0.00f, 1.00f)   // VolumeColor
    };
    
    themes_["Nord"] = {
        ImVec4(0.13f, 0.16f, 0.20f, 1.00f),  // Background
        ImVec4(0.18f, 0.21f, 0.26f, 1.00f),  // WindowBg
        ImVec4(0.22f, 0.25f, 0.30f, 1.00f),  // ChildBg
        ImVec4(0.54f, 0.72f, 0.83f, 1.00f),  // Accent
        ImVec4(0.35f, 0.39f, 0.45f, 1.00f),  // Border
        ImVec4(0.92f, 0.93f, 0.94f, 1.00f),  // Text
        ImVec4(0.63f, 0.80f, 0.58f, 1.00f),  // BullColor
        ImVec4(0.86f, 0.55f, 0.59f, 1.00f),  // BearColor
        ImVec4(0.88f, 0.75f, 0.53f, 1.00f)   // VolumeColor
    };
    
    // Copy current theme to working colors
    working_colors_ = themes_[current_theme_];
}

void ThemeCustomizationPanel::initialize() {
    // Stub implementation
}

void ThemeCustomizationPanel::render() {
    begin_panel_window();
    
    if (!is_visible()) {
        end_panel_window();
        return;
    }
    
    render_theme_selector();
    ImGui::Separator();
    render_color_editor();
    ImGui::Separator();
    render_preview();
    ImGui::Separator();
    render_actions();
    
    end_panel_window();
}

void ThemeCustomizationPanel::render_theme_selector() {
    ImGui::Text("Theme Presets");
    
    // Theme dropdown
    if (ImGui::BeginCombo("Select Theme", current_theme_.c_str())) {
        for (const auto& [name, colors] : themes_) {
            bool is_selected = (current_theme_ == name);
            if (ImGui::Selectable(name.c_str(), is_selected)) {
                current_theme_ = name;
                working_colors_ = colors;
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    
    ImGui::SameLine();
    if (ImGui::Button("New Theme")) {
        ImGui::OpenPopup("NewThemePopup");
    }
    
    if (ImGui::BeginPopup("NewThemePopup")) {
        static char new_name[64] = "";
        ImGui::InputText("Theme Name", new_name, sizeof(new_name));
        if (ImGui::Button("Create") && new_name[0] != '\0') {
            themes_[std::string(new_name)] = working_colors_;
            current_theme_ = new_name;
            new_name[0] = '\0';
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void ThemeCustomizationPanel::render_color_editor() {
    ImGui::Text("Color Customization");
    
    // Color editors for each theme element
    struct ColorEntry {
        const char* name;
        ImVec4* color;
    };
    
    ColorEntry colors[] = {
        {"Background", &working_colors_.background},
        {"Window Background", &working_colors_.window_bg},
        {"Child Background", &working_colors_.child_bg},
        {"Accent Color", &working_colors_.accent},
        {"Border Color", &working_colors_.border},
        {"Text Color", &working_colors_.text},
        {"Bull (Buy) Color", &working_colors_.bull_color},
        {"Bear (Sell) Color", &working_colors_.bear_color},
        {"Volume Color", &working_colors_.volume_color}
    };
    
    for (auto& entry : colors) {
        ImGui::ColorEdit4(entry.name, (float*)entry.color, 
                          ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaPreview);
    }
}

void ThemeCustomizationPanel::render_preview() {
    ImGui::Text("Preview");
    
    // Draw a preview panel with current colors
    ImGui::PushStyleColor(ImGuiCol_ChildBg, working_colors_.child_bg);
    ImGui::PushStyleColor(ImGuiCol_Border, working_colors_.border);
    ImGui::PushStyleColor(ImGuiCol_Text, working_colors_.text);
    
    if (ImGui::BeginChild("Preview", ImVec2(0, 120), true)) {
        // Simulated chart preview
        ImGui::Text("Sample Chart Panel");
        
        // Draw sample candlesticks
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        
        for (int i = 0; i < 10; ++i) {
            float x = p.x + i * 30.0f + 15.0f;
            bool is_bull = (i % 2 == 0);
            ImVec4 color = is_bull ? working_colors_.bull_color : working_colors_.bear_color;
            ImU32 col = ImGui::ColorConvertFloat4ToU32(color);
            
            // Draw wick
            draw_list->AddLine(ImVec2(x, p.y + 20), ImVec2(x, p.y + 80), col, 1.0f);
            
            // Draw body
            float body_top = is_bull ? p.y + 35 : p.y + 45;
            float body_bottom = is_bull ? p.y + 65 : p.y + 55;
            draw_list->AddRectFilled(ImVec2(x - 8, body_top), ImVec2(x + 8, body_bottom), col);
        }
        
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 90);
        
        // Sample volume bars
        ImGui::Text("Volume:");
        for (int i = 0; i < 10; ++i) {
            ImGui::SameLine();
            ImGui::ColorButton("##vol", working_colors_.volume_color, 
                              ImGuiColorEditFlags_NoInputs, ImVec2(20, 10));
        }
    }
    ImGui::EndChild();
    
    ImGui::PopStyleColor(3);
}

void ThemeCustomizationPanel::render_actions() {
    // Apply button
    if (ImGui::Button("Apply Theme")) {
        apply_theme();
    }
    
    ImGui::SameLine();
    
    // Save button
    if (ImGui::Button("Save Theme")) {
        save_theme(current_theme_);
    }
    
    ImGui::SameLine();
    
    // Export button
    if (ImGui::Button("Export...")) {
        ImGui::OpenPopup("ExportPopup");
    }
    
    if (ImGui::BeginPopup("ExportPopup")) {
        static char filename[256] = "theme.json";
        ImGui::InputText("Filename", filename, sizeof(filename));
        if (ImGui::Button("Export")) {
            export_theme(filename);
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    
    ImGui::SameLine();
    
    // Import button
    if (ImGui::Button("Import...")) {
        ImGui::OpenPopup("ImportPopup");
    }
    
    if (ImGui::BeginPopup("ImportPopup")) {
        static char filename[256] = "";
        ImGui::InputText("Filename", filename, sizeof(filename));
        if (ImGui::Button("Import")) {
            import_theme(filename);
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    
    // Delete theme button (not for default themes)
    if (current_theme_ != "Dark" && current_theme_ != "Light" && current_theme_ != "Nord") {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        if (ImGui::Button("Delete Theme")) {
            themes_.erase(current_theme_);
            current_theme_ = "Dark";
            working_colors_ = themes_[current_theme_];
        }
        ImGui::PopStyleColor();
    }
}

void ThemeCustomizationPanel::apply_theme() {
    // Update the stored theme
    themes_[current_theme_] = working_colors_;
    
    // Apply to ImGui style (stub - would need full implementation)
    ImGuiStyle& style = ImGui::GetStyle();
    style.Colors[ImGuiCol_WindowBg] = working_colors_.window_bg;
    style.Colors[ImGuiCol_ChildBg] = working_colors_.child_bg;
    style.Colors[ImGuiCol_Border] = working_colors_.border;
    style.Colors[ImGuiCol_Text] = working_colors_.text;
    style.Colors[ImGuiCol_Button] = working_colors_.accent;
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(
        working_colors_.accent.x * 1.2f,
        working_colors_.accent.y * 1.2f,
        working_colors_.accent.z * 1.2f,
        working_colors_.accent.w
    );
}

void ThemeCustomizationPanel::save_theme(const std::string& name) {
    themes_[name] = working_colors_;
    // Stub: Would save to config file
}

bool ThemeCustomizationPanel::export_theme(const std::string& filename) {
    // Stub: Would export to JSON file
    return true;
}

bool ThemeCustomizationPanel::import_theme(const std::string& filename) {
    // Stub: Would import from JSON file
    return true;
}

}  // namespace BTQuant
