#pragma once

#include <string>
#include <vector>
#include <map>

#include "imgui.h"
#include "settings_manager.hpp"

namespace BTQuant {
namespace UI {

// Structure to define a simple color theme (for appearance settings)
// Note: This is different from the full ThemeDefinition in unified_theme_system.hpp
struct SimpleColorTheme {
    ImVec4 background_primary;
    ImVec4 background_secondary;
    ImVec4 background_panel;
    ImVec4 text_primary;
    ImVec4 text_secondary;
    ImVec4 text_muted;
    ImVec4 price_up;
    ImVec4 price_down;
    ImVec4 price_neutral;
    ImVec4 accent_primary;
    ImVec4 accent_secondary;
    ImVec4 border_color;
    ImVec4 status_connected;
    ImVec4 status_disconnected;
    ImVec4 status_warning;
};

class AppearanceSettings {
public:
    explicit AppearanceSettings(SettingsManager& settings_manager);

    // Initialize all appearance settings
    void initialize_appearance_settings();

    // Apply current appearance settings to the UI
    void apply_appearance_settings();

    // Get available themes
    std::vector<std::string> get_available_themes() const;

    // Add a custom theme
    void add_custom_theme(const std::string& name, const SimpleColorTheme& theme_def);

    // Render appearance settings UI
    void render_appearance_settings_ui();

private:
    SettingsManager& settings_manager_;
    std::map<std::string, SimpleColorTheme> custom_themes_;

    // Initialize theme-specific settings
    void initialize_theme_settings();

    // Initialize font settings
    void initialize_font_settings();

    // Initialize opacity settings
    void initialize_opacity_settings();

    // Initialize border settings
    void initialize_border_settings();

    // Apply current theme
    void apply_current_theme();

    // Apply font settings
    void apply_font_settings();

    // Apply border settings
    void apply_border_settings();
};

}  // namespace UI
}  // namespace BTQuant