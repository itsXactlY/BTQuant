#include "../../include/ui/settings_manager.hpp"
#include "../../include/ui/appearance_settings.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <algorithm>
#include <vector>
#include <string>
#include <map>

namespace BTQuant {
namespace UI {

// ============================================================================
// Appearance Settings Implementation
// ============================================================================

AppearanceSettings::AppearanceSettings(SettingsManager& settings_manager) 
    : settings_manager_(settings_manager) {
    initialize_appearance_settings();
}

void AppearanceSettings::initialize_appearance_settings() {
    // Initialize theme settings
    initialize_theme_settings();
    
    // Initialize font settings
    initialize_font_settings();
    
    // Initialize opacity settings
    initialize_opacity_settings();
    
    // Initialize border settings
    initialize_border_settings();
}

void AppearanceSettings::initialize_theme_settings() {
    // Theme selection (Dark, Light, Custom)
    SettingInfo theme_setting;
    theme_setting.key = "appearance.theme";
    theme_setting.display_name = "Theme";
    theme_setting.description = "Select the application theme (Dark, Light, or Custom)";
    theme_setting.type = SettingType::ENUM;
    theme_setting.category = SettingCategory::APPEARANCE;
    theme_setting.enum_options = {"Dark Professional", "Light Professional", "High Contrast", "Custom"};
    theme_setting.enum_selected_index = 0;

    // Add callback to apply theme when changed
    theme_setting.on_change_callback = [this]() {
        apply_current_theme();
    };

    settings_manager_.register_setting(theme_setting);

    // Custom theme color settings
    SettingInfo bg_primary_color;
    bg_primary_color.key = "appearance.colors.background_primary";
    bg_primary_color.display_name = "Primary Background";
    bg_primary_color.description = "Primary background color for the interface";
    bg_primary_color.type = SettingType::COLOR;
    bg_primary_color.category = SettingCategory::APPEARANCE;
    bg_primary_color.color_value = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);  // Default dark

    // Add callback to apply theme when color changes
    bg_primary_color.on_change_callback = [this]() {
        apply_current_theme();
    };

    settings_manager_.register_setting(bg_primary_color);

    SettingInfo bg_secondary_color;
    bg_secondary_color.key = "appearance.colors.background_secondary";
    bg_secondary_color.display_name = "Secondary Background";
    bg_secondary_color.description = "Secondary background color for panels";
    bg_secondary_color.type = SettingType::COLOR;
    bg_secondary_color.category = SettingCategory::APPEARANCE;
    bg_secondary_color.color_value = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);  // Default dark

    // Add callback to apply theme when color changes
    bg_secondary_color.on_change_callback = [this]() {
        apply_current_theme();
    };

    settings_manager_.register_setting(bg_secondary_color);

    SettingInfo text_primary_color;
    text_primary_color.key = "appearance.colors.text_primary";
    text_primary_color.display_name = "Primary Text";
    text_primary_color.description = "Primary text color";
    text_primary_color.type = SettingType::COLOR;
    text_primary_color.category = SettingCategory::APPEARANCE;
    text_primary_color.color_value = ImVec4(0.9f, 0.9f, 0.9f, 1.0f);  // Default light text

    // Add callback to apply theme when color changes
    text_primary_color.on_change_callback = [this]() {
        apply_current_theme();
    };

    settings_manager_.register_setting(text_primary_color);

    SettingInfo accent_primary_color;
    accent_primary_color.key = "appearance.colors.accent_primary";
    accent_primary_color.display_name = "Accent Color";
    accent_primary_color.description = "Primary accent color for highlights";
    accent_primary_color.type = SettingType::COLOR;
    accent_primary_color.category = SettingCategory::APPEARANCE;
    accent_primary_color.color_value = ImVec4(0.2f, 0.6f, 1.0f, 1.0f);  // Default blue accent

    // Add callback to apply theme when color changes
    accent_primary_color.on_change_callback = [this]() {
        apply_current_theme();
    };

    settings_manager_.register_setting(accent_primary_color);

    SettingInfo border_color;
    border_color.key = "appearance.colors.border";
    border_color.display_name = "Border Color";
    border_color.description = "Color for panel borders";
    border_color.type = SettingType::COLOR;
    border_color.category = SettingCategory::APPEARANCE;
    border_color.color_value = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);  // Default dark border

    // Add callback to apply theme when color changes
    border_color.on_change_callback = [this]() {
        apply_current_theme();
    };

    settings_manager_.register_setting(border_color);
}

void AppearanceSettings::initialize_font_settings() {
    // Font family selection
    SettingInfo font_family_setting;
    font_family_setting.key = "appearance.font_family";
    font_family_setting.display_name = "Font Family";
    font_family_setting.description = "Select the primary font family for the interface";
    font_family_setting.type = SettingType::ENUM;
    font_family_setting.category = SettingCategory::APPEARANCE;
    font_family_setting.enum_options = {"Roboto", "Segoe UI", "Arial", "Consolas", "Custom"};
    font_family_setting.enum_selected_index = 0;  // Roboto as default

    // Add callback to apply font settings when changed
    font_family_setting.on_change_callback = [this]() {
        apply_font_settings();
    };

    settings_manager_.register_setting(font_family_setting);

    // Base font size
    SettingInfo font_size_setting;
    font_size_setting.key = "appearance.font_size";
    font_size_setting.display_name = "Base Font Size";
    font_size_setting.description = "Set the base font size for the interface";
    font_size_setting.type = SettingType::INTEGER;
    font_size_setting.category = SettingCategory::APPEARANCE;
    font_size_setting.int_value = 14;
    font_size_setting.min_int = 8;
    font_size_setting.max_int = 24;

    // Add callback to apply font settings when changed
    font_size_setting.on_change_callback = [this]() {
        apply_font_settings();
    };

    settings_manager_.register_setting(font_size_setting);

    // Header font size multiplier
    SettingInfo header_font_multiplier;
    header_font_multiplier.key = "appearance.header_font_multiplier";
    header_font_multiplier.display_name = "Header Font Scale";
    header_font_multiplier.description = "Multiplier for header font sizes relative to base font";
    header_font_multiplier.type = SettingType::FLOAT;
    header_font_multiplier.category = SettingCategory::APPEARANCE;
    header_font_multiplier.float_value = 1.2f;
    header_font_multiplier.min_float = 1.0f;
    header_font_multiplier.max_float = 2.0f;

    // Add callback to apply font settings when changed
    header_font_multiplier.on_change_callback = [this]() {
        apply_font_settings();
    };

    settings_manager_.register_setting(header_font_multiplier);

    // Monospace font size for code/data display
    SettingInfo mono_font_size_setting;
    mono_font_size_setting.key = "appearance.mono_font_size";
    mono_font_size_setting.display_name = "Monospace Font Size";
    mono_font_size_setting.description = "Font size for monospaced text (charts, tables, etc.)";
    mono_font_size_setting.type = SettingType::INTEGER;
    mono_font_size_setting.category = SettingCategory::APPEARANCE;
    mono_font_size_setting.int_value = 12;
    mono_font_size_setting.min_int = 8;
    mono_font_size_setting.max_int = 20;

    // Add callback to apply font settings when changed
    mono_font_size_setting.on_change_callback = [this]() {
        apply_font_settings();
    };

    settings_manager_.register_setting(mono_font_size_setting);
}

void AppearanceSettings::initialize_opacity_settings() {
    // Panel opacity
    SettingInfo panel_opacity_setting;
    panel_opacity_setting.key = "appearance.panel_opacity";
    panel_opacity_setting.display_name = "Panel Opacity";
    panel_opacity_setting.description = "Set the opacity level for panels";
    panel_opacity_setting.type = SettingType::FLOAT;
    panel_opacity_setting.category = SettingCategory::APPEARANCE;
    panel_opacity_setting.float_value = 1.0f;
    panel_opacity_setting.min_float = 0.1f;
    panel_opacity_setting.max_float = 1.0f;

    // Add callback to apply opacity settings when changed
    panel_opacity_setting.on_change_callback = [this]() {
        apply_current_theme();  // Opacity affects theme application
    };

    settings_manager_.register_setting(panel_opacity_setting);

    // Window opacity
    SettingInfo window_opacity_setting;
    window_opacity_setting.key = "appearance.window_opacity";
    window_opacity_setting.display_name = "Window Opacity";
    window_opacity_setting.description = "Set the opacity level for windows";
    window_opacity_setting.type = SettingType::FLOAT;
    window_opacity_setting.category = SettingCategory::APPEARANCE;
    window_opacity_setting.float_value = 1.0f;
    window_opacity_setting.min_float = 0.1f;
    window_opacity_setting.max_float = 1.0f;

    // Add callback to apply opacity settings when changed
    window_opacity_setting.on_change_callback = [this]() {
        apply_current_theme();  // Opacity affects theme application
    };

    settings_manager_.register_setting(window_opacity_setting);

    // Chart overlay opacity
    SettingInfo chart_overlay_opacity;
    chart_overlay_opacity.key = "appearance.chart_overlay_opacity";
    chart_overlay_opacity.display_name = "Chart Overlay Opacity";
    chart_overlay_opacity.description = "Opacity for chart overlays like grid lines and indicators";
    chart_overlay_opacity.type = SettingType::FLOAT;
    chart_overlay_opacity.category = SettingCategory::APPEARANCE;
    chart_overlay_opacity.float_value = 0.7f;
    chart_overlay_opacity.min_float = 0.1f;
    chart_overlay_opacity.max_float = 1.0f;

    // Add callback to apply opacity settings when changed
    chart_overlay_opacity.on_change_callback = [this]() {
        apply_current_theme();  // Opacity affects theme application
    };

    settings_manager_.register_setting(chart_overlay_opacity);
}

void AppearanceSettings::initialize_border_settings() {
    // Border width
    SettingInfo border_width_setting;
    border_width_setting.key = "appearance.border_width";
    border_width_setting.display_name = "Border Width";
    border_width_setting.description = "Width of panel borders in pixels";
    border_width_setting.type = SettingType::INTEGER;
    border_width_setting.category = SettingCategory::APPEARANCE;
    border_width_setting.int_value = 1;
    border_width_setting.min_int = 0;
    border_width_setting.max_int = 5;

    // Add callback to apply border settings when changed
    border_width_setting.on_change_callback = [this]() {
        apply_border_settings();
    };

    settings_manager_.register_setting(border_width_setting);

    // Border style
    SettingInfo border_style_setting;
    border_style_setting.key = "appearance.border_style";
    border_style_setting.display_name = "Border Style";
    border_style_setting.description = "Style of panel borders";
    border_style_setting.type = SettingType::ENUM;
    border_style_setting.category = SettingCategory::APPEARANCE;
    border_style_setting.enum_options = {"Solid", "Dashed", "Dotted", "Rounded", "None"};
    border_style_setting.enum_selected_index = 0;  // Solid as default

    // Add callback to apply border settings when changed
    border_style_setting.on_change_callback = [this]() {
        apply_border_settings();
    };

    settings_manager_.register_setting(border_style_setting);

    // Corner radius for rounded borders
    SettingInfo corner_radius_setting;
    corner_radius_setting.key = "appearance.corner_radius";
    corner_radius_setting.display_name = "Corner Radius";
    corner_radius_setting.description = "Radius for rounded corners (when rounded border style is selected)";
    corner_radius_setting.type = SettingType::INTEGER;
    corner_radius_setting.category = SettingCategory::APPEARANCE;
    corner_radius_setting.int_value = 4;
    corner_radius_setting.min_int = 0;
    corner_radius_setting.max_int = 20;

    // Add callback to apply border settings when changed
    corner_radius_setting.on_change_callback = [this]() {
        apply_border_settings();
    };

    settings_manager_.register_setting(corner_radius_setting);

    // Panel spacing
    SettingInfo panel_spacing_setting;
    panel_spacing_setting.key = "appearance.panel_spacing";
    panel_spacing_setting.display_name = "Panel Spacing";
    panel_spacing_setting.description = "Spacing between panels in the layout";
    panel_spacing_setting.type = SettingType::INTEGER;
    panel_spacing_setting.category = SettingCategory::APPEARANCE;
    panel_spacing_setting.int_value = 4;
    panel_spacing_setting.min_int = 0;
    panel_spacing_setting.max_int = 20;

    // Add callback to apply border settings when changed
    panel_spacing_setting.on_change_callback = [this]() {
        apply_border_settings();
    };

    settings_manager_.register_setting(panel_spacing_setting);
}

void AppearanceSettings::apply_current_theme() {
    // Get the current theme selection
    int theme_index = settings_manager_.get_enum("appearance.theme", 0);

    // Apply theme-specific settings based on selection
    if (theme_index == 3) {  // Custom theme
        // Use custom colors defined in settings
        ImVec4 bg_primary = settings_manager_.get_color("appearance.colors.background_primary", ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
        ImVec4 bg_secondary = settings_manager_.get_color("appearance.colors.background_secondary", ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        ImVec4 text_primary = settings_manager_.get_color("appearance.colors.text_primary", ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
        ImVec4 accent_primary = settings_manager_.get_color("appearance.colors.accent_primary", ImVec4(0.2f, 0.6f, 1.0f, 1.0f));
        ImVec4 border_color = settings_manager_.get_color("appearance.colors.border", ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

        // Apply custom theme to ImGui
        ImGuiStyle& style = ImGui::GetStyle();
        style.Colors[ImGuiCol_WindowBg] = bg_primary;
        style.Colors[ImGuiCol_ChildBg] = bg_secondary;
        style.Colors[ImGuiCol_PopupBg] = bg_secondary;
        style.Colors[ImGuiCol_Text] = text_primary;
        style.Colors[ImGuiCol_Button] = accent_primary;
        style.Colors[ImGuiCol_Border] = border_color;

        // Apply opacity settings
        float panel_opacity = settings_manager_.get_float("appearance.panel_opacity", 1.0f);
        float window_opacity = settings_manager_.get_float("appearance.window_opacity", 1.0f);
        float chart_overlay_opacity = settings_manager_.get_float("appearance.chart_overlay_opacity", 0.7f);

        style.Colors[ImGuiCol_WindowBg].w = window_opacity;
        style.Colors[ImGuiCol_ChildBg].w = panel_opacity;
        style.Colors[ImGuiCol_PopupBg].w = panel_opacity;
        style.Colors[ImGuiCol_FrameBg].w = panel_opacity;
        style.Colors[ImGuiCol_FrameBgHovered].w = panel_opacity;
        style.Colors[ImGuiCol_FrameBgActive].w = panel_opacity;
    }
    // Note: Default themes are handled by the existing settings system
}

void AppearanceSettings::apply_font_settings() {
    // Get font settings
    int base_font_size = settings_manager_.get_int("appearance.font_size", 14);
    int mono_font_size = settings_manager_.get_int("appearance.mono_font_size", 12);
    
    // In a real implementation, this would load and apply fonts to ImGui
    // For now, we'll just store the settings which can be applied elsewhere
    // The font family would typically be used to load specific font files
    
    // Apply font scale factors
    float header_scale = settings_manager_.get_float("appearance.header_font_multiplier", 1.2f);
    
    // These settings would be used by the UI rendering system
    // to determine appropriate font sizes for different elements
}

void AppearanceSettings::apply_border_settings() {
    // Get border settings
    int border_width = settings_manager_.get_int("appearance.border_width", 1);
    int corner_radius = settings_manager_.get_int("appearance.corner_radius", 4);
    int border_style = settings_manager_.get_enum("appearance.border_style", 0);  // 0 = Solid

    // Apply to ImGui style
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameBorderSize = (border_style == 4) ? 0.0f : static_cast<float>(border_width);  // None style means no border
    style.WindowRounding = (border_style == 3) ? static_cast<float>(corner_radius) : 0.0f;  // Rounded style means rounded corners
    style.ChildRounding = (border_style == 3) ? static_cast<float>(corner_radius) : 0.0f;
    style.FrameRounding = (border_style == 3) ? static_cast<float>(corner_radius) : 0.0f;
    style.GrabRounding = (border_style == 3) ? static_cast<float>(corner_radius) : 0.0f;
    style.ScrollbarRounding = (border_style == 3) ? static_cast<float>(corner_radius) : 0.0f;

    // Panel spacing
    int panel_spacing = settings_manager_.get_int("appearance.panel_spacing", 4);
    style.ItemSpacing = ImVec2(static_cast<float>(panel_spacing), static_cast<float>(panel_spacing));
    style.ItemInnerSpacing = ImVec2(static_cast<float>(panel_spacing), static_cast<float>(panel_spacing));
    style.WindowPadding = ImVec2(static_cast<float>(panel_spacing * 2), static_cast<float>(panel_spacing * 2));
}

void AppearanceSettings::apply_appearance_settings() {
    apply_current_theme();
    apply_font_settings();
    apply_border_settings();
}

std::vector<std::string> AppearanceSettings::get_available_themes() const {
    std::vector<std::string> themes;
    themes.push_back("Dark Professional");
    themes.push_back("Light Professional");
    themes.push_back("High Contrast");
    themes.push_back("Custom");
    return themes;
}

void AppearanceSettings::add_custom_theme(const std::string& name, const ThemeDefinition& theme_def) {
    // In a real implementation, this would store custom themes
    // For now, we just acknowledge the setting system handles this
    custom_themes_[name] = theme_def;
}

void AppearanceSettings::render_appearance_settings_ui() {
    // This would normally render a dedicated UI for appearance settings
    // Since the main settings UI already handles this, we'll just ensure
    // all appearance settings are properly registered
}

}  // namespace UI
}  // namespace BTQuant