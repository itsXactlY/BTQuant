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
    ImGuiStyle& style = ImGui::GetStyle();

    switch (theme_index) {
        case 0: // Dark Professional
            // Apply dark theme colors
            style.Colors[ImGuiCol_Text] = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
            style.Colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.10f, 0.94f);
            style.Colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
            style.Colors[ImGuiCol_PopupBg] = ImVec4(0.10f, 0.10f, 0.10f, 0.94f);
            style.Colors[ImGuiCol_Border] = ImVec4(0.30f, 0.30f, 0.30f, 0.50f);
            style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
            style.Colors[ImGuiCol_FrameBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.54f);
            style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.30f, 0.30f, 0.30f, 0.67f);
            style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
            style.Colors[ImGuiCol_TitleBg] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
            style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
            style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
            style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
            style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
            style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
            style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
            style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.24f, 0.52f, 0.88f, 1.00f);
            style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
            style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
            style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
            style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_Separator] = ImVec4(0.39f, 0.39f, 0.39f, 0.62f);
            style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
            style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
            style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
            style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
            style.Colors[ImGuiCol_Tab] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
            style.Colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
            style.Colors[ImGuiCol_TabActive] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
            style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.07f, 0.10f, 0.15f, 0.97f);
            style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.14f, 0.26f, 0.42f, 1.00f);
            style.Colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
            style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
            style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
            style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
            style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
            style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
            style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
            style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
            style.Colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
            style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
            style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
            style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
            break;

        case 1: // Light Professional
            // Apply light theme colors
            style.Colors[ImGuiCol_Text] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
            style.Colors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
            style.Colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
            style.Colors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.98f);
            style.Colors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.30f);
            style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
            style.Colors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
            style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
            style.Colors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
            style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
            style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
            style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
            style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.53f);
            style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.69f, 0.69f, 0.69f, 0.80f);
            style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.49f, 0.49f, 0.49f, 0.80f);
            style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.49f, 0.49f, 0.49f, 1.00f);
            style.Colors[ImGuiCol_CheckMark] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
            style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_Header] = ImVec4(0.26f, 0.59f, 0.98f, 0.31f);
            style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
            style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_Separator] = ImVec4(0.39f, 0.39f, 0.39f, 0.62f);
            style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.78f);
            style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.26f, 0.59f, 0.98f, 0.25f);
            style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.67f);
            style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
            style.Colors[ImGuiCol_Tab] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
            style.Colors[ImGuiCol_TabHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.80f);
            style.Colors[ImGuiCol_TabActive] = ImVec4(0.76f, 0.76f, 0.76f, 1.00f);
            style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.92f, 0.92f, 0.92f, 1.00f);
            style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.76f, 0.76f, 0.76f, 1.00f);
            style.Colors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
            style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
            style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.45f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.78f, 0.87f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.57f, 0.57f, 0.64f, 1.00f);
            style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.68f, 0.68f, 0.74f, 1.00f);
            style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
            style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(0.30f, 0.30f, 0.30f, 0.09f);
            style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
            style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.26f, 0.59f, 0.98f, 0.90f);
            style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
            style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.70f, 0.70f, 0.70f, 0.70f);
            style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.20f);
            style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);
            break;

        case 2: // High Contrast
            // Apply high contrast theme colors
            style.Colors[ImGuiCol_Text] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
            style.Colors[ImGuiCol_WindowBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
            style.Colors[ImGuiCol_PopupBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_Border] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
            style.Colors[ImGuiCol_FrameBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);
            style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.20f, 0.20f, 0.20f, 0.80f);
            style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
            style.Colors[ImGuiCol_TitleBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
            style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.53f);
            style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.80f, 0.80f, 0.80f, 1.00f);
            style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
            style.Colors[ImGuiCol_CheckMark] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_SliderGrab] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
            style.Colors[ImGuiCol_Button] = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);
            style.Colors[ImGuiCol_ButtonHovered] = ImVec4(1.00f, 1.00f, 1.00f, 0.80f);
            style.Colors[ImGuiCol_ButtonActive] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_Header] = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);
            style.Colors[ImGuiCol_HeaderHovered] = ImVec4(1.00f, 1.00f, 1.00f, 0.80f);
            style.Colors[ImGuiCol_HeaderActive] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_Separator] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.80f, 0.80f, 0.80f, 1.00f);
            style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
            style.Colors[ImGuiCol_ResizeGrip] = ImVec4(1.00f, 1.00f, 1.00f, 0.50f);
            style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(1.00f, 1.00f, 1.00f, 0.80f);
            style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_Tab] = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);
            style.Colors[ImGuiCol_TabHovered] = ImVec4(1.00f, 1.00f, 1.00f, 0.80f);
            style.Colors[ImGuiCol_TabActive] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);
            style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_PlotLines] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_PlotHistogram] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
            style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_TableBorderLight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
            style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
            style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
            style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.35f);
            style.Colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
            style.Colors[ImGuiCol_NavHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
            style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
            style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.20f);
            style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.35f);
            break;

        case 3: // Custom theme
        default:
            // Use custom colors defined in settings
            ImVec4 bg_primary = settings_manager_.get_color("appearance.colors.background_primary", ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
            ImVec4 bg_secondary = settings_manager_.get_color("appearance.colors.background_secondary", ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
            ImVec4 text_primary = settings_manager_.get_color("appearance.colors.text_primary", ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
            ImVec4 accent_primary = settings_manager_.get_color("appearance.colors.accent_primary", ImVec4(0.2f, 0.6f, 0.6f, 1.0f));
            ImVec4 border_color = settings_manager_.get_color("appearance.colors.border", ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

            // Apply custom theme to ImGui
            style.Colors[ImGuiCol_WindowBg] = bg_primary;
            style.Colors[ImGuiCol_ChildBg] = bg_secondary;
            style.Colors[ImGuiCol_PopupBg] = bg_secondary;
            style.Colors[ImGuiCol_Text] = text_primary;
            style.Colors[ImGuiCol_Button] = accent_primary;
            style.Colors[ImGuiCol_Border] = border_color;
            break;
    }

    // Apply opacity settings regardless of theme
    float panel_opacity = settings_manager_.get_float("appearance.panel_opacity", 1.0f);
    float window_opacity = settings_manager_.get_float("appearance.window_opacity", 1.0f);
    float chart_overlay_opacity = settings_manager_.get_float("appearance.chart_overlay_opacity", 0.7f);

    // Adjust opacities for different elements
    style.Colors[ImGuiCol_WindowBg].w = window_opacity;
    style.Colors[ImGuiCol_ChildBg].w = panel_opacity;
    style.Colors[ImGuiCol_PopupBg].w = panel_opacity;
    style.Colors[ImGuiCol_FrameBg].w = panel_opacity;
    style.Colors[ImGuiCol_FrameBgHovered].w = panel_opacity;
    style.Colors[ImGuiCol_FrameBgActive].w = panel_opacity;
    style.Colors[ImGuiCol_TitleBg].w *= window_opacity;
    style.Colors[ImGuiCol_TitleBgActive].w *= window_opacity;
    style.Colors[ImGuiCol_MenuBarBg].w *= panel_opacity;
    style.Colors[ImGuiCol_ScrollbarBg].w *= panel_opacity;
    style.Colors[ImGuiCol_CheckMark].w *= panel_opacity;
    style.Colors[ImGuiCol_SliderGrab].w *= panel_opacity;
    style.Colors[ImGuiCol_SliderGrabActive].w *= panel_opacity;
    style.Colors[ImGuiCol_Button].w *= panel_opacity;
    style.Colors[ImGuiCol_ButtonHovered].w *= panel_opacity;
    style.Colors[ImGuiCol_ButtonActive].w *= panel_opacity;
    style.Colors[ImGuiCol_Header].w *= panel_opacity;
    style.Colors[ImGuiCol_HeaderHovered].w *= panel_opacity;
    style.Colors[ImGuiCol_HeaderActive].w *= panel_opacity;
    style.Colors[ImGuiCol_Separator].w *= panel_opacity;
    style.Colors[ImGuiCol_SeparatorHovered].w *= panel_opacity;
    style.Colors[ImGuiCol_SeparatorActive].w *= panel_opacity;
    style.Colors[ImGuiCol_ResizeGrip].w *= panel_opacity;
    style.Colors[ImGuiCol_ResizeGripHovered].w *= panel_opacity;
    style.Colors[ImGuiCol_ResizeGripActive].w *= panel_opacity;
    style.Colors[ImGuiCol_Tab].w *= panel_opacity;
    style.Colors[ImGuiCol_TabHovered].w *= panel_opacity;
    style.Colors[ImGuiCol_TabActive].w *= panel_opacity;
    style.Colors[ImGuiCol_TabUnfocused].w *= panel_opacity;
    style.Colors[ImGuiCol_TabUnfocusedActive].w *= panel_opacity;
    style.Colors[ImGuiCol_TextSelectedBg].w *= panel_opacity;
    style.Colors[ImGuiCol_ModalWindowDimBg].w *= panel_opacity;
}

void AppearanceSettings::apply_font_settings() {
    // Get font settings
    int font_family_index = settings_manager_.get_enum("appearance.font_family", 0);
    int base_font_size = settings_manager_.get_int("appearance.font_size", 14);
    int mono_font_size = settings_manager_.get_int("appearance.mono_font_size", 12);
    float header_scale = settings_manager_.get_float("appearance.header_font_multiplier", 1.2f);

    // Determine font path based on selection
    std::string font_path;
    switch (font_family_index) {
        case 0: // Roboto
            font_path = "fonts/Roboto-Regular.ttf";
            break;
        case 1: // Segoe UI
            font_path = "fonts/segoeui.ttf";
            break;
        case 2: // Arial
            font_path = "fonts/arial.ttf";
            break;
        case 3: // Consolas
            font_path = "fonts/consolas.ttf";
            break;
        case 4: // Custom
        default:
            font_path = "fonts/custom.ttf"; // Default fallback
            break;
    }

    // In a real implementation, we would load the font file and configure ImGui
    // For now, we'll use ImGui's default font configuration with the selected size
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear(); // Clear existing fonts

    // Configure base font
    ImFontConfig config;
    config.SizePixels = static_cast<float>(base_font_size);

    // Add the selected font to ImGui
    // For now, we'll use the default font with the selected size
    io.Fonts->AddFontDefault(&config);

    // Configure header font (larger size)
    ImFontConfig header_config;
    header_config.SizePixels = static_cast<float>(base_font_size * header_scale);
    ImFont* header_font = io.Fonts->AddFontDefault(&header_config);

    // Configure monospace font for code/data display
    ImFontConfig mono_config;
    mono_config.SizePixels = static_cast<float>(mono_font_size);
    ImFont* mono_font = io.Fonts->AddFontDefault(&mono_config);

    // In a real implementation, we would load actual font files:
    /*
    // Attempt to load the selected font file
    if (std::filesystem::exists(font_path)) {
        io.Fonts->AddFontFromFileTTF(font_path.c_str(), static_cast<float>(base_font_size), &config);
    } else {
        // Fallback to default font if file not found
        io.Fonts->AddFontDefault(&config);
    }
    */

    // Associate fonts with ImGui context
    io.FontDefault = io.Fonts->Fonts[0];  // Base font

    // Mark fonts for rebuild
    io.Fonts->Build();
}

void AppearanceSettings::apply_border_settings() {
    // Get border settings
    int border_width = settings_manager_.get_int("appearance.border_width", 1);
    int corner_radius = settings_manager_.get_int("appearance.corner_radius", 4);
    int border_style = settings_manager_.get_enum("appearance.border_style", 0);  // 0 = Solid

    // Apply to ImGui style
    ImGuiStyle& style = ImGui::GetStyle();

    // Set border size based on style
    style.FrameBorderSize = (border_style == 4) ? 0.0f : static_cast<float>(border_width);  // None style means no border
    style.WindowBorderSize = 0.0f;  // Always no window borders
    style.ChildBorderSize = (border_style == 4) ? 0.0f : static_cast<float>(border_width);
    style.PopupBorderSize = (border_style == 4) ? 0.0f : static_cast<float>(border_width);

    // Set rounding based on style
    float rounding_value = (border_style == 3) ? static_cast<float>(corner_radius) : 0.0f;
    style.WindowRounding = rounding_value;
    style.ChildRounding = rounding_value;
    style.FrameRounding = rounding_value;
    style.GrabRounding = rounding_value;
    style.ScrollbarRounding = rounding_value;
    style.TabRounding = rounding_value;
    style.PopupRounding = rounding_value;

    // Panel spacing
    int panel_spacing = settings_manager_.get_int("appearance.panel_spacing", 4);
    style.ItemSpacing = ImVec2(static_cast<float>(panel_spacing), static_cast<float>(panel_spacing));
    style.ItemInnerSpacing = ImVec2(static_cast<float>(panel_spacing), static_cast<float>(panel_spacing));
    style.WindowPadding = ImVec2(static_cast<float>(panel_spacing * 2), static_cast<float>(panel_spacing * 2));

    // For dashed and dotted borders, we would need custom rendering
    // Since ImGui doesn't directly support dashed/dotted borders, we'll note this
    // In a real implementation, custom drawing functions would be needed for these styles
    if (border_style == 1 || border_style == 2) {  // Dashed or Dotted
        // These styles would require custom rendering code
        // For now, we'll just use solid borders but note the intended style
        // A full implementation would draw custom borders using ImDrawList
    }
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

void AppearanceSettings::add_custom_theme(const std::string& name, const SimpleColorTheme& theme_def) {
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