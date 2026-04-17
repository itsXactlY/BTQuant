#include "../../include/components/theme_manager.hpp"

#include <iostream>

#include "../../include/ui/unified_theme_system.hpp"

namespace BTQuant {

void ThemeManager::initialize() {
  loadFonts();
  applyTheme(ThemeType::DarkNeon);
}

void ThemeManager::loadFonts() {
  // Ideally, we would load "Inter" font here.
  // For now, we rely on ImGui's default font or current setup.
  // If the IO has fonts loaded, we pick them.
  ImGuiIO& io = ImGui::GetIO();
  if (!io.Fonts->Fonts.empty()) {
    main_font_ = io.Fonts->Fonts[0];
    // If there's a second font loaded, assume it's large, otherwise reuse main
    large_font_ = (io.Fonts->Fonts.size() > 1) ? io.Fonts->Fonts[1] : main_font_;
  }
}

void ThemeManager::applyTheme(ThemeType type) {
  current_theme_type_ = type;

  if (type == ThemeType::DarkNeon) {
    // MMT Deep Void - dark, clean, professional Quantower-style palette
    current_colors_.background = ImVec4(0.043f, 0.055f, 0.067f, 1.0f);  // #0B0E11
    current_colors_.text = ImVec4(0.82f, 0.84f, 0.86f, 1.0f);           // #D1D7DE
    current_colors_.text_dim = ImVec4(0.45f, 0.48f, 0.52f, 1.0f);
    current_colors_.accent_green = ImVec4(0.0f, 0.9f, 0.4f, 1.0f);      // #00E566 - Neon Mint
    current_colors_.accent_red = ImVec4(0.9f, 0.1f, 0.15f, 1.0f);       // #E61926 - Crimson
    current_colors_.accent_cyan = ImVec4(0.0f, 0.9f, 0.4f, 1.0f);       // #00E566 - Neon Mint
    current_colors_.accent_magenta = ImVec4(0.9f, 0.1f, 0.15f, 1.0f);   // #E61926 - Crimson

    // Panel backgrounds - uniform deep void
    current_colors_.panel_bg = ImVec4(0.082f, 0.098f, 0.118f, 1.0f);    // #15191E
    current_colors_.border = ImVec4(0.165f, 0.180f, 0.196f, 1.0f);      // #2A2E33
    current_colors_.header_bg = ImVec4(0.082f, 0.098f, 0.118f, 1.0f);   // #15191E

    // Chart-specific
    current_colors_.chart_grid = ImVec4(0.10f, 0.12f, 0.14f, 0.4f);     // #1A1E24
    current_colors_.candle_up = ImVec4(0.0f, 0.9f, 0.4f, 1.0f);         // #00E566
    current_colors_.candle_down = ImVec4(0.9f, 0.1f, 0.15f, 1.0f);      // #E61926
  } else {
    // Light Clean (Placeholder)
    ImGui::StyleColorsLight();
    return;  // Built-in light style
  }

  updateImGuiStyle();
}

void ThemeManager::updateImGuiStyle() {
  ImGuiStyle& style = ImGui::GetStyle();

  // MMT Deep Void - zero rounding, clean edges, Quantower look
  style.WindowRounding = 0.0f;
  style.FrameRounding = 0.0f;
  style.PopupRounding = 0.0f;
  style.ScrollbarRounding = 0.0f;
  style.GrabRounding = 0.0f;
  style.TabRounding = 0.0f;

  // No borders
  style.WindowBorderSize = 0.0f;
  style.FrameBorderSize = 0.0f;

  // Compact padding/spacing
  style.WindowPadding = ImVec2(4, 4);
  style.FramePadding = ImVec2(4, 3);
  style.ItemSpacing = ImVec2(4, 3);
  style.ItemInnerSpacing = ImVec2(4, 2);

  // Thin scrollbars
  style.ScrollbarSize = 8.0f;
  style.GrabMinSize = 6.0f;

  // Colors - MMT Deep Void palette
  const auto& c = current_colors_;

  style.Colors[ImGuiCol_Text] = c.text;
  style.Colors[ImGuiCol_TextDisabled] = c.text_dim;
  style.Colors[ImGuiCol_WindowBg] = c.panel_bg;
  style.Colors[ImGuiCol_ChildBg] = ImVec4(0.043f, 0.055f, 0.067f, 0.0f);  // Transparent child bg
  style.Colors[ImGuiCol_PopupBg] = ImVec4(0.082f, 0.098f, 0.118f, 0.98f);
  style.Colors[ImGuiCol_Border] = c.border;
  style.Colors[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);

  // Frame background
  style.Colors[ImGuiCol_FrameBg] = ImVec4(0.06f, 0.07f, 0.09f, 0.9f);
  style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.10f, 0.12f, 0.14f, 0.95f);
  style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.14f, 0.16f, 0.18f, 1.0f);

  // Title bars - match panel bg, no accent bleed
  style.Colors[ImGuiCol_TitleBg] = ImVec4(0.06f, 0.07f, 0.09f, 1.0f);
  style.Colors[ImGuiCol_TitleBgActive] = c.header_bg;
  style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.06f, 0.07f, 0.09f, 0.75f);

  style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.07f, 0.08f, 0.10f, 0.9f);

  // Scrollbar - thin and dark
  style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.043f, 0.055f, 0.067f, 0.4f);
  style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.16f, 0.18f, 0.20f, 0.7f);
  style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.20f, 0.22f, 0.24f, 0.8f);
  style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.25f, 0.27f, 0.29f, 1.0f);

  // Interactive elements - use accent (Neon Mint)
  style.Colors[ImGuiCol_CheckMark] = c.accent_green;
  style.Colors[ImGuiCol_SliderGrab] = c.accent_green;
  style.Colors[ImGuiCol_SliderGrabActive] = c.accent_green;

  // Buttons
  style.Colors[ImGuiCol_Button] = ImVec4(0.10f, 0.12f, 0.14f, 0.7f);
  style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.14f, 0.16f, 0.18f, 0.85f);
  style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.18f, 0.20f, 0.22f, 1.0f);

  // Headers
  style.Colors[ImGuiCol_Header] = ImVec4(0.10f, 0.12f, 0.14f, 0.7f);
  style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.14f, 0.16f, 0.18f, 0.85f);
  style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.18f, 0.20f, 0.22f, 0.95f);

  // Separators
  style.Colors[ImGuiCol_Separator] = c.border;
  style.Colors[ImGuiCol_SeparatorHovered] = c.accent_green;
  style.Colors[ImGuiCol_SeparatorActive] = c.accent_green;

  // Resize grips
  style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.10f, 0.12f, 0.14f, 0.5f);
  style.Colors[ImGuiCol_ResizeGripHovered] = c.accent_green;
  style.Colors[ImGuiCol_ResizeGripActive] = c.accent_green;

  // Tabs
  style.Colors[ImGuiCol_Tab] = ImVec4(0.06f, 0.07f, 0.09f, 0.85f);
  style.Colors[ImGuiCol_TabHovered] = ImVec4(0.10f, 0.12f, 0.14f, 0.95f);
  style.Colors[ImGuiCol_TabActive] = c.panel_bg;
  style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.06f, 0.07f, 0.09f, 0.8f);
  style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.08f, 0.09f, 0.11f, 0.9f);

  // Plot colors - Neon Mint for lines, Crimson for hover
  style.Colors[ImGuiCol_PlotLines] = c.accent_green;
  style.Colors[ImGuiCol_PlotLinesHovered] = c.accent_red;
  style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.165f, 0.227f, 0.294f, 0.6f);  // Volume #2A3A4A
  style.Colors[ImGuiCol_PlotHistogramHovered] = c.accent_green;

  // Selection / Nav
  style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.0f, 0.9f, 0.4f, 0.25f);
  style.Colors[ImGuiCol_DragDropTarget] = c.accent_green;
  style.Colors[ImGuiCol_NavHighlight] = c.accent_green;
  style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.0f, 0.9f, 0.4f, 0.5f);
  style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.043f, 0.055f, 0.067f, 0.6f);
  style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.043f, 0.055f, 0.067f, 0.6f);
}

void ThemeManager::pushGlassStyle() {
  // MMT Deep Void - no border, dark panel background
  ImGui::PushStyleColor(ImGuiCol_WindowBg, current_colors_.panel_bg);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
}

void ThemeManager::popGlassStyle() {
  ImGui::PopStyleVar(1);
  ImGui::PopStyleColor(1);
}

void ThemeManager::toggleTheme() {
  if (current_theme_type_ == ThemeType::DarkNeon) {
    applyTheme(ThemeType::LightClean);
  } else {
    applyTheme(ThemeType::DarkNeon);
  }
}

// Integration with unified theme system
void ThemeManager::apply_unified_theme(const std::string& theme_name) {
  // Get the unified theme manager instance
  auto& unified_manager = UI::UnifiedThemeManager::getInstance();

  // Set the theme in the unified system
  unified_manager.set_current_theme(theme_name);

  // Apply the theme to ImGui
  unified_manager.apply_to_imgui();

  // Update our internal theme type based on the unified theme
  auto theme = unified_manager.get_theme(theme_name);
  if (theme && theme->is_dark_theme) {
    current_theme_type_ = ThemeType::DarkNeon;
    // Apply our enhanced dark theme to ensure consistency across all panels
    applyTheme(ThemeType::DarkNeon);
  } else {
    current_theme_type_ = ThemeType::LightClean;
    applyTheme(ThemeType::LightClean);
  }
}

void ThemeManager::sync_with_layout_manager() {
  // This method would sync the current theme with the layout manager
  // For now, it's a placeholder implementation
}

}  // namespace BTQuant
