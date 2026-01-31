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
    current_colors_.background = ImVec4(0.04f, 0.04f, 0.04f, 1.0f);  // #0A0A0A
    current_colors_.text = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    current_colors_.text_dim = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
    current_colors_.accent_green = ImVec4(0.2f, 1.0f, 0.2f, 1.0f);  // Neon Green
    current_colors_.accent_red = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);    // Neon Red
    current_colors_.accent_cyan = ImVec4(0.0f, 1.0f, 1.0f, 1.0f);   // Cyan

    // Glass effect
    current_colors_.panel_bg =
        ImVec4(0.08f, 0.08f, 0.08f, 0.85f);  // Slightly opaque for readability
    current_colors_.border = ImVec4(0.2f, 0.2f, 0.2f, 0.5f);
    current_colors_.header_bg = ImVec4(0.1f, 0.1f, 0.1f, 0.9f);
  } else {
    // Light Clean (Placeholder)
    ImGui::StyleColorsLight();
    return;  // Built-in light style
  }

  updateImGuiStyle();
}

void ThemeManager::updateImGuiStyle() {
  ImGuiStyle& style = ImGui::GetStyle();

  // Modern rounding
  style.WindowRounding = 6.0f;
  style.FrameRounding = 4.0f;
  style.PopupRounding = 4.0f;
  style.ScrollbarRounding = 4.0f;
  style.GrabRounding = 4.0f;
  style.TabRounding = 4.0f;

  // Padding/Spacing
  style.WindowPadding = ImVec2(10, 10);
  style.FramePadding = ImVec2(5, 5);
  style.ItemSpacing = ImVec2(6, 6);

  // Colors
  const auto& c = current_colors_;

  style.Colors[ImGuiCol_Text] = c.text;
  style.Colors[ImGuiCol_TextDisabled] = c.text_dim;
  style.Colors[ImGuiCol_WindowBg] = c.panel_bg;
  style.Colors[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);  // Transparent to inherit
  style.Colors[ImGuiCol_PopupBg] = ImVec4(0.08f, 0.08f, 0.08f, 0.95f);
  style.Colors[ImGuiCol_Border] = c.border;
  style.Colors[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);

  style.Colors[ImGuiCol_FrameBg] = ImVec4(0.15f, 0.15f, 0.15f, 0.6f);
  style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.2f, 0.2f, 0.2f, 0.8f);
  style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.25f, 0.25f, 0.25f, 1.0f);

  style.Colors[ImGuiCol_TitleBg] = c.header_bg;
  style.Colors[ImGuiCol_TitleBgActive] = c.header_bg;
  style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.05f, 0.05f, 0.05f, 0.5f);

  style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.1f, 0.1f, 0.1f, 0.8f);

  style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.02f, 0.02f, 0.02f, 0.0f);
  style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.3f, 0.3f, 0.3f, 0.6f);
  style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.4f, 0.4f, 0.4f, 0.8f);
  style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);

  style.Colors[ImGuiCol_CheckMark] = c.accent_cyan;
  style.Colors[ImGuiCol_SliderGrab] = c.accent_cyan;
  style.Colors[ImGuiCol_SliderGrabActive] = c.accent_cyan;

  style.Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.2f, 0.2f, 0.4f);
  style.Colors[ImGuiCol_ButtonHovered] =
      ImVec4(0.2f, 0.2f, 0.2f, 0.7f);  // Hover grow effect logic can be external
  style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);

  style.Colors[ImGuiCol_Header] = ImVec4(0.2f, 0.2f, 0.2f, 0.4f);
  style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.25f, 0.25f, 0.7f);
  style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);

  style.Colors[ImGuiCol_Separator] = c.border;
  style.Colors[ImGuiCol_SeparatorHovered] = c.accent_cyan;
  style.Colors[ImGuiCol_SeparatorActive] = c.accent_cyan;

  style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.3f, 0.3f, 0.3f, 0.5f);
  style.Colors[ImGuiCol_ResizeGripHovered] = c.accent_cyan;
  style.Colors[ImGuiCol_ResizeGripActive] = c.accent_cyan;

  style.Colors[ImGuiCol_Tab] = ImVec4(0.15f, 0.15f, 0.15f, 0.6f);
  style.Colors[ImGuiCol_TabHovered] = ImVec4(0.25f, 0.25f, 0.25f, 0.8f);
  style.Colors[ImGuiCol_TabActive] = c.panel_bg;  // Matches window bg
  style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.1f, 0.1f, 0.1f, 0.6f);
  style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.15f, 0.15f, 0.15f, 0.8f);

  style.Colors[ImGuiCol_PlotLines] = c.accent_cyan;
  style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  style.Colors[ImGuiCol_PlotHistogram] = c.accent_cyan;
  style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

  style.Colors[ImGuiCol_TextSelectedBg] =
      ImVec4(c.accent_cyan.x, c.accent_cyan.y, c.accent_cyan.z, 0.35f);
  style.Colors[ImGuiCol_DragDropTarget] = c.accent_cyan;
  style.Colors[ImGuiCol_NavHighlight] = c.accent_cyan;
  style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.0f, 1.0f, 1.0f, 0.7f);
  style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.8f, 0.8f, 0.8f, 0.20f);
  style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.6f);
}

void ThemeManager::pushGlassStyle() {
  // Glassmorphism - usually handled by WindowBg alpha + Blur (done in shader or
  // composition) Here we can enforce the alpha for specific panels if needed
  // But since we updated global style, this might be redundant unless we want
  // VARIATIONS. For now, let's allow overlapping windows to blur background
  // ImGui doesn't support backdrop blur natively without backend shaders.
  // We will simulate the look with semi-transparent dark backgrounds.
  ImGui::PushStyleColor(ImGuiCol_WindowBg, current_colors_.panel_bg);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
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
  } else {
    current_theme_type_ = ThemeType::LightClean;
  }
}

void ThemeManager::sync_with_layout_manager() {
  // This method would sync the current theme with the layout manager
  // For now, it's a placeholder implementation
}

}  // namespace BTQuant
