#include "../include/ui/unified_theme_system.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "imgui.h"  // Include imgui.h for the actual implementation

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace BTQuant {
namespace UI {

UnifiedThemeManager::UnifiedThemeManager() {
  initialize_themes_directory();
  load_builtin_themes();
  current_theme_name_ = "Dark Professional";
}

UnifiedThemeManager::~UnifiedThemeManager() {
  // Cleanup if needed
}

void UnifiedThemeManager::initialize() {
  // Initialization logic if needed
}

void UnifiedThemeManager::initialize_themes_directory() {
  // Create themes directory if it doesn't exist
  themes_directory_ = "themes";
  std::filesystem::create_directories(themes_directory_);
}

void UnifiedThemeManager::load_builtin_themes() {
  // Dark Professional Theme
  ThemeDefinition dark_professional;
  dark_professional.name = "Dark Professional";
  dark_professional.description = "Professional dark theme for trading";
  dark_professional.is_dark_theme = true;

  // Set colors
  dark_professional.colors.background_primary[0] = 0.1f;  // R
  dark_professional.colors.background_primary[1] = 0.1f;  // G
  dark_professional.colors.background_primary[2] = 0.1f;  // B
  dark_professional.colors.background_primary[3] = 1.0f;  // A

  dark_professional.colors.background_secondary[0] = 0.15f;
  dark_professional.colors.background_secondary[1] = 0.15f;
  dark_professional.colors.background_secondary[2] = 0.15f;
  dark_professional.colors.background_secondary[3] = 1.0f;

  dark_professional.colors.background_panel[0] = 0.12f;
  dark_professional.colors.background_panel[1] = 0.12f;
  dark_professional.colors.background_panel[2] = 0.12f;
  dark_professional.colors.background_panel[3] = 1.0f;

  dark_professional.colors.text_primary[0] = 0.9f;
  dark_professional.colors.text_primary[1] = 0.9f;
  dark_professional.colors.text_primary[2] = 0.9f;
  dark_professional.colors.text_primary[3] = 1.0f;

  dark_professional.colors.text_secondary[0] = 0.7f;
  dark_professional.colors.text_secondary[1] = 0.7f;
  dark_professional.colors.text_secondary[2] = 0.7f;
  dark_professional.colors.text_secondary[3] = 1.0f;

  dark_professional.colors.text_muted[0] = 0.5f;
  dark_professional.colors.text_muted[1] = 0.5f;
  dark_professional.colors.text_muted[2] = 0.5f;
  dark_professional.colors.text_muted[3] = 1.0f;

  dark_professional.colors.price_up[0] = 0.0f;
  dark_professional.colors.price_up[1] = 0.8f;
  dark_professional.colors.price_up[2] = 0.0f;
  dark_professional.colors.price_up[3] = 1.0f;

  dark_professional.colors.price_down[0] = 0.8f;
  dark_professional.colors.price_down[1] = 0.0f;
  dark_professional.colors.price_down[2] = 0.0f;
  dark_professional.colors.price_down[3] = 1.0f;

  dark_professional.colors.price_neutral[0] = 0.6f;
  dark_professional.colors.price_neutral[1] = 0.6f;
  dark_professional.colors.price_neutral[2] = 0.6f;
  dark_professional.colors.price_neutral[3] = 1.0f;

  dark_professional.colors.accent_primary[0] = 0.2f;
  dark_professional.colors.accent_primary[1] = 0.6f;
  dark_professional.colors.accent_primary[2] = 1.0f;
  dark_professional.colors.accent_primary[3] = 1.0f;

  dark_professional.colors.accent_secondary[0] = 0.8f;
  dark_professional.colors.accent_secondary[1] = 0.4f;
  dark_professional.colors.accent_secondary[2] = 0.0f;
  dark_professional.colors.accent_secondary[3] = 1.0f;

  dark_professional.colors.border_color[0] = 0.3f;
  dark_professional.colors.border_color[1] = 0.3f;
  dark_professional.colors.border_color[2] = 0.3f;
  dark_professional.colors.border_color[3] = 1.0f;

  dark_professional.colors.status_connected[0] = 0.0f;
  dark_professional.colors.status_connected[1] = 0.8f;
  dark_professional.colors.status_connected[2] = 0.0f;
  dark_professional.colors.status_connected[3] = 1.0f;

  dark_professional.colors.status_disconnected[0] = 0.8f;
  dark_professional.colors.status_disconnected[1] = 0.0f;
  dark_professional.colors.status_disconnected[2] = 0.0f;
  dark_professional.colors.status_disconnected[3] = 1.0f;

  dark_professional.colors.status_warning[0] = 0.8f;
  dark_professional.colors.status_warning[1] = 0.8f;
  dark_professional.colors.status_warning[2] = 0.0f;
  dark_professional.colors.status_warning[3] = 1.0f;

  // Typography
  dark_professional.font_family = "Inter";
  dark_professional.font_size_normal = 14.0f;
  dark_professional.font_size_small = 12.0f;
  dark_professional.font_size_large = 18.0f;

  // Spacing
  dark_professional.padding_small = 4.0f;
  dark_professional.padding_medium = 8.0f;
  dark_professional.padding_large = 16.0f;

  // Border radius
  dark_professional.border_radius_small = 2.0f;
  dark_professional.border_radius_medium = 4.0f;
  dark_professional.border_radius_large = 8.0f;

  // Shadow properties
  dark_professional.shadow_normal.offset_x = 0.0f;
  dark_professional.shadow_normal.offset_y = 2.0f;
  dark_professional.shadow_normal.blur_radius = 4.0f;
  dark_professional.shadow_normal.spread = 0.0f;
  dark_professional.shadow_normal.color[0] = 0.0f;
  dark_professional.shadow_normal.color[1] = 0.0f;
  dark_professional.shadow_normal.color[2] = 0.0f;
  dark_professional.shadow_normal.color[3] = 0.2f;

  dark_professional.is_builtin = true;

  themes_[dark_professional.name] = dark_professional;

  // Light Professional Theme
  ThemeDefinition light_professional;
  light_professional.name = "Light Professional";
  light_professional.description = "Professional light theme for trading";
  light_professional.is_dark_theme = false;

  // Set colors
  light_professional.colors.background_primary[0] = 0.95f;
  light_professional.colors.background_primary[1] = 0.95f;
  light_professional.colors.background_primary[2] = 0.95f;
  light_professional.colors.background_primary[3] = 1.0f;

  light_professional.colors.background_secondary[0] = 0.9f;
  light_professional.colors.background_secondary[1] = 0.9f;
  light_professional.colors.background_secondary[2] = 0.9f;
  light_professional.colors.background_secondary[3] = 1.0f;

  light_professional.colors.background_panel[0] = 1.0f;
  light_professional.colors.background_panel[1] = 1.0f;
  light_professional.colors.background_panel[2] = 1.0f;
  light_professional.colors.background_panel[3] = 1.0f;

  light_professional.colors.text_primary[0] = 0.1f;
  light_professional.colors.text_primary[1] = 0.1f;
  light_professional.colors.text_primary[2] = 0.1f;
  light_professional.colors.text_primary[3] = 1.0f;

  light_professional.colors.text_secondary[0] = 0.3f;
  light_professional.colors.text_secondary[1] = 0.3f;
  light_professional.colors.text_secondary[2] = 0.3f;
  light_professional.colors.text_secondary[3] = 1.0f;

  light_professional.colors.text_muted[0] = 0.5f;
  light_professional.colors.text_muted[1] = 0.5f;
  light_professional.colors.text_muted[2] = 0.5f;
  light_professional.colors.text_muted[3] = 1.0f;

  light_professional.colors.price_up[0] = 0.0f;
  light_professional.colors.price_up[1] = 0.6f;
  light_professional.colors.price_up[2] = 0.0f;
  light_professional.colors.price_up[3] = 1.0f;

  light_professional.colors.price_down[0] = 0.8f;
  light_professional.colors.price_down[1] = 0.0f;
  light_professional.colors.price_down[2] = 0.0f;
  light_professional.colors.price_down[3] = 1.0f;

  light_professional.colors.price_neutral[0] = 0.4f;
  light_professional.colors.price_neutral[1] = 0.4f;
  light_professional.colors.price_neutral[2] = 0.4f;
  light_professional.colors.price_neutral[3] = 1.0f;

  light_professional.colors.accent_primary[0] = 0.0f;
  light_professional.colors.accent_primary[1] = 0.4f;
  light_professional.colors.accent_primary[2] = 0.8f;
  light_professional.colors.accent_primary[3] = 1.0f;

  light_professional.colors.accent_secondary[0] = 0.6f;
  light_professional.colors.accent_secondary[1] = 0.3f;
  light_professional.colors.accent_secondary[2] = 0.0f;
  light_professional.colors.accent_secondary[3] = 1.0f;

  light_professional.colors.border_color[0] = 0.7f;
  light_professional.colors.border_color[1] = 0.7f;
  light_professional.colors.border_color[2] = 0.7f;
  light_professional.colors.border_color[3] = 1.0f;

  light_professional.colors.status_connected[0] = 0.0f;
  light_professional.colors.status_connected[1] = 0.6f;
  light_professional.colors.status_connected[2] = 0.0f;
  light_professional.colors.status_connected[3] = 1.0f;

  light_professional.colors.status_disconnected[0] = 0.8f;
  light_professional.colors.status_disconnected[1] = 0.0f;
  light_professional.colors.status_disconnected[2] = 0.0f;
  light_professional.colors.status_disconnected[3] = 1.0f;

  light_professional.colors.status_warning[0] = 0.8f;
  light_professional.colors.status_warning[1] = 0.6f;
  light_professional.colors.status_warning[2] = 0.0f;
  light_professional.colors.status_warning[3] = 1.0f;

  // Typography
  light_professional.font_family = "Inter";
  light_professional.font_size_normal = 14.0f;
  light_professional.font_size_small = 12.0f;
  light_professional.font_size_large = 18.0f;

  // Spacing
  light_professional.padding_small = 4.0f;
  light_professional.padding_medium = 8.0f;
  light_professional.padding_large = 16.0f;

  // Border radius
  light_professional.border_radius_small = 2.0f;
  light_professional.border_radius_medium = 4.0f;
  light_professional.border_radius_large = 8.0f;

  // Shadow properties
  light_professional.shadow_normal.offset_x = 0.0f;
  light_professional.shadow_normal.offset_y = 2.0f;
  light_professional.shadow_normal.blur_radius = 4.0f;
  light_professional.shadow_normal.spread = 0.0f;
  light_professional.shadow_normal.color[0] = 0.0f;
  light_professional.shadow_normal.color[1] = 0.0f;
  light_professional.shadow_normal.color[2] = 0.0f;
  light_professional.shadow_normal.color[3] = 0.2f;

  light_professional.is_builtin = true;

  themes_[light_professional.name] = light_professional;

  // High Contrast Theme
  ThemeDefinition high_contrast;
  high_contrast.name = "High Contrast";
  high_contrast.description = "High contrast theme for accessibility";
  high_contrast.is_dark_theme = true;

  // Set colors
  high_contrast.colors.background_primary[0] = 0.0f;
  high_contrast.colors.background_primary[1] = 0.0f;
  high_contrast.colors.background_primary[2] = 0.0f;
  high_contrast.colors.background_primary[3] = 1.0f;

  high_contrast.colors.background_secondary[0] = 0.1f;
  high_contrast.colors.background_secondary[1] = 0.1f;
  high_contrast.colors.background_secondary[2] = 0.1f;
  high_contrast.colors.background_secondary[3] = 1.0f;

  high_contrast.colors.background_panel[0] = 0.05f;
  high_contrast.colors.background_panel[1] = 0.05f;
  high_contrast.colors.background_panel[2] = 0.05f;
  high_contrast.colors.background_panel[3] = 1.0f;

  high_contrast.colors.text_primary[0] = 1.0f;
  high_contrast.colors.text_primary[1] = 1.0f;
  high_contrast.colors.text_primary[2] = 1.0f;
  high_contrast.colors.text_primary[3] = 1.0f;

  high_contrast.colors.text_secondary[0] = 0.9f;
  high_contrast.colors.text_secondary[1] = 0.9f;
  high_contrast.colors.text_secondary[2] = 0.9f;
  high_contrast.colors.text_secondary[3] = 1.0f;

  high_contrast.colors.text_muted[0] = 0.7f;
  high_contrast.colors.text_muted[1] = 0.7f;
  high_contrast.colors.text_muted[2] = 0.7f;
  high_contrast.colors.text_muted[3] = 1.0f;

  high_contrast.colors.price_up[0] = 0.0f;
  high_contrast.colors.price_up[1] = 1.0f;
  high_contrast.colors.price_up[2] = 0.0f;
  high_contrast.colors.price_up[3] = 1.0f;

  high_contrast.colors.price_down[0] = 1.0f;
  high_contrast.colors.price_down[1] = 0.0f;
  high_contrast.colors.price_down[2] = 0.0f;
  high_contrast.colors.price_down[3] = 1.0f;

  high_contrast.colors.price_neutral[0] = 0.8f;
  high_contrast.colors.price_neutral[1] = 0.8f;
  high_contrast.colors.price_neutral[2] = 0.8f;
  high_contrast.colors.price_neutral[3] = 1.0f;

  high_contrast.colors.accent_primary[0] = 0.0f;
  high_contrast.colors.accent_primary[1] = 0.8f;
  high_contrast.colors.accent_primary[2] = 1.0f;
  high_contrast.colors.accent_primary[3] = 1.0f;

  high_contrast.colors.accent_secondary[0] = 1.0f;
  high_contrast.colors.accent_secondary[1] = 0.5f;
  high_contrast.colors.accent_secondary[2] = 0.0f;
  high_contrast.colors.accent_secondary[3] = 1.0f;

  high_contrast.colors.border_color[0] = 0.5f;
  high_contrast.colors.border_color[1] = 0.5f;
  high_contrast.colors.border_color[2] = 0.5f;
  high_contrast.colors.border_color[3] = 1.0f;

  high_contrast.colors.status_connected[0] = 0.0f;
  high_contrast.colors.status_connected[1] = 1.0f;
  high_contrast.colors.status_connected[2] = 0.0f;
  high_contrast.colors.status_connected[3] = 1.0f;

  high_contrast.colors.status_disconnected[0] = 1.0f;
  high_contrast.colors.status_disconnected[1] = 0.0f;
  high_contrast.colors.status_disconnected[2] = 0.0f;
  high_contrast.colors.status_disconnected[3] = 1.0f;

  high_contrast.colors.status_warning[0] = 1.0f;
  high_contrast.colors.status_warning[1] = 1.0f;
  high_contrast.colors.status_warning[2] = 0.0f;
  high_contrast.colors.status_warning[3] = 1.0f;

  // Typography
  high_contrast.font_family = "Inter";
  high_contrast.font_size_normal = 16.0f;  // Larger for accessibility
  high_contrast.font_size_small = 14.0f;
  high_contrast.font_size_large = 20.0f;

  // Spacing
  high_contrast.padding_small = 6.0f;
  high_contrast.padding_medium = 12.0f;
  high_contrast.padding_large = 20.0f;

  // Border radius
  high_contrast.border_radius_small = 3.0f;
  high_contrast.border_radius_medium = 6.0f;
  high_contrast.border_radius_large = 10.0f;

  // Shadow properties
  high_contrast.shadow_normal.offset_x = 0.0f;
  high_contrast.shadow_normal.offset_y = 3.0f;
  high_contrast.shadow_normal.blur_radius = 6.0f;
  high_contrast.shadow_normal.spread = 0.0f;
  high_contrast.shadow_normal.color[0] = 0.0f;
  high_contrast.shadow_normal.color[1] = 0.0f;
  high_contrast.shadow_normal.color[2] = 0.0f;
  high_contrast.shadow_normal.color[3] = 0.3f;

  high_contrast.is_builtin = true;

  themes_[high_contrast.name] = high_contrast;

  // Deep Void Theme
  ThemeDefinition deep_void;
  deep_void.name = "Deep Void";
  deep_void.description = "Deep space inspired dark theme with cosmic accents";
  deep_void.is_dark_theme = true;

  // Set colors - deep space inspired
  deep_void.colors.background_primary[0] = 0.05f;  // R - Very dark space black
  deep_void.colors.background_primary[1] = 0.05f;  // G
  deep_void.colors.background_primary[2] = 0.1f;   // B - With a hint of deep blue
  deep_void.colors.background_primary[3] = 1.0f;   // A

  deep_void.colors.background_secondary[0] = 0.08f;
  deep_void.colors.background_secondary[1] = 0.08f;
  deep_void.colors.background_secondary[2] = 0.15f;
  deep_void.colors.background_secondary[3] = 1.0f;

  deep_void.colors.background_panel[0] = 0.07f;
  deep_void.colors.background_panel[1] = 0.07f;
  deep_void.colors.background_panel[2] = 0.12f;
  deep_void.colors.background_panel[3] = 1.0f;

  deep_void.colors.text_primary[0] = 0.95f;    // R - Bright white for primary text
  deep_void.colors.text_primary[1] = 0.95f;    // G
  deep_void.colors.text_primary[2] = 0.98f;    // B - Slightly blue-white
  deep_void.colors.text_primary[3] = 1.0f;     // A

  deep_void.colors.text_secondary[0] = 0.75f;
  deep_void.colors.text_secondary[1] = 0.78f;
  deep_void.colors.text_secondary[2] = 0.85f;
  deep_void.colors.text_secondary[3] = 1.0f;

  deep_void.colors.text_muted[0] = 0.45f;
  deep_void.colors.text_muted[1] = 0.5f;
  deep_void.colors.text_muted[2] = 0.6f;
  deep_void.colors.text_muted[3] = 1.0f;

  // Cosmic green for positive prices
  deep_void.colors.price_up[0] = 0.2f;
  deep_void.colors.price_up[1] = 0.9f;
  deep_void.colors.price_up[2] = 0.7f;
  deep_void.colors.price_up[3] = 1.0f;

  // Nebula red for negative prices
  deep_void.colors.price_down[0] = 0.9f;
  deep_void.colors.price_down[1] = 0.3f;
  deep_void.colors.price_down[2] = 0.5f;
  deep_void.colors.price_down[3] = 1.0f;

  deep_void.colors.price_neutral[0] = 0.6f;
  deep_void.colors.price_neutral[1] = 0.65f;
  deep_void.colors.price_neutral[2] = 0.75f;
  deep_void.colors.price_neutral[3] = 1.0f;

  // Cosmic blue accent
  deep_void.colors.accent_primary[0] = 0.3f;
  deep_void.colors.accent_primary[1] = 0.6f;
  deep_void.colors.accent_primary[2] = 1.0f;
  deep_void.colors.accent_primary[3] = 1.0f;

  // Nebula orange accent
  deep_void.colors.accent_secondary[0] = 1.0f;
  deep_void.colors.accent_secondary[1] = 0.5f;
  deep_void.colors.accent_secondary[2] = 0.2f;
  deep_void.colors.accent_secondary[3] = 1.0f;

  deep_void.colors.border_color[0] = 0.25f;
  deep_void.colors.border_color[1] = 0.3f;
  deep_void.colors.border_color[2] = 0.4f;
  deep_void.colors.border_color[3] = 1.0f;

  deep_void.colors.status_connected[0] = 0.3f;
  deep_void.colors.status_connected[1] = 0.9f;
  deep_void.colors.status_connected[2] = 0.6f;
  deep_void.colors.status_connected[3] = 1.0f;

  deep_void.colors.status_disconnected[0] = 0.9f;
  deep_void.colors.status_disconnected[1] = 0.4f;
  deep_void.colors.status_disconnected[2] = 0.4f;
  deep_void.colors.status_disconnected[3] = 1.0f;

  deep_void.colors.status_warning[0] = 1.0f;
  deep_void.colors.status_warning[1] = 0.7f;
  deep_void.colors.status_warning[2] = 0.2f;
  deep_void.colors.status_warning[3] = 1.0f;

  // Typography
  deep_void.font_family = "Inter";
  deep_void.font_size_normal = 14.0f;
  deep_void.font_size_small = 12.0f;
  deep_void.font_size_large = 18.0f;

  // Spacing
  deep_void.padding_small = 4.0f;
  deep_void.padding_medium = 8.0f;
  deep_void.padding_large = 16.0f;

  // Border radius
  deep_void.border_radius_small = 2.0f;
  deep_void.border_radius_medium = 4.0f;
  deep_void.border_radius_large = 8.0f;

  // Shadow properties
  deep_void.shadow_normal.offset_x = 0.0f;
  deep_void.shadow_normal.offset_y = 2.0f;
  deep_void.shadow_normal.blur_radius = 4.0f;
  deep_void.shadow_normal.spread = 0.0f;
  deep_void.shadow_normal.color[0] = 0.0f;
  deep_void.shadow_normal.color[1] = 0.0f;
  deep_void.shadow_normal.color[2] = 0.0f;
  deep_void.shadow_normal.color[3] = 0.3f;

  deep_void.is_builtin = true;

  themes_[deep_void.name] = deep_void;
}

bool UnifiedThemeManager::register_theme(const ThemeDefinition& theme) {
  if (theme.name.empty()) {
    return false;
  }

  themes_[theme.name] = theme;
  return true;
}

bool UnifiedThemeManager::set_current_theme(const std::string& theme_name) {
  auto it = themes_.find(theme_name);
  if (it != themes_.end()) {
    current_theme_name_ = theme_name;
    return true;
  }
  return false;
}

const ThemeDefinition& UnifiedThemeManager::get_current_theme() const {
  static ThemeDefinition empty_theme;
  auto it = themes_.find(current_theme_name_);
  if (it != themes_.end()) {
    return it->second;
  }
  return empty_theme;
}

std::vector<std::string> UnifiedThemeManager::get_available_themes() const {
  std::vector<std::string> theme_names;
  for (const auto& pair : themes_) {
    theme_names.push_back(pair.first);
  }
  return theme_names;
}

const ThemeDefinition* UnifiedThemeManager::get_theme(const std::string& theme_name) const {
  auto it = themes_.find(theme_name);
  if (it != themes_.end()) {
    return &(it->second);
  }
  return nullptr;
}

void UnifiedThemeManager::apply_to_imgui() const {
  auto it = themes_.find(current_theme_name_);
  if (it == themes_.end()) {
    return;
  }

  const auto& theme = it->second;
  auto& style = ImGui::GetStyle();

  // Apply colors
  style.Colors[ImGuiCol_Text] = ImVec4(theme.colors.text_primary[0], theme.colors.text_primary[1],
                                       theme.colors.text_primary[2], theme.colors.text_primary[3]);
  style.Colors[ImGuiCol_TextDisabled] =
      ImVec4(theme.colors.text_muted[0], theme.colors.text_muted[1], theme.colors.text_muted[2],
             theme.colors.text_muted[3]);
  style.Colors[ImGuiCol_WindowBg] =
      ImVec4(theme.colors.background_panel[0], theme.colors.background_panel[1],
             theme.colors.background_panel[2], theme.colors.background_panel[3]);
  style.Colors[ImGuiCol_ChildBg] =
      ImVec4(theme.colors.background_secondary[0], theme.colors.background_secondary[1],
             theme.colors.background_secondary[2], theme.colors.background_secondary[3]);
  style.Colors[ImGuiCol_PopupBg] =
      ImVec4(theme.colors.background_primary[0], theme.colors.background_primary[1],
             theme.colors.background_primary[2], theme.colors.background_primary[3]);
  style.Colors[ImGuiCol_Border] =
      ImVec4(theme.colors.border_color[0], theme.colors.border_color[1],
             theme.colors.border_color[2], theme.colors.border_color[3]);
  style.Colors[ImGuiCol_FrameBg] =
      ImVec4(theme.colors.background_secondary[0], theme.colors.background_secondary[1],
             theme.colors.background_secondary[2], theme.colors.background_secondary[3] * 0.7f);
  style.Colors[ImGuiCol_TitleBg] =
      ImVec4(theme.colors.background_primary[0], theme.colors.background_primary[1],
             theme.colors.background_primary[2], theme.colors.background_primary[3]);
  style.Colors[ImGuiCol_TitleBgActive] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3]);
  style.Colors[ImGuiCol_MenuBarBg] =
      ImVec4(theme.colors.background_secondary[0], theme.colors.background_secondary[1],
             theme.colors.background_secondary[2], theme.colors.background_secondary[3]);
  style.Colors[ImGuiCol_ScrollbarBg] =
      ImVec4(theme.colors.background_secondary[0], theme.colors.background_secondary[1],
             theme.colors.background_secondary[2], theme.colors.background_secondary[3] * 0.2f);
  style.Colors[ImGuiCol_ScrollbarGrab] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.6f);
  style.Colors[ImGuiCol_ScrollbarGrabHovered] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.8f);
  style.Colors[ImGuiCol_ScrollbarGrabActive] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3]);
  style.Colors[ImGuiCol_CheckMark] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3]);
  style.Colors[ImGuiCol_SliderGrab] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3]);
  style.Colors[ImGuiCol_SliderGrabActive] =
      ImVec4(theme.colors.accent_secondary[0], theme.colors.accent_secondary[1],
             theme.colors.accent_secondary[2], theme.colors.accent_secondary[3]);
  style.Colors[ImGuiCol_Button] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.4f);
  style.Colors[ImGuiCol_ButtonHovered] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.6f);
  style.Colors[ImGuiCol_ButtonActive] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.8f);
  style.Colors[ImGuiCol_Header] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.4f);
  style.Colors[ImGuiCol_HeaderHovered] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.6f);
  style.Colors[ImGuiCol_HeaderActive] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.8f);
  style.Colors[ImGuiCol_Separator] =
      ImVec4(theme.colors.border_color[0], theme.colors.border_color[1],
             theme.colors.border_color[2], theme.colors.border_color[3]);
  style.Colors[ImGuiCol_SeparatorHovered] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3]);
  style.Colors[ImGuiCol_SeparatorActive] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3]);
  style.Colors[ImGuiCol_ResizeGrip] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.4f);
  style.Colors[ImGuiCol_ResizeGripHovered] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.6f);
  style.Colors[ImGuiCol_ResizeGripActive] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.8f);
  style.Colors[ImGuiCol_Tab] =
      ImVec4(theme.colors.background_secondary[0], theme.colors.background_secondary[1],
             theme.colors.background_secondary[2], theme.colors.background_secondary[3] * 0.8f);
  style.Colors[ImGuiCol_TabHovered] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.6f);
  style.Colors[ImGuiCol_TabActive] =
      ImVec4(theme.colors.background_panel[0], theme.colors.background_panel[1],
             theme.colors.background_panel[2], theme.colors.background_panel[3]);
  style.Colors[ImGuiCol_PlotLines] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3]);
  style.Colors[ImGuiCol_PlotHistogram] =
      ImVec4(theme.colors.accent_secondary[0], theme.colors.accent_secondary[1],
             theme.colors.accent_secondary[2], theme.colors.accent_secondary[3]);
  style.Colors[ImGuiCol_TextSelectedBg] =
      ImVec4(theme.colors.accent_primary[0], theme.colors.accent_primary[1],
             theme.colors.accent_primary[2], theme.colors.accent_primary[3] * 0.3f);

  // Apply spacing and rounding
  style.WindowPadding = ImVec2(theme.padding_medium, theme.padding_medium);
  style.FramePadding = ImVec2(theme.padding_small, theme.padding_small);
  style.ItemSpacing = ImVec2(theme.padding_medium, theme.padding_small);
  style.TouchExtraPadding = ImVec2(0.0f, 0.0f);
  style.IndentSpacing = theme.padding_large;
  style.ScrollbarSize = 14.0f;
  style.GrabMinSize = 10.0f;

  style.WindowBorderSize = 1.0f;
  style.ChildBorderSize = 1.0f;
  style.PopupBorderSize = 1.0f;
  style.FrameBorderSize = 1.0f;
  style.TabBorderSize = 1.0f;

  style.WindowRounding = theme.border_radius_medium;
  style.ChildRounding = theme.border_radius_small;
  style.FrameRounding = theme.border_radius_small;
  style.PopupRounding = theme.border_radius_medium;
  style.ScrollbarRounding = theme.border_radius_small;
  style.GrabRounding = theme.border_radius_small;
  style.TabRounding = theme.border_radius_small;

// Apply shadow effects (would need custom rendering for full shadow support)
// For now, just set the shadow color as a reference
#ifdef ImGuiCol_DockingPreview
  style.Colors[ImGuiCol_DockingPreview] =
      ImVec4(theme.shadow_normal.color[0], theme.shadow_normal.color[1],
             theme.shadow_normal.color[2], theme.shadow_normal.color[3]);
#endif
}

void UnifiedThemeManager::apply_to_components() const {
  // Apply theme to custom components
  // This would typically involve notifying registered components about theme changes
  // For now, just a placeholder implementation
}

void UnifiedThemeManager::get_color(const std::string& color_name, float* rgba) const {
  auto it = themes_.find(current_theme_name_);
  if (it == themes_.end()) {
    // Return default color if theme not found
    rgba[0] = 0.5f;
    rgba[1] = 0.5f;
    rgba[2] = 0.5f;
    rgba[3] = 1.0f;
    return;
  }

  const auto& colors = it->second.colors;

  if (color_name == "background_primary") {
    rgba[0] = colors.background_primary[0];
    rgba[1] = colors.background_primary[1];
    rgba[2] = colors.background_primary[2];
    rgba[3] = colors.background_primary[3];
  } else if (color_name == "background_secondary") {
    rgba[0] = colors.background_secondary[0];
    rgba[1] = colors.background_secondary[1];
    rgba[2] = colors.background_secondary[2];
    rgba[3] = colors.background_secondary[3];
  } else if (color_name == "background_panel") {
    rgba[0] = colors.background_panel[0];
    rgba[1] = colors.background_panel[1];
    rgba[2] = colors.background_panel[2];
    rgba[3] = colors.background_panel[3];
  } else if (color_name == "text_primary") {
    rgba[0] = colors.text_primary[0];
    rgba[1] = colors.text_primary[1];
    rgba[2] = colors.text_primary[2];
    rgba[3] = colors.text_primary[3];
  } else if (color_name == "text_secondary") {
    rgba[0] = colors.text_secondary[0];
    rgba[1] = colors.text_secondary[1];
    rgba[2] = colors.text_secondary[2];
    rgba[3] = colors.text_secondary[3];
  } else if (color_name == "text_muted") {
    rgba[0] = colors.text_muted[0];
    rgba[1] = colors.text_muted[1];
    rgba[2] = colors.text_muted[2];
    rgba[3] = colors.text_muted[3];
  } else if (color_name == "price_up") {
    rgba[0] = colors.price_up[0];
    rgba[1] = colors.price_up[1];
    rgba[2] = colors.price_up[2];
    rgba[3] = colors.price_up[3];
  } else if (color_name == "price_down") {
    rgba[0] = colors.price_down[0];
    rgba[1] = colors.price_down[1];
    rgba[2] = colors.price_down[2];
    rgba[3] = colors.price_down[3];
  } else if (color_name == "price_neutral") {
    rgba[0] = colors.price_neutral[0];
    rgba[1] = colors.price_neutral[1];
    rgba[2] = colors.price_neutral[2];
    rgba[3] = colors.price_neutral[3];
  } else if (color_name == "accent_primary") {
    rgba[0] = colors.accent_primary[0];
    rgba[1] = colors.accent_primary[1];
    rgba[2] = colors.accent_primary[2];
    rgba[3] = colors.accent_primary[3];
  } else if (color_name == "accent_secondary") {
    rgba[0] = colors.accent_secondary[0];
    rgba[1] = colors.accent_secondary[1];
    rgba[2] = colors.accent_secondary[2];
    rgba[3] = colors.accent_secondary[3];
  } else if (color_name == "border_color") {
    rgba[0] = colors.border_color[0];
    rgba[1] = colors.border_color[1];
    rgba[2] = colors.border_color[2];
    rgba[3] = colors.border_color[3];
  } else if (color_name == "status_connected") {
    rgba[0] = colors.status_connected[0];
    rgba[1] = colors.status_connected[1];
    rgba[2] = colors.status_connected[2];
    rgba[3] = colors.status_connected[3];
  } else if (color_name == "status_disconnected") {
    rgba[0] = colors.status_disconnected[0];
    rgba[1] = colors.status_disconnected[1];
    rgba[2] = colors.status_disconnected[2];
    rgba[3] = colors.status_disconnected[3];
  } else if (color_name == "status_warning") {
    rgba[0] = colors.status_warning[0];
    rgba[1] = colors.status_warning[1];
    rgba[2] = colors.status_warning[2];
    rgba[3] = colors.status_warning[3];
  } else {
    // Return default color if color name not found
    rgba[0] = 0.5f;
    rgba[1] = 0.5f;
    rgba[2] = 0.5f;
    rgba[3] = 1.0f;
  }
}

bool UnifiedThemeManager::save_theme(const std::string& theme_name,
                                     const std::string& file_path) const {
  auto it = themes_.find(theme_name);
  if (it == themes_.end()) {
    return false;
  }

  return save_theme_to_file(it->second, file_path);
}

bool UnifiedThemeManager::load_theme(const std::string& file_path) {
  auto theme = load_theme_from_file(file_path);
  if (theme.name.empty()) {
    return false;
  }

  themes_[theme.name] = theme;
  return true;
}

void UnifiedThemeManager::create_preview(const std::string& theme_name,
                                         const std::string& output_path) const {
  // This would create a visual preview of the theme
  // For now, just a placeholder implementation
}

bool UnifiedThemeManager::validate_theme(const ThemeDefinition& theme) const {
  // Basic validation
  return !theme.name.empty();
}

std::vector<std::string> UnifiedThemeManager::get_theme_categories() const {
  std::vector<std::string> categories = {"Trading", "Analysis", "Monitoring", "Accessibility"};
  return categories;
}

std::vector<std::string> UnifiedThemeManager::get_themes_by_category(
    const std::string& category) const {
  std::vector<std::string> theme_names;
  for (const auto& pair : themes_) {
    // For now, just return all themes
    // In a real implementation, themes would have categories
    theme_names.push_back(pair.first);
  }
  return theme_names;
}

#ifdef HAS_NLOHMANN_JSON
ThemeDefinition UnifiedThemeManager::load_theme_from_file(const std::string& file_path) const {
  ThemeDefinition theme;
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return theme;
  }

  try {
    nlohmann::json j;
    file >> j;

    theme.name = j.value("name", "");
    theme.description = j.value("description", "");
    theme.is_dark_theme = j.value("is_dark_theme", false);

    // Load colors
    if (j.contains("colors")) {
      auto colors = j["colors"];
      if (colors.contains("background_primary")) {
        auto bg_primary = colors["background_primary"];
        for (int i = 0; i < 4 && i < bg_primary.size(); ++i) {
          theme.colors.background_primary[i] = bg_primary[i].get<float>();
        }
      }
      // Load other colors similarly...
    }

    // Load typography
    theme.font_family = j.value("font_family", "Inter");
    theme.font_size_normal = j.value("font_size_normal", 14.0f);
    theme.font_size_small = j.value("font_size_small", 12.0f);
    theme.font_size_large = j.value("font_size_large", 18.0f);

    // Load spacing
    theme.padding_small = j.value("padding_small", 4.0f);
    theme.padding_medium = j.value("padding_medium", 8.0f);
    theme.padding_large = j.value("padding_large", 16.0f);

    // Load border radius
    theme.border_radius_small = j.value("border_radius_small", 2.0f);
    theme.border_radius_medium = j.value("border_radius_medium", 4.0f);
    theme.border_radius_large = j.value("border_radius_large", 8.0f);

    theme.is_builtin = false;
  } catch (const std::exception& e) {
    std::cerr << "Error loading theme " << file_path << ": " << e.what() << std::endl;
  }

  file.close();
  return theme;
}

bool UnifiedThemeManager::save_theme_to_file(const ThemeDefinition& theme,
                                             const std::string& file_path) const {
  nlohmann::json j;

  j["name"] = theme.name;
  j["description"] = theme.description;
  j["is_dark_theme"] = theme.is_dark_theme;

  // Save colors
  nlohmann::json colors;
  colors["background_primary"] = nlohmann::json::array(
      {theme.colors.background_primary[0], theme.colors.background_primary[1],
       theme.colors.background_primary[2], theme.colors.background_primary[3]});
  colors["background_secondary"] = nlohmann::json::array(
      {theme.colors.background_secondary[0], theme.colors.background_secondary[1],
       theme.colors.background_secondary[2], theme.colors.background_secondary[3]});
  colors["background_panel"] =
      nlohmann::json::array({theme.colors.background_panel[0], theme.colors.background_panel[1],
                             theme.colors.background_panel[2], theme.colors.background_panel[3]});
  colors["text_primary"] =
      nlohmann::json::array({theme.colors.text_primary[0], theme.colors.text_primary[1],
                             theme.colors.text_primary[2], theme.colors.text_primary[3]});
  colors["text_secondary"] =
      nlohmann::json::array({theme.colors.text_secondary[0], theme.colors.text_secondary[1],
                             theme.colors.text_secondary[2], theme.colors.text_secondary[3]});
  colors["text_muted"] =
      nlohmann::json::array({theme.colors.text_muted[0], theme.colors.text_muted[1],
                             theme.colors.text_muted[2], theme.colors.text_muted[3]});
  colors["price_up"] = nlohmann::json::array({theme.colors.price_up[0], theme.colors.price_up[1],
                                              theme.colors.price_up[2], theme.colors.price_up[3]});
  colors["price_down"] =
      nlohmann::json::array({theme.colors.price_down[0], theme.colors.price_down[1],
                             theme.colors.price_down[2], theme.colors.price_down[3]});
  colors["price_neutral"] =
      nlohmann::json::array({theme.colors.price_neutral[0], theme.colors.price_neutral[1],
                             theme.colors.price_neutral[2], theme.colors.price_neutral[3]});
  colors["accent_primary"] =
      nlohmann::json::array({theme.colors.accent_primary[0], theme.colors.accent_primary[1],
                             theme.colors.accent_primary[2], theme.colors.accent_primary[3]});
  colors["accent_secondary"] =
      nlohmann::json::array({theme.colors.accent_secondary[0], theme.colors.accent_secondary[1],
                             theme.colors.accent_secondary[2], theme.colors.accent_secondary[3]});
  colors["border_color"] =
      nlohmann::json::array({theme.colors.border_color[0], theme.colors.border_color[1],
                             theme.colors.border_color[2], theme.colors.border_color[3]});
  colors["status_connected"] =
      nlohmann::json::array({theme.colors.status_connected[0], theme.colors.status_connected[1],
                             theme.colors.status_connected[2], theme.colors.status_connected[3]});
  colors["status_disconnected"] = nlohmann::json::array(
      {theme.colors.status_disconnected[0], theme.colors.status_disconnected[1],
       theme.colors.status_disconnected[2], theme.colors.status_disconnected[3]});
  colors["status_warning"] =
      nlohmann::json::array({theme.colors.status_warning[0], theme.colors.status_warning[1],
                             theme.colors.status_warning[2], theme.colors.status_warning[3]});
  j["colors"] = colors;

  // Save typography
  j["font_family"] = theme.font_family;
  j["font_size_normal"] = theme.font_size_normal;
  j["font_size_small"] = theme.font_size_small;
  j["font_size_large"] = theme.font_size_large;

  // Save spacing
  j["padding_small"] = theme.padding_small;
  j["padding_medium"] = theme.padding_medium;
  j["padding_large"] = theme.padding_large;

  // Save border radius
  j["border_radius_small"] = theme.border_radius_small;
  j["border_radius_medium"] = theme.border_radius_medium;
  j["border_radius_large"] = theme.border_radius_large;

  std::ofstream file(file_path);
  if (file.is_open()) {
    file << j.dump(4);
    file.close();
    return true;
  }

  return false;
}
#else
// Fallback implementations when nlohmann/json is not available
ThemeDefinition UnifiedThemeManager::load_theme_from_file(const std::string& file_path) const {
  ThemeDefinition theme;
  // Return a default theme when JSON parsing is not available
  theme.name = "Default";
  theme.description = "Default theme when JSON support is not available";
  theme.is_dark_theme = true;
  theme.is_builtin = true;
  return theme;
}

bool UnifiedThemeManager::save_theme_to_file(const ThemeDefinition& theme,
                                             const std::string& file_path) const {
  // Return false when JSON serialization is not available
  return false;
}
#endif

std::string UnifiedThemeManager::get_theme_file_path(const std::string& theme_name) const {
  return themes_directory_ + "/" + theme_name + ".json";
}

}  // namespace UI
}  // namespace BTQuant