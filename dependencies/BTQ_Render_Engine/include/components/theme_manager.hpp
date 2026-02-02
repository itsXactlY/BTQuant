#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "imgui.h"

namespace BTQuant {

struct ThemeColors {
  ImVec4 background = ImVec4(0.06f, 0.06f, 0.06f, 1.0f);  // #0F0F0F - Dark background
  ImVec4 text = ImVec4(0.95f, 0.95f, 0.95f, 1.0f);        // Bright white text for good contrast
  ImVec4 text_dim = ImVec4(0.65f, 0.65f, 0.65f, 1.0f);    // Muted text with sufficient contrast

  // Accents
  ImVec4 accent_green = ImVec4(0.3f, 0.9f, 0.3f, 1.0f);   // Improved green for better contrast
  ImVec4 accent_red = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);     // Improved red for better contrast
  ImVec4 accent_cyan = ImVec4(0.2f, 0.8f, 0.9f, 1.0f);    // Improved cyan for better contrast

  // UI Elements
  ImVec4 panel_bg = ImVec4(0.11f, 0.11f, 0.11f, 0.9f);   // Dark panel background with good contrast
  ImVec4 border = ImVec4(0.25f, 0.25f, 0.25f, 0.6f);      // Borders with sufficient contrast
  ImVec4 header_bg = ImVec4(0.15f, 0.15f, 0.15f, 0.95f);  // Header background with good contrast

  // Chart specific
  ImVec4 chart_grid = ImVec4(0.25f, 0.25f, 0.25f, 0.3f);  // Improved grid contrast
  ImVec4 candle_up = ImVec4(0.3f, 0.9f, 0.3f, 1.0f);      // Consistent with accent green
  ImVec4 candle_down = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);    // Consistent with accent red
};

enum class ThemeType {
  DarkNeon,
  LightClean  // Future proofing
};

class ThemeManager {
 public:
  static ThemeManager& getInstance() {
    static ThemeManager instance;
    return instance;
  }

  ThemeManager(const ThemeManager&) = delete;
  ThemeManager& operator=(const ThemeManager&) = delete;

  void initialize();
  void applyTheme(ThemeType type);

  // Glass-morphism helpers
  void pushGlassStyle();
  void popGlassStyle();

  // Accessors
  const ThemeColors& getColors() const { return current_colors_; }
  ImFont* getMainFont() const { return main_font_; }
  ImFont* getLargeFont() const { return large_font_; }

  // Dynamic updates
  void toggleTheme();  // Switching between Dark/Light or variants

  // Integration with unified theme system
  void apply_unified_theme(const std::string& theme_name);
  void sync_with_layout_manager();

 private:
  ThemeManager() = default;

  ThemeColors current_colors_;
  ThemeType current_theme_type_ = ThemeType::DarkNeon;

  ImFont* main_font_ = nullptr;
  ImFont* large_font_ = nullptr;

  void loadFonts();
  void updateImGuiStyle();
};

}  // namespace BTQuant
