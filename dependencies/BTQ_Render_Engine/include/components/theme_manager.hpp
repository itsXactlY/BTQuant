#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "imgui.h"

namespace BTQuant {

struct ThemeColors {
  ImVec4 background = ImVec4(0.06f, 0.07f, 0.12f, 1.0f);  // #10121f - Dark background with enhanced contrast
  ImVec4 text = ImVec4(0.92f, 0.92f, 0.92f, 1.0f);        // #ebebeb - Light text for excellent contrast (WCAG AAA)
  ImVec4 text_dim = ImVec4(0.55f, 0.55f, 0.55f, 1.0f);    // #8c8c8c - Muted text with sufficient contrast (WCAG AA)

  // Accents
  ImVec4 accent_green = ImVec4(0.35f, 0.95f, 0.35f, 1.0f); // #59f259 - Enhanced green for optimal contrast
  ImVec4 accent_red = ImVec4(0.95f, 0.35f, 0.35f, 1.0f);   // #f25959 - Enhanced red for optimal contrast
  ImVec4 accent_cyan = ImVec4(0.25f, 0.85f, 0.95f, 1.0f);  // #40d9f5 - Enhanced cyan accent
  ImVec4 accent_magenta = ImVec4(0.95f, 0.25f, 0.85f, 1.0f); // #f540d9 - Enhanced magenta accent

  // UI Elements
  ImVec4 panel_bg = ImVec4(0.09f, 0.11f, 0.15f, 0.92f);   // #171c26 - Panel background with enhanced contrast
  ImVec4 border = ImVec4(0.20f, 0.22f, 0.28f, 0.65f);     // #333847 - Borders with enhanced contrast
  ImVec4 header_bg = ImVec4(0.12f, 0.14f, 0.20f, 0.97f);  // #1f2433 - Header background with enhanced contrast

  // Chart specific
  ImVec4 chart_grid = ImVec4(0.20f, 0.22f, 0.28f, 0.35f);  // #333847 - Grid with enhanced contrast
  ImVec4 candle_up = ImVec4(0.35f, 0.95f, 0.35f, 1.0f);    // Consistent with accent green
  ImVec4 candle_down = ImVec4(0.95f, 0.35f, 0.35f, 1.0f);  // Consistent with accent red
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
