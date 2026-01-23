#pragma once

#include "imgui.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace BTQuant {

struct ThemeColors {
  ImVec4 background = ImVec4(0.04f, 0.04f, 0.04f, 1.0f); // #0A0A0A
  ImVec4 text = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  ImVec4 text_dim = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);

  // Accents
  ImVec4 accent_green = ImVec4(0.2f, 1.0f, 0.2f, 1.0f); // #33FF33 Neon Green
  ImVec4 accent_red = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);   // #FF3333 Neon Red
  ImVec4 accent_cyan = ImVec4(0.0f, 1.0f, 1.0f, 1.0f);  // #00FFFF Cyan

  // UI Elements
  ImVec4 panel_bg = ImVec4(0.08f, 0.08f, 0.08f, 0.6f); // Glass effect base
  ImVec4 border = ImVec4(0.2f, 0.2f, 0.2f, 0.5f);
  ImVec4 header_bg = ImVec4(0.1f, 0.1f, 0.1f, 0.8f);

  // Chart specific
  ImVec4 chart_grid = ImVec4(0.2f, 0.2f, 0.2f, 0.2f);
  ImVec4 candle_up = ImVec4(0.2f, 1.0f, 0.2f, 1.0f);
  ImVec4 candle_down = ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
};

enum class ThemeType {
  DarkNeon,
  LightClean // Future proofing
};

class ThemeManager {
public:
  static ThemeManager &getInstance() {
    static ThemeManager instance;
    return instance;
  }

  ThemeManager(const ThemeManager &) = delete;
  ThemeManager &operator=(const ThemeManager &) = delete;

  void initialize();
  void applyTheme(ThemeType type);

  // Glass-morphism helpers
  void pushGlassStyle();
  void popGlassStyle();

  // Accessors
  const ThemeColors &getColors() const { return current_colors_; }
  ImFont *getMainFont() const { return main_font_; }
  ImFont *getLargeFont() const { return large_font_; }

  // Dynamic updates
  void toggleTheme(); // Switching between Dark/Light or variants

private:
  ThemeManager() = default;

  ThemeColors current_colors_;
  ThemeType current_theme_type_ = ThemeType::DarkNeon;

  ImFont *main_font_ = nullptr;
  ImFont *large_font_ = nullptr;

  void loadFonts();
  void updateImGuiStyle();
};

} // namespace BTQuant
