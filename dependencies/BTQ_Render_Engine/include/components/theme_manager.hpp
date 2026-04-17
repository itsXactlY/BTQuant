#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "imgui.h"

namespace BTQuant {

struct ThemeColors {
  // MMT Deep Void Palette - dark, clean, professional
  ImVec4 background = ImVec4(0.043f, 0.055f, 0.067f, 1.0f);  // #0B0E11 - Deep void background
  ImVec4 text = ImVec4(0.82f, 0.84f, 0.86f, 1.0f);           // #D1D7DE - Clean text
  ImVec4 text_dim = ImVec4(0.45f, 0.48f, 0.52f, 1.0f);       // Muted text

  // Accents - MMT Neon Mint / Crimson
  ImVec4 accent_green = ImVec4(0.0f, 0.9f, 0.4f, 1.0f);      // #00E566 - Neon Mint
  ImVec4 accent_red = ImVec4(0.9f, 0.1f, 0.15f, 1.0f);       // #E61926 - Crimson
  ImVec4 accent_cyan = ImVec4(0.0f, 0.9f, 0.4f, 1.0f);       // #00E566 - Accent = Neon Mint
  ImVec4 accent_magenta = ImVec4(0.9f, 0.1f, 0.15f, 1.0f);   // #E61926 - Accent = Crimson

  // UI Elements
  ImVec4 panel_bg = ImVec4(0.082f, 0.098f, 0.118f, 1.0f);    // #15191E - Panel background
  ImVec4 border = ImVec4(0.165f, 0.180f, 0.196f, 1.0f);      // #2A2E33 - Border
  ImVec4 header_bg = ImVec4(0.082f, 0.098f, 0.118f, 1.0f);   // #15191E - Header matches panel

  // Chart specific
  ImVec4 chart_grid = ImVec4(0.10f, 0.12f, 0.14f, 0.4f);     // #1A1E24 - Grid lines
  ImVec4 candle_up = ImVec4(0.0f, 0.9f, 0.4f, 1.0f);         // #00E566 - Neon Mint
  ImVec4 candle_down = ImVec4(0.9f, 0.1f, 0.15f, 1.0f);      // #E61926 - Crimson
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
