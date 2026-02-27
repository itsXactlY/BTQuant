#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// We'll include imgui.h in the implementation file to avoid conflicts
// Forward declarations are not needed here since we're including imgui.h in the implementation

namespace BTQuant {
namespace UI {

// ============================================================================
// Unified Theme System
// ============================================================================

struct ThemeDefinition {
  std::string name;
  std::string description;
  bool is_dark_theme;

  // Color definitions
  struct Colors {
    float background_primary[4];    // RGBA
    float background_secondary[4];  // RGBA
    float background_panel[4];      // RGBA
    float text_primary[4];          // RGBA
    float text_secondary[4];        // RGBA
    float text_muted[4];            // RGBA
    float price_up[4];              // RGBA
    float price_down[4];            // RGBA
    float price_neutral[4];         // RGBA
    float accent_primary[4];        // RGBA
    float accent_secondary[4];      // RGBA
    float border_color[4];          // RGBA
    float status_connected[4];      // RGBA
    float status_disconnected[4];   // RGBA
    float status_warning[4];        // RGBA
  } colors;

  // Typography
  std::string font_family;
  float font_size_normal;
  float font_size_small;
  float font_size_large;

  // Spacing
  float padding_small;
  float padding_medium;
  float padding_large;

  // Border radius
  float border_radius_small;
  float border_radius_medium;
  float border_radius_large;

  // Shadow properties
  struct Shadow {
    float offset_x;
    float offset_y;
    float blur_radius;
    float spread;
    float color[4];  // RGBA
  } shadow_normal;

  bool is_builtin;
};

class UnifiedThemeManager {
 public:
  static UnifiedThemeManager& getInstance() {
    static UnifiedThemeManager instance;
    return instance;
  }

  UnifiedThemeManager(const UnifiedThemeManager&) = delete;
  UnifiedThemeManager& operator=(const UnifiedThemeManager&) = delete;

  // Initialize the theme system
  void initialize();

  // Register a new theme
  bool register_theme(const ThemeDefinition& theme);

  // Set the current theme
  bool set_current_theme(const std::string& theme_name);

  // Get the current theme
  const ThemeDefinition& get_current_theme() const;

  // Get all available themes
  std::vector<std::string> get_available_themes() const;

  // Get theme by name
  const ThemeDefinition* get_theme(const std::string& theme_name) const;

  // Apply theme to ImGui
  void apply_to_imgui() const;

  // Apply theme to custom components
  void apply_to_components() const;

  // Get color by name from current theme
  void get_color(const std::string& color_name, float* rgba) const;

  // Save a custom theme to file
  bool save_theme(const std::string& theme_name, const std::string& file_path) const;

  // Load a theme from file
  bool load_theme(const std::string& file_path);

  // Create a theme preview
  void create_preview(const std::string& theme_name, const std::string& output_path) const;

  // Validate a theme
  bool validate_theme(const ThemeDefinition& theme) const;

  // Get theme categories
  std::vector<std::string> get_theme_categories() const;

  // Get themes by category
  std::vector<std::string> get_themes_by_category(const std::string& category) const;

 private:
  UnifiedThemeManager();   // Private constructor for singleton
  ~UnifiedThemeManager();  // Private destructor for singleton

  std::unordered_map<std::string, ThemeDefinition> themes_;
  std::string current_theme_name_;
  std::string themes_directory_;

  void initialize_themes_directory();
  void load_builtin_themes();
  ThemeDefinition load_theme_from_file(const std::string& file_path) const;
  bool save_theme_to_file(const ThemeDefinition& theme, const std::string& file_path) const;
  std::string get_theme_file_path(const std::string& theme_name) const;
};

}  // namespace UI
}  // namespace BTQuant