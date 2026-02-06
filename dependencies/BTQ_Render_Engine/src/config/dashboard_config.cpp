#include "dashboard_config.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace BTQuant {
namespace RenderEngine {

DashboardConfig::DashboardConfig()
    : config_file_("dashboard_config.yaml"),
      auto_save_enabled_(true),
      config_version_(1),
      layouts_dir_("layouts"),
      current_layout_("default") {
  // Create layouts directory if it doesn't exist
  std::filesystem::create_directories(layouts_dir_);

  // Initialize default configuration
  initializeDefaults();
  std::cout << "[DashboardConfig] Initialized with default settings" << std::endl;
}

DashboardConfig::~DashboardConfig() {
  if (auto_save_enabled_) {
    saveConfiguration();
  }
}

bool DashboardConfig::loadConfiguration(const std::string& config_file) {
  config_file_ = config_file;

  std::ifstream file(config_file_);
  if (!file.is_open()) {
    std::cout << "[DashboardConfig] Config file not found, using defaults: " << config_file_
              << std::endl;
    return false;
  }

  try {
    std::string line;
    std::string current_section;

    while (std::getline(file, line)) {
      // Remove comments and trim whitespace
      size_t comment_pos = line.find('#');
      if (comment_pos != std::string::npos) {
        line = line.substr(0, comment_pos);
      }

      line = trim(line);
      if (line.empty()) continue;

      // Check for section headers
      if (line.front() == '[' && line.back() == ']') {
        current_section = line.substr(1, line.length() - 2);
        continue;
      }

      // Parse key-value pairs
      size_t equals_pos = line.find('=');
      if (equals_pos != std::string::npos) {
        std::string key = trim(line.substr(0, equals_pos));
        std::string value = trim(line.substr(equals_pos + 1));

        parseConfigValue(current_section, key, value);
      }
    }

    file.close();
    std::cout << "[DashboardConfig] Loaded configuration from: " << config_file_ << std::endl;
    return true;

  } catch (const std::exception& e) {
    std::cerr << "[DashboardConfig] Error loading config: " << e.what() << std::endl;
    return false;
  }
}

bool DashboardConfig::saveConfiguration() const {
  if (config_file_.empty()) {
    std::cerr << "[DashboardConfig] No config file specified for saving" << std::endl;
    return false;
  }

  std::ofstream file(config_file_);
  if (!file.is_open()) {
    std::cerr << "[DashboardConfig] Failed to open config file for writing: " << config_file_
              << std::endl;
    return false;
  }

  try {
    file << "# BTQuant Dashboard Configuration" << std::endl;
    file << "# Generated automatically - edit with care" << std::endl;
    file << "version=" << config_version_ << std::endl;
    file << std::endl;

    // Data source configuration
    file << "[data_source]" << std::endl;
    file << "data_source_uri=" << data_source_config_.data_source_uri << std::endl;
    file << "symbols_file=" << data_source_config_.symbols_file << std::endl;
    file << "auto_reconnect=" << (data_source_config_.auto_reconnect ? "true" : "false")
         << std::endl;
    file << "reconnect_interval_ms=" << data_source_config_.reconnect_interval_ms << std::endl;
    file << "max_symbols=" << data_source_config_.max_symbols << std::endl;
    file << std::endl;

    // Display configuration
    file << "[display]" << std::endl;
    file << "window_width=" << display_config_.window_width << std::endl;
    file << "window_height=" << display_config_.window_height << std::endl;
    file << "fullscreen=" << (display_config_.fullscreen ? "true" : "false") << std::endl;
    file << "vsync=" << (display_config_.vsync ? "true" : "false") << std::endl;
    file << "target_fps=" << display_config_.target_fps << std::endl;
    file << "msaa_samples=" << display_config_.msaa_samples << std::endl;
    file << std::endl;

    // Theme configuration
    file << "[theme]" << std::endl;
    file << "name=" << theme_config_.name << std::endl;
    file << "background_color=" << colorToString(theme_config_.background_color) << std::endl;
    file << "text_color=" << colorToString(theme_config_.text_color) << std::endl;
    file << "accent_color=" << colorToString(theme_config_.accent_color) << std::endl;
    file << "positive_color=" << colorToString(theme_config_.positive_color) << std::endl;
    file << "negative_color=" << colorToString(theme_config_.negative_color) << std::endl;
    file << "neutral_color=" << colorToString(theme_config_.neutral_color) << std::endl;
    file << "grid_line_color=" << colorToString(theme_config_.grid_line_color) << std::endl;
    file << "font_size=" << theme_config_.font_size << std::endl;
    file << "line_height=" << theme_config_.line_height << std::endl;
    file << std::endl;

    // Layout configuration
    file << "[layout]" << std::endl;
    file << "grid_columns=" << layout_config_.grid_columns << std::endl;
    file << "grid_rows=" << layout_config_.grid_rows << std::endl;
    file << "show_grid=" << (layout_config_.show_grid ? "true" : "false") << std::endl;
    file << "show_heatmap=" << (layout_config_.show_heatmap ? "true" : "false") << std::endl;
    file << "show_charts=" << (layout_config_.show_charts ? "true" : "false") << std::endl;
    file << "show_orderbook=" << (layout_config_.show_orderbook ? "true" : "false") << std::endl;
    file << "show_logs=" << (layout_config_.show_logs ? "true" : "false") << std::endl;
    file << "show_performance=" << (layout_config_.show_performance ? "true" : "false")
         << std::endl;
    file << "panel_spacing=" << layout_config_.panel_spacing << std::endl;
    file << "panel_padding=" << layout_config_.panel_padding << std::endl;
    file << std::endl;

    // Performance configuration
    file << "[performance]" << std::endl;
    file << "monitoring_enabled=" << (performance_config_.monitoring_enabled ? "true" : "false")
         << std::endl;
    file << "monitoring_interval_ms=" << performance_config_.monitoring_interval_ms << std::endl;
    file << "history_size=" << performance_config_.history_size << std::endl;
    file << "fps_alert_threshold=" << performance_config_.fps_alert_threshold << std::endl;
    file << "latency_alert_threshold_ms=" << performance_config_.latency_alert_threshold_ms
         << std::endl;
    file << "memory_alert_threshold_mb=" << performance_config_.memory_alert_threshold_mb
         << std::endl;
    file << "enable_profiling=" << (performance_config_.enable_profiling ? "true" : "false")
         << std::endl;
    file << std::endl;

    // User preferences
    file << "[user_preferences]" << std::endl;
    file << "default_exchange=" << user_preferences_.default_exchange << std::endl;
    file << "default_symbol=" << user_preferences_.default_symbol << std::endl;
    file << "auto_save_layout=" << (user_preferences_.auto_save_layout ? "true" : "false")
         << std::endl;
    file << "show_tooltips=" << (user_preferences_.show_tooltips ? "true" : "false") << std::endl;
    file << "animation_speed=" << user_preferences_.animation_speed << std::endl;
    file << "update_frequency_hz=" << user_preferences_.update_frequency_hz << std::endl;
    file << "current_layout=" << current_layout_ << std::endl;

    file.close();
    std::cout << "[DashboardConfig] Saved configuration to: " << config_file_ << std::endl;
    return true;

  } catch (const std::exception& e) {
    std::cerr << "[DashboardConfig] Error saving config: " << e.what() << std::endl;
    return false;
  }
}

void DashboardConfig::resetToDefaults() {
  initializeDefaults();
  std::cout << "[DashboardConfig] Reset to default configuration" << std::endl;
}

// Getters
const DataSourceConfig& DashboardConfig::getDataSourceConfig() const { return data_source_config_; }

const DisplayConfig& DashboardConfig::getDisplayConfig() const { return display_config_; }

const ThemeConfig& DashboardConfig::getThemeConfig() const { return theme_config_; }

const LayoutConfig& DashboardConfig::getLayoutConfig() const { return layout_config_; }

const PerformanceConfig& DashboardConfig::getPerformanceConfig() const {
  return performance_config_;
}

const UserPreferences& DashboardConfig::getUserPreferences() const { return user_preferences_; }

// Setters
void DashboardConfig::setDataSourceConfig(const DataSourceConfig& config) {
  data_source_config_ = config;
}

void DashboardConfig::setDisplayConfig(const DisplayConfig& config) { display_config_ = config; }

void DashboardConfig::setThemeConfig(const ThemeConfig& config) { theme_config_ = config; }

void DashboardConfig::setLayoutConfig(const LayoutConfig& config) { layout_config_ = config; }

void DashboardConfig::setPerformanceConfig(const PerformanceConfig& config) {
  performance_config_ = config;
}

void DashboardConfig::setUserPreferences(const UserPreferences& preferences) {
  user_preferences_ = preferences;
}

void DashboardConfig::enableAutoSave(bool enabled) { auto_save_enabled_ = enabled; }

std::vector<std::string> DashboardConfig::getAvailableThemes() const {
  return {"dark", "light", "blue", "green", "custom"};
}

void DashboardConfig::applyDarkTheme() {
  theme_config_.name = "dark";
  theme_config_.background_color = {0.1f, 0.1f, 0.1f, 1.0f};
  theme_config_.text_color = {0.9f, 0.9f, 0.9f, 1.0f};
  theme_config_.accent_color = {0.2f, 0.6f, 1.0f, 1.0f};
  theme_config_.positive_color = {0.0f, 0.8f, 0.0f, 1.0f};
  theme_config_.negative_color = {0.8f, 0.0f, 0.0f, 1.0f};
  theme_config_.neutral_color = {0.5f, 0.5f, 0.5f, 1.0f};
  theme_config_.grid_line_color = {0.3f, 0.3f, 0.3f, 1.0f};
}

void DashboardConfig::applyLightTheme() {
  theme_config_.name = "light";
  theme_config_.background_color = {0.95f, 0.95f, 0.95f, 1.0f};
  theme_config_.text_color = {0.1f, 0.1f, 0.1f, 1.0f};
  theme_config_.accent_color = {0.0f, 0.4f, 0.8f, 1.0f};
  theme_config_.positive_color = {0.0f, 0.6f, 0.0f, 1.0f};
  theme_config_.negative_color = {0.8f, 0.0f, 0.0f, 1.0f};
  theme_config_.neutral_color = {0.4f, 0.4f, 0.4f, 1.0f};
  theme_config_.grid_line_color = {0.7f, 0.7f, 0.7f, 1.0f};
}

void DashboardConfig::applyBlueTheme() {
  theme_config_.name = "blue";
  theme_config_.background_color = {0.05f, 0.1f, 0.2f, 1.0f};
  theme_config_.text_color = {0.8f, 0.9f, 1.0f, 1.0f};
  theme_config_.accent_color = {0.3f, 0.7f, 1.0f, 1.0f};
  theme_config_.positive_color = {0.0f, 0.8f, 0.4f, 1.0f};
  theme_config_.negative_color = {1.0f, 0.3f, 0.3f, 1.0f};
  theme_config_.neutral_color = {0.5f, 0.6f, 0.7f, 1.0f};
  theme_config_.grid_line_color = {0.2f, 0.3f, 0.4f, 1.0f};
}

void DashboardConfig::applyGreenTheme() {
  theme_config_.name = "green";
  theme_config_.background_color = {0.05f, 0.15f, 0.1f, 1.0f};
  theme_config_.text_color = {0.8f, 0.95f, 0.8f, 1.0f};
  theme_config_.accent_color = {0.2f, 0.8f, 0.4f, 1.0f};
  theme_config_.positive_color = {0.3f, 0.9f, 0.3f, 1.0f};
  theme_config_.negative_color = {0.9f, 0.3f, 0.3f, 1.0f};
  theme_config_.neutral_color = {0.5f, 0.6f, 0.5f, 1.0f};
  theme_config_.grid_line_color = {0.2f, 0.3f, 0.2f, 1.0f};
}

bool DashboardConfig::applyTheme(const std::string& theme_name) {
  if (theme_name == "dark") {
    applyDarkTheme();
  } else if (theme_name == "light") {
    applyLightTheme();
  } else if (theme_name == "blue") {
    applyBlueTheme();
  } else if (theme_name == "green") {
    applyGreenTheme();
  } else if (theme_name == "custom") {
    theme_config_.name = "custom";
    // Keep existing colors when applying custom theme
  } else {
    return false;  // Unknown theme
  }

  std::cout << "[DashboardConfig] Applied theme: " << theme_name << std::endl;
  return true;
}

bool DashboardConfig::saveLayout(const std::string& layout_name) {
  try {
    std::string file_path = getLayoutFilePath(layout_name);
    std::ofstream file(file_path);

    if (!file.is_open()) {
      std::cerr << "[DashboardConfig] Failed to open layout file for writing: " << file_path
                << std::endl;
      return false;
    }

    // Save current layout configuration
    file << "# BTQuant Dashboard Layout Configuration" << std::endl;
    file << "# Generated automatically - edit with care" << std::endl;
    file << "version=" << config_version_ << std::endl;
    file << std::endl;

    file << "[layout]" << std::endl;
    file << "name=" << layout_name << std::endl;
    file << "grid_columns=" << layout_config_.grid_columns << std::endl;
    file << "grid_rows=" << layout_config_.grid_rows << std::endl;
    file << "show_grid=" << (layout_config_.show_grid ? "true" : "false") << std::endl;
    file << "show_heatmap=" << (layout_config_.show_heatmap ? "true" : "false") << std::endl;
    file << "show_charts=" << (layout_config_.show_charts ? "true" : "false") << std::endl;
    file << "show_orderbook=" << (layout_config_.show_orderbook ? "true" : "false") << std::endl;
    file << "show_logs=" << (layout_config_.show_logs ? "true" : "false") << std::endl;
    file << "show_performance=" << (layout_config_.show_performance ? "true" : "false")
         << std::endl;
    file << "panel_spacing=" << layout_config_.panel_spacing << std::endl;
    file << "panel_padding=" << layout_config_.panel_padding << std::endl;

    file.close();
    current_layout_ = layout_name;
    std::cout << "[DashboardConfig] Saved layout: " << layout_name << " to " << file_path
              << std::endl;
    return true;

  } catch (const std::exception& e) {
    std::cerr << "[DashboardConfig] Error saving layout: " << e.what() << std::endl;
    return false;
  }
}

bool DashboardConfig::loadLayout(const std::string& layout_name) {
  try {
    std::string file_path = getLayoutFilePath(layout_name);
    std::ifstream file(file_path);

    if (!file.is_open()) {
      std::cerr << "[DashboardConfig] Layout file not found: " << file_path << std::endl;
      return false;
    }

    std::string line;
    std::string current_section;

    while (std::getline(file, line)) {
      // Remove comments and trim whitespace
      size_t comment_pos = line.find('#');
      if (comment_pos != std::string::npos) {
        line = line.substr(0, comment_pos);
      }

      line = trim(line);
      if (line.empty()) continue;

      // Check for section headers
      if (line.front() == '[' && line.back() == ']') {
        current_section = line.substr(1, line.length() - 2);
        continue;
      }

      // Parse key-value pairs
      size_t equals_pos = line.find('=');
      if (equals_pos != std::string::npos) {
        std::string key = trim(line.substr(0, equals_pos));
        std::string value = trim(line.substr(equals_pos + 1));

        parseConfigValue(current_section, key, value);
      }
    }

    file.close();
    current_layout_ = layout_name;
    std::cout << "[DashboardConfig] Loaded layout: " << layout_name << " from " << file_path
              << std::endl;
    return true;

  } catch (const std::exception& e) {
    std::cerr << "[DashboardConfig] Error loading layout: " << e.what() << std::endl;
    return false;
  }
}

std::vector<std::string> DashboardConfig::getAvailableLayouts() const {
  std::vector<std::string> layouts;
  try {
    if (std::filesystem::exists(layouts_dir_)) {
      for (const auto& entry : std::filesystem::directory_iterator(layouts_dir_)) {
        if (entry.is_regular_file() && entry.path().extension() == ".yaml") {
          std::string filename = entry.path().filename().string();
          layouts.push_back(filename.substr(0, filename.size() - 5));  // Remove .yaml extension
        }
      }
    }

    // Always include default layout if it doesn't exist
    if (layouts.empty()) {
      layouts.push_back("default");
    }

  } catch (const std::exception& e) {
    std::cerr << "[DashboardConfig] Error loading available layouts: " << e.what() << std::endl;
    layouts.push_back("default");
  }

  return layouts;
}

bool DashboardConfig::deleteLayout(const std::string& layout_name) {
  if (layout_name == "default") {
    std::cerr << "[DashboardConfig] Cannot delete default layout" << std::endl;
    return false;
  }

  try {
    std::string file_path = getLayoutFilePath(layout_name);
    if (std::filesystem::exists(file_path)) {
      std::filesystem::remove(file_path);
      std::cout << "[DashboardConfig] Deleted layout: " << layout_name << std::endl;
      return true;
    } else {
      std::cerr << "[DashboardConfig] Layout file not found: " << file_path << std::endl;
      return false;
    }

  } catch (const std::exception& e) {
    std::cerr << "[DashboardConfig] Error deleting layout: " << e.what() << std::endl;
    return false;
  }
}

std::string DashboardConfig::getLayoutFilePath(const std::string& layout_name) const {
  return layouts_dir_ + "/" + layout_name + ".yaml";
}

void DashboardConfig::initializeDefaults() {
  // Data source defaults
  data_source_config_.data_source_uri = "tcp://localhost:5555";
  data_source_config_.symbols_file = "/dev/shm/btquant_symbols.json";
  data_source_config_.auto_reconnect = true;
  data_source_config_.reconnect_interval_ms = 5000;
  data_source_config_.max_symbols = 1000;

  // Display defaults
  display_config_.window_width = 1920;
  display_config_.window_height = 1080;
  display_config_.fullscreen = false;
  display_config_.vsync = true;
  display_config_.target_fps = 60;
  display_config_.msaa_samples = 4;

  // Theme defaults (dark theme)
  applyTheme("dark");
  theme_config_.font_size = 14;
  theme_config_.line_height = 1.2f;

  // Layout defaults
  layout_config_.grid_columns = 10;
  layout_config_.grid_rows = 20;
  layout_config_.show_grid = true;
  layout_config_.show_heatmap = true;
  layout_config_.show_charts = true;
  layout_config_.show_orderbook = true;
  layout_config_.show_logs = true;
  layout_config_.show_performance = true;
  layout_config_.panel_spacing = 8.0f;
  layout_config_.panel_padding = 12.0f;

  // Performance defaults
  performance_config_.monitoring_enabled = true;
  performance_config_.monitoring_interval_ms = 1000;
  performance_config_.history_size = 300;
  performance_config_.fps_alert_threshold = 30.0;
  performance_config_.latency_alert_threshold_ms = 10.0;
  performance_config_.memory_alert_threshold_mb = 1024.0;
  performance_config_.enable_profiling = false;

  // User preferences defaults
  user_preferences_.default_exchange = "binance";
  user_preferences_.default_symbol = "BTCUSDT";
  user_preferences_.auto_save_layout = true;
  user_preferences_.show_tooltips = true;
  user_preferences_.animation_speed = 1.0f;
  user_preferences_.update_frequency_hz = 30.0;
}

void DashboardConfig::parseConfigValue(const std::string& section, const std::string& key,
                                       const std::string& value) {
  if (section == "data_source") {
    if (key == "data_source_uri")
      data_source_config_.data_source_uri = value;
    else if (key == "symbols_file")
      data_source_config_.symbols_file = value;
    else if (key == "auto_reconnect")
      data_source_config_.auto_reconnect = (value == "true");
    else if (key == "reconnect_interval_ms")
      data_source_config_.reconnect_interval_ms = std::stoul(value);
    else if (key == "max_symbols")
      data_source_config_.max_symbols = std::stoul(value);
  } else if (section == "display") {
    if (key == "window_width")
      display_config_.window_width = std::stoul(value);
    else if (key == "window_height")
      display_config_.window_height = std::stoul(value);
    else if (key == "fullscreen")
      display_config_.fullscreen = (value == "true");
    else if (key == "vsync")
      display_config_.vsync = (value == "true");
    else if (key == "target_fps")
      display_config_.target_fps = std::stoul(value);
    else if (key == "msaa_samples")
      display_config_.msaa_samples = std::stoul(value);
  } else if (section == "theme") {
    if (key == "name")
      theme_config_.name = value;
    else if (key == "background_color")
      theme_config_.background_color = parseColor(value);
    else if (key == "text_color")
      theme_config_.text_color = parseColor(value);
    else if (key == "accent_color")
      theme_config_.accent_color = parseColor(value);
    else if (key == "positive_color")
      theme_config_.positive_color = parseColor(value);
    else if (key == "negative_color")
      theme_config_.negative_color = parseColor(value);
    else if (key == "neutral_color")
      theme_config_.neutral_color = parseColor(value);
    else if (key == "grid_line_color")
      theme_config_.grid_line_color = parseColor(value);
    else if (key == "font_size")
      theme_config_.font_size = std::stoul(value);
    else if (key == "line_height")
      theme_config_.line_height = std::stof(value);
  } else if (section == "layout") {
    if (key == "grid_columns")
      layout_config_.grid_columns = std::stoul(value);
    else if (key == "grid_rows")
      layout_config_.grid_rows = std::stoul(value);
    else if (key == "show_grid")
      layout_config_.show_grid = (value == "true");
    else if (key == "show_heatmap")
      layout_config_.show_heatmap = (value == "true");
    else if (key == "show_charts")
      layout_config_.show_charts = (value == "true");
    else if (key == "show_orderbook")
      layout_config_.show_orderbook = (value == "true");
    else if (key == "show_logs")
      layout_config_.show_logs = (value == "true");
    else if (key == "show_performance")
      layout_config_.show_performance = (value == "true");
    else if (key == "panel_spacing")
      layout_config_.panel_spacing = std::stof(value);
    else if (key == "panel_padding")
      layout_config_.panel_padding = std::stof(value);
  } else if (section == "performance") {
    if (key == "monitoring_enabled")
      performance_config_.monitoring_enabled = (value == "true");
    else if (key == "monitoring_interval_ms")
      performance_config_.monitoring_interval_ms = std::stoul(value);
    else if (key == "history_size")
      performance_config_.history_size = std::stoul(value);
    else if (key == "fps_alert_threshold")
      performance_config_.fps_alert_threshold = std::stod(value);
    else if (key == "latency_alert_threshold_ms")
      performance_config_.latency_alert_threshold_ms = std::stod(value);
    else if (key == "memory_alert_threshold_mb")
      performance_config_.memory_alert_threshold_mb = std::stod(value);
    else if (key == "enable_profiling")
      performance_config_.enable_profiling = (value == "true");
  } else if (section == "user_preferences") {
    if (key == "default_exchange")
      user_preferences_.default_exchange = value;
    else if (key == "default_symbol")
      user_preferences_.default_symbol = value;
    else if (key == "auto_save_layout")
      user_preferences_.auto_save_layout = (value == "true");
    else if (key == "show_tooltips")
      user_preferences_.show_tooltips = (value == "true");
    else if (key == "animation_speed")
      user_preferences_.animation_speed = std::stof(value);
    else if (key == "update_frequency_hz")
      user_preferences_.update_frequency_hz = std::stod(value);
  }
}

ColorRGBA DashboardConfig::parseColor(const std::string& color_str) const {
  // Parse color in format "r,g,b,a" or "#RRGGBB" or "#RRGGBBAA"
  ColorRGBA color = {1.0f, 1.0f, 1.0f, 1.0f};  // Default white

  if (color_str.empty()) return color;

  if (color_str[0] == '#') {
    // Hex format
    std::string hex = color_str.substr(1);
    if (hex.length() == 6 || hex.length() == 8) {
      unsigned long value = std::stoul(hex, nullptr, 16);
      if (hex.length() == 6) {
        color.r = ((value >> 16) & 0xFF) / 255.0f;
        color.g = ((value >> 8) & 0xFF) / 255.0f;
        color.b = (value & 0xFF) / 255.0f;
        color.a = 1.0f;
      } else {
        color.r = ((value >> 24) & 0xFF) / 255.0f;
        color.g = ((value >> 16) & 0xFF) / 255.0f;
        color.b = ((value >> 8) & 0xFF) / 255.0f;
        color.a = (value & 0xFF) / 255.0f;
      }
    }
  } else {
    // Comma-separated format
    std::istringstream iss(color_str);
    std::string component;
    int i = 0;
    while (std::getline(iss, component, ',') && i < 4) {
      float value = std::stof(trim(component));
      switch (i) {
        case 0:
          color.r = value;
          break;
        case 1:
          color.g = value;
          break;
        case 2:
          color.b = value;
          break;
        case 3:
          color.a = value;
          break;
      }
      i++;
    }
  }

  return color;
}

std::string DashboardConfig::colorToString(const ColorRGBA& color) const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << color.r << "," << color.g << "," << color.b << ","
      << color.a;
  return oss.str();
}

std::string DashboardConfig::trim(const std::string& str) const {
  size_t start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";

  size_t end = str.find_last_not_of(" \t\r\n");
  return str.substr(start, end - start + 1);
}

}  // namespace RenderEngine
}  // namespace BTQuant