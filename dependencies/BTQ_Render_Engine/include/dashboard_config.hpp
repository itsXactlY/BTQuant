#pragma once

#include "data_visualization_engine.hpp"
#include <string>
#include <vector>
#include <iomanip>

namespace BTQuant {
namespace RenderEngine {

// Data source configuration
struct DataSourceConfig {
    std::string hotspine_shm_name = "/btquant_hotspine";
    std::string symbols_file = "/dev/shm/btquant_symbols.json";
    bool auto_reconnect = true;
    uint32_t reconnect_interval_ms = 5000;
    size_t max_symbols = 1000;
};

// Display configuration
struct DisplayConfig {
    uint32_t window_width = 1920;
    uint32_t window_height = 1080;
    bool fullscreen = false;
    bool vsync = true;
    uint32_t target_fps = 60;
    uint32_t msaa_samples = 4;
};

// Theme configuration
struct ThemeConfig {
    std::string name = "dark";
    ColorRGBA background_color = {0.1f, 0.1f, 0.1f, 1.0f};
    ColorRGBA text_color = {0.9f, 0.9f, 0.9f, 1.0f};
    ColorRGBA accent_color = {0.2f, 0.6f, 1.0f, 1.0f};
    ColorRGBA positive_color = {0.0f, 0.8f, 0.0f, 1.0f};
    ColorRGBA negative_color = {0.8f, 0.0f, 0.0f, 1.0f};
    ColorRGBA neutral_color = {0.5f, 0.5f, 0.5f, 1.0f};
    ColorRGBA grid_line_color = {0.3f, 0.3f, 0.3f, 1.0f};
    uint32_t font_size = 14;
    float line_height = 1.2f;
};

// Layout configuration
struct LayoutConfig {
    uint32_t grid_columns = 10;
    uint32_t grid_rows = 20;
    bool show_grid = true;
    bool show_heatmap = true;
    bool show_charts = true;
    bool show_orderbook = true;
    bool show_logs = true;
    bool show_performance = true;
    float panel_spacing = 8.0f;
    float panel_padding = 12.0f;
};

// Performance configuration
struct PerformanceConfig {
    bool monitoring_enabled = true;
    uint32_t monitoring_interval_ms = 1000;
    size_t history_size = 300;
    double fps_alert_threshold = 30.0;
    double latency_alert_threshold_ms = 10.0;
    double memory_alert_threshold_mb = 1024.0;
    bool enable_profiling = false;
};

// User preferences
struct UserPreferences {
    std::string default_exchange = "binance";
    std::string default_symbol = "BTCUSDT";
    bool auto_save_layout = true;
    bool show_tooltips = true;
    float animation_speed = 1.0f;
    double update_frequency_hz = 30.0;
};

/**
 * DashboardConfig - Comprehensive configuration management system
 * 
 * This class manages all configuration aspects of the dashboard including:
 * - Data source settings (HotSpine connection, symbol files)
 * - Display settings (resolution, fullscreen, performance)
 * - Theme and color management
 * - Layout and UI preferences
 * - Performance monitoring configuration
 * - User preferences and customization
 * - Configuration persistence and loading
 */
class DashboardConfig {
public:
    DashboardConfig();
    ~DashboardConfig();
    
    // Non-copyable, non-movable
    DashboardConfig(const DashboardConfig&) = delete;
    DashboardConfig& operator=(const DashboardConfig&) = delete;
    DashboardConfig(DashboardConfig&&) = delete;
    DashboardConfig& operator=(DashboardConfig&&) = delete;
    
    /**
     * Load configuration from file
     * @param config_file Path to configuration file
     * @return true if loaded successfully
     */
    bool loadConfiguration(const std::string& config_file = "dashboard_config.yaml");
    
    /**
     * Save current configuration to file
     * @return true if saved successfully
     */
    bool saveConfiguration() const;
    
    /**
     * Reset all settings to defaults
     */
    void resetToDefaults();
    
    /**
     * Configuration getters
     */
    const DataSourceConfig& getDataSourceConfig() const;
    const DisplayConfig& getDisplayConfig() const;
    const ThemeConfig& getThemeConfig() const;
    const LayoutConfig& getLayoutConfig() const;
    const PerformanceConfig& getPerformanceConfig() const;
    const UserPreferences& getUserPreferences() const;
    
    /**
     * Configuration setters
     */
    void setDataSourceConfig(const DataSourceConfig& config);
    void setDisplayConfig(const DisplayConfig& config);
    void setThemeConfig(const ThemeConfig& config);
    void setLayoutConfig(const LayoutConfig& config);
    void setPerformanceConfig(const PerformanceConfig& config);
    void setUserPreferences(const UserPreferences& preferences);
    
    /**
     * Auto-save configuration
     */
    void enableAutoSave(bool enabled);
    
    /**
     * Theme management
     */
    std::vector<std::string> getAvailableThemes() const;
    bool applyTheme(const std::string& theme_name);
    
    /**
     * Get configuration file path
     */
    const std::string& getConfigFile() const { return config_file_; }

private:
    // Configuration storage
    DataSourceConfig data_source_config_;
    DisplayConfig display_config_;
    ThemeConfig theme_config_;
    LayoutConfig layout_config_;
    PerformanceConfig performance_config_;
    UserPreferences user_preferences_;
    
    // Configuration management
    std::string config_file_;
    bool auto_save_enabled_;
    uint32_t config_version_;
    
    // Private methods
    void initializeDefaults();
    void parseConfigValue(const std::string& section, const std::string& key, const std::string& value);
    ColorRGBA parseColor(const std::string& color_str) const;
    std::string colorToString(const ColorRGBA& color) const;
    std::string trim(const std::string& str) const;
};

} // namespace RenderEngine
} // namespace BTQuant