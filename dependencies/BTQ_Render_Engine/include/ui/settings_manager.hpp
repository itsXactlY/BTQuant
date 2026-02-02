#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

#include "imgui.h"

namespace BTQuant {
namespace UI {

// ============================================================================
// Unified Settings System
// ============================================================================

// Enum for setting types
enum class SettingType {
    BOOLEAN,
    INTEGER,
    FLOAT,
    STRING,
    COLOR,
    ENUM
};

// Enum for setting categories
enum class SettingCategory {
    APPEARANCE,
    DATA,
    PERFORMANCE,
    ALERTS,
    KEYBOARD_SHORTCUTS
};

// Structure to hold individual setting information
struct SettingInfo {
    std::string key;
    std::string display_name;
    std::string description;
    SettingType type;
    SettingCategory category;
    
    // Value storage
    union {
        bool bool_value;
        int int_value;
        float float_value;
    };
    
    std::string string_value;
    ImVec4 color_value;
    
    // For enum values
    std::vector<std::string> enum_options;
    int enum_selected_index;
    
    // Validation and range limits
    int min_int = 0;
    int max_int = 100;
    float min_float = 0.0f;
    float max_float = 100.0f;
    
    // Callback for when setting changes
    std::function<void()> on_change_callback;
};

class SettingsManager {
public:
    static SettingsManager& getInstance() {
        static SettingsManager instance;
        return instance;
    }

    SettingsManager(const SettingsManager&) = delete;
    SettingsManager& operator=(const SettingsManager&) = delete;

    // Initialize the settings system
    void initialize();

    // Register a new setting
    bool register_setting(const SettingInfo& setting);

    // Update an existing setting
    bool update_setting(const std::string& key, const SettingInfo& setting);

    // Get a setting by key
    SettingInfo* get_setting(const std::string& key);

    // Get all settings in a category
    std::vector<SettingInfo*> get_settings_by_category(SettingCategory category);

    // Get all settings
    std::vector<SettingInfo*> get_all_settings();

    // Set boolean value
    bool set_bool(const std::string& key, bool value);

    // Set integer value
    bool set_int(const std::string& key, int value);

    // Set float value
    bool set_float(const std::string& key, float value);

    // Set string value
    bool set_string(const std::string& key, const std::string& value);

    // Set color value
    bool set_color(const std::string& key, const ImVec4& value);

    // Set enum value
    bool set_enum(const std::string& key, int index);

    // Get boolean value
    bool get_bool(const std::string& key, bool default_value = false) const;

    // Get integer value
    int get_int(const std::string& key, int default_value = 0) const;

    // Get float value
    float get_float(const std::string& key, float default_value = 0.0f) const;

    // Get string value
    std::string get_string(const std::string& key, const std::string& default_value = "") const;

    // Get color value
    ImVec4 get_color(const std::string& key, const ImVec4& default_value = ImVec4(0.5f, 0.5f, 0.5f, 1.0f)) const;

    // Get enum value
    int get_enum(const std::string& key, int default_index = 0) const;

    // Save settings to file
    bool save_settings(const std::string& file_path) const;

    // Load settings from file
    bool load_settings(const std::string& file_path);

    // Reset all settings to defaults
    void reset_to_defaults();

    // Render the settings UI
    void render_settings_ui();

    // Get category name as string
    std::string get_category_name(SettingCategory category) const;

private:
    SettingsManager();   // Private constructor for singleton
    ~SettingsManager();  // Private destructor for singleton

    std::unordered_map<std::string, SettingInfo> settings_;
    std::string settings_directory_;

    void initialize_default_settings();
    void initialize_appearance_settings();
    void initialize_data_settings();
    void initialize_performance_settings();
    void initialize_alert_settings();
    void initialize_keyboard_shortcut_settings();
    void render_category_settings(SettingCategory category);
    void render_setting_control(SettingInfo& setting);
    void trigger_on_change_callback(const std::string& key);
};

}  // namespace UI
}  // namespace BTQuant