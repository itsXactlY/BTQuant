#include "../include/ui/settings_manager.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../include/ui/alert_settings.hpp"
#include "../include/ui/appearance_settings.hpp"
#include "../include/ui/data_settings.hpp"
#include "../include/ui/performance_settings.hpp"

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace BTQuant {
namespace UI {

// ============================================================================
// SettingsManager Implementation
// ============================================================================

SettingsManager::SettingsManager() {
  initialize_default_settings();

  // Load settings from file on startup
  load_settings(settings_file_path_);
}

SettingsManager::~SettingsManager() = default;

void SettingsManager::initialize() {
  // Initialization logic if needed
}

void SettingsManager::initialize_default_settings() {
  // Initialize all default settings organized by category
  initialize_appearance_settings();
  initialize_data_settings();
  initialize_performance_settings();
  initialize_alert_settings();
  initialize_keyboard_shortcut_settings();
}

void SettingsManager::initialize_appearance_settings() {
  // Create AppearanceSettings instance to handle all appearance-related settings
  appearance_settings_ = std::make_unique<AppearanceSettings>(*this);
}

void SettingsManager::initialize_data_settings() {
  // Auto-refresh interval
  SettingInfo refresh_interval_setting;
  refresh_interval_setting.key = "data.refresh_interval";
  refresh_interval_setting.display_name = "Refresh Interval";
  refresh_interval_setting.description = "Interval in seconds for data refresh";
  refresh_interval_setting.type = SettingType::INTEGER;
  refresh_interval_setting.category = SettingCategory::DATA;
  refresh_interval_setting.int_value = 5;
  refresh_interval_setting.min_int = 1;
  refresh_interval_setting.max_int = 60;
  register_setting(refresh_interval_setting);

  // Max data points
  SettingInfo max_data_points_setting;
  max_data_points_setting.key = "data.max_data_points";
  max_data_points_setting.display_name = "Max Data Points";
  max_data_points_setting.description = "Maximum number of data points to store";
  max_data_points_setting.type = SettingType::INTEGER;
  max_data_points_setting.category = SettingCategory::DATA;
  max_data_points_setting.int_value = 10000;
  max_data_points_setting.min_int = 1000;
  max_data_points_setting.max_int = 1000000;
  register_setting(max_data_points_setting);

  // Data precision
  SettingInfo data_precision_setting;
  data_precision_setting.key = "data.precision";
  data_precision_setting.display_name = "Data Precision";
  data_precision_setting.description = "Decimal places for numerical data display";
  data_precision_setting.type = SettingType::INTEGER;
  data_precision_setting.category = SettingCategory::DATA;
  data_precision_setting.int_value = 2;
  data_precision_setting.min_int = 0;
  data_precision_setting.max_int = 8;
  register_setting(data_precision_setting);

  // Enable data compression
  SettingInfo enable_compression_setting;
  enable_compression_setting.key = "data.enable_compression";
  enable_compression_setting.display_name = "Enable Compression";
  enable_compression_setting.description = "Compress data to reduce memory usage";
  enable_compression_setting.type = SettingType::BOOLEAN;
  enable_compression_setting.category = SettingCategory::DATA;
  enable_compression_setting.bool_value = false;
  register_setting(enable_compression_setting);

  // Cache size
  SettingInfo cache_size_setting;
  cache_size_setting.key = "data.cache_size";
  cache_size_setting.display_name = "Cache Size (MB)";
  cache_size_setting.description = "Size of the data cache in megabytes";
  cache_size_setting.type = SettingType::INTEGER;
  cache_size_setting.category = SettingCategory::DATA;
  cache_size_setting.int_value = 512;
  cache_size_setting.min_int = 64;
  cache_size_setting.max_int = 4096;
  register_setting(cache_size_setting);

  // Create DataSettings instance to handle additional data-related settings
  data_settings_ = std::make_unique<DataSettings>(*this);
}

void SettingsManager::initialize_performance_settings() {
  // Create PerformanceSettings instance to handle all performance-related settings
  performance_settings_ = std::make_unique<PerformanceSettings>(*this);
}

void SettingsManager::initialize_alert_settings() {
  // Create AlertSettings instance to handle all alert-related settings
  alert_settings_ = std::make_unique<AlertSettings>(*this);
}

void SettingsManager::initialize_keyboard_shortcut_settings() {
  // Toggle fullscreen
  SettingInfo toggle_fullscreen_setting;
  toggle_fullscreen_setting.key = "keyboard.toggle_fullscreen";
  toggle_fullscreen_setting.display_name = "Toggle Fullscreen";
  toggle_fullscreen_setting.description = "Keyboard shortcut for toggling fullscreen";
  toggle_fullscreen_setting.type = SettingType::STRING;
  toggle_fullscreen_setting.category = SettingCategory::KEYBOARD_SHORTCUTS;
  toggle_fullscreen_setting.string_value = "F11";
  register_setting(toggle_fullscreen_setting);

  // Save layout
  SettingInfo save_layout_setting;
  save_layout_setting.key = "keyboard.save_layout";
  save_layout_setting.display_name = "Save Layout";
  save_layout_setting.description = "Keyboard shortcut for saving current layout";
  save_layout_setting.type = SettingType::STRING;
  save_layout_setting.category = SettingCategory::KEYBOARD_SHORTCUTS;
  save_layout_setting.string_value = "Ctrl+S";
  register_setting(save_layout_setting);

  // Load layout
  SettingInfo load_layout_setting;
  load_layout_setting.key = "keyboard.load_layout";
  load_layout_setting.display_name = "Load Layout";
  load_layout_setting.description = "Keyboard shortcut for loading a layout";
  load_layout_setting.type = SettingType::STRING;
  load_layout_setting.category = SettingCategory::KEYBOARD_SHORTCUTS;
  load_layout_setting.string_value = "Ctrl+L";
  register_setting(load_layout_setting);

  // Take screenshot
  SettingInfo take_screenshot_setting;
  take_screenshot_setting.key = "keyboard.take_screenshot";
  take_screenshot_setting.display_name = "Take Screenshot";
  take_screenshot_setting.description = "Keyboard shortcut for taking screenshots";
  take_screenshot_setting.type = SettingType::STRING;
  take_screenshot_setting.category = SettingCategory::KEYBOARD_SHORTCUTS;
  take_screenshot_setting.string_value = "Ctrl+Shift+S";
  register_setting(take_screenshot_setting);

  // Open settings
  SettingInfo open_settings_setting;
  open_settings_setting.key = "keyboard.open_settings";
  open_settings_setting.display_name = "Open Settings";
  open_settings_setting.description = "Keyboard shortcut for opening settings dialog";
  open_settings_setting.type = SettingType::STRING;
  open_settings_setting.category = SettingCategory::KEYBOARD_SHORTCUTS;
  open_settings_setting.string_value = "Ctrl+,";
  register_setting(open_settings_setting);
}

bool SettingsManager::register_setting(const SettingInfo& setting) {
  if (setting.key.empty()) {
    return false;
  }

  settings_[setting.key] = setting;
  return true;
}

bool SettingsManager::update_setting(const std::string& key, const SettingInfo& setting) {
  auto it = settings_.find(key);
  if (it != settings_.end()) {
    settings_[key] = setting;
    return true;
  }
  return false;
}

SettingInfo* SettingsManager::get_setting(const std::string& key) {
  auto it = settings_.find(key);
  if (it != settings_.end()) {
    return &(it->second);
  }
  return nullptr;
}

std::vector<SettingInfo*> SettingsManager::get_settings_by_category(SettingCategory category) {
  std::vector<SettingInfo*> result;
  for (auto& pair : settings_) {
    if (pair.second.category == category) {
      result.push_back(&(pair.second));
    }
  }
  return result;
}

std::vector<SettingInfo*> SettingsManager::get_all_settings() {
  std::vector<SettingInfo*> result;
  for (auto& pair : settings_) {
    result.push_back(&(pair.second));
  }
  return result;
}

bool SettingsManager::set_bool(const std::string& key, bool value) {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::BOOLEAN) {
    it->second.bool_value = value;
    trigger_on_change_callback(key);
    return true;
  }
  return false;
}

bool SettingsManager::set_int(const std::string& key, int value) {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::INTEGER) {
    // Clamp value to range if applicable
    if (value < it->second.min_int) value = it->second.min_int;
    if (value > it->second.max_int) value = it->second.max_int;

    it->second.int_value = value;
    trigger_on_change_callback(key);
    return true;
  }
  return false;
}

bool SettingsManager::set_float(const std::string& key, float value) {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::FLOAT) {
    // Clamp value to range if applicable
    if (value < it->second.min_float) value = it->second.min_float;
    if (value > it->second.max_float) value = it->second.max_float;

    it->second.float_value = value;
    trigger_on_change_callback(key);
    return true;
  }
  return false;
}

bool SettingsManager::set_string(const std::string& key, const std::string& value) {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::STRING) {
    it->second.string_value = value;
    trigger_on_change_callback(key);
    return true;
  }
  return false;
}

bool SettingsManager::set_color(const std::string& key, const ImVec4& value) {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::COLOR) {
    it->second.color_value = value;
    trigger_on_change_callback(key);
    return true;
  }
  return false;
}

bool SettingsManager::set_enum(const std::string& key, int index) {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::ENUM) {
    if (index >= 0 && index < static_cast<int>(it->second.enum_options.size())) {
      it->second.enum_selected_index = index;
      trigger_on_change_callback(key);
      return true;
    }
  }
  return false;
}

bool SettingsManager::get_bool(const std::string& key, bool default_value) const {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::BOOLEAN) {
    return it->second.bool_value;
  }
  return default_value;
}

int SettingsManager::get_int(const std::string& key, int default_value) const {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::INTEGER) {
    return it->second.int_value;
  }
  return default_value;
}

float SettingsManager::get_float(const std::string& key, float default_value) const {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::FLOAT) {
    return it->second.float_value;
  }
  return default_value;
}

std::string SettingsManager::get_string(const std::string& key,
                                        const std::string& default_value) const {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::STRING) {
    return it->second.string_value;
  }
  return default_value;
}

ImVec4 SettingsManager::get_color(const std::string& key, const ImVec4& default_value) const {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::COLOR) {
    return it->second.color_value;
  }
  return default_value;
}

int SettingsManager::get_enum(const std::string& key, int default_index) const {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.type == SettingType::ENUM) {
    return it->second.enum_selected_index;
  }
  return default_index;
}

bool SettingsManager::save_settings(const std::string& file_path) const {
#ifdef HAS_NLOHMANN_JSON
  try {
    nlohmann::json settings_json;

    for (const auto& pair : settings_) {
      const auto& key = pair.first;
      const auto& setting = pair.second;

      nlohmann::json setting_json;
      setting_json["key"] = setting.key;
      setting_json["display_name"] = setting.display_name;
      setting_json["description"] = setting.description;
      setting_json["type"] = static_cast<int>(setting.type);
      setting_json["category"] = static_cast<int>(setting.category);

      // Store the value based on type
      switch (setting.type) {
        case SettingType::BOOLEAN:
          setting_json["value"] = setting.bool_value;
          break;
        case SettingType::INTEGER:
          setting_json["value"] = setting.int_value;
          setting_json["min_int"] = setting.min_int;
          setting_json["max_int"] = setting.max_int;
          break;
        case SettingType::FLOAT:
          setting_json["value"] = setting.float_value;
          setting_json["min_float"] = setting.min_float;
          setting_json["max_float"] = setting.max_float;
          break;
        case SettingType::STRING:
          setting_json["value"] = setting.string_value;
          break;
        case SettingType::COLOR:
          setting_json["value"] = {setting.color_value.x, setting.color_value.y,
                                   setting.color_value.z, setting.color_value.w};
          break;
        case SettingType::ENUM:
          setting_json["value"] = setting.enum_selected_index;
          setting_json["options"] = setting.enum_options;
          break;
      }

      settings_json[key] = setting_json;
    }

    // Add version information
    settings_json["version"] = current_version_;

    std::ofstream file(file_path);
    if (file.is_open()) {
      file << settings_json.dump(4);
      file.close();
      std::cout << "Settings saved to: " << file_path << std::endl;
      return true;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error saving settings: " << e.what() << std::endl;
  }
#else
  std::cerr << "JSON support not available for saving settings" << std::endl;
#endif
  return false;
}

bool SettingsManager::load_settings(const std::string& file_path) {
#ifdef HAS_NLOHMANN_JSON
  try {
    std::ifstream file(file_path);
    if (!file.is_open()) {
      std::cerr << "Could not open settings file: " << file_path << std::endl;
      return false;
    }

    nlohmann::json settings_json;
    file >> settings_json;
    file.close();

    // Check if version information exists and perform migration if needed
    std::string saved_version = "";
    if (settings_json.contains("version")) {
      saved_version = settings_json["version"].get<std::string>();

      // Perform migration if the saved version is different from current version
      if (saved_version != current_version_) {
        std::cout << "Detected settings version " << saved_version << ", performing migration to "
                  << current_version_ << std::endl;
        migrate_settings(saved_version);
      }
    }

    for (auto& pair : settings_json.items()) {
      std::string key = pair.key();
      auto setting_json = pair.value();

      // Skip the version entry as it's not a setting
      if (key == "version") {
        continue;
      }

      auto it = settings_.find(key);
      if (it != settings_.end()) {
        // Update the setting value based on type
        SettingType type = static_cast<SettingType>(setting_json["type"].get<int>());

        switch (type) {
          case SettingType::BOOLEAN:
            if (setting_json.contains("value")) {
              it->second.bool_value = setting_json["value"].get<bool>();
            }
            break;
          case SettingType::INTEGER:
            if (setting_json.contains("value")) {
              it->second.int_value = setting_json["value"].get<int>();
            }
            if (setting_json.contains("min_int")) {
              it->second.min_int = setting_json["min_int"].get<int>();
            }
            if (setting_json.contains("max_int")) {
              it->second.max_int = setting_json["max_int"].get<int>();
            }
            break;
          case SettingType::FLOAT:
            if (setting_json.contains("value")) {
              it->second.float_value = setting_json["value"].get<float>();
            }
            if (setting_json.contains("min_float")) {
              it->second.min_float = setting_json["min_float"].get<float>();
            }
            if (setting_json.contains("max_float")) {
              it->second.max_float = setting_json["max_float"].get<float>();
            }
            break;
          case SettingType::STRING:
            if (setting_json.contains("value")) {
              it->second.string_value = setting_json["value"].get<std::string>();
            }
            break;
          case SettingType::COLOR:
            if (setting_json.contains("value")) {
              auto color_array = setting_json["value"].get<std::vector<float>>();
              if (color_array.size() >= 4) {
                it->second.color_value.x = color_array[0];
                it->second.color_value.y = color_array[1];
                it->second.color_value.z = color_array[2];
                it->second.color_value.w = color_array[3];
              }
            }
            break;
          case SettingType::ENUM:
            if (setting_json.contains("value")) {
              it->second.enum_selected_index = setting_json["value"].get<int>();
            }
            if (setting_json.contains("options")) {
              it->second.enum_options = setting_json["options"].get<std::vector<std::string>>();
            }
            break;
        }
      }
    }

    std::cout << "Settings loaded from: " << file_path << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Error loading settings: " << e.what() << std::endl;
  }
#else
  std::cerr << "JSON support not available for loading settings" << std::endl;
#endif
  return false;
}

void SettingsManager::reset_to_defaults() {
  // Reinitialize all settings to their default values
  initialize_default_settings();
}

void SettingsManager::render_settings_ui() {
  // Check if we're in a valid ImGui frame scope to prevent assertion errors
  // We can check this by attempting to get the current context and checking if it's valid
  ImGuiContext* g = ImGui::GetCurrentContext();
  if (g == nullptr) {
    // If there's no valid ImGui context, skip rendering this frame
    return;
  }

  // In newer versions of ImGui, we can't directly access WithinFrameScope
  // Instead, we'll just check if the context is valid and proceed with rendering
  // If we're not in a proper frame, ImGui will handle the error internally

  ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

  // Category selection
  static SettingCategory selected_category = SettingCategory::APPEARANCE;

  ImGui::Text("Settings Categories:");
  ImGui::Separator();

  // Display category buttons
  if (ImGui::Button("Appearance")) selected_category = SettingCategory::APPEARANCE;
  ImGui::SameLine();
  if (ImGui::Button("Data")) selected_category = SettingCategory::DATA;
  ImGui::SameLine();
  if (ImGui::Button("Performance")) selected_category = SettingCategory::PERFORMANCE;
  ImGui::SameLine();
  if (ImGui::Button("Alerts")) selected_category = SettingCategory::ALERTS;
  ImGui::SameLine();
  if (ImGui::Button("Keyboard")) selected_category = SettingCategory::KEYBOARD_SHORTCUTS;

  ImGui::Separator();

  // Render settings for the selected category
  render_category_settings(selected_category);

  ImGui::Separator();

  // Action buttons
  if (ImGui::Button("Save Settings")) {
    save_settings(settings_file_path_);
  }
  ImGui::SameLine();
  if (ImGui::Button("Load Settings")) {
    load_settings(settings_file_path_);
  }
  ImGui::SameLine();
  if (ImGui::Button("Reset to Defaults")) {
    reset_to_defaults();
  }

  ImGui::End();
}

void SettingsManager::render_category_settings(SettingCategory category) {
  auto settings_list = get_settings_by_category(category);

  ImGui::Text("%s Settings", get_category_name(category).c_str());
  ImGui::Separator();

  for (auto* setting : settings_list) {
    render_setting_control(*setting);
  }
}

void SettingsManager::render_setting_control(SettingInfo& setting) {
  ImGui::PushID(setting.key.c_str());  // Use the setting key as ID to ensure uniqueness

  ImGui::Text("%s", setting.display_name.c_str());
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");  // Tooltip indicator
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(setting.description.c_str());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }

  switch (setting.type) {
    case SettingType::BOOLEAN:
      ImGui::Checkbox("##checkbox", &setting.bool_value);
      break;

    case SettingType::INTEGER:
      ImGui::SliderInt("##slider", &setting.int_value, setting.min_int, setting.max_int);
      break;

    case SettingType::FLOAT:
      ImGui::SliderFloat("##slider", &setting.float_value, setting.min_float, setting.max_float);
      break;

    case SettingType::STRING: {
      static char buffer[256];
      strcpy(buffer, setting.string_value.c_str());
      if (ImGui::InputText("##input", buffer, sizeof(buffer))) {
        setting.string_value = std::string(buffer);
      }
    } break;

    case SettingType::COLOR:
      ImGui::ColorEdit4("##color", &setting.color_value.x);
      break;

    case SettingType::ENUM:
      if (!setting.enum_options.empty()) {
        // Create combo box with options
        const char* preview_value = setting.enum_options[setting.enum_selected_index].c_str();

        if (ImGui::BeginCombo("##combo", preview_value)) {
          for (int n = 0; n < static_cast<int>(setting.enum_options.size()); n++) {
            bool is_selected = (setting.enum_selected_index == n);
            if (ImGui::Selectable(setting.enum_options[n].c_str(), is_selected)) {
              setting.enum_selected_index = n;
            }
            if (is_selected) {
              ImGui::SetItemDefaultFocus();
            }
          }
          ImGui::EndCombo();
        }
      }
      break;
  }

  ImGui::PopID();  // Pop the ID we pushed earlier
}

std::string SettingsManager::get_category_name(SettingCategory category) const {
  switch (category) {
    case SettingCategory::APPEARANCE:
      return "Appearance";
    case SettingCategory::DATA:
      return "Data";
    case SettingCategory::PERFORMANCE:
      return "Performance";
    case SettingCategory::ALERTS:
      return "Alerts";
    case SettingCategory::KEYBOARD_SHORTCUTS:
      return "Keyboard";
    default:
      return "Unknown";
  }
}

void SettingsManager::trigger_on_change_callback(const std::string& key) {
  auto it = settings_.find(key);
  if (it != settings_.end() && it->second.on_change_callback) {
    it->second.on_change_callback();
  }

  // Auto-save settings if enabled
  if (auto_save_enabled_) {
    auto_save_settings();
  }
}

void SettingsManager::auto_save_settings() {
  // Ensure the config directory exists
  std::filesystem::path config_dir = std::filesystem::path(settings_file_path_).parent_path();
  if (!config_dir.empty()) {
    std::filesystem::create_directories(config_dir);
  }

  // Save settings to file
  save_settings(settings_file_path_);
}

bool SettingsManager::migrate_settings(const std::string& old_version) {
  // Parse version numbers
  std::istringstream old_stream(old_version);
  std::istringstream current_stream(current_version_);
  std::string old_part, current_part;
  std::vector<int> old_ver, current_ver;

  // Split version by dots
  while (std::getline(old_stream, old_part, '.')) {
    try {
      old_ver.push_back(std::stoi(old_part));
    } catch (...) {
      old_ver.push_back(0);
    }
  }

  while (std::getline(current_stream, current_part, '.')) {
    try {
      current_ver.push_back(std::stoi(current_part));
    } catch (...) {
      current_ver.push_back(0);
    }
  }

  // Perform migrations based on version differences
  if (old_ver.size() >= 3 && current_ver.size() >= 3) {
    // Migration from version 0.x.x to 1.x.x
    if (old_ver[0] < 1 && current_ver[0] >= 1) {
      // Example migration: adjust any deprecated settings
      // For now, we'll just log the migration
      std::cout << "Migrating settings from version " << old_version << " to " << current_version_
                << std::endl;

      // Add any specific migration logic here if needed
      // For example, if a setting key changed between versions:
      // if (settings_.count("old.setting.key") > 0 && settings_.count("new.setting.key") == 0) {
      //     settings_["new.setting.key"] = settings_["old.setting.key"];
      //     settings_.erase("old.setting.key");
      // }
    }

    // Add more version-specific migrations as needed
    // if (old_ver[0] == 1 && old_ver[1] < 1 && current_ver[0] == 1 && current_ver[1] >= 1) {
    //     // Migration from 1.0.x to 1.1.x
    // }
  }

  return true;
}

std::string SettingsManager::get_saved_version() const {
#ifdef HAS_NLOHMANN_JSON
  try {
    std::ifstream file(settings_file_path_);
    if (!file.is_open()) {
      return "";  // File doesn't exist yet
    }

    nlohmann::json settings_json;
    file >> settings_json;
    file.close();

    if (settings_json.contains("version")) {
      return settings_json["version"].get<std::string>();
    }
  } catch (const std::exception& e) {
    std::cerr << "Error reading version from settings: " << e.what() << std::endl;
  }
#endif
  return "";
}

void SettingsManager::set_saved_version(const std::string& version) {
#ifdef HAS_NLOHMANN_JSON
  try {
    nlohmann::json settings_json;

    for (const auto& pair : settings_) {
      const auto& key = pair.first;
      const auto& setting = pair.second;

      nlohmann::json setting_json;
      setting_json["key"] = setting.key;
      setting_json["display_name"] = setting.display_name;
      setting_json["description"] = setting.description;
      setting_json["type"] = static_cast<int>(setting.type);
      setting_json["category"] = static_cast<int>(setting.category);

      // Store the value based on type
      switch (setting.type) {
        case SettingType::BOOLEAN:
          setting_json["value"] = setting.bool_value;
          break;
        case SettingType::INTEGER:
          setting_json["value"] = setting.int_value;
          setting_json["min_int"] = setting.min_int;
          setting_json["max_int"] = setting.max_int;
          break;
        case SettingType::FLOAT:
          setting_json["value"] = setting.float_value;
          setting_json["min_float"] = setting.min_float;
          setting_json["max_float"] = setting.max_float;
          break;
        case SettingType::STRING:
          setting_json["value"] = setting.string_value;
          break;
        case SettingType::COLOR:
          setting_json["value"] = {setting.color_value.x, setting.color_value.y,
                                   setting.color_value.z, setting.color_value.w};
          break;
        case SettingType::ENUM:
          setting_json["value"] = setting.enum_selected_index;
          setting_json["options"] = setting.enum_options;
          break;
      }

      settings_json[key] = setting_json;
    }

    // Add version information
    settings_json["version"] = version;

    std::ofstream file(settings_file_path_);
    if (file.is_open()) {
      file << settings_json.dump(4);
      file.close();
      std::cout << "Settings saved with version " << version << " to: " << settings_file_path_
                << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error saving settings with version: " << e.what() << std::endl;
  }
#else
  std::cerr << "JSON support not available for saving settings with version" << std::endl;
#endif
}

}  // namespace UI
}  // namespace BTQuant