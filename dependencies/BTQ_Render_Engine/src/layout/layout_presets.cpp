#include "layout/layout_presets.hpp"

#include <filesystem>
#include <fstream>

namespace BTQuant {
namespace Layout {

LayoutPresetManager::LayoutPresetManager() {
  initialize_presets_directory();
  load_presets();
}

LayoutPresetManager::~LayoutPresetManager() = default;

void LayoutPresetManager::load_presets() {
  presets_.clear();
  load_builtin_presets();
  load_user_presets();
}

bool LayoutPresetManager::save_preset(const std::string& name, const std::string& description,
                                      const std::string& json_data, const std::string& category) {
  LayoutPreset preset;
  preset.name = name;
  preset.description = description;
  preset.json_data = json_data;
  preset.category = category;
  preset.is_builtin = false;
  preset.author = "User";
  preset.version = "1.0";

  if (!validate_preset(preset)) {
    return false;
  }

  // Check if preset already exists and update it
  for (auto& existing : presets_) {
    if (existing.name == name) {
      existing = preset;
      return save_preset_to_file(preset);
    }
  }

  presets_.push_back(preset);
  return save_preset_to_file(preset);
}

bool LayoutPresetManager::apply_preset(const std::string& preset_name) {
  for (const auto& preset : presets_) {
    if (preset.name == preset_name) {
      // In a real implementation, this would apply the json_data to the layout
      return true;
    }
  }
  return false;
}

bool LayoutPresetManager::delete_preset(const std::string& preset_name) {
  for (auto it = presets_.begin(); it != presets_.end(); ++it) {
    if (it->name == preset_name && !it->is_builtin) {
      // Delete the file
      std::string file_path = get_preset_file_path(preset_name);
      std::filesystem::remove(file_path);
      presets_.erase(it);
      return true;
    }
  }
  return false;
}

std::vector<LayoutPreset> LayoutPresetManager::get_all_presets() const { return presets_; }

std::vector<LayoutPreset> LayoutPresetManager::get_presets_by_category(
    const std::string& category) const {
  std::vector<LayoutPreset> result;
  for (const auto& preset : presets_) {
    if (preset.category == category) {
      result.push_back(preset);
    }
  }
  return result;
}

bool LayoutPresetManager::export_preset(const std::string& preset_name,
                                        const std::string& file_path) {
  for (const auto& preset : presets_) {
    if (preset.name == preset_name) {
      std::ofstream file(file_path);
      if (file.is_open()) {
        file << preset.json_data;
        return true;
      }
    }
  }
  return false;
}

bool LayoutPresetManager::import_preset(const std::string& file_path) {
  auto preset = load_preset_from_file(file_path);
  if (!preset.name.empty()) {
    presets_.push_back(preset);
    return true;
  }
  return false;
}

bool LayoutPresetManager::create_thumbnail(const std::string& /*preset_name*/,
                                           const std::string& /*thumbnail_path*/) {
  // Thumbnail creation would be implemented here
  return true;
}

bool LayoutPresetManager::validate_preset(const LayoutPreset& preset) const {
  return !preset.name.empty() && !preset.json_data.empty();
}

void LayoutPresetManager::initialize_presets_directory() {
  presets_directory_ = "./presets";
  std::filesystem::create_directories(presets_directory_);
}

void LayoutPresetManager::load_builtin_presets() {
  // Default trading layout
  LayoutPreset trading;
  trading.name = "Trading Default";
  trading.description = "Default layout for trading operations";
  trading.category = "Trading";
  trading.json_data = "{}";
  trading.is_builtin = true;
  trading.author = "BTQuant";
  trading.version = "1.0";
  presets_.push_back(trading);

  // Default analysis layout
  LayoutPreset analysis;
  analysis.name = "Analysis Default";
  analysis.description = "Default layout for market analysis";
  analysis.category = "Analysis";
  analysis.json_data = "{}";
  analysis.is_builtin = true;
  analysis.author = "BTQuant";
  analysis.version = "1.0";
  presets_.push_back(analysis);
}

void LayoutPresetManager::load_user_presets() {
  if (!std::filesystem::exists(presets_directory_)) {
    return;
  }

  for (const auto& entry : std::filesystem::directory_iterator(presets_directory_)) {
    if (entry.path().extension() == ".json") {
      auto preset = load_preset_from_file(entry.path().string());
      if (!preset.name.empty()) {
        presets_.push_back(preset);
      }
    }
  }
}

LayoutPreset LayoutPresetManager::load_preset_from_file(const std::string& file_path) {
  LayoutPreset preset;
  std::ifstream file(file_path);
  if (file.is_open()) {
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    preset.json_data = content;
    preset.name = std::filesystem::path(file_path).stem().string();
    preset.category = "Custom";
    preset.is_builtin = false;
  }
  return preset;
}

bool LayoutPresetManager::save_preset_to_file(const LayoutPreset& preset) {
  std::string file_path = get_preset_file_path(preset.name);
  std::ofstream file(file_path);
  if (file.is_open()) {
    file << preset.json_data;
    return true;
  }
  return false;
}

std::string LayoutPresetManager::get_preset_file_path(const std::string& preset_name) const {
  return presets_directory_ + "/" + preset_name + ".json";
}

}  // namespace Layout
}  // namespace BTQuant
