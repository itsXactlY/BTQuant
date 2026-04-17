#include "layout/layout_presets.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace BTQuant {
namespace Layout {

LayoutPresetManager::LayoutPresetManager() {
    presets_directory_ = "presets";
    initialize_presets_directory();
    load_presets();
}

LayoutPresetManager::~LayoutPresetManager() = default;

void LayoutPresetManager::load_presets() {
    load_builtin_presets();
    load_user_presets();
}

bool LayoutPresetManager::save_preset(const std::string& name,
                                       const std::string& description,
                                       const std::string& json_data,
                                       const std::string& category) {
    LayoutPreset preset;
    preset.name = name;
    preset.description = description;
    preset.category = category;
    preset.json_data = json_data;
    preset.is_builtin = false;

    // Remove existing preset with same name
    presets_.erase(
        std::remove_if(presets_.begin(), presets_.end(),
                        [&name](const LayoutPreset& p) { return p.name == name; }),
        presets_.end());

    presets_.push_back(preset);
    return save_preset_to_file(preset);
}

bool LayoutPresetManager::apply_preset(const std::string& preset_name) {
    for (const auto& preset : presets_) {
        if (preset.name == preset_name) {
            // Apply preset logic would go here
            return true;
        }
    }
    return false;
}

bool LayoutPresetManager::delete_preset(const std::string& preset_name) {
    auto it = std::find_if(presets_.begin(), presets_.end(),
                            [&preset_name](const LayoutPreset& p) {
                                return p.name == preset_name && !p.is_builtin;
                            });
    if (it != presets_.end()) {
        std::string file_path = get_preset_file_path(preset_name);
        if (fs::exists(file_path)) {
            fs::remove(file_path);
        }
        presets_.erase(it);
        return true;
    }
    return false;
}

std::vector<LayoutPreset> LayoutPresetManager::get_all_presets() const {
    return presets_;
}

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
    std::ifstream file(file_path);
    if (!file.is_open()) return false;

    std::string json_data((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    LayoutPreset preset;
    preset.name = fs::path(file_path).stem().string();
    preset.json_data = json_data;
    preset.is_builtin = false;
    preset.category = "Imported";

    presets_.push_back(preset);
    return save_preset_to_file(preset);
}

bool LayoutPresetManager::create_thumbnail(const std::string& preset_name,
                                            const std::string& thumbnail_path) {
    (void)preset_name;
    (void)thumbnail_path;
    return false; // Not implemented
}

bool LayoutPresetManager::validate_preset(const LayoutPreset& preset) const {
    return !preset.name.empty() && !preset.json_data.empty();
}

void LayoutPresetManager::initialize_presets_directory() {
    if (!fs::exists(presets_directory_)) {
        fs::create_directories(presets_directory_);
    }
}

void LayoutPresetManager::load_builtin_presets() {
    LayoutPreset default_preset;
    default_preset.name = "Default";
    default_preset.description = "Default trading layout";
    default_preset.category = "Trading";
    default_preset.is_builtin = true;
    default_preset.json_data = "{}";
    presets_.push_back(default_preset);
}

void LayoutPresetManager::load_user_presets() {
    if (!fs::exists(presets_directory_)) return;

    for (const auto& entry : fs::directory_iterator(presets_directory_)) {
        if (entry.path().extension() == ".json") {
            LayoutPreset preset = load_preset_from_file(entry.path().string());
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
        preset.name = fs::path(file_path).stem().string();
        preset.json_data = std::string((std::istreambuf_iterator<char>(file)),
                                        std::istreambuf_iterator<char>());
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

std::string LayoutPresetManager::get_preset_file_path(
    const std::string& preset_name) const {
    return presets_directory_ + "/" + preset_name + ".json";
}

}  // namespace Layout
}  // namespace BTQuant
