#include "../include/ui/layout_manager.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace BTQuant {
namespace UI {

// ============================================================================
// LayoutManager Implementation
// ============================================================================

LayoutManager::LayoutManager() {
    initialize_presets_directory();
}

LayoutManager::~LayoutManager() = default;

void LayoutManager::initialize_presets_directory() {
    // Create presets directory if it doesn't exist
    presets_directory_ = "presets";
    std::filesystem::create_directories(presets_directory_);
}

// Save current panel arrangement as named preset
bool LayoutManager::save_current_layout_as_preset(const std::string& preset_name,
                                                  const std::string& description,
                                                  const std::string& category) {
    if (preset_name.empty()) {
        std::cerr << "Error: Preset name cannot be empty" << std::endl;
        return false;
    }

    // Validate preset name to ensure it contains only valid characters
    for (char c : preset_name) {
        if (!std::isalnum(c) && c != '_' && c != '-' && c != ' ') {
            std::cerr << "Error: Preset name contains invalid characters" << std::endl;
            return false;
        }
    }

    // Get current layout from dashboard layout manager
    std::string json_data = get_current_layout_json();

    if (json_data.empty()) {
        std::cerr << "Error: Could not retrieve current layout data" << std::endl;
        return false;
    }

    // In the new architecture, layout saving is handled differently
    // This is a placeholder implementation
    std::cout << "Layout saved as preset: " << preset_name << std::endl;
    return true;
}

// Load preset to restore layout
bool LayoutManager::load_preset_layout(const std::string& preset_name) {
    if (preset_name.empty()) {
        std::cerr << "Error: Preset name cannot be empty" << std::endl;
        return false;
    }

    // In the new architecture, layout loading is handled differently
    // This is a placeholder implementation
    std::cout << "Loading preset: " << preset_name << std::endl;

    // Apply the preset to the layout
    // For now, just return true indicating success
    return true;
}

// Delete preset
bool LayoutManager::delete_preset(const std::string& preset_name) {
    if (preset_name.empty()) {
        std::cerr << "Error: Preset name cannot be empty" << std::endl;
        return false;
    }

    // In the new architecture, layout deletion is handled differently
    // This is a placeholder implementation
    std::cout << "Deleting preset: " << preset_name << std::endl;

    // For now, just return true indicating success
    return true;
}

bool LayoutManager::preset_exists(const std::string& preset_name) const {
    if (preset_name.empty()) {
        return false;
    }

    // In the new architecture, preset existence check is handled differently
    // This is a placeholder implementation
    return true; // Assume preset exists
}

std::vector<std::string> LayoutManager::get_all_presets() const {
    // Return a vector of preset names instead of LayoutPreset objects
    return {"Default Layout", "Trading Layout", "Analysis Layout"};
}

std::vector<std::string> LayoutManager::get_presets_by_category(const std::string& category) const {
    // Return presets by category
    return {"Default Layout", "Trading Layout", "Analysis Layout"};
}

std::string LayoutManager::get_current_layout_json() {
#ifdef HAS_NLOHMANN_JSON
    // Create a default layout
    nlohmann::json layout_json;
    layout_json["layout_name"] = "default_layout";
    layout_json["theme"] = "dark";
    layout_json["created_time"] = "unknown";
    layout_json["grid_columns"] = 6;
    layout_json["grid_rows"] = 9;
    layout_json["min_width"] = 200.0f;
    layout_json["min_height"] = 150.0f;

    // Create empty panels array
    layout_json["panels"] = nlohmann::json::array();

    return layout_json.dump(4);
#else
    // Return empty JSON when JSON support is not available
    return "{}";
#endif
}

bool LayoutManager::apply_layout_from_preset(const std::string& preset_name) {
    // Placeholder implementation - in a real scenario, this would connect to the actual layout system
    std::cout << "Applied preset: " << preset_name << std::endl;
    return true;
}

std::string LayoutManager::get_preset_file_path(const std::string& preset_name) const {
    return presets_directory_ + "/" + preset_name + ".json";
}


// Quick-save functionality for F5-F8 hotkeys
bool LayoutManager::quick_save_layout(int slot_num) {
    if (slot_num < 1 || slot_num > 4) {
        std::cerr << "Error: Invalid quick save slot number: " << slot_num << std::endl;
        return false;
    }

    std::string preset_name = get_quick_save_name(slot_num);
    std::string description = "Quick save layout slot " + std::to_string(slot_num);

    std::cout << "[Layout] Saving to Quick Save " << slot_num << " (" << preset_name << ")" << std::endl;

    // Save the current layout as a preset
    bool success = save_current_layout_as_preset(preset_name, description, "Quick Save");

    return success;
}

bool LayoutManager::quick_load_layout(int slot_num) {
    if (slot_num < 1 || slot_num > 4) {
        std::cerr << "Error: Invalid quick save slot number: " << slot_num << std::endl;
        return false;
    }

    std::string preset_name = get_quick_save_name(slot_num);

    std::cout << "[Layout] Loading Quick Save " << slot_num << " (" << preset_name << ")" << std::endl;

    // Load the preset
    bool success = load_preset_layout(preset_name);

    if (success) {
        // Set the active quick slot to the loaded slot
        set_active_quick_slot(slot_num);
    } else {
        // If loading failed, set active slot to 0 (no active quick slot)
        set_active_quick_slot(0);
    }

    return success;
}

std::string LayoutManager::get_quick_save_name(int slot_num) const {
    switch (slot_num) {
        case 1: return "Quick Save 1";
        case 2: return "Quick Save 2";
        case 3: return "Quick Save 3";
        case 4: return "Quick Save 4";
        default: return "Quick Save " + std::to_string(slot_num);
    }
}

std::string LayoutManager::get_quick_save_filename(int slot_num) const {
    return "quick_save_" + std::to_string(slot_num) + ".json";
}

std::string LayoutManager::get_active_layout_name() const {
    return "Default Layout";
}

void LayoutManager::set_active_quick_slot(int slot_num) {
    if (slot_num >= 0 && slot_num <= 4) {
        active_quick_slot_ = slot_num;
    }
}

int LayoutManager::get_active_quick_slot() const {
    return active_quick_slot_;
}

}  // namespace UI
}  // namespace BTQuant