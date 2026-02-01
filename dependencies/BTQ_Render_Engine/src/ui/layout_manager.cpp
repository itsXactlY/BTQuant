#include "../include/ui/layout_manager.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "layout/dashboard_layout_manager.hpp"
#include "layout/layout_presets.hpp"

#ifdef HAS_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace BTQuant {
namespace UI {

// ============================================================================
// LayoutManager Implementation
// ============================================================================

LayoutManager::LayoutManager() {
    preset_manager_ = std::make_unique<Layout::LayoutPresetManager>();
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

    // Get current layout from dashboard layout manager
    std::string json_data = get_current_layout_json();
    
    if (json_data.empty()) {
        std::cerr << "Error: Could not retrieve current layout data" << std::endl;
        return false;
    }

    // Save the preset using the preset manager
    return preset_manager_->save_preset(preset_name, description, json_data, category);
}

// Load preset to restore layout
bool LayoutManager::load_preset_layout(const std::string& preset_name) {
    if (preset_name.empty()) {
        std::cerr << "Error: Preset name cannot be empty" << std::endl;
        return false;
    }

    // Find the preset
    auto all_presets = preset_manager_->get_all_presets();
    auto it = std::find_if(all_presets.begin(), all_presets.end(),
                          [&preset_name](const Layout::LayoutPreset& preset) {
                              return preset.name == preset_name;
                          });

    if (it == all_presets.end()) {
        std::cerr << "Error: Preset '" << preset_name << "' not found" << std::endl;
        return false;
    }

    // Apply the preset to the layout
    return apply_layout_from_preset(*it);
}

// Delete preset
bool LayoutManager::delete_preset(const std::string& preset_name) {
    if (preset_name.empty()) {
        std::cerr << "Error: Preset name cannot be empty" << std::endl;
        return false;
    }

    // Attempt to delete the preset using the preset manager
    return preset_manager_->delete_preset(preset_name);
}

std::vector<Layout::LayoutPreset> LayoutManager::get_all_presets() const {
    return preset_manager_->get_all_presets();
}

std::vector<Layout::LayoutPreset> LayoutManager::get_presets_by_category(const std::string& category) const {
    return preset_manager_->get_presets_by_category(category);
}

std::string LayoutManager::get_current_layout_json() {
#ifdef HAS_NLOHMANN_JSON
    // This would typically get the current layout state from the dashboard layout manager
    // For now, we'll create a sample JSON representation
    nlohmann::json layout_json;
    
    // In a real implementation, this would gather the current panel arrangements
    // from the active dashboard layout manager
    
    // Sample structure - in reality this would come from the actual layout manager
    layout_json["layout_name"] = "Current Layout";
    layout_json["theme"] = "dark";
    
    // This would be populated with actual panel data from the current layout
    nlohmann::json panels = nlohmann::json::array();
    
    // Example panel data (this would come from the actual layout manager)
    nlohmann::json panel;
    panel["panel_id"] = "sample_panel_1";
    panel["panel_name"] = "Sample Panel";
    panel["x"] = 0.0f;
    panel["y"] = 0.0f;
    panel["width"] = 0.5f;
    panel["height"] = 0.5f;
    panels.push_back(panel);
    
    layout_json["panels"] = panels;
    
    return layout_json.dump(4);
#else
    // Return empty JSON when JSON support is not available
    return "{}";
#endif
}

bool LayoutManager::apply_layout_from_preset(const Layout::LayoutPreset& preset) {
#ifdef HAS_NLOHMANN_JSON
    try {
        // Parse the JSON data from the preset
        auto layout_json = nlohmann::json::parse(preset.json_data);
        
        // Extract layout information
        std::string layout_name = layout_json.value("layout_name", "Unknown");
        std::string theme = layout_json.value("theme", "dark");
        
        // Process panels if they exist
        if (layout_json.contains("panels") && layout_json["panels"].is_array()) {
            auto panels = layout_json["panels"];
            
            // In a real implementation, this would update the actual layout manager
            // with the panel configurations from the preset
            
            std::cout << "Applying preset: " << preset.name << std::endl;
            std::cout << "Layout: " << layout_name << ", Theme: " << theme << std::endl;
            std::cout << "Number of panels: " << panels.size() << std::endl;
            
            // Here you would actually apply the layout to the UI
            // For now, just return true to indicate success
            return true;
        }
        
        std::cerr << "Preset '" << preset.name << "' contains no panels" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error applying preset '" << preset.name << "': " << e.what() << std::endl;
        return false;
    }
#else
    // Return false when JSON support is not available
    return false;
#endif
}

std::string LayoutManager::get_preset_file_path(const std::string& preset_name) const {
    return presets_directory_ + "/" + preset_name + ".json";
}

// Integration methods with dashboard layout manager
bool LayoutManager::save_current_dashboard_layout(const std::string& layout_name) {
    // This would delegate to the dashboard layout manager
    // For now, return true as a placeholder
    std::cout << "Saving current dashboard layout as: " << layout_name << std::endl;
    return true;
}

bool LayoutManager::load_dashboard_layout(const std::string& layout_name) {
    // This would delegate to the dashboard layout manager
    // For now, return true as a placeholder
    std::cout << "Loading dashboard layout: " << layout_name << std::endl;
    return true;
}

}  // namespace UI
}  // namespace BTQuant