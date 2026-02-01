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
    dashboard_layout_manager_ = std::make_unique<Layout::DashboardLayoutManager>();
    dashboard_layout_manager_->set_preset_manager(preset_manager_.get());
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
    if (!dashboard_layout_manager_) {
        std::cerr << "Error: Dashboard layout manager not initialized" << std::endl;
        return "{}";
    }

    // Get the current layout from the dashboard layout manager
    auto current_layout = dashboard_layout_manager_->get_current_layout();

    // If there's no current layout, create a default one
    if (current_layout.layout_name.empty()) {
        std::cout << "Warning: No current layout exists, creating default layout" << std::endl;

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
    }

    // Convert the current layout to JSON
    nlohmann::json layout_json;
    layout_json["layout_name"] = current_layout.layout_name;
    layout_json["theme"] = current_layout.theme;
    layout_json["created_time"] = current_layout.created_time;
    layout_json["grid_columns"] = current_layout.grid_columns;
    layout_json["grid_rows"] = current_layout.grid_rows;
    layout_json["min_width"] = current_layout.min_width;
    layout_json["min_height"] = current_layout.min_height;

    // Convert panels to JSON
    nlohmann::json panels = nlohmann::json::array();
    for (const auto& panel : current_layout.panels) {
        nlohmann::json panel_json;
        panel_json["panel_id"] = panel.panel_id;
        panel_json["panel_name"] = panel.panel_name;
        panel_json["type"] = static_cast<int>(panel.type);
        panel_json["x"] = panel.x;
        panel_json["y"] = panel.y;
        panel_json["width"] = panel.width;
        panel_json["height"] = panel.height;
        panel_json["is_docked"] = panel.is_docked;
        panel_json["dock_node_id"] = panel.dock_node_id;
        panel_json["symbol"] = panel.symbol;
        panel_json["timeframe"] = panel.timeframe;
        panel_json["is_collapsed"] = panel.is_collapsed;
        panel_json["is_focused"] = panel.is_focused;
        panels.push_back(panel_json);
    }

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
        if (!dashboard_layout_manager_) {
            std::cerr << "Error: Dashboard layout manager not initialized" << std::endl;
            return false;
        }

        // Parse the JSON data from the preset
        auto layout_json = nlohmann::json::parse(preset.json_data);

        // Extract layout information
        std::string layout_name = layout_json.value("layout_name", "Unknown");
        std::string theme = layout_json.value("theme", "dark");
        std::string created_time = layout_json.value("created_time", "");
        int grid_columns = layout_json.value("grid_columns", 6);
        int grid_rows = layout_json.value("grid_rows", 9);
        float min_width = layout_json.value("min_width", 200.0f);
        float min_height = layout_json.value("min_height", 150.0f);

        // Create a new layout based on the preset
        Layout::DashboardLayoutManager::DashboardLayout new_layout;
        new_layout.layout_name = layout_name;
        new_layout.theme = theme;
        new_layout.created_time = created_time;
        new_layout.grid_columns = grid_columns;
        new_layout.grid_rows = grid_rows;
        new_layout.min_width = min_width;
        new_layout.min_height = min_height;

        // Process panels if they exist
        if (layout_json.contains("panels") && layout_json["panels"].is_array()) {
            auto panels_json = layout_json["panels"];

            for (const auto& panel_json : panels_json) {
                Layout::DashboardLayoutManager::PanelLayout panel;
                panel.panel_id = panel_json.value("panel_id", "");
                panel.panel_name = panel_json.value("panel_name", "");

                // Convert integer back to PanelType enum
                int panel_type_int = panel_json.value("type", static_cast<int>(Layout::PanelType::CUSTOM));
                panel.type = static_cast<Layout::PanelType>(panel_type_int);

                panel.x = panel_json.value("x", 0.0f);
                panel.y = panel_json.value("y", 0.0f);
                panel.width = panel_json.value("width", 400.0f);
                panel.height = panel_json.value("height", 300.0f);
                panel.is_docked = panel_json.value("is_docked", false);
                panel.dock_node_id = panel_json.value("dock_node_id", 0);
                panel.symbol = panel_json.value("symbol", "");
                panel.timeframe = panel_json.value("timeframe", "");
                panel.is_collapsed = panel_json.value("is_collapsed", false);
                panel.is_focused = panel_json.value("is_focused", false);

                new_layout.panels.push_back(panel);
            }

            // Create or update the layout in the dashboard layout manager
            if (!dashboard_layout_manager_->has_layout(layout_name)) {
                // Create a new layout entry in the manager
                dashboard_layout_manager_->create_new_layout();
            }

            // Load the layout to make it current
            dashboard_layout_manager_->load_layout(layout_name);

            // Update the current layout with the preset data
            dashboard_layout_manager_->update_current_layout(new_layout);

            // Set grid dimensions
            dashboard_layout_manager_->set_grid_dimensions(grid_columns, grid_rows);

            std::cout << "Applied preset: " << preset.name << std::endl;
            std::cout << "Layout: " << layout_name << ", Theme: " << theme << std::endl;
            std::cout << "Number of panels: " << new_layout.panels.size() << std::endl;

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
    if (!dashboard_layout_manager_) {
        std::cerr << "Error: Dashboard layout manager not initialized" << std::endl;
        return false;
    }

    dashboard_layout_manager_->save_layout(layout_name);
    return true;
}

bool LayoutManager::load_dashboard_layout(const std::string& layout_name) {
    if (!dashboard_layout_manager_) {
        std::cerr << "Error: Dashboard layout manager not initialized" << std::endl;
        return false;
    }

    dashboard_layout_manager_->load_layout(layout_name);
    return true;
}

}  // namespace UI
}  // namespace BTQuant