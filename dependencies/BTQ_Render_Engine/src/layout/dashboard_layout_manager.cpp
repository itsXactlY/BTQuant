#include "layout/dashboard_layout_manager.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace BTQuant {
namespace Layout {

// ============================================================================
// DashboardLayoutManager Implementation
// ============================================================================

DashboardLayoutManager::DashboardLayoutManager() {
    initialize_layouts_directory();
    load_all_layouts();
}

DashboardLayoutManager::~DashboardLayoutManager() {
    // Save current layout before destruction
    if (!current_layout_name_.empty()) {
        save_layout(current_layout_name_);
    }
}

void DashboardLayoutManager::initialize_layouts_directory() {
    // Create layouts directory if it doesn't exist
    layouts_directory_ = "layouts";
    std::filesystem::create_directories(layouts_directory_);
}

void DashboardLayoutManager::load_all_layouts() {
    layouts_.clear();
    
    if (!std::filesystem::exists(layouts_directory_)) {
        return;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(layouts_directory_)) {
        if (entry.path().extension() == ".json") {
            std::string layout_name = entry.path().stem().string();
            auto layout = load_layout_from_file(layout_name);
            if (!layout.layout_name.empty()) {
                layouts_[layout_name] = layout;
            }
        }
    }
    
    std::cout << "Loaded " << layouts_.size() << " layouts" << std::endl;
}

void DashboardLayoutManager::create_new_layout() {
    DashboardLayout new_layout;
    new_layout.layout_name = "custom_" + std::to_string(layouts_.size());
    new_layout.theme = "dark";
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    new_layout.created_time = std::ctime(&time_t);
    new_layout.created_time.pop_back(); // Remove newline
    
    layouts_[new_layout.layout_name] = new_layout;
    current_layout_name_ = new_layout.layout_name;
    
    save_layout_to_file(new_layout);
    std::cout << "Created new layout: " << new_layout.layout_name << std::endl;
}

void DashboardLayoutManager::load_layout(const std::string& layout_name) {
    auto it = layouts_.find(layout_name);
    if (it != layouts_.end()) {
        current_layout_name_ = layout_name;
        std::cout << "Loaded layout: " << layout_name << std::endl;
    } else {
        std::cerr << "Layout not found: " << layout_name << std::endl;
    }
}

void DashboardLayoutManager::save_layout(const std::string& layout_name) {
    auto it = layouts_.find(layout_name);
    if (it != layouts_.end()) {
        save_layout_to_file(it->second);
        std::cout << "Saved layout: " << layout_name << std::endl;
    } else {
        std::cerr << "Cannot save non-existent layout: " << layout_name << std::endl;
    }
}

void DashboardLayoutManager::reset_layout() {
    if (!current_layout_name_.empty()) {
        load_layout("default");
    }
}

void DashboardLayoutManager::delete_layout(const std::string& layout_name) {
    auto it = layouts_.find(layout_name);
    if (it != layouts_.end()) {
        layouts_.erase(it);
        
        // Delete file
        std::string file_path = get_layout_file_path(layout_name);
        std::filesystem::remove(file_path);
        
        std::cout << "Deleted layout: " << layout_name << std::endl;
        
        // Reset to default if current layout was deleted
        if (current_layout_name_ == layout_name) {
            current_layout_name_.clear();
        }
    }
}

std::vector<std::string> DashboardLayoutManager::get_available_layouts() const {
    std::vector<std::string> layout_names;
    for (const auto& [name, layout] : layouts_) {
        layout_names.push_back(name);
    }
    return layout_names;
}

DashboardLayoutManager::DashboardLayout DashboardLayoutManager::get_current_layout() const {
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        return it->second;
    }
    return DashboardLayout{};
}

bool DashboardLayoutManager::has_layout(const std::string& layout_name) const {
    return layouts_.find(layout_name) != layouts_.end();
}

void DashboardLayoutManager::add_panel_to_layout(const PanelLayout& panel) {
    if (!current_layout_name_.empty()) {
        auto it = layouts_.find(current_layout_name_);
        if (it != layouts_.end()) {
            it->second.panels.push_back(panel);
            save_layout(current_layout_name_);
        }
    }
}

void DashboardLayoutManager::remove_panel_from_layout(const std::string& panel_id) {
    if (!current_layout_name_.empty()) {
        auto it = layouts_.find(current_layout_name_);
        if (it != layouts_.end()) {
            auto& panels = it->second.panels;
            panels.erase(
                std::remove_if(panels.begin(), panels.end(),
                    [&panel_id](const PanelLayout& p) { return p.panel_id == panel_id; }),
                panels.end()
            );
            save_layout(current_layout_name_);
        }
    }
}

void DashboardLayoutManager::update_panel_layout(const std::string& panel_id, const PanelLayout& layout) {
    if (!current_layout_name_.empty()) {
        auto it = layouts_.find(current_layout_name_);
        if (it != layouts_.end()) {
            auto& panels = it->second.panels;
            for (auto& panel : panels) {
                if (panel.panel_id == panel_id) {
                    panel = layout;
                    break;
                }
            }
            save_layout(current_layout_name_);
        }
    }
}

void DashboardLayoutManager::save_layout_to_file(const DashboardLayout& layout) {
    std::string file_path = get_layout_file_path(layout.layout_name);
    
    nlohmann::json j;
    j["layout_name"] = layout.layout_name;
    j["theme"] = layout.theme;
    j["created_time"] = layout.created_time;
    
    nlohmann::json panels_json = nlohmann::json::array();
    for (const auto& panel : layout.panels) {
        nlohmann::json panel_json;
        panel_json["panel_id"] = panel.panel_id;
        panel_json["panel_name"] = panel.panel_name;
        panel_json["x"] = panel.x;
        panel_json["y"] = panel.y;
        panel_json["width"] = panel.width;
        panel_json["height"] = panel.height;
        panel_json["is_docked"] = panel.is_docked;
        panel_json["dock_node_id"] = panel.dock_node_id;
        panels_json.push_back(panel_json);
    }
    j["panels"] = panels_json;
    
    std::ofstream file(file_path);
    if (file.is_open()) {
        file << j.dump(4);
        file.close();
    }
}

DashboardLayoutManager::DashboardLayout DashboardLayoutManager::load_layout_from_file(const std::string& layout_name) {
    DashboardLayout layout;
    std::string file_path = get_layout_file_path(layout_name);
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return layout;
    }
    
    try {
        nlohmann::json j;
        file >> j;
        
        layout.layout_name = j.value("layout_name", "");
        layout.theme = j.value("theme", "dark");
        layout.created_time = j.value("created_time", "");
        
        if (j.contains("panels") && j["panels"].is_array()) {
            for (const auto& panel_json : j["panels"]) {
                PanelLayout panel;
                panel.panel_id = panel_json.value("panel_id", "");
                panel.panel_name = panel_json.value("panel_name", "");
                panel.x = panel_json.value("x", 0.0f);
                panel.y = panel_json.value("y", 0.0f);
                panel.width = panel_json.value("width", 400.0f);
                panel.height = panel_json.value("height", 300.0f);
                panel.is_docked = panel_json.value("is_docked", false);
                panel.dock_node_id = panel_json.value("dock_node_id", 0);
                layout.panels.push_back(panel);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading layout " << layout_name << ": " << e.what() << std::endl;
    }
    
    file.close();
    return layout;
}

std::string DashboardLayoutManager::get_layout_file_path(const std::string& layout_name) const {
    return layouts_directory_ + "/" + layout_name + ".json";
}

} // namespace Layout
} // namespace BTQuant
