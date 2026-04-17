#include "layout/dashboard_layout_manager.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace BTQuant {
namespace Layout {

DashboardLayoutManager::DashboardLayoutManager() {
    layouts_directory_ = "layouts";
    initialize_layouts_directory();
    load_all_layouts();
}

DashboardLayoutManager::~DashboardLayoutManager() = default;

void DashboardLayoutManager::create_new_layout() {
    DashboardLayout layout;
    layout.layout_name = "New Layout";
    layout.theme = "Dark";
    current_layout_name_ = layout.layout_name;
    layouts_[layout.layout_name] = layout;
}

void DashboardLayoutManager::load_layout(const std::string& layout_name) {
    auto it = layouts_.find(layout_name);
    if (it != layouts_.end()) {
        current_layout_name_ = layout_name;
    } else {
        // Try to load from file
        DashboardLayout layout = load_layout_from_file(layout_name);
        if (!layout.layout_name.empty()) {
            layouts_[layout_name] = layout;
            current_layout_name_ = layout_name;
        }
    }
}

void DashboardLayoutManager::save_layout(const std::string& layout_name) {
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        it->second.layout_name = layout_name;
        layouts_[layout_name] = it->second;
        if (layout_name != current_layout_name_) {
            layouts_.erase(current_layout_name_);
        }
        current_layout_name_ = layout_name;
        save_layout_to_file(it->second);
    }
}

void DashboardLayoutManager::reset_layout() {
    layouts_.clear();
    create_new_layout();
}

void DashboardLayoutManager::delete_layout(const std::string& layout_name) {
    layouts_.erase(layout_name);
    std::string file_path = get_layout_file_path(layout_name);
    if (fs::exists(file_path)) {
        fs::remove(file_path);
    }
    if (current_layout_name_ == layout_name) {
        current_layout_name_.clear();
    }
}

std::vector<std::string> DashboardLayoutManager::get_available_layouts() const {
    std::vector<std::string> names;
    for (const auto& [name, layout] : layouts_) {
        names.push_back(name);
    }
    return names;
}

DashboardLayoutManager::DashboardLayout
DashboardLayoutManager::get_current_layout() const {
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        return it->second;
    }
    return DashboardLayout();
}

std::string DashboardLayoutManager::get_current_layout_name() const {
    return current_layout_name_;
}

bool DashboardLayoutManager::has_layout(const std::string& layout_name) const {
    return layouts_.find(layout_name) != layouts_.end();
}

void DashboardLayoutManager::add_panel_to_layout(const PanelLayout& panel) {
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        it->second.panels.push_back(panel);
    }
}

void DashboardLayoutManager::remove_panel_from_layout(const std::string& panel_id) {
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        auto& panels = it->second.panels;
        panels.erase(
            std::remove_if(panels.begin(), panels.end(),
                            [&panel_id](const PanelLayout& p) {
                                return p.panel_id == panel_id;
                            }),
            panels.end());
    }
}

void DashboardLayoutManager::update_panel_layout(
    const std::string& panel_id, const PanelLayout& layout) {
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        for (auto& panel : it->second.panels) {
            if (panel.panel_id == panel_id) {
                panel = layout;
                break;
            }
        }
    }
}

std::vector<DashboardLayoutManager::PanelLayout>
DashboardLayoutManager::get_panels_for_symbol(const std::string& symbol) const {
    std::vector<PanelLayout> result;
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        for (const auto& panel : it->second.panels) {
            if (panel.symbol == symbol) {
                result.push_back(panel);
            }
        }
    }
    return result;
}

void DashboardLayoutManager::update_symbol_for_all_panels(
    const std::string& old_symbol, const std::string& new_symbol) {
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        for (auto& panel : it->second.panels) {
            if (panel.symbol == old_symbol) {
                panel.symbol = new_symbol;
            }
        }
    }
}

void DashboardLayoutManager::set_grid_dimensions(int columns, int rows) {
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        it->second.grid_columns = columns;
        it->second.grid_rows = rows;
    }
}

std::pair<int, int> DashboardLayoutManager::get_grid_dimensions() const {
    auto it = layouts_.find(current_layout_name_);
    if (it != layouts_.end()) {
        return {it->second.grid_columns, it->second.grid_rows};
    }
    return {6, 9};
}

void DashboardLayoutManager::auto_arrange_panels() {
    // Placeholder - auto arrangement logic
}

void DashboardLayoutManager::center_layout() {
    // Placeholder
}

void DashboardLayoutManager::maximize_panel(const std::string& panel_id) {
    (void)panel_id;
    // Placeholder
}

void DashboardLayoutManager::restore_panel_sizes() {
    // Placeholder
}

void DashboardLayoutManager::update_current_layout(const DashboardLayout& new_layout) {
    layouts_[current_layout_name_] = new_layout;
}

void DashboardLayoutManager::initialize_layouts_directory() {
    if (!fs::exists(layouts_directory_)) {
        fs::create_directories(layouts_directory_);
    }
}

void DashboardLayoutManager::load_all_layouts() {
    if (!fs::exists(layouts_directory_)) return;

    for (const auto& entry : fs::directory_iterator(layouts_directory_)) {
        if (entry.path().extension() == ".json") {
            DashboardLayout layout = load_layout_from_file(entry.path().stem().string());
            if (!layout.layout_name.empty()) {
                layouts_[layout.layout_name] = layout;
            }
        }
    }
}

void DashboardLayoutManager::save_layout_to_file(const DashboardLayout& layout) {
    std::string file_path = get_layout_file_path(layout.layout_name);
    std::ofstream file(file_path);
    if (file.is_open()) {
        file << "{\"name\":\"" << layout.layout_name << "\"}";
    }
}

DashboardLayoutManager::DashboardLayout
DashboardLayoutManager::load_layout_from_file(const std::string& layout_name) {
    DashboardLayout layout;
    std::string file_path = get_layout_file_path(layout_name);
    std::ifstream file(file_path);
    if (file.is_open()) {
        layout.layout_name = layout_name;
    }
    return layout;
}

std::string DashboardLayoutManager::get_layout_file_path(
    const std::string& layout_name) const {
    return layouts_directory_ + "/" + layout_name + ".json";
}

}  // namespace Layout
}  // namespace BTQuant
