#include "data/ui_data_manager.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace BTQuant {
namespace Data {

// ============================================================================
// UIDataManager Implementation
// ============================================================================

UIDataManager::UIDataManager() {
    initialize_state_directory();
    load_state("default");
}

UIDataManager::~UIDataManager() {
    save_state("default");
}

void UIDataManager::initialize_state_directory() {
    state_directory_ = "ui_state";
    std::filesystem::create_directories(state_directory_);
}

void UIDataManager::set_ui_state(const UIState& state) {
    ui_state_ = state;
}

UIDataManager::UIState UIDataManager::get_ui_state() const {
    return ui_state_;
}

void UIDataManager::add_panel(const PanelState& panel) {
    panels_[panel.panel_id] = panel;
}

void UIDataManager::remove_panel(const std::string& panel_id) {
    panels_.erase(panel_id);
}

void UIDataManager::update_panel(const std::string& panel_id, const PanelState& state) {
    auto it = panels_.find(panel_id);
    if (it != panels_.end()) {
        it->second = state;
    }
}

std::vector<UIDataManager::PanelState> UIDataManager::get_panels() const {
    std::vector<PanelState> panel_list;
    for (const auto& [id, state] : panels_) {
        panel_list.push_back(state);
    }
    return panel_list;
}

UIDataManager::PanelState UIDataManager::get_panel(const std::string& panel_id) const {
    auto it = panels_.find(panel_id);
    if (it != panels_.end()) {
        return it->second;
    }
    return PanelState{};
}

void UIDataManager::save_state(const std::string& filename) {
    std::string file_path = get_state_file_path(filename);
    
    nlohmann::json j;
    j["current_symbol"] = ui_state_.current_symbol;
    j["current_timeframe"] = ui_state_.current_timeframe;
    j["current_theme"] = ui_state_.current_theme;
    j["show_performance_overlay"] = ui_state_.show_performance_overlay;
    j["show_debug_info"] = ui_state_.show_debug_info;
    j["show_demo_window"] = ui_state_.show_demo_window;
    
    nlohmann::json panels_json = nlohmann::json::array();
    for (const auto& [id, state] : panels_) {
        nlohmann::json panel_json;
        panel_json["panel_id"] = state.panel_id;
        panel_json["is_visible"] = state.is_visible;
        panel_json["is_docked"] = state.is_docked;
        panel_json["x"] = state.x;
        panel_json["y"] = state.y;
        panel_json["width"] = state.width;
        panel_json["height"] = state.height;
        panels_json.push_back(panel_json);
    }
    j["panels"] = panels_json;
    
    std::ofstream file(file_path);
    if (file.is_open()) {
        file << j.dump(4);
        file.close();
    }
}

void UIDataManager::load_state(const std::string& filename) {
    std::string file_path = get_state_file_path(filename);
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return;
    }
    
    try {
        nlohmann::json j;
        file >> j;
        
        ui_state_.current_symbol = j.value("current_symbol", "BTCUSDT");
        ui_state_.current_timeframe = j.value("current_timeframe", "15m");
        ui_state_.current_theme = j.value("current_theme", "dark");
        ui_state_.show_performance_overlay = j.value("show_performance_overlay", true);
        ui_state_.show_debug_info = j.value("show_debug_info", false);
        ui_state_.show_demo_window = j.value("show_demo_window", false);
        
        if (j.contains("panels") && j["panels"].is_array()) {
            for (const auto& panel_json : j["panels"]) {
                PanelState state;
                state.panel_id = panel_json.value("panel_id", "");
                state.is_visible = panel_json.value("is_visible", true);
                state.is_docked = panel_json.value("is_docked", false);
                state.x = panel_json.value("x", 0.0f);
                state.y = panel_json.value("y", 0.0f);
                state.width = panel_json.value("width", 400.0f);
                state.height = panel_json.value("height", 300.0f);
                panels_[state.panel_id] = state;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading UI state: " << e.what() << std::endl;
    }
    
    file.close();
}

void UIDataManager::set_theme(const std::string& theme) {
    ui_state_.current_theme = theme;
}

std::string UIDataManager::get_theme() const {
    return ui_state_.current_theme;
}

void UIDataManager::set_current_symbol(const std::string& symbol) {
    ui_state_.current_symbol = symbol;
}

std::string UIDataManager::get_current_symbol() const {
    return ui_state_.current_symbol;
}

void UIDataManager::set_current_timeframe(const std::string& timeframe) {
    ui_state_.current_timeframe = timeframe;
}

std::string UIDataManager::get_current_timeframe() const {
    return ui_state_.current_timeframe;
}

std::string UIDataManager::get_state_file_path(const std::string& filename) const {
    return state_directory_ + "/" + filename + ".json";
}

} // namespace Data
} // namespace BTQuant
