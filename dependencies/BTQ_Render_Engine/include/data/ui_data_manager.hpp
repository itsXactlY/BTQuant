#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace BTQuant {
namespace Data {

// ============================================================================
// UI Data Manager
// ============================================================================

class UIDataManager {
public:
    struct UIState {
        std::string current_symbol;
        std::string current_timeframe;
        std::string current_theme;
        bool show_performance_overlay;
        bool show_debug_info;
        bool show_demo_window;
    };

    struct PanelState {
        std::string panel_id;
        bool is_visible;
        bool is_docked;
        float x;
        float y;
        float width;
        float height;
    };

    UIDataManager();
    ~UIDataManager();

    // State management
    void set_ui_state(const UIState& state);
    UIState get_ui_state() const;
    
    // Panel management
    void add_panel(const PanelState& panel);
    void remove_panel(const std::string& panel_id);
    void update_panel(const std::string& panel_id, const PanelState& state);
    std::vector<PanelState> get_panels() const;
    PanelState get_panel(const std::string& panel_id) const;
    
    // Persistence
    void save_state(const std::string& filename);
    void load_state(const std::string& filename);
    
    // Theme management
    void set_theme(const std::string& theme);
    std::string get_theme() const;
    
    // Symbol management
    void set_current_symbol(const std::string& symbol);
    std::string get_current_symbol() const;
    
    // Timeframe management
    void set_current_timeframe(const std::string& timeframe);
    std::string get_current_timeframe() const;

private:
    UIState ui_state_;
    std::unordered_map<std::string, PanelState> panels_;
    std::string state_directory_;
    
    void initialize_state_directory();
    std::string get_state_file_path(const std::string& filename) const;
};

} // namespace Data
} // namespace BTQuant
