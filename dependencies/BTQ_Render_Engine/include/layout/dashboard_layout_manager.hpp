#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace BTQuant {
namespace Layout {

// ============================================================================
// Dashboard Layout Manager
// ============================================================================

class DashboardLayoutManager {
public:
    struct PanelLayout {
        std::string panel_id;
        std::string panel_name;
        float x = 0.0f;
        float y = 0.0f;
        float width = 400.0f;
        float height = 300.0f;
        bool is_docked = false;
        int dock_node_id = 0;
    };

    struct DashboardLayout {
        std::string layout_name;
        std::vector<PanelLayout> panels;
        std::string theme;
        std::string created_time;
    };

    DashboardLayoutManager();
    ~DashboardLayoutManager();

    // Layout management
    void create_new_layout();
    void load_layout(const std::string& layout_name);
    void save_layout(const std::string& layout_name);
    void reset_layout();
    void delete_layout(const std::string& layout_name);

    // Layout queries
    std::vector<std::string> get_available_layouts() const;
    DashboardLayout get_current_layout() const;
    bool has_layout(const std::string& layout_name) const;

    // Panel management
    void add_panel_to_layout(const PanelLayout& panel);
    void remove_panel_from_layout(const std::string& panel_id);
    void update_panel_layout(const std::string& panel_id, const PanelLayout& layout);

private:
    std::unordered_map<std::string, DashboardLayout> layouts_;
    std::string current_layout_name_;
    std::string layouts_directory_;

    void initialize_layouts_directory();
    void load_all_layouts();
    void save_layout_to_file(const DashboardLayout& layout);
    DashboardLayout load_layout_from_file(const std::string& layout_name);
    std::string get_layout_file_path(const std::string& layout_name) const;
};

} // namespace Layout
} // namespace BTQuant
