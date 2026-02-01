#pragma once

#include <memory>
#include <string>
#include <vector>

#include "layout/layout_presets.hpp"

namespace BTQuant {
namespace Layout {
    class DashboardLayoutManager;
}

namespace UI {

// ============================================================================
// Layout Manager System
// ============================================================================

class LayoutManager {
public:
    LayoutManager();
    ~LayoutManager();

    // Save current panel arrangement as named preset
    bool save_current_layout_as_preset(const std::string& preset_name,
                                      const std::string& description = "",
                                      const std::string& category = "Custom");

    // Load preset to restore layout
    bool load_preset_layout(const std::string& preset_name);

    // Delete preset
    bool delete_preset(const std::string& preset_name);

    // Get all available presets
    std::vector<Layout::LayoutPreset> get_all_presets() const;

    // Get presets by category
    std::vector<Layout::LayoutPreset> get_presets_by_category(const std::string& category) const;

    // Integration methods with dashboard layout manager
    bool save_current_dashboard_layout(const std::string& layout_name);
    bool load_dashboard_layout(const std::string& layout_name);

private:
    std::unique_ptr<Layout::LayoutPresetManager> preset_manager_;
    std::unique_ptr<Layout::DashboardLayoutManager> dashboard_layout_manager_;
    std::string presets_directory_;

    void initialize_presets_directory();
    std::string get_current_layout_json();
    bool apply_layout_from_preset(const Layout::LayoutPreset& preset);
    std::string get_preset_file_path(const std::string& preset_name) const;
};

}  // namespace UI
}  // namespace BTQuant