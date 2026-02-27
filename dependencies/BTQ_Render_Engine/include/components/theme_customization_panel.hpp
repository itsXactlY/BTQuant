#pragma once

#include "panel_base.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace BTQuant {

/**
 * ThemeCustomizationPanel - UI theme and styling configuration
 * 
 * Features (Stub):
 * - Preset themes (Dark Neon, Light, Classic, Custom)
 * - Color customization for all UI elements
 * - Font selection and sizing
 * - Chart color schemes
 * - Export/Import theme settings
 */
class ThemeCustomizationPanel : public PanelBase {
public:
    explicit ThemeCustomizationPanel(const PanelConfig& config);
    ~ThemeCustomizationPanel() override = default;

    void initialize() override;
    void render() override;

    // Theme management
    void apply_theme();
    void save_theme(const std::string& name);
    bool export_theme(const std::string& filename);
    bool import_theme(const std::string& filename);

private:
    // Theme colors structure
    struct ThemeColors {
        ImVec4 background;
        ImVec4 window_bg;
        ImVec4 child_bg;
        ImVec4 accent;
        ImVec4 border;
        ImVec4 text;
        ImVec4 bull_color;
        ImVec4 bear_color;
        ImVec4 volume_color;
    };
    
    std::string current_theme_;
    std::unordered_map<std::string, ThemeColors> themes_;
    ThemeColors working_colors_;
    
    void render_theme_selector();
    void render_color_editor();
    void render_preview();
    void render_actions();
};

}  // namespace BTQuant
