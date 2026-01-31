#pragma once

#include "../vulkan_dashboard_advanced.hpp"
#include "../dashboard_config.hpp"
#include "imgui.h"

namespace BTQuant {

class ThemeCustomizationComponent : public UIComponent {
public:
    explicit ThemeCustomizationComponent(RenderEngine::DashboardConfig& config);
    virtual ~ThemeCustomizationComponent() = default;

    void update(float dt) override;
    void render_gui() override;

    void initialize_vulkan_resources(VulkanCore* core) override {}
    void clear_data() override {}

private:
    RenderEngine::DashboardConfig& config_;
    bool show_theme_window_ = false;
    bool show_layout_window_ = false;
    char new_layout_name_[64] = "";

    // Theme customization
    RenderEngine::ThemeConfig custom_theme_;
    char custom_theme_name_[64] = "Custom";

    // Layout management
    std::vector<std::string> available_layouts_;
    int selected_layout_index_ = 0;

    void render_theme_window();
    void render_layout_window();
    void apply_current_theme();
    void save_custom_theme();
    void load_available_layouts();
};

} // namespace BTQuant
