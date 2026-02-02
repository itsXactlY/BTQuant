#include "../../include/rendering/panel_culler.hpp"

#include "imgui.h"

namespace BTQuant {
namespace RenderEngine {

PanelCuller::PanelCuller() = default;

PanelCuller::~PanelCuller() = default;

bool PanelCuller::should_render_panel(const PanelBase& panel) const {
    const auto& config = panel.get_config();

    // Don't render if panel is not visible (hidden)
    if (!config.visible) {
        return false;
    }

    // Don't render if panel is minimized/collapsed
    if (config.minimized) {
        return false;
    }

    // Calculate panel bounds
    ImVec2 panel_pos = config.position;
    ImVec2 panel_size = config.size;

    // Check if panel is completely off-screen based on viewport bounds
    if (panel_pos.x > viewport_max_.x ||                         // Panel is too far right
        panel_pos.y > viewport_max_.y ||                         // Panel is too far down
        (panel_pos.x + panel_size.x) < viewport_min_.x ||       // Panel is too far left
        (panel_pos.y + panel_size.y) < viewport_min_.y) {       // Panel is too far up
        return false;
    }

    // Additional check: if panel size is zero or negative, don't render
    if (panel_size.x <= 0.0f || panel_size.y <= 0.0f) {
        return false;
    }

    // If we got here, the panel should be rendered
    return true;
}

void PanelCuller::set_viewport_bounds(const ImVec2& min_bound, const ImVec2& max_bound) {
    viewport_min_ = min_bound;
    viewport_max_ = max_bound;
}

std::vector<const PanelBase*> PanelCuller::cull_panels(const std::vector<const PanelBase*>& panels) const {
    std::vector<const PanelBase*> visible_panels;
    
    for (const auto* panel : panels) {
        if (should_render_panel(*panel)) {
            visible_panels.push_back(panel);
        }
    }
    
    return visible_panels;
}

} // namespace RenderEngine
} // namespace BTQuant