#include "../../include/rendering/panel_culler.hpp"

#include "imgui.h"
#include <algorithm> // for std::max and std::min

namespace BTQuant {
namespace RenderEngine {

PanelCuller::PanelCuller() = default;

PanelCuller::~PanelCuller() = default;

bool PanelCuller::should_render_panel(const PanelBase& panel) const {
    const auto& config = panel.get_config();

    // Early exit: Don't render if panel is not visible (hidden)
    if (!config.visible) {
        return false;
    }

    // Early exit: Don't render if panel is minimized/collapsed
    if (config.minimized) {
        return false;
    }

    // Calculate panel bounds
    ImVec2 panel_pos = config.position;
    ImVec2 panel_size = config.size;

    // Early exit: Additional check: if panel size is zero or negative, don't render
    if (panel_size.x <= 0.0f || panel_size.y <= 0.0f) {
        return false;
    }

    // Quick off-screen check using bounding box
    // Check if panel is completely off-screen based on viewport bounds
    if (panel_pos.x > viewport_max_.x ||                         // Panel is too far right
        panel_pos.y > viewport_max_.y ||                         // Panel is too far down
        (panel_pos.x + panel_size.x) < viewport_min_.x ||       // Panel is too far left
        (panel_pos.y + panel_size.y) < viewport_min_.y) {       // Panel is too far up
        return false;
    }

    // Calculate intersection between panel and viewport to determine visible area
    // Using local variables for better performance
    const float intersect_left = std::max(panel_pos.x, viewport_min_.x);
    const float intersect_top = std::max(panel_pos.y, viewport_min_.y);
    const float intersect_right = std::min(panel_pos.x + panel_size.x, viewport_max_.x);
    const float intersect_bottom = std::min(panel_pos.y + panel_size.y, viewport_max_.y);

    // Calculate intersection area
    const float intersect_width = intersect_right - intersect_left;
    const float intersect_height = intersect_bottom - intersect_top;

    // If intersection area is not positive, the panel is not visible
    if (intersect_width <= 0.0f || intersect_height <= 0.0f) {
        return false;
    }

    // Optional: Skip rendering if only a very small portion of the panel is visible
    // This threshold can be adjusted based on performance needs
    const float panel_area = panel_size.x * panel_size.y;
    const float visible_area = intersect_width * intersect_height;

    // If less than 1% of the panel is visible and the visible area is very small, skip rendering
    if (visible_area < 100.0f && (visible_area / panel_area) < 0.01f) {
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

    // Reserve space to avoid repeated allocations
    visible_panels.reserve(panels.size());

    for (const auto* panel : panels) {
        if (should_render_panel(*panel)) {
            visible_panels.push_back(panel);
        }
    }

    // Shrink to fit to optimize memory usage
    visible_panels.shrink_to_fit();

    return visible_panels;
}

} // namespace RenderEngine
} // namespace BTQuant