#include "../../include/rendering/panel_culler.hpp"

#include "imgui.h"
#include <algorithm> // for std::max and std::min

namespace BTQuant {
namespace RenderEngine {

PanelCuller::PanelCuller() = default;

PanelCuller::~PanelCuller() = default;

void PanelCuller::set_visibility_thresholds(float small_area_threshold, float percentage_threshold) {
    visibility_threshold_small_area_ = small_area_threshold;
    visibility_threshold_percentage_ = percentage_threshold;
}

bool PanelCuller::should_render_panel(const PanelBase& panel) const {
    const auto& config = panel.get_config();

    // Early exit: Don't render if panel is not visible or minimized
    if (!config.visible || config.minimized) {
        return false;
    }

    // Calculate panel bounds
    const ImVec2 panel_pos = config.position;
    const ImVec2 panel_size = config.size;

    // Early exit: Don't render if panel size is zero or negative
    if (panel_size.x <= 0.0f || panel_size.y <= 0.0f) {
        return false;
    }

    // Quick off-screen check using bounding box
    // Check if panel is completely off-screen based on viewport bounds
    const float right_edge = panel_pos.x + panel_size.x;
    const float bottom_edge = panel_pos.y + panel_size.y;

    if (panel_pos.x > viewport_max_.x ||           // Panel is too far right
        panel_pos.y > viewport_max_.y ||           // Panel is too far down
        right_edge < viewport_min_.x ||            // Panel is too far left
        bottom_edge < viewport_min_.y) {           // Panel is too far up
        return false;
    }

    // Calculate intersection between panel and viewport to determine visible area
    // Using local variables for better performance
    const float intersect_left = std::max(panel_pos.x, viewport_min_.x);
    const float intersect_top = std::max(panel_pos.y, viewport_min_.y);
    const float intersect_right = std::min(right_edge, viewport_max_.x);
    const float intersect_bottom = std::min(bottom_edge, viewport_max_.y);

    // Calculate intersection area
    const float intersect_width = intersect_right - intersect_left;
    const float intersect_height = intersect_bottom - intersect_top;

    // If intersection area is not positive, the panel is not visible
    if (intersect_width <= 0.0f || intersect_height <= 0.0f) {
        return false;
    }

    // Optional: Skip rendering if only a very small portion of the panel is visible
    // Thresholds can be adjusted based on performance needs
    const float panel_area = panel_size.x * panel_size.y;
    const float visible_area = intersect_width * intersect_height;
    const float visible_percentage = panel_area > 0.0f ? (visible_area / panel_area) : 0.0f;

    // Skip rendering if the visible area is too small OR the visible percentage is too low
    // Both area and percentage thresholds must be met for the panel to render
    // This prevents rendering of tiny slivers of panels that barely intersect with the viewport
    if (visible_area < visibility_threshold_small_area_ ||
        visible_percentage < visibility_threshold_percentage_) {
        return false;
    }

    // Additional check: If the panel is mostly off-screen (>90% outside viewport),
    // consider it for culling even if it meets the thresholds
    const float panel_outside_x = std::max(0.0f,
        std::max(viewport_min_.x - panel_pos.x, panel_pos.x + panel_size.x - viewport_max_.x) / panel_size.x);
    const float panel_outside_y = std::max(0.0f,
        std::max(viewport_min_.y - panel_pos.y, panel_pos.y + panel_size.y - viewport_max_.y) / panel_size.y);

    // If more than 90% of the panel is outside the viewport in either dimension, cull it
    if (panel_outside_x > 0.9f || panel_outside_y > 0.9f) {
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