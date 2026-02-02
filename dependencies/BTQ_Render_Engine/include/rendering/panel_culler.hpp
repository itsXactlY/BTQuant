#pragma once

#include <vector>

#include "../components/panel_base.hpp"

struct ImVec2;

namespace BTQuant {
namespace RenderEngine {

/**
 * @brief PanelCuller - Implements off-screen and minimized panel culling
 *
 * This class determines which panels should be rendered based on their visibility
 * and position relative to the current viewport. Panels that are minimized,
 * off-screen, or otherwise not visible to the user will be culled to improve
 * rendering performance.
 */
class PanelCuller {
public:
    PanelCuller();
    ~PanelCuller();

    /**
     * @brief Determines if a panel should be rendered
     *
     * Checks if the panel is visible, not minimized, and within the viewport bounds.
     *
     * @param panel The panel to check
     * @return true if the panel should be rendered, false otherwise
     */
    bool should_render_panel(const PanelBase& panel) const;

    /**
     * @brief Sets the viewport bounds for culling calculations
     *
     * @param min_bound Minimum viewport coordinate
     * @param max_bound Maximum viewport coordinate
     */
    void set_viewport_bounds(const ImVec2& min_bound, const ImVec2& max_bound);

    /**
     * @brief Filters a list of panels to only those that should be rendered
     *
     * @param panels Vector of panel pointers to cull
     * @return Vector containing only the panels that should be rendered
     */
    std::vector<const PanelBase*> cull_panels(const std::vector<const PanelBase*>& panels) const;

    /**
     * @brief Sets the visibility thresholds for small area culling
     *
     * @param small_area_threshold Minimum visible area in pixels^2 to trigger percentage check
     * @param percentage_threshold Percentage of total panel area that must be visible (0.0-1.0)
     */
    void set_visibility_thresholds(float small_area_threshold = 100.0f, float percentage_threshold = 0.01f);

private:
    ImVec2 viewport_min_{0.0f, 0.0f};
    ImVec2 viewport_max_{0.0f, 0.0f};

    // Visibility thresholds for determining when to skip rendering very small visible portions
    float visibility_threshold_small_area_{100.0f};    // Minimum visible area in pixels^2
    float visibility_threshold_percentage_{0.01f};    // Minimum percentage of panel area (0.0-1.0)
};

} // namespace RenderEngine
} // namespace BTQuant