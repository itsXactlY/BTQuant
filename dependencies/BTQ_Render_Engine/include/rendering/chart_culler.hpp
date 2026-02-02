#pragma once

#include <vector>

#include "../components/chart_manager.hpp"
#include "../ChartMath.hpp"

struct ImVec2;

namespace BTQuant {
namespace RenderEngine {

/**
 * @brief ChartCuller - Implements off-screen chart item culling and polygon reduction
 *
 * This class determines which chart elements should be rendered based on their visibility
 * and position relative to the current viewport. Chart elements that are off-screen
 * or outside the visible time/price range will be culled to improve rendering performance.
 * Additionally, implements polygon reduction strategies at lower zoom levels to reduce
 * computational overhead.
 */
class ChartCuller {
public:
    ChartCuller();
    ~ChartCuller();

    /**
     * @brief Sets the viewport bounds for culling calculations
     *
     * @param view_port Current viewport with time/price boundaries
     */
    void set_viewport_bounds(const ViewPort& view_port);

    /**
     * @brief Determines if a chart element should be rendered based on visibility
     *
     * @param time The time value of the chart element
     * @param price The price value of the chart element
     * @return true if the element should be rendered, false otherwise
     */
    bool should_render_element(double time, double price) const;

    /**
     * @brief Determines if a chart element with padding should be rendered based on visibility
     *        Useful for elements that extend beyond single points (e.g., candles, bars)
     *
     * @param time The time value of the chart element
     * @param price The price value of the chart element
     * @param time_padding Additional time padding to consider for visibility
     * @param price_padding Additional price padding to consider for visibility
     * @return true if the element with padding should be rendered, false otherwise
     */
    bool should_render_element_with_padding(double time, double price, double time_padding, double price_padding) const;

    /**
     * @brief Calculates the appropriate level of detail based on zoom level
     *
     * @param zoom_factor Current zoom factor (higher means more zoomed in)
     * @return Level of detail factor (1.0 = full detail, < 1.0 = reduced detail)
     */
    float calculate_lod_factor(double zoom_factor) const;

    /**
     * @brief Calculates adaptive LOD factor considering data density in viewport
     *
     * @param zoom_factor Current zoom factor
     * @param data_point_count Number of data points in the visible range
     * @param viewport_width_pixels Width of the viewport in pixels
     * @return Adaptive level of detail factor
     */
    float calculate_adaptive_lod_factor(double zoom_factor, size_t data_point_count, float viewport_width_pixels) const;

    /**
     * @brief Filters chart data points to only those within the viewport
     *
     * @param chart The chart instance to cull
     * @param start_index Output parameter for the start index of visible data
     * @param end_index Output parameter for the end index of visible data
     */
    void get_visible_data_range(const ChartInstance& chart, size_t& start_index, size_t& end_index) const;

    /**
     * @brief Applies culling and LOD to chart data for rendering
     *
     * @param chart The chart instance to process
     * @param lod_factor Level of detail factor to apply
     * @return Processed chart data with culling and LOD applied
     */
    ChartInstance apply_culling_and_lod(const ChartInstance& chart, float lod_factor = 1.0f) const;

    /**
     * @brief Applies advanced culling and adaptive LOD considering off-screen elements and polygon reduction
     *
     * @param chart The chart instance to process
     * @param lod_factor Base level of detail factor to apply
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Processed chart data with advanced culling and adaptive LOD applied
     */
    ChartInstance apply_advanced_culling_and_lod(const ChartInstance& chart, float lod_factor = 1.0f,
                                                float viewport_width_pixels = 0.0f, float viewport_height_pixels = 0.0f) const;

    /**
     * @brief Determines if a chart should be rendered based on its visibility
     *
     * @param chart The chart to check
     * @return true if the chart should be rendered, false otherwise
     */
    bool should_render_chart(const ChartInstance& chart) const;

private:
    ViewPort viewport_{};
    bool viewport_set_ = false;
};

} // namespace RenderEngine
} // namespace BTQuant