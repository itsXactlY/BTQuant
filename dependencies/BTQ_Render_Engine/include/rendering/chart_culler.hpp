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
     * @brief Determines if a bounding box should be rendered based on visibility
     *        Useful for rectangular elements like candlesticks that have width and height
     *
     * @param min_time The minimum time value of the bounding box
     * @param max_time The maximum time value of the bounding box
     * @param min_price The minimum price value of the bounding box
     * @param max_price The maximum price value of the bounding box
     * @return true if the bounding box intersects with the viewport, false otherwise
     */
    bool should_render_bounding_box(double min_time, double max_time, double min_price, double max_price) const;

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
     * @brief Calculates advanced LOD factor considering both zoom and data density
     *
     * @param zoom_factor Current zoom factor
     * @param data_point_count Number of data points in the visible range
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Advanced adaptive level of detail factor
     */
    float calculate_advanced_lod_factor(double zoom_factor, size_t data_point_count,
                                      float viewport_width_pixels, float viewport_height_pixels) const;

    /**
     * @brief Filters chart data points to only those within the viewport
     *
     * @param chart The chart instance to cull
     * @param start_index Output parameter for the start index of visible data
     * @param end_index Output parameter for the end index of visible data
     */
    void get_visible_data_range(const ChartInstance& chart, size_t& start_index, size_t& end_index) const;

    /**
     * @brief Optimized version that uses binary search to find visible data range (better for large datasets)
     *
     * @param chart The chart instance to cull
     * @param start_index Output parameter for the start index of visible data
     * @param end_index Output parameter for the end index of visible data
     */
    void get_visible_data_range_optimized(const ChartInstance& chart, size_t& start_index, size_t& end_index) const;

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
     * @brief Applies optimized culling with binary search range finding and enhanced polygon reduction
     *
     * @param chart The chart instance to process
     * @param lod_factor Level of detail factor to apply
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Processed chart data with optimized culling and LOD applied
     */
    ChartInstance apply_optimized_culling_and_lod(const ChartInstance& chart, float lod_factor = 1.0f,
                                                 float viewport_width_pixels = 0.0f, float viewport_height_pixels = 0.0f) const;

    /**
     * @brief Determines if a chart should be rendered based on its visibility
     *
     * @param chart The chart to check
     * @return true if the chart should be rendered, false otherwise
     */
    bool should_render_chart(const ChartInstance& chart) const;

    /**
     * @brief Applies polygon reduction based on zoom level and data density
     *
     * @param chart The chart instance to process
     * @param zoom_factor Current zoom factor affecting polygon count
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Processed chart data with polygon reduction applied
     */
    ChartInstance apply_polygon_reduction(const ChartInstance& chart, float zoom_factor = 1.0f,
                                        float viewport_width_pixels = 0.0f, float viewport_height_pixels = 0.0f) const;

    /**
     * @brief Determines if an off-screen element should be rendered based on a threshold
     *        Useful for elements that might contribute to rendering (shadows, highlights, etc.)
     *
     * @param time The time value of the chart element
     * @param price The price value of the chart element
     * @param offscreen_threshold Threshold for how far off-screen an element can be and still render
     * @return true if the off-screen element should be rendered, false otherwise
     */
    bool should_render_offscreen_element(double time, double price, double offscreen_threshold) const;

    /**
     * @brief Applies importance-based polygon reduction that prioritizes visually significant elements
     *
     * @param chart The chart instance to process
     * @param zoom_factor Current zoom factor affecting polygon count
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Processed chart data with importance-based polygon reduction applied
     */
    ChartInstance apply_importance_based_polygon_reduction(const ChartInstance& chart, float zoom_factor = 1.0f,
                                                         float viewport_width_pixels = 0.0f, float viewport_height_pixels = 0.0f) const;

    /**
     * @brief Calculates dynamic padding based on zoom level for better off-screen rendering
     *
     * @param zoom_factor Current zoom factor
     * @param base_padding_ratio Base padding ratio to adjust based on zoom
     * @return Dynamic padding value adjusted for the current zoom level
     */
    double calculate_dynamic_padding(double zoom_factor, double base_padding_ratio) const;

    /**
     * @brief Applies polygon reduction using an enhanced algorithm that preserves visual significance
     *        Uses Douglas-Peucker-like approach for line simplification
     *
     * @param chart The chart instance to process
     * @param tolerance_factor Tolerance factor affecting how aggressively to reduce polygons
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Processed chart data with enhanced polygon reduction applied
     */
    ChartInstance apply_douglas_peucker_polygon_reduction(const ChartInstance& chart, float tolerance_factor = 0.1f,
                                                        float viewport_width_pixels = 0.0f, float viewport_height_pixels = 0.0f) const;

    /**
     * @brief Applies advanced off-screen culling combined with intelligent polygon reduction
     *        Combines visibility checks with importance-based filtering for optimal performance
     *
     * @param chart The chart instance to process
     * @param zoom_factor Current zoom factor affecting polygon count and culling behavior
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Processed chart data with advanced off-screen culling and polygon reduction applied
     */
    ChartInstance apply_advanced_offscreen_culling(const ChartInstance& chart, float zoom_factor = 1.0f,
                                                 float viewport_width_pixels = 0.0f, float viewport_height_pixels = 0.0f) const;

    /**
     * @brief Applies frustum-based culling with polygon reduction for improved performance
     *        Uses extended viewport bounds to account for off-screen elements that might contribute to rendering
     *
     * @param chart The chart instance to process
     * @param zoom_factor Current zoom factor affecting polygon count and culling behavior
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Processed chart data with frustum culling and polygon reduction applied
     */
    ChartInstance apply_frustum_culling_and_polygon_reduction(const ChartInstance& chart, float zoom_factor = 1.0f,
                                                           float viewport_width_pixels = 0.0f, float viewport_height_pixels = 0.0f) const;

    /**
     * @brief Applies clustering-based polygon reduction that reduces polygon count at lower zoom levels
     *        Groups nearby points into clusters and selects the most representative point from each cluster
     *
     * @param chart The chart instance to process
     * @param zoom_factor Current zoom factor affecting polygon count and culling behavior
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Processed chart data with clustering-based polygon reduction applied
     */
    ChartInstance apply_clustering_polygon_reduction(const ChartInstance& chart, float zoom_factor = 1.0f,
                                                  float viewport_width_pixels = 0.0f, float viewport_height_pixels = 0.0f) const;

    /**
     * @brief Applies aggressive off-screen culling combined with zoom-based polygon reduction
     *        Performs strict visibility checks and reduces polygon count more aggressively at lower zoom levels
     *
     * @param chart The chart instance to process
     * @param zoom_factor Current zoom factor affecting polygon count and culling behavior
     * @param viewport_width_pixels Width of the viewport in pixels
     * @param viewport_height_pixels Height of the viewport in pixels
     * @return Processed chart data with aggressive culling and zoom-based polygon reduction applied
     */
    ChartInstance apply_aggressive_culling_and_lod(const ChartInstance& chart, float zoom_factor = 1.0f,
                                                 float viewport_width_pixels = 0.0f, float viewport_height_pixels = 0.0f) const;

private:
    ViewPort viewport_{};
    bool viewport_set_ = false;
};

} // namespace RenderEngine
} // namespace BTQuant