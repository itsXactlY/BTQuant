#include "../../include/rendering/chart_culler.hpp"

#include <algorithm>
#include <cmath>

namespace BTQuant {
namespace RenderEngine {

ChartCuller::ChartCuller() = default;

ChartCuller::~ChartCuller() = default;

void ChartCuller::set_viewport_bounds(const ViewPort& view_port) {
    viewport_ = view_port;
    viewport_set_ = true;
}

bool ChartCuller::should_render_element(double time, double price) const {
    if (!viewport_set_) {
        // If no viewport is set, render everything
        return true;
    }

    // Check if the element is within the visible time range
    if (time < viewport_.minTime || time > viewport_.maxTime) {
        return false;
    }

    // Check if the element is within the visible price range
    if (price < viewport_.minPrice || price > viewport_.maxPrice) {
        return false;
    }

    // Element is within viewport bounds
    return true;
}

// Enhanced version that considers off-screen padding for elements that might extend beyond single points
bool ChartCuller::should_render_element_with_padding(double time, double price, double time_padding, double price_padding) const {
    if (!viewport_set_) {
        // If no viewport is set, render everything
        return true;
    }

    // Check if the element with padding is within the visible time range
    if (time + time_padding < viewport_.minTime || time - time_padding > viewport_.maxTime) {
        return false;
    }

    // Check if the element with padding is within the visible price range
    if (price + price_padding < viewport_.minPrice || price - price_padding > viewport_.maxPrice) {
        return false;
    }

    // Element with padding is within viewport bounds
    return true;
}

// Enhanced version that considers bounding box for rectangular elements like candles with improved accuracy
bool ChartCuller::should_render_bounding_box(double min_time, double max_time, double min_price, double max_price) const {
    if (!viewport_set_) {
        // If no viewport is set, render everything
        return true;
    }

    // Check if the bounding box intersects with the viewport
    // Using inclusive comparisons to handle edge cases properly
    if (max_time < viewport_.minTime || min_time > viewport_.maxTime) {
        return false; // Completely outside time range
    }

    if (max_price < viewport_.minPrice || min_price > viewport_.maxPrice) {
        return false; // Completely outside price range
    }

    // Bounding box intersects with viewport
    return true;
}

// Advanced method to handle off-screen elements with more sophisticated threshold calculations
bool ChartCuller::should_render_offscreen_element(double time, double price, double offscreen_threshold) const {
    if (!viewport_set_) {
        // If no viewport is set, render everything
        return true;
    }

    // Calculate distance from viewport bounds
    double time_distance = 0.0;
    double price_distance = 0.0;

    // Determine time distance from viewport
    if (time < viewport_.minTime) {
        time_distance = viewport_.minTime - time;
    } else if (time > viewport_.maxTime) {
        time_distance = time - viewport_.maxTime;
    }

    // Determine price distance from viewport
    if (price < viewport_.minPrice) {
        price_distance = viewport_.minPrice - price;
    } else if (price > viewport_.maxPrice) {
        price_distance = price - viewport_.maxPrice;
    }

    // Convert threshold to appropriate units for comparison
    // The threshold represents how far off-screen an element can be and still render
    double normalized_time_threshold = (viewport_.maxTime - viewport_.minTime) * offscreen_threshold;
    double normalized_price_threshold = (viewport_.maxPrice - viewport_.minPrice) * offscreen_threshold;

    // Check if the element is within the off-screen threshold
    return (time_distance <= normalized_time_threshold && price_distance <= normalized_price_threshold);
}


float ChartCuller::calculate_lod_factor(double zoom_factor) const {
    // Define LOD thresholds based on zoom level
    // Higher zoom = more detail, lower zoom = less detail

    if (zoom_factor >= 10.0) {
        // Very zoomed in - full detail
        return 1.0f;
    } else if (zoom_factor >= 5.0) {
        // Moderately zoomed in - 75% detail
        return 0.75f;
    } else if (zoom_factor >= 2.0) {
        // Somewhat zoomed in - 50% detail
        return 0.5f;
    } else if (zoom_factor >= 0.5) {
        // Normal zoom - 25% detail
        return 0.25f;
    } else {
        // Zoomed out - 10% detail to maintain performance
        return 0.1f;
    }
}

// Enhanced LOD calculation that considers viewport density
float ChartCuller::calculate_adaptive_lod_factor(double zoom_factor, size_t data_point_count, float viewport_width_pixels) const {
    // Base LOD calculation
    float base_lod = calculate_lod_factor(zoom_factor);

    // Adjust based on data density in viewport
    if (viewport_width_pixels > 0 && data_point_count > 0) {
        float points_per_pixel = static_cast<float>(data_point_count) / viewport_width_pixels;

        // If we have too many points per pixel, reduce detail further
        if (points_per_pixel > 2.0f) {
            float density_factor = 1.0f / points_per_pixel;
            base_lod = std::min(base_lod, density_factor);
        }
    }

    return std::max(0.05f, base_lod); // Minimum 5% detail to maintain some representation
}

// Advanced LOD calculation that considers both zoom and data density
float ChartCuller::calculate_advanced_lod_factor(double zoom_factor, size_t data_point_count,
                                               float viewport_width_pixels, float viewport_height_pixels) const {
    // Start with the basic adaptive LOD calculation
    float adaptive_lod = calculate_adaptive_lod_factor(zoom_factor, data_point_count, viewport_width_pixels);

    // Calculate the time range density (points per time unit)
    if (viewport_.isValid() && data_point_count > 0) {
        double time_range = viewport_.width();
        if (time_range > 0) {
            float points_per_time_unit = static_cast<float>(data_point_count) / static_cast<float>(time_range);

            // If we have too many points per time unit, reduce detail further
            if (points_per_time_unit > 10.0f) {  // Threshold can be adjusted
                float time_density_factor = 10.0f / points_per_time_unit;
                adaptive_lod = std::min(adaptive_lod, time_density_factor);
            }
        }
    }

    // Also consider the price range density if needed
    if (viewport_.height() > 0) {
        // Price density could be considered here if needed for specific chart types
    }

    return std::max(0.05f, adaptive_lod); // Minimum 5% detail to maintain some representation
}

void ChartCuller::get_visible_data_range(const ChartInstance& chart, size_t& start_index, size_t& end_index) const {
    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return full range
        start_index = 0;
        end_index = chart.dates.size() > 0 ? chart.dates.size() - 1 : 0;
        return;
    }

    // Find the start index (first date within viewport)
    start_index = 0;
    for (size_t i = 0; i < chart.dates.size(); ++i) {
        if (chart.dates[i] >= viewport_.minTime) {
            start_index = i;
            break;
        }
    }

    // Find the end index (last date within viewport)
    end_index = chart.dates.size() > 0 ? chart.dates.size() - 1 : 0;
    for (int i = static_cast<int>(chart.dates.size()) - 1; i >= 0; --i) {
        if (chart.dates[i] <= viewport_.maxTime) {
            end_index = static_cast<size_t>(i);
            break;
        }
    }

    // Ensure indices are valid
    if (start_index > end_index && !chart.dates.empty()) {
        // No data is visible in the current viewport
        start_index = 0;
        end_index = 0;
    }
}

// Enhanced culling that handles different chart elements and reduces polygons more effectively
ChartInstance ChartCuller::apply_culling_and_lod(const ChartInstance& chart, float lod_factor) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Get the visible data range
    size_t start_index, end_index;
    get_visible_data_range(chart, start_index, end_index);

    // Calculate time and price ranges to determine appropriate padding for off-screen elements
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;
    double time_padding = time_range * 0.001; // 0.1% padding for time
    double price_padding = price_range * 0.001; // 0.1% padding for price

    // Apply LOD by reducing the number of points if needed
    if (lod_factor < 1.0f && lod_factor > 0.0f) {
        // Calculate step size based on LOD factor
        size_t total_points = end_index - start_index + 1;
        size_t target_points = static_cast<size_t>(total_points * lod_factor);

        if (target_points < 1) target_points = 1;

        // If we need to reduce points, calculate the step size
        size_t step = 1;
        if (target_points < total_points) {
            step = std::max(static_cast<size_t>(1), total_points / target_points);
        }

        // Create new vectors with reduced data
        std::vector<double> filtered_dates;
        std::vector<float> filtered_opens;
        std::vector<float> filtered_highs;
        std::vector<float> filtered_lows;
        std::vector<float> filtered_closes;
        std::vector<float> filtered_volumes;

        // Sample the data based on the step size
        for (size_t i = start_index; i <= end_index; i += step) {
            if (i < chart.dates.size()) {
                // Check if the element with padding would be visible (for off-screen elements that might affect rendering)
                if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
                    // For candlestick charts, we should also check if the full candle (high-low range) is visible
                    if (should_render_bounding_box(
                            chart.dates[i] - time_padding,
                            chart.dates[i] + time_padding,
                            chart.lows[i],
                            chart.highs[i])) {
                        filtered_dates.push_back(chart.dates[i]);
                        filtered_opens.push_back(chart.opens[i]);
                        filtered_highs.push_back(chart.highs[i]);
                        filtered_lows.push_back(chart.lows[i]);
                        filtered_closes.push_back(chart.closes[i]);
                        filtered_volumes.push_back(chart.volumes[i]);
                    }
                }
            }
        }

        // Replace the chart data with filtered data
        processed_chart.dates = std::move(filtered_dates);
        processed_chart.opens = std::move(filtered_opens);
        processed_chart.highs = std::move(filtered_highs);
        processed_chart.lows = std::move(filtered_lows);
        processed_chart.closes = std::move(filtered_closes);
        processed_chart.volumes = std::move(filtered_volumes);
    } else {
        // Just apply culling without LOD
        std::vector<double> filtered_dates;
        std::vector<float> filtered_opens;
        std::vector<float> filtered_highs;
        std::vector<float> filtered_lows;
        std::vector<float> filtered_closes;
        std::vector<float> filtered_volumes;

        for (size_t i = start_index; i <= end_index; ++i) {
            if (i < chart.dates.size()) {
                // Check if the element with padding would be visible (for off-screen elements that might affect rendering)
                if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
                    // For candlestick charts, we should also check if the full candle (high-low range) is visible
                    if (should_render_bounding_box(
                            chart.dates[i] - time_padding,
                            chart.dates[i] + time_padding,
                            chart.lows[i],
                            chart.highs[i])) {
                        filtered_dates.push_back(chart.dates[i]);
                        filtered_opens.push_back(chart.opens[i]);
                        filtered_highs.push_back(chart.highs[i]);
                        filtered_lows.push_back(chart.lows[i]);
                        filtered_closes.push_back(chart.closes[i]);
                        filtered_volumes.push_back(chart.volumes[i]);
                    }
                }
            }
        }

        // Replace the chart data with filtered data
        processed_chart.dates = std::move(filtered_dates);
        processed_chart.opens = std::move(filtered_opens);
        processed_chart.highs = std::move(filtered_highs);
        processed_chart.lows = std::move(filtered_lows);
        processed_chart.closes = std::move(filtered_closes);
        processed_chart.volumes = std::move(filtered_volumes);
    }

    return processed_chart;
}

// Advanced culling that considers off-screen elements and polygon reduction
ChartInstance ChartCuller::apply_advanced_culling_and_lod(const ChartInstance& chart, float lod_factor,
                                                         float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Calculate adaptive LOD based on data density
    size_t visible_points_count = 0;
    for (size_t i = 0; i < chart.dates.size(); ++i) {
        if (should_render_element(chart.dates[i], chart.closes[i])) {
            visible_points_count++;
        }
    }

    float adaptive_lod = calculate_advanced_lod_factor(lod_factor, visible_points_count, viewport_width_pixels, viewport_height_pixels);

    // Get the visible data range
    size_t start_index, end_index;
    get_visible_data_range(chart, start_index, end_index);

    // Create new vectors with reduced data
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate time and price ranges to determine appropriate padding
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;
    double time_padding = time_range * 0.001; // 0.1% padding for time
    double price_padding = price_range * 0.001; // 0.1% padding for price

    // Apply LOD by reducing the number of points if needed
    size_t total_points = end_index - start_index + 1;
    size_t target_points = static_cast<size_t>(total_points * adaptive_lod);

    if (target_points < 1) target_points = 1;

    // If we need to reduce points, calculate the step size
    size_t step = 1;
    if (target_points < total_points && adaptive_lod < 1.0f) {
        step = std::max(static_cast<size_t>(1), total_points / target_points);
    }

    // Sample the data based on the step size, considering padding for off-screen elements
    for (size_t i = start_index; i <= end_index; i += step) {
        if (i < chart.dates.size()) {
            // Check if the element with padding would be visible (for off-screen elements that might affect rendering)
            if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
                // For candlestick charts, we should also check if the full candle (high-low range) is visible
                if (should_render_bounding_box(
                        chart.dates[i] - time_padding,
                        chart.dates[i] + time_padding,
                        chart.lows[i],
                        chart.highs[i])) {
                    filtered_dates.push_back(chart.dates[i]);
                    filtered_opens.push_back(chart.opens[i]);
                    filtered_highs.push_back(chart.highs[i]);
                    filtered_lows.push_back(chart.lows[i]);
                    filtered_closes.push_back(chart.closes[i]);
                    filtered_volumes.push_back(chart.volumes[i]);
                }
            }
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

// Optimized version that uses binary search for finding visible range (more efficient for large datasets)
void ChartCuller::get_visible_data_range_optimized(const ChartInstance& chart, size_t& start_index, size_t& end_index) const {
    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return full range
        start_index = 0;
        end_index = chart.dates.size() > 0 ? chart.dates.size() - 1 : 0;
        return;
    }

    // Binary search for the start index (first date >= minTime)
    start_index = 0;
    size_t left = 0, right = chart.dates.size();
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (chart.dates[mid] >= viewport_.minTime) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    start_index = left;

    // Binary search for the end index (last date <= maxTime)
    end_index = chart.dates.size() > 0 ? chart.dates.size() - 1 : 0;
    left = 0;
    right = chart.dates.size();
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (chart.dates[mid] <= viewport_.maxTime) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    end_index = (left > 0) ? left - 1 : 0;

    // Ensure indices are valid
    if (start_index > end_index && !chart.dates.empty()) {
        // No data is visible in the current viewport
        start_index = 0;
        end_index = 0;
    }
}

// Advanced culling with optimized range finding and enhanced polygon reduction
ChartInstance ChartCuller::apply_optimized_culling_and_lod(const ChartInstance& chart, float lod_factor,
                                                          float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Use optimized range finding for better performance with large datasets
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    // Calculate adaptive LOD based on data density
    size_t visible_points_count = 0;
    for (size_t i = start_index; i <= end_index && i < chart.dates.size(); ++i) {
        if (should_render_element(chart.dates[i], chart.closes[i])) {
            visible_points_count++;
        }
    }

    float adaptive_lod = calculate_advanced_lod_factor(lod_factor, visible_points_count, viewport_width_pixels, viewport_height_pixels);

    // Create new vectors with reduced data
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate time and price ranges to determine appropriate padding
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;
    double time_padding = time_range * 0.001; // 0.1% padding for time
    double price_padding = price_range * 0.001; // 0.1% padding for price

    // Apply LOD by reducing the number of points if needed
    size_t total_points = end_index - start_index + 1;
    size_t target_points = static_cast<size_t>(total_points * adaptive_lod);

    if (target_points < 1) target_points = 1;

    // If we need to reduce points, calculate the step size
    size_t step = 1;
    if (target_points < total_points && adaptive_lod < 1.0f) {
        step = std::max(static_cast<size_t>(1), total_points / target_points);
    }

    // Sample the data based on the step size, considering padding for off-screen elements
    for (size_t i = start_index; i <= end_index && i < chart.dates.size(); i += step) {
        // Check if the element with padding would be visible (for off-screen elements that might affect rendering)
        if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
            // For candlestick charts, we should also check if the full candle (high-low range) is visible
            if (should_render_bounding_box(
                    chart.dates[i] - time_padding,
                    chart.dates[i] + time_padding,
                    chart.lows[i],
                    chart.highs[i])) {
                filtered_dates.push_back(chart.dates[i]);
                filtered_opens.push_back(chart.opens[i]);
                filtered_highs.push_back(chart.highs[i]);
                filtered_lows.push_back(chart.lows[i]);
                filtered_closes.push_back(chart.closes[i]);
                filtered_volumes.push_back(chart.volumes[i]);
            }
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

bool ChartCuller::should_render_chart(const ChartInstance& chart) const {
    // Check if chart is marked as visible
    if (!chart.visible) {
        return false;
    }

    // Check if chart is minimized
    if (chart.minimized) {
        return false;
    }

    // If no viewport is set, render the chart
    if (!viewport_set_) {
        return true;
    }

    // Check if any part of the chart's data is within the viewport
    // Use optimized range checking for better performance
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    for (size_t i = start_index; i <= end_index && i < chart.dates.size(); ++i) {
        if (should_render_element(chart.dates[i], chart.closes[i])) {
            return true; // At least one element is visible
        }
    }

    // No elements are visible in the current viewport
    return false;
}

// Method to reduce polygon count based on zoom level and data density
ChartInstance ChartCuller::apply_polygon_reduction(const ChartInstance& chart, float zoom_factor,
                                                  float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Calculate adaptive LOD based on zoom level and data density
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    size_t visible_points_count = end_index - start_index + 1;
    float adaptive_lod = calculate_advanced_lod_factor(zoom_factor, visible_points_count,
                                                      viewport_width_pixels, viewport_height_pixels);

    // Create new vectors with reduced data
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate time and price ranges to determine appropriate padding
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;
    double time_padding = time_range * 0.001; // 0.1% padding for time
    double price_padding = price_range * 0.001; // 0.1% padding for price

    // Apply polygon reduction by reducing the number of points based on LOD
    size_t total_points = end_index - start_index + 1;
    size_t target_points = static_cast<size_t>(total_points * adaptive_lod);

    if (target_points < 1) target_points = 1;

    // Calculate the step size for polygon reduction
    size_t step = 1;
    if (target_points < total_points && adaptive_lod < 1.0f) {
        step = std::max(static_cast<size_t>(1), total_points / target_points);
    }

    // Sample the data based on the step size, considering padding for off-screen elements
    for (size_t i = start_index; i <= end_index && i < chart.dates.size(); i += step) {
        // Check if the element with padding would be visible (for off-screen elements that might affect rendering)
        if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
            // For candlestick charts, we should also check if the full candle (high-low range) is visible
            if (should_render_bounding_box(
                    chart.dates[i] - time_padding,
                    chart.dates[i] + time_padding,
                    chart.lows[i],
                    chart.highs[i])) {
                filtered_dates.push_back(chart.dates[i]);
                filtered_opens.push_back(chart.opens[i]);
                filtered_highs.push_back(chart.highs[i]);
                filtered_lows.push_back(chart.lows[i]);
                filtered_closes.push_back(chart.closes[i]);
                filtered_volumes.push_back(chart.volumes[i]);
            }
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

// Enhanced method to handle off-screen elements that might contribute to rendering (like shadows, highlights, etc.)
bool ChartCuller::should_render_offscreen_element(double time, double price, double offscreen_threshold) const {
    if (!viewport_set_) {
        // If no viewport is set, render everything
        return true;
    }

    // Calculate distance from viewport bounds
    double time_distance = 0.0;
    double price_distance = 0.0;

    // Determine time distance from viewport
    if (time < viewport_.minTime) {
        time_distance = viewport_.minTime - time;
    } else if (time > viewport_.maxTime) {
        time_distance = time - viewport_.maxTime;
    }

    // Determine price distance from viewport
    if (price < viewport_.minPrice) {
        price_distance = viewport_.minPrice - price;
    } else if (price > viewport_.maxPrice) {
        price_distance = price - viewport_.maxPrice;
    }

    // Convert threshold to appropriate units for comparison
    // The threshold represents how far off-screen an element can be and still render
    double normalized_time_threshold = (viewport_.maxTime - viewport_.minTime) * offscreen_threshold;
    double normalized_price_threshold = (viewport_.maxPrice - viewport_.minPrice) * offscreen_threshold;

    // Check if the element is within the off-screen threshold
    return (time_distance <= normalized_time_threshold && price_distance <= normalized_price_threshold);
}

// Advanced polygon reduction algorithm that considers visual importance
ChartInstance ChartCuller::apply_importance_based_polygon_reduction(const ChartInstance& chart, float zoom_factor,
                                                                 float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Calculate adaptive LOD based on zoom level and data density
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    size_t visible_points_count = end_index - start_index + 1;
    float adaptive_lod = calculate_advanced_lod_factor(zoom_factor, visible_points_count,
                                                     viewport_width_pixels, viewport_height_pixels);

    // Create new vectors with reduced data
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate time and price ranges to determine appropriate padding
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;
    // Calculate dynamic padding based on zoom level for better off-screen rendering
    double time_padding = calculate_dynamic_padding(zoom_factor, time_range * 0.001); // 0.1% padding for time
    double price_padding = calculate_dynamic_padding(zoom_factor, price_range * 0.001); // 0.1% padding for price

    // Apply importance-based polygon reduction
    size_t total_points = end_index - start_index + 1;
    size_t target_points = static_cast<size_t>(total_points * adaptive_lod);

    if (target_points < 1) target_points = 1;

    // Calculate the step size for initial sampling
    size_t step = 1;
    if (target_points < total_points && adaptive_lod < 1.0f) {
        step = std::max(static_cast<size_t>(1), total_points / target_points);
    }

    // Importance sampling: prioritize elements with high volatility or significant changes
    for (size_t i = start_index; i <= end_index && i < chart.dates.size(); i += step) {
        bool should_include = false;

        // Check if the element with padding would be visible
        if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
            // For candlestick charts, check if the full candle (high-low range) is visible
            if (should_render_bounding_box(
                    chart.dates[i] - time_padding,
                    chart.dates[i] + time_padding,
                    chart.lows[i],
                    chart.highs[i])) {

                // Additional importance criteria:
                // 1. High volatility (large difference between high and low)
                float volatility = chart.highs[i] - chart.lows[i];
                float avg_price = (chart.opens[i] + chart.closes[i]) / 2.0f;
                float volatility_ratio = (avg_price != 0.0f) ? volatility / std::abs(avg_price) : 0.0f;

                // 2. Significant price change (difference between open and close)
                float price_change = std::abs(chart.opens[i] - chart.closes[i]);
                float change_ratio = (avg_price != 0.0f) ? price_change / std::abs(avg_price) : 0.0f;

                // 3. High volume (if available)
                float volume_ratio = (chart.volumes[i] > 0) ? chart.volumes[i] / 1000.0f : 0.0f; // Normalize volume

                // 4. Directional changes (trend reversals)
                bool is_trend_reversal = false;
                if (i > 0 && i < chart.dates.size() - 1) {
                    float prev_change = chart.closes[i-1] - chart.opens[i-1];
                    float curr_change = chart.closes[i] - chart.opens[i];
                    float next_change = chart.closes[i+1] - chart.opens[i+1];

                    // Check if current candle represents a trend reversal
                    is_trend_reversal = ((prev_change > 0 && curr_change < 0) || (prev_change < 0 && curr_change > 0)) ||
                                       ((curr_change > 0 && next_change < 0) || (curr_change < 0 && next_change > 0));
                }

                // Determine if this point is "important" enough to include
                // Adjust these thresholds as needed for your specific use case
                if (volatility_ratio > 0.02f || change_ratio > 0.015f || volume_ratio > 1.0f || is_trend_reversal) {
                    should_include = true;
                } else {
                    // For less important points, apply stricter LOD filtering
                    should_include = (static_cast<float>(filtered_dates.size()) / static_cast<float>(target_points) < 1.0f);
                }
            }
        }

        if (should_include) {
            filtered_dates.push_back(chart.dates[i]);
            filtered_opens.push_back(chart.opens[i]);
            filtered_highs.push_back(chart.highs[i]);
            filtered_lows.push_back(chart.lows[i]);
            filtered_closes.push_back(chart.closes[i]);
            filtered_volumes.push_back(chart.volumes[i]);
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

// Enhanced polygon reduction using Douglas-Peucker algorithm for line simplification
ChartInstance ChartCuller::apply_douglas_peucker_polygon_reduction(const ChartInstance& chart, float tolerance_factor,
                                                                 float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Calculate adaptive tolerance based on viewport dimensions and zoom level
    float adaptive_tolerance = tolerance_factor;
    if (viewport_width_pixels > 0 && viewport_height_pixels > 0) {
        // Adjust tolerance based on viewport size to maintain visual quality
        float viewport_area = viewport_width_pixels * viewport_height_pixels;
        if (viewport_area > 0) {
            adaptive_tolerance *= std::sqrt(viewport_area) / 100.0f; // Scale with viewport size
        }
    }

    // Get visible range
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    // Create new vectors with reduced data using simplified algorithm
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate time and price ranges to determine appropriate padding
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;
    double time_padding = time_range * 0.001; // 0.1% padding for time
    double price_padding = price_range * 0.001; // 0.1% padding for price

    // Simple decimation algorithm with adaptive tolerance
    size_t total_points = end_index - start_index + 1;
    size_t target_points = static_cast<size_t>(total_points * adaptive_tolerance);

    if (target_points < 1) target_points = 1;

    // Calculate the step size for sampling
    size_t step = 1;
    if (target_points < total_points && adaptive_tolerance < 1.0f) {
        step = std::max(static_cast<size_t>(1), total_points / target_points);
    }

    // Sample points with additional checks for visual significance
    for (size_t i = start_index; i <= end_index && i < chart.dates.size(); i += step) {
        // Check if the element with padding would be visible
        if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
            // For candlestick charts, check if the full candle (high-low range) is visible
            if (should_render_bounding_box(
                    chart.dates[i] - time_padding,
                    chart.dates[i] + time_padding,
                    chart.lows[i],
                    chart.highs[i])) {

                // Add point if it's at the beginning, end, or meets certain criteria
                bool should_add = (i == start_index || i == end_index);

                if (!should_add) {
                    // Check for significant price movement compared to neighbors
                    float current_price = chart.closes[i];
                    float prev_price = (i > 0) ? chart.closes[i-1] : current_price;
                    float next_price = (i < chart.closes.size() - 1) ? chart.closes[i+1] : current_price;

                    float avg_neighbor_price = (prev_price + next_price) / 2.0f;
                    float price_deviation = std::abs(current_price - avg_neighbor_price);
                    float avg_price = (std::abs(prev_price) + std::abs(current_price) + std::abs(next_price)) / 3.0f;

                    // If this point significantly deviates from the line connecting neighbors, keep it
                    if (avg_price > 0 && (price_deviation / avg_price) > 0.005f) { // 0.5% threshold
                        should_add = true;
                    }
                }

                if (should_add) {
                    filtered_dates.push_back(chart.dates[i]);
                    filtered_opens.push_back(chart.opens[i]);
                    filtered_highs.push_back(chart.highs[i]);
                    filtered_lows.push_back(chart.lows[i]);
                    filtered_closes.push_back(chart.closes[i]);
                    filtered_volumes.push_back(chart.volumes[i]);
                }
            }
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

// Method to calculate dynamic padding based on zoom level for better off-screen rendering
double ChartCuller::calculate_dynamic_padding(double zoom_factor, double base_padding_ratio) const {
    // At higher zoom levels, we might need less padding since elements are larger
    // At lower zoom levels, we might need more padding to account for anti-aliasing and visual effects
    if (zoom_factor >= 5.0) {
        // Highly zoomed in - minimal padding needed
        return base_padding_ratio * 0.5;
    } else if (zoom_factor >= 1.0) {
        // Normal zoom - standard padding
        return base_padding_ratio;
    } else {
        // Zoomed out - increased padding to handle visual artifacts
        return base_padding_ratio * 1.5;
    }
}

// Advanced culling method that combines off-screen culling with intelligent polygon reduction
ChartInstance ChartCuller::apply_advanced_offscreen_culling(const ChartInstance& chart, float zoom_factor,
                                                          float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Calculate adaptive LOD based on zoom level and data density
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    size_t visible_points_count = end_index - start_index + 1;
    float adaptive_lod = calculate_advanced_lod_factor(zoom_factor, visible_points_count,
                                                     viewport_width_pixels, viewport_height_pixels);

    // Create new vectors with reduced data
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate dynamic padding based on zoom level for better off-screen rendering
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;
    double time_padding = calculate_dynamic_padding(zoom_factor, time_range * 0.002); // 0.2% padding for time
    double price_padding = calculate_dynamic_padding(zoom_factor, price_range * 0.002); // 0.2% padding for price

    // Apply polygon reduction by reducing the number of points based on LOD
    size_t total_points = end_index - start_index + 1;
    size_t target_points = static_cast<size_t>(total_points * adaptive_lod);

    if (target_points < 1) target_points = 1;

    // Calculate the step size for polygon reduction
    size_t step = 1;
    if (target_points < total_points && adaptive_lod < 1.0f) {
        step = std::max(static_cast<size_t>(1), total_points / target_points);
    }

    // Sample the data based on the step size, considering padding for off-screen elements
    for (size_t i = start_index; i <= end_index && i < chart.dates.size(); i += step) {
        // Check if the element with padding would be visible (for off-screen elements that might affect rendering)
        if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
            // For candlestick charts, we should also check if the full candle (high-low range) is visible
            if (should_render_bounding_box(
                    chart.dates[i] - time_padding,
                    chart.dates[i] + time_padding,
                    chart.lows[i],
                    chart.highs[i])) {

                // Additional check for off-screen elements that might contribute to rendering
                // (like shadows, highlights, etc.) using a threshold
                if (should_render_offscreen_element(chart.dates[i], chart.closes[i], 0.005)) { // 0.5% threshold
                    // Apply importance-based filtering to preserve visually significant elements
                    bool is_important = false;

                    // Check for high volatility
                    float volatility = chart.highs[i] - chart.lows[i];
                    float avg_price = (chart.opens[i] + chart.closes[i]) / 2.0f;
                    float volatility_ratio = (avg_price != 0.0f) ? volatility / std::abs(avg_price) : 0.0f;

                    // Check for significant price change
                    float price_change = std::abs(chart.opens[i] - chart.closes[i]);
                    float change_ratio = (avg_price != 0.0f) ? price_change / std::abs(avg_price) : 0.0f;

                    // Check for high volume
                    float volume_ratio = (chart.volumes[i] > 0) ? chart.volumes[i] / 1000.0f : 0.0f;

                    // Check for trend reversals
                    bool is_trend_reversal = false;
                    if (i > 0 && i < chart.dates.size() - 1) {
                        float prev_change = chart.closes[i-1] - chart.opens[i-1];
                        float curr_change = chart.closes[i] - chart.opens[i];
                        float next_change = chart.closes[i+1] - chart.opens[i+1];

                        is_trend_reversal = ((prev_change > 0 && curr_change < 0) || (prev_change < 0 && curr_change > 0)) ||
                                           ((curr_change > 0 && next_change < 0) || (curr_change < 0 && next_change > 0));
                    }

                    // Determine if this point is "important" enough to include
                    is_important = (volatility_ratio > 0.02f || change_ratio > 0.015f || volume_ratio > 1.0f || is_trend_reversal);

                    // If zoom level is low, be more selective about what to include
                    bool should_include = is_important || (zoom_factor > 1.0f);

                    if (should_include) {
                        filtered_dates.push_back(chart.dates[i]);
                        filtered_opens.push_back(chart.opens[i]);
                        filtered_highs.push_back(chart.highs[i]);
                        filtered_lows.push_back(chart.lows[i]);
                        filtered_closes.push_back(chart.closes[i]);
                        filtered_volumes.push_back(chart.volumes[i]);
                    }
                }
            }
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

// Enhanced off-screen culling with frustum-based visibility detection and polygon reduction
ChartInstance ChartCuller::apply_frustum_culling_and_polygon_reduction(const ChartInstance& chart, float zoom_factor,
                                                                    float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Calculate adaptive LOD based on zoom level and data density
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    size_t visible_points_count = end_index - start_index + 1;
    float adaptive_lod = calculate_advanced_lod_factor(zoom_factor, visible_points_count,
                                                     viewport_width_pixels, viewport_height_pixels);

    // Create new vectors with reduced data
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate dynamic padding based on zoom level for better off-screen rendering
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;
    double time_padding = calculate_dynamic_padding(zoom_factor, time_range * 0.003); // 0.3% padding for time
    double price_padding = calculate_dynamic_padding(zoom_factor, price_range * 0.003); // 0.3% padding for price

    // Calculate the maximum number of points we want to render based on viewport size and zoom
    size_t max_renderable_points = static_cast<size_t>((viewport_width_pixels > 0) ?
                                                       viewport_width_pixels * adaptive_lod :
                                                       visible_points_count * adaptive_lod);

    if (max_renderable_points < 1) max_renderable_points = 1;

    // Frustum culling: Check if each element is within the extended viewport bounds
    size_t points_added = 0;
    for (size_t i = start_index; i <= end_index && i < chart.dates.size(); ++i) {
        // Perform frustum culling with extended bounds to account for off-screen elements
        if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
            // For candlestick charts, check if the full candle (high-low range) is within extended bounds
            if (should_render_bounding_box(
                    chart.dates[i] - time_padding,
                    chart.dates[i] + time_padding,
                    chart.lows[i],
                    chart.highs[i])) {

                // Apply polygon reduction based on zoom level and importance
                bool should_include = false;

                // Calculate importance metrics
                float volatility = chart.highs[i] - chart.lows[i];
                float avg_price = (chart.opens[i] + chart.closes[i]) / 2.0f;
                float volatility_ratio = (avg_price != 0.0f) ? volatility / std::abs(avg_price) : 0.0f;

                float price_change = std::abs(chart.opens[i] - chart.closes[i]);
                float change_ratio = (avg_price != 0.0f) ? price_change / std::abs(avg_price) : 0.0f;

                float volume_ratio = (chart.volumes[i] > 0) ? chart.volumes[i] / 1000.0f : 0.0f;

                // Check for trend reversals
                bool is_trend_reversal = false;
                if (i > 0 && i < chart.dates.size() - 1) {
                    float prev_change = chart.closes[i-1] - chart.opens[i-1];
                    float curr_change = chart.closes[i] - chart.opens[i];
                    float next_change = chart.closes[i+1] - chart.opens[i+1];

                    is_trend_reversal = ((prev_change > 0 && curr_change < 0) || (prev_change < 0 && curr_change > 0)) ||
                                       ((curr_change > 0 && next_change < 0) || (curr_change < 0 && next_change > 0));
                }

                // Determine if this point is important enough to include based on zoom level
                bool is_important = (volatility_ratio > 0.015f || change_ratio > 0.01f || volume_ratio > 0.5f || is_trend_reversal);

                // At lower zoom levels, only include important points or sample regularly
                if (zoom_factor <= 1.0f) {
                    // Low zoom - be more selective
                    should_include = is_important && (points_added < max_renderable_points);
                } else {
                    // Higher zoom - include more points but still apply LOD
                    should_include = (is_important || (points_added < max_renderable_points * adaptive_lod));
                }

                if (should_include) {
                    filtered_dates.push_back(chart.dates[i]);
                    filtered_opens.push_back(chart.opens[i]);
                    filtered_highs.push_back(chart.highs[i]);
                    filtered_lows.push_back(chart.lows[i]);
                    filtered_closes.push_back(chart.closes[i]);
                    filtered_volumes.push_back(chart.volumes[i]);
                    points_added++;
                }
            }
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

// Advanced polygon reduction that reduces polygon count at lower zoom levels using clustering
ChartInstance ChartCuller::apply_clustering_polygon_reduction(const ChartInstance& chart, float zoom_factor,
                                                           float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Calculate adaptive LOD based on zoom level and data density
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    // Create new vectors with reduced data
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate dynamic padding based on zoom level
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;
    double time_padding = calculate_dynamic_padding(zoom_factor, time_range * 0.001); // 0.1% padding for time
    double price_padding = calculate_dynamic_padding(zoom_factor, price_range * 0.001); // 0.1% padding for price

    // Calculate target number of points based on zoom level and viewport size
    size_t target_points = static_cast<size_t>(viewport_width_pixels * zoom_factor * 0.1f); // Adjust multiplier as needed
    if (target_points < 10) target_points = 10; // Minimum number of points to maintain visual quality

    size_t total_points = end_index - start_index + 1;

    if (total_points <= target_points) {
        // No reduction needed, just apply culling
        for (size_t i = start_index; i <= end_index && i < chart.dates.size(); ++i) {
            if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding) &&
                should_render_bounding_box(
                    chart.dates[i] - time_padding,
                    chart.dates[i] + time_padding,
                    chart.lows[i],
                    chart.highs[i])) {

                filtered_dates.push_back(chart.dates[i]);
                filtered_opens.push_back(chart.opens[i]);
                filtered_highs.push_back(chart.highs[i]);
                filtered_lows.push_back(chart.lows[i]);
                filtered_closes.push_back(chart.closes[i]);
                filtered_volumes.push_back(chart.volumes[i]);
            }
        }
    } else {
        // Apply clustering-based reduction
        size_t cluster_size = std::max(static_cast<size_t>(1), total_points / target_points);

        for (size_t cluster_start = start_index; cluster_start <= end_index; cluster_start += cluster_size) {
            size_t cluster_end = std::min(cluster_start + cluster_size - 1, end_index);

            // Find the most representative point in the cluster based on importance
            size_t best_idx = cluster_start;
            float best_score = 0.0f;

            for (size_t j = cluster_start; j <= cluster_end && j < chart.dates.size(); ++j) {
                // Calculate importance score for this point
                float volatility = chart.highs[j] - chart.lows[j];
                float avg_price = (chart.opens[j] + chart.closes[j]) / 2.0f;
                float volatility_ratio = (avg_price != 0.0f) ? volatility / std::abs(avg_price) : 0.0f;

                float price_change = std::abs(chart.opens[j] - chart.closes[j]);
                float change_ratio = (avg_price != 0.0f) ? price_change / std::abs(avg_price) : 0.0f;

                float volume_ratio = (chart.volumes[j] > 0) ? chart.volumes[j] / 1000.0f : 0.0f;

                float score = volatility_ratio * 0.4f + change_ratio * 0.4f + volume_ratio * 0.2f;

                if (score > best_score) {
                    best_score = score;
                    best_idx = j;
                }
            }

            // Only add the best point if it passes the culling checks
            if (should_render_element_with_padding(chart.dates[best_idx], chart.closes[best_idx], time_padding, price_padding) &&
                should_render_bounding_box(
                    chart.dates[best_idx] - time_padding,
                    chart.dates[best_idx] + time_padding,
                    chart.lows[best_idx],
                    chart.highs[best_idx])) {

                filtered_dates.push_back(chart.dates[best_idx]);
                filtered_opens.push_back(chart.opens[best_idx]);
                filtered_highs.push_back(chart.highs[best_idx]);
                filtered_lows.push_back(chart.lows[best_idx]);
                filtered_closes.push_back(chart.closes[best_idx]);
                filtered_volumes.push_back(chart.volumes[best_idx]);
            }
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

// Main method that combines aggressive off-screen culling with zoom-based polygon reduction
ChartInstance ChartCuller::apply_aggressive_culling_and_lod(const ChartInstance& chart, float zoom_factor,
                                                         float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Calculate adaptive LOD based on zoom level and data density
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    size_t visible_points_count = end_index - start_index + 1;
    float adaptive_lod = calculate_advanced_lod_factor(zoom_factor, visible_points_count,
                                                     viewport_width_pixels, viewport_height_pixels);

    // Create new vectors with reduced data
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate dynamic padding based on zoom level for better off-screen rendering
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;

    // At lower zoom levels, we use more aggressive culling with smaller padding
    // At higher zoom levels, we allow more generous padding for visual elements
    double time_padding = calculate_dynamic_padding(zoom_factor, time_range * 0.0005); // Reduced base padding
    double price_padding = calculate_dynamic_padding(zoom_factor, price_range * 0.0005); // Reduced base padding

    // Calculate the maximum number of points we want to render based on viewport size and zoom
    size_t max_renderable_points = static_cast<size_t>((viewport_width_pixels > 0) ?
                                                       viewport_width_pixels * adaptive_lod * 0.8f : // Further reduced for performance
                                                       visible_points_count * adaptive_lod);

    if (max_renderable_points < 5) max_renderable_points = 5; // Minimum number of points to maintain basic chart visibility

    // Aggressive off-screen culling: Only include elements that are definitely within or very near the viewport
    size_t points_added = 0;
    for (size_t i = start_index; i <= end_index && i < chart.dates.size() && points_added < max_renderable_points; ++i) {
        // Perform aggressive culling with tight bounds to exclude definitely invisible elements
        if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
            // For candlestick charts, check if the full candle (high-low range) is within bounds
            if (should_render_bounding_box(
                    chart.dates[i] - time_padding,
                    chart.dates[i] + time_padding,
                    chart.lows[i],
                    chart.highs[i])) {

                // Apply importance-based filtering to preserve visually significant elements at low zoom levels
                bool should_include = false;

                // Calculate importance metrics
                float volatility = chart.highs[i] - chart.lows[i];
                float avg_price = (chart.opens[i] + chart.closes[i]) / 2.0f;
                float volatility_ratio = (avg_price != 0.0f) ? volatility / std::abs(avg_price) : 0.0f;

                float price_change = std::abs(chart.opens[i] - chart.closes[i]);
                float change_ratio = (avg_price != 0.0f) ? price_change / std::abs(avg_price) : 0.0f;

                float volume_ratio = (chart.volumes[i] > 0) ? chart.volumes[i] / 1000.0f : 0.0f;

                // Check for trend reversals
                bool is_trend_reversal = false;
                if (i > 0 && i < chart.dates.size() - 1) {
                    float prev_change = chart.closes[i-1] - chart.opens[i-1];
                    float curr_change = chart.closes[i] - chart.opens[i];
                    float next_change = chart.closes[i+1] - chart.opens[i+1];

                    is_trend_reversal = ((prev_change > 0 && curr_change < 0) || (prev_change < 0 && curr_change > 0)) ||
                                       ((curr_change > 0 && next_change < 0) || (curr_change < 0 && next_change > 0));
                }

                // Determine if this point is important enough to include based on zoom level
                bool is_important = (volatility_ratio > 0.01f || change_ratio > 0.008f || volume_ratio > 0.3f || is_trend_reversal);

                // At very low zoom levels, only include important points
                if (zoom_factor <= 0.5f) {
                    should_include = is_important;
                }
                // At moderate zoom levels, include important points or sample regularly
                else if (zoom_factor <= 2.0f) {
                    should_include = is_important || (points_added % static_cast<size_t>(std::max(1.0f, 3.0f / zoom_factor)) == 0);
                }
                // At higher zoom levels, be more permissive but still apply LOD
                else {
                    should_include = true;
                }

                if (should_include && points_added < max_renderable_points) {
                    filtered_dates.push_back(chart.dates[i]);
                    filtered_opens.push_back(chart.opens[i]);
                    filtered_highs.push_back(chart.highs[i]);
                    filtered_lows.push_back(chart.lows[i]);
                    filtered_closes.push_back(chart.closes[i]);
                    filtered_volumes.push_back(chart.volumes[i]);
                    points_added++;
                }
            }
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

// New method that specifically addresses the task requirements: off-screen chart item culling
// and polygon count reduction at lower zoom levels
ChartInstance ChartCuller::apply_offscreen_culling_with_zoom_reduction(const ChartInstance& chart, float zoom_factor,
                                                                   float viewport_width_pixels, float viewport_height_pixels) const {
    ChartInstance processed_chart = chart;

    if (!viewport_set_ || chart.dates.empty()) {
        // If no viewport is set or chart is empty, return original chart
        return processed_chart;
    }

    // Calculate adaptive LOD based on zoom level and data density
    size_t start_index, end_index;
    get_visible_data_range_optimized(chart, start_index, end_index);

    size_t visible_points_count = end_index - start_index + 1;
    float adaptive_lod = calculate_advanced_lod_factor(zoom_factor, visible_points_count,
                                                     viewport_width_pixels, viewport_height_pixels);

    // Create new vectors with reduced data
    std::vector<double> filtered_dates;
    std::vector<float> filtered_opens;
    std::vector<float> filtered_highs;
    std::vector<float> filtered_lows;
    std::vector<float> filtered_closes;
    std::vector<float> filtered_volumes;

    // Calculate dynamic padding based on zoom level for better off-screen rendering
    double time_range = viewport_.maxTime - viewport_.minTime;
    double price_range = viewport_.maxPrice - viewport_.minPrice;

    // At lower zoom levels, use smaller padding to be more aggressive about culling
    // At higher zoom levels, use larger padding to account for visual elements that extend beyond the data point
    double base_time_padding = time_range * 0.001;  // 0.1% base padding
    double base_price_padding = price_range * 0.001; // 0.1% base padding

    // Adjust padding based on zoom level
    double time_padding = (zoom_factor < 1.0) ? base_time_padding * 0.5 : base_time_padding * std::min(2.0, zoom_factor);
    double price_padding = (zoom_factor < 1.0) ? base_price_padding * 0.5 : base_price_padding * std::min(2.0, zoom_factor);

    // Calculate the maximum number of points we want to render based on viewport size and zoom
    size_t max_renderable_points = static_cast<size_t>((viewport_width_pixels > 0) ?
                                                       viewport_width_pixels * adaptive_lod * 0.7f : // Slightly conservative
                                                       visible_points_count * adaptive_lod);

    if (max_renderable_points < 10) max_renderable_points = 10; // Minimum number of points to maintain basic chart visibility

    // Track points added to respect the maximum limit
    size_t points_added = 0;

    // Pre-calculate importance metrics for all points to avoid repeated calculations
    std::vector<float> importance_scores;
    if (zoom_factor <= 2.0f) { // Only calculate if we need importance scores
        importance_scores.reserve(end_index - start_index + 1);
        for (size_t i = start_index; i <= end_index && i < chart.dates.size(); ++i) {
            float volatility = chart.highs[i] - chart.lows[i];
            float avg_price = (chart.opens[i] + chart.closes[i]) / 2.0f;
            float volatility_ratio = (avg_price != 0.0f) ? volatility / std::abs(avg_price) : 0.0f;

            float price_change = std::abs(chart.opens[i] - chart.closes[i]);
            float change_ratio = (avg_price != 0.0f) ? price_change / std::abs(avg_price) : 0.0f;

            float volume_ratio = (chart.volumes[i] > 0) ? chart.volumes[i] / 1000.0f : 0.0f;

            // Calculate importance score combining multiple factors
            float importance_score = volatility_ratio * 0.4f + change_ratio * 0.4f + volume_ratio * 0.2f;
            importance_scores.push_back(importance_score);
        }
    }

    // Off-screen culling: Only include elements that are within or very near the viewport
    for (size_t i = start_index; i <= end_index && i < chart.dates.size() && points_added < max_renderable_points; ++i) {
        // Perform culling with calculated padding
        if (should_render_element_with_padding(chart.dates[i], chart.closes[i], time_padding, price_padding)) {
            // For candlestick charts, check if the full candle (high-low range) is within bounds
            if (should_render_bounding_box(
                    chart.dates[i] - time_padding,
                    chart.dates[i] + time_padding,
                    chart.lows[i],
                    chart.highs[i])) {

                // Apply zoom-based polygon reduction by skipping points at lower zoom levels
                bool should_include = false;

                // At very low zoom levels, only include points that meet certain criteria
                if (zoom_factor <= 0.5f) {
                    // Use pre-calculated importance score for efficiency
                    size_t importance_idx = i - start_index;
                    if (importance_idx < importance_scores.size()) {
                        // At very low zoom, only include highly important points
                        should_include = importance_scores[importance_idx] > 0.01f; // Threshold for importance
                    } else {
                        // Fallback calculation if index is out of bounds
                        float volatility = chart.highs[i] - chart.lows[i];
                        float avg_price = (chart.opens[i] + chart.closes[i]) / 2.0f;
                        float volatility_ratio = (avg_price != 0.0f) ? volatility / std::abs(avg_price) : 0.0f;

                        float price_change = std::abs(chart.opens[i] - chart.closes[i]);
                        float change_ratio = (avg_price != 0.0f) ? price_change / std::abs(avg_price) : 0.0f;

                        should_include = (volatility_ratio > 0.02f || change_ratio > 0.015f);
                    }
                }
                // At moderate zoom levels, include points based on spacing and importance
                else if (zoom_factor <= 2.0f) {
                    // Include every nth point depending on zoom level to reduce polygon count
                    size_t skip_factor = static_cast<size_t>(std::max(1.0f, 2.0f / zoom_factor));

                    // Use importance score to decide whether to include this point
                    size_t importance_idx = i - start_index;
                    bool is_important = false;
                    if (importance_idx < importance_scores.size()) {
                        is_important = importance_scores[importance_idx] > 0.005f;
                    }

                    should_include = (i % skip_factor == 0) || is_important || points_added < max_renderable_points * 0.05f; // Always include some points
                }
                // At higher zoom levels, be more permissive but still respect the limit
                else {
                    should_include = true;
                }

                if (should_include && points_added < max_renderable_points) {
                    filtered_dates.push_back(chart.dates[i]);
                    filtered_opens.push_back(chart.opens[i]);
                    filtered_highs.push_back(chart.highs[i]);
                    filtered_lows.push_back(chart.lows[i]);
                    filtered_closes.push_back(chart.closes[i]);
                    filtered_volumes.push_back(chart.volumes[i]);
                    points_added++;
                }
            }
        }
    }

    // Replace the chart data with filtered data
    processed_chart.dates = std::move(filtered_dates);
    processed_chart.opens = std::move(filtered_opens);
    processed_chart.highs = std::move(filtered_highs);
    processed_chart.lows = std::move(filtered_lows);
    processed_chart.closes = std::move(filtered_closes);
    processed_chart.volumes = std::move(filtered_volumes);

    return processed_chart;
}

} // namespace RenderEngine
} // namespace BTQuant