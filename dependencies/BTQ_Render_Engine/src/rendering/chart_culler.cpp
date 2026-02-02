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

// Enhanced version that considers bounding box for rectangular elements like candles
bool ChartCuller::should_render_bounding_box(double min_time, double max_time, double min_price, double max_price) const {
    if (!viewport_set_) {
        // If no viewport is set, render everything
        return true;
    }

    // Check if the bounding box intersects with the viewport
    if (max_time < viewport_.minTime || min_time > viewport_.maxTime) {
        return false; // Completely outside time range
    }

    if (max_price < viewport_.minPrice || min_price > viewport_.maxPrice) {
        return false; // Completely outside price range
    }

    // Bounding box intersects with viewport
    return true;
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

} // namespace RenderEngine
} // namespace BTQuant