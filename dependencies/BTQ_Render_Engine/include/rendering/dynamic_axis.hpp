#pragma once

/**
 * @file dynamic_axis.hpp
 * @brief Dynamic X-Axis (Time) and Y-Axis (Price) Rendering with Zoom-based Grid Steps
 * 
 * This implementation provides:
 * - Automatic grid step calculation based on zoom level
 * - Sub-pixel accurate axis rendering
 * - Time formatting for X-axis (seconds, minutes, hours, days)
 * - Price formatting for Y-axis with appropriate decimal places
 * - Adaptive label density based on available screen space
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <glm/glm.hpp>

namespace btq {
namespace rendering {

/**
 * @brief Time interval types for X-axis
 */
enum class TimeInterval : uint32_t {
    SECOND_1 = 1,
    SECOND_5 = 5,
    SECOND_15 = 15,
    SECOND_30 = 30,
    MINUTE_1 = 60,
    MINUTE_5 = 300,
    MINUTE_15 = 900,
    MINUTE_30 = 1800,
    HOUR_1 = 3600,
    HOUR_4 = 14400,
    HOUR_12 = 43200,
    DAY_1 = 86400,
    DAY_3 = 259200,
    WEEK_1 = 604800,
    MONTH_1 = 2592000
};

/**
 * @brief Grid line data for rendering
 */
struct GridLine {
    float position;         // Position in chart coordinates
    float screen_position;  // Position in screen pixels
    std::string label;      // Display label
    bool is_major;          // Major or minor grid line
};

/**
 * @brief Axis configuration
 */
struct AxisConfig {
    float min_label_spacing = 60.0f;    // Minimum pixels between labels
    float major_line_width = 1.0f;      // Major grid line width
    float minor_line_width = 0.5f;      // Minor grid line width
    float label_font_size = 12.0f;      // Font size for labels
    float axis_margin = 60.0f;          // Margin for axis labels
    glm::vec4 major_color = {0.3f, 0.3f, 0.3f, 0.8f};
    glm::vec4 minor_color = {0.2f, 0.2f, 0.2f, 0.4f};
    glm::vec4 label_color = {0.8f, 0.8f, 0.8f, 1.0f};
};

/**
 * @brief Dynamic X-Axis (Time) generator
 */
class DynamicTimeAxis {
public:
    DynamicTimeAxis() = default;
    
    /**
     * @brief Calculate grid lines for the current view
     * @param time_range Visible time range (start, end)
     * @param screen_width Available screen width in pixels
     * @param config Axis configuration
     * @return Vector of grid lines to render
     */
    std::vector<GridLine> calculateGridLines(
        const glm::vec2& time_range,
        float screen_width,
        const AxisConfig& config = AxisConfig{})
    {
        std::vector<GridLine> lines;
        
        if (screen_width <= 0 || time_range.y <= time_range.x) {
            return lines;
        }
        
        // Calculate time span and pixels per second
        float time_span = time_range.y - time_range.x;
        float pixels_per_second = screen_width / time_span;
        
        // Determine appropriate time interval based on zoom
        TimeInterval interval = calculateOptimalInterval(pixels_per_second, config);
        
        // Calculate grid positions
        float interval_seconds = static_cast<float>(interval);
        
        // Find the first grid line position
        float start_time = std::floor(time_range.x / interval_seconds) * interval_seconds;
        
        // Generate grid lines
        for (float t = start_time; t <= time_range.y; t += interval_seconds) {
            if (t < time_range.x) continue;
            
            GridLine line;
            line.position = t;
            line.screen_position = (t - time_range.x) * pixels_per_second;
            line.label = formatTimeLabel(t, interval);
            line.is_major = isMajorLine(t, interval);
            
            lines.push_back(line);
        }
        
        return lines;
    }
    
    /**
     * @brief Calculate the optimal time interval for the current zoom level
     */
    TimeInterval calculateOptimalInterval(float pixels_per_second, const AxisConfig& config) {
        // Target: approximately one label every min_label_spacing pixels
        float target_interval = config.min_label_spacing / pixels_per_second;
        
        // Find the smallest interval that's larger than or equal to target
        static const TimeInterval intervals[] = {
            TimeInterval::SECOND_1,
            TimeInterval::SECOND_5,
            TimeInterval::SECOND_15,
            TimeInterval::SECOND_30,
            TimeInterval::MINUTE_1,
            TimeInterval::MINUTE_5,
            TimeInterval::MINUTE_15,
            TimeInterval::MINUTE_30,
            TimeInterval::HOUR_1,
            TimeInterval::HOUR_4,
            TimeInterval::HOUR_12,
            TimeInterval::DAY_1,
            TimeInterval::DAY_3,
            TimeInterval::WEEK_1,
            TimeInterval::MONTH_1
        };
        
        for (TimeInterval interval : intervals) {
            if (static_cast<float>(interval) >= target_interval) {
                return interval;
            }
        }
        
        return TimeInterval::MONTH_1;
    }
    
    /**
     * @brief Format time label based on interval
     */
    std::string formatTimeLabel(float timestamp, TimeInterval interval) {
        // Convert to time_point
        auto time = std::chrono::system_clock::time_point{
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::seconds(static_cast<int64_t>(timestamp))
            )
        };
        
        std::time_t time_t_val = std::chrono::system_clock::to_time_t(time);
        std::tm* tm = std::localtime(&time_t_val);
        
        char buffer[32];
        
        switch (interval) {
            case TimeInterval::SECOND_1:
            case TimeInterval::SECOND_5:
            case TimeInterval::SECOND_15:
            case TimeInterval::SECOND_30:
                // Show HH:MM:SS
                std::strftime(buffer, sizeof(buffer), "%H:%M:%S", tm);
                break;
                
            case TimeInterval::MINUTE_1:
            case TimeInterval::MINUTE_5:
            case TimeInterval::MINUTE_15:
            case TimeInterval::MINUTE_30:
                // Show HH:MM
                std::strftime(buffer, sizeof(buffer), "%H:%M", tm);
                break;
                
            case TimeInterval::HOUR_1:
            case TimeInterval::HOUR_4:
            case TimeInterval::HOUR_12:
                // Show DD HH:MM
                std::strftime(buffer, sizeof(buffer), "%d %H:%M", tm);
                break;
                
            case TimeInterval::DAY_1:
            case TimeInterval::DAY_3:
            case TimeInterval::WEEK_1:
                // Show MMM DD
                std::strftime(buffer, sizeof(buffer), "%b %d", tm);
                break;
                
            case TimeInterval::MONTH_1:
                // Show MMM YYYY
                std::strftime(buffer, sizeof(buffer), "%b %Y", tm);
                break;
                
            default:
                std::strftime(buffer, sizeof(buffer), "%H:%M", tm);
                break;
        }
        
        return std::string(buffer);
    }
    
    /**
     * @brief Check if this is a major grid line
     */
    bool isMajorLine(float timestamp, TimeInterval interval) {
        auto time = std::chrono::system_clock::time_point{
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::seconds(static_cast<int64_t>(timestamp))
            )
        };
        
        std::time_t time_t_val = std::chrono::system_clock::to_time_t(time);
        std::tm* tm = std::localtime(&time_t_val);
        
        switch (interval) {
            case TimeInterval::SECOND_1:
            case TimeInterval::SECOND_5:
            case TimeInterval::SECOND_15:
            case TimeInterval::SECOND_30:
                // Major on minute boundary
                return tm->tm_sec == 0;
                
            case TimeInterval::MINUTE_1:
            case TimeInterval::MINUTE_5:
            case TimeInterval::MINUTE_15:
            case TimeInterval::MINUTE_30:
                // Major on hour boundary
                return tm->tm_min == 0;
                
            case TimeInterval::HOUR_1:
            case TimeInterval::HOUR_4:
            case TimeInterval::HOUR_12:
                // Major on day boundary
                return tm->tm_hour == 0;
                
            case TimeInterval::DAY_1:
            case TimeInterval::DAY_3:
            case TimeInterval::WEEK_1:
                // Major on month boundary
                return tm->tm_mday == 1;
                
            case TimeInterval::MONTH_1:
                // Major on year boundary
                return tm->tm_mon == 0;
                
            default:
                return false;
        }
    }
};

/**
 * @brief Dynamic Y-Axis (Price) generator
 */
class DynamicPriceAxis {
public:
    DynamicPriceAxis() = default;
    
    /**
     * @brief Calculate grid lines for the current view
     * @param price_range Visible price range (low, high)
     * @param screen_height Available screen height in pixels
     * @param config Axis configuration
     * @return Vector of grid lines to render
     */
    std::vector<GridLine> calculateGridLines(
        const glm::vec2& price_range,
        float screen_height,
        const AxisConfig& config = AxisConfig{})
    {
        std::vector<GridLine> lines;
        
        if (screen_height <= 0 || price_range.y <= price_range.x) {
            return lines;
        }
        
        // Calculate price span and pixels per price unit
        float price_span = price_range.y - price_range.x;
        float pixels_per_price = screen_height / price_span;
        
        // Determine appropriate price step based on zoom
        float price_step = calculateOptimalPriceStep(price_span, screen_height, config);
        
        // Find the first grid line position
        float start_price = std::floor(price_range.x / price_step) * price_step;
        
        // Generate grid lines
        for (float p = start_price; p <= price_range.y; p += price_step) {
            if (p < price_range.x) continue;
            
            GridLine line;
            line.position = p;
            // Y-axis is inverted (0 at top)
            line.screen_position = screen_height - (p - price_range.x) * pixels_per_price;
            line.label = formatPriceLabel(p, price_step);
            line.is_major = isMajorLine(p, price_step);
            
            lines.push_back(line);
        }
        
        return lines;
    }
    
    /**
     * @brief Calculate the optimal price step for the current zoom level
     */
    float calculateOptimalPriceStep(float price_span, float screen_height, const AxisConfig& config) {
        // Target: approximately one label every min_label_spacing pixels
        float target_steps = screen_height / config.min_label_spacing;
        float raw_step = price_span / target_steps;
        
        // Round to a "nice" number (1, 2, 5, 10, 20, 50, etc.)
        return roundToNiceNumber(raw_step);
    }
    
    /**
     * @brief Round to a "nice" number for display
     */
    float roundToNiceNumber(float value) {
        if (value <= 0) return 0.0001f;
        
        // Find the magnitude
        float magnitude = std::pow(10.0f, std::floor(std::log10(value)));
        float normalized = value / magnitude;
        
        // Round to 1, 2, or 5
        float nice_normalized;
        if (normalized < 1.5f) {
            nice_normalized = 1.0f;
        } else if (normalized < 3.0f) {
            nice_normalized = 2.0f;
        } else if (normalized < 7.0f) {
            nice_normalized = 5.0f;
        } else {
            nice_normalized = 10.0f;
        }
        
        return nice_normalized * magnitude;
    }
    
    /**
     * @brief Format price label with appropriate decimal places
     */
    std::string formatPriceLabel(float price, float step) {
        // Determine decimal places based on step size
        int decimal_places = 0;
        if (step < 0.0001f) {
            decimal_places = 6;
        } else if (step < 0.001f) {
            decimal_places = 5;
        } else if (step < 0.01f) {
            decimal_places = 4;
        } else if (step < 0.1f) {
            decimal_places = 3;
        } else if (step < 1.0f) {
            decimal_places = 2;
        } else if (step < 10.0f) {
            decimal_places = 1;
        } else {
            decimal_places = 0;
        }
        
        // Format with thousands separator for large numbers
        if (price >= 10000.0f) {
            return formatWithThousands(price, decimal_places);
        }
        
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.*f", decimal_places, price);
        return std::string(buffer);
    }
    
    /**
     * @brief Format large numbers with thousands separators
     */
    std::string formatWithThousands(float price, int decimal_places) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.*f", decimal_places, price);
        
        std::string result;
        std::string num(buffer);
        
        // Find decimal point
        size_t decimal_pos = num.find('.');
        if (decimal_pos == std::string::npos) {
            decimal_pos = num.length();
        }
        
        // Add thousands separators
        int count = 0;
        for (int i = static_cast<int>(decimal_pos) - 1; i >= 0; --i) {
            if (count > 0 && count % 3 == 0) {
                result = ',' + result;
            }
            result = num[i] + result;
            ++count;
        }
        
        // Add decimal part
        if (decimal_pos < num.length()) {
            result += num.substr(decimal_pos);
        }
        
        return result;
    }
    
    /**
     * @brief Check if this is a major grid line
     */
    bool isMajorLine(float price, float step) {
        // Every 5th line is major
        float major_interval = step * 5;
        float remainder = std::fmod(price, major_interval);
        return std::abs(remainder) < step * 0.01f || std::abs(remainder - major_interval) < step * 0.01f;
    }
};

/**
 * @brief Combined axis renderer for both X and Y axes
 */
class AxisRenderer {
public:
    AxisRenderer() = default;
    
    /**
     * @brief Calculate all grid lines for the current view
     */
    struct AxisData {
        std::vector<GridLine> x_axis_lines;
        std::vector<GridLine> y_axis_lines;
    };
    
    AxisData calculateAxes(
        const glm::vec2& time_range,
        const glm::vec2& price_range,
        float screen_width,
        float screen_height,
        const AxisConfig& config = AxisConfig{})
    {
        AxisData data;
        data.x_axis_lines = time_axis_.calculateGridLines(time_range, screen_width, config);
        data.y_axis_lines = price_axis_.calculateGridLines(price_range, screen_height, config);
        return data;
    }
    
private:
    DynamicTimeAxis time_axis_;
    DynamicPriceAxis price_axis_;
};

} // namespace rendering
} // namespace btq
