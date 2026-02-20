#pragma once

/**
 * @file crosshair.hpp
 * @brief Hardware-Accelerated Crosshair with Price/Time Interpolation
 * 
 * This implementation provides:
 * - GPU-rendered crosshair lines
 * - Real-time price/time coordinate interpolation from camera matrix
 * - Sub-pixel accurate positioning
 * - Magnet mode support for snapping to OHLC values
 * - Price and time label rendering
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <string>

namespace btq {
namespace rendering {

/**
 * @brief Crosshair style configuration
 */
struct CrosshairStyle {
    glm::vec4 line_color = {0.5f, 0.5f, 0.5f, 0.8f};
    glm::vec4 label_background = {0.1f, 0.1f, 0.1f, 0.9f};
    glm::vec4 label_text = {1.0f, 1.0f, 1.0f, 1.0f};
    float line_width = 1.0f;
    float label_padding = 4.0f;
    float label_font_size = 12.0f;
    bool show_price_label = true;
    bool show_time_label = true;
    bool show_dot = true;
    float dot_radius = 4.0f;
};

/**
 * @brief Crosshair position data
 */
struct CrosshairPosition {
    glm::vec2 screen_pos;       // Screen coordinates (pixels)
    glm::vec2 chart_pos;        // Chart coordinates (time, price)
    float interpolated_price;   // Interpolated price at cursor
    float interpolated_time;    // Interpolated time at cursor
    int candle_index;           // Index of hovered candle (-1 if none)
    bool is_valid;              // Whether position is valid
};

/**
 * @brief Hardware-accelerated crosshair renderer
 */
class CrosshairRenderer {
public:
    CrosshairRenderer() = default;
    
    /**
     * @brief Initialize the crosshair renderer
     */
    void initialize(const CrosshairStyle& style = CrosshairStyle{}) {
        style_ = style;
    }
    
    /**
     * @brief Calculate crosshair position from mouse coordinates
     * @param mouse_pos Mouse position in screen coordinates
     * @param view_proj_matrix View-projection matrix of the chart camera
     * @param chart_bounds Bounds of the chart area (x, y, width, height)
     * @param time_range Visible time range
     * @param price_range Visible price range
     * @return Crosshair position data
     */
    CrosshairPosition calculatePosition(
        const glm::vec2& mouse_pos,
        const glm::mat4& view_proj_matrix,
        const glm::vec4& chart_bounds,
        const glm::vec2& time_range,
        const glm::vec2& price_range) const
    {
        CrosshairPosition pos;
        pos.screen_pos = mouse_pos;
        pos.is_valid = false;
        pos.candle_index = -1;
        
        // Check if mouse is within chart bounds
        if (mouse_pos.x < chart_bounds.x || 
            mouse_pos.x > chart_bounds.x + chart_bounds.z ||
            mouse_pos.y < chart_bounds.y || 
            mouse_pos.y > chart_bounds.y + chart_bounds.w) {
            return pos;
        }
        
        pos.is_valid = true;
        
        // Calculate inverse view-projection matrix
        glm::mat4 inv_view_proj = glm::inverse(view_proj_matrix);
        
        // Convert screen position to normalized device coordinates
        float ndc_x = (2.0f * (mouse_pos.x - chart_bounds.x) / chart_bounds.z) - 1.0f;
        float ndc_y = 1.0f - (2.0f * (mouse_pos.y - chart_bounds.y) / chart_bounds.w);
        
        // Transform to chart coordinates
        glm::vec4 ndc_pos = glm::vec4(ndc_x, ndc_y, 0.0f, 1.0f);
        glm::vec4 chart_pos_4 = inv_view_proj * ndc_pos;
        
        pos.chart_pos = glm::vec2(chart_pos_4.x, chart_pos_4.y);
        pos.interpolated_time = pos.chart_pos.x;
        pos.interpolated_price = pos.chart_pos.y;
        
        // Clamp to visible ranges
        pos.interpolated_time = std::clamp(pos.interpolated_time, time_range.x, time_range.y);
        pos.interpolated_price = std::clamp(pos.interpolated_price, price_range.x, price_range.y);
        
        return pos;
    }
    
    /**
     * @brief Find the nearest candle to the cursor
     * @param time Cursor time position
     * @param candle_times Array of candle times
     * @param candle_count Number of candles
     * @return Index of nearest candle, or -1 if none
     */
    int findNearestCandle(float time, const float* candle_times, size_t candle_count) const {
        if (candle_count == 0 || candle_times == nullptr) {
            return -1;
        }
        
        // Binary search for nearest candle
        int left = 0;
        int right = static_cast<int>(candle_count) - 1;
        
        while (left < right) {
            int mid = (left + right) / 2;
            if (candle_times[mid] < time) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        // Check which of left-1, left, left+1 is closest
        int best = left;
        float best_dist = std::abs(candle_times[left] - time);
        
        if (left > 0) {
            float dist = std::abs(candle_times[left - 1] - time);
            if (dist < best_dist) {
                best = left - 1;
                best_dist = dist;
            }
        }
        
        if (left < static_cast<int>(candle_count) - 1) {
            float dist = std::abs(candle_times[left + 1] - time);
            if (dist < best_dist) {
                best = left + 1;
            }
        }
        
        return best;
    }
    
    /**
     * @brief Snap price to nearest OHLC value (magnet mode)
     * @param price Current price
     * @param open Candle open price
     * @param high Candle high price
     * @param low Candle low price
     * @param close Candle close price
     * @return Nearest OHLC price
     */
    float snapToOHLC(float price, float open, float high, float low, float close) const {
        float nearest = open;
        float min_dist = std::abs(price - open);
        
        float dist_high = std::abs(price - high);
        if (dist_high < min_dist) {
            min_dist = dist_high;
            nearest = high;
        }
        
        float dist_low = std::abs(price - low);
        if (dist_low < min_dist) {
            min_dist = dist_low;
            nearest = low;
        }
        
        float dist_close = std::abs(price - close);
        if (dist_close < min_dist) {
            nearest = close;
        }
        
        return nearest;
    }
    
    /**
     * @brief Format price for display
     */
    std::string formatPrice(float price, float reference_price = 0.0f) const {
        // Determine decimal places based on price magnitude
        int decimal_places = 2;
        float abs_price = std::abs(price);
        
        if (abs_price < 0.01f) {
            decimal_places = 6;
        } else if (abs_price < 1.0f) {
            decimal_places = 4;
        } else if (abs_price < 100.0f) {
            decimal_places = 2;
        } else if (abs_price < 10000.0f) {
            decimal_places = 2;
        } else {
            decimal_places = 1;
        }
        
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.*f", decimal_places, price);
        return std::string(buffer);
    }
    
    /**
     * @brief Format time for display
     */
    std::string formatTime(float timestamp) const {
        auto time = std::chrono::system_clock::time_point{
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::seconds(static_cast<int64_t>(timestamp))
            )
        };
        
        std::time_t time_t_val = std::chrono::system_clock::to_time_t(time);
        std::tm* tm = std::localtime(&time_t_val);
        
        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm);
        return std::string(buffer);
    }
    
    /**
     * @brief Get the current style
     */
    const CrosshairStyle& getStyle() const { return style_; }
    
    /**
     * @brief Set the style
     */
    void setStyle(const CrosshairStyle& style) { style_ = style; }

private:
    CrosshairStyle style_;
};

/**
 * @brief Crosshair vertex data for GPU rendering
 */
struct CrosshairVertex {
    glm::vec2 position;
    glm::vec4 color;
};

/**
 * @brief Crosshair geometry generator
 */
class CrosshairGeometry {
public:
    /**
     * @brief Generate vertices for crosshair lines
     * @param screen_pos Screen position of crosshair
     * @param chart_bounds Chart bounds (x, y, width, height)
     * @param style Crosshair style
     * @return Array of vertices for line rendering
     */
    static std::vector<CrosshairVertex> generateLines(
        const glm::vec2& screen_pos,
        const glm::vec4& chart_bounds,
        const CrosshairStyle& style)
    {
        std::vector<CrosshairVertex> vertices;
        
        // Horizontal line (full width)
        // Left segment
        vertices.push_back({{chart_bounds.x, screen_pos.y}, style.line_color});
        vertices.push_back({{screen_pos.x - 1.0f, screen_pos.y}, style.line_color});
        
        // Right segment
        vertices.push_back({{screen_pos.x + 1.0f, screen_pos.y}, style.line_color});
        vertices.push_back({{chart_bounds.x + chart_bounds.z, screen_pos.y}, style.line_color});
        
        // Vertical line (full height)
        // Top segment
        vertices.push_back({{screen_pos.x, chart_bounds.y}, style.line_color});
        vertices.push_back({{screen_pos.x, screen_pos.y - 1.0f}, style.line_color});
        
        // Bottom segment
        vertices.push_back({{screen_pos.x, screen_pos.y + 1.0f}, style.line_color});
        vertices.push_back({{screen_pos.x, chart_bounds.y + chart_bounds.w}, style.line_color});
        
        return vertices;
    }
    
    /**
     * @brief Generate vertices for crosshair dot
     */
    static std::vector<CrosshairVertex> generateDot(
        const glm::vec2& screen_pos,
        float radius,
        const glm::vec4& color,
        int segments = 16)
    {
        std::vector<CrosshairVertex> vertices;
        
        for (int i = 0; i < segments; ++i) {
            float angle1 = (2.0f * 3.14159f * i) / segments;
            float angle2 = (2.0f * 3.14159f * (i + 1)) / segments;
            
            // Center
            vertices.push_back({screen_pos, color});
            
            // First point on circle
            glm::vec2 p1 = screen_pos + glm::vec2(std::cos(angle1), std::sin(angle1)) * radius;
            vertices.push_back({p1, color});
            
            // Second point on circle
            glm::vec2 p2 = screen_pos + glm::vec2(std::cos(angle2), std::sin(angle2)) * radius;
            vertices.push_back({p2, color});
        }
        
        return vertices;
    }
};

/**
 * @brief OHLC data for magnet mode snapping
 */
struct OHLCData {
    float open;
    float high;
    float low;
    float close;
    float time;
};

/**
 * @brief Magnet mode helper for cursor snapping
 */
class MagnetMode {
public:
    MagnetMode() = default;
    
    /**
     * @brief Enable or disable magnet mode
     */
    void setEnabled(bool enabled) { enabled_ = enabled; }
    
    /**
     * @brief Check if magnet mode is enabled
     */
    bool isEnabled() const { return enabled_; }
    
    /**
     * @brief Find the snap position for the cursor
     * @param cursor_pos Current cursor position in chart coordinates
     * @param candles Array of OHLC data
     * @param candle_count Number of candles
     * @param snap_threshold_pixels Snap threshold in screen pixels
     * @param pixels_per_price Pixels per price unit
     * @return Snapped position, or original if no snap
     */
    glm::vec2 findSnapPosition(
        const glm::vec2& cursor_pos,
        const OHLCData* candles,
        size_t candle_count,
        float snap_threshold_pixels = 10.0f,
        float pixels_per_price = 1.0f) const
    {
        if (!enabled_ || candles == nullptr || candle_count == 0) {
            return cursor_pos;
        }
        
        // Find nearest candle by time
        size_t nearest_idx = 0;
        float min_time_dist = std::abs(candles[0].time - cursor_pos.x);
        
        for (size_t i = 1; i < candle_count; ++i) {
            float dist = std::abs(candles[i].time - cursor_pos.x);
            if (dist < min_time_dist) {
                min_time_dist = dist;
                nearest_idx = i;
            }
        }
        
        const OHLCData& candle = candles[nearest_idx];
        
        // Find nearest OHLC price
        float snap_threshold_price = snap_threshold_pixels / pixels_per_price;
        
        float prices[] = {candle.open, candle.high, candle.low, candle.close};
        float nearest_price = cursor_pos.y;
        float min_price_dist = snap_threshold_price + 1.0f;  // Start above threshold
        
        for (float price : prices) {
            float dist = std::abs(price - cursor_pos.y);
            if (dist < min_price_dist && dist <= snap_threshold_price) {
                min_price_dist = dist;
                nearest_price = price;
            }
        }
        
        // Return snapped position
        return glm::vec2(candle.time, nearest_price);
    }

private:
    bool enabled_ = false;
};

} // namespace rendering
} // namespace btq
