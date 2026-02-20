#pragma once

/**
 * @file volume_profile.hpp
 * @brief Volume Profile / Visible Range Rendering on Vulkan Chart
 * 
 * This implementation provides:
 * - Volume-at-price histogram rendering
 * - Visible range volume profile
 * - Session volume profile
 * - Delta volume (buy vs sell pressure)
 * - Point of Control (POC) identification
 * - Value Area calculation (70% volume)
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <glm/glm.hpp>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace btq {
namespace rendering {

/**
 * @brief Volume profile configuration
 */
struct VolumeProfileConfig {
    int num_price_levels = 100;         // Number of price levels
    float histogram_width = 150.0f;     // Width of histogram bars
    float row_height = 2.0f;            // Height of each price row
    bool show_poc = true;               // Show Point of Control
    bool show_value_area = true;        // Show Value Area
    float value_area_percent = 0.70f;   // Value Area percentage (70%)
    bool show_delta = true;             // Show buy/sell delta
    
    glm::vec4 bid_volume_color = {0.0f, 0.6f, 0.0f, 0.7f};
    glm::vec4 ask_volume_color = {0.8f, 0.0f, 0.0f, 0.7f};
    glm::vec4 poc_color = {1.0f, 1.0f, 0.0f, 1.0f};
    glm::vec4 value_area_color = {0.5f, 0.5f, 0.5f, 0.3f};
    glm::vec4 neutral_color = {0.5f, 0.5f, 0.5f, 0.5f};
};

/**
 * @brief Single price level volume data
 */
struct VolumeLevel {
    double price = 0.0;
    double volume = 0.0;
    double bid_volume = 0.0;
    double ask_volume = 0.0;
    double delta = 0.0;             // bid_volume - ask_volume
    float y_position = 0.0f;
    float bar_width = 0.0f;
    float bid_width = 0.0f;
    float ask_width = 0.0f;
    bool is_poc = false;
    bool in_value_area = false;
};

/**
 * @brief Volume profile result
 */
struct VolumeProfileResult {
    std::vector<VolumeLevel> levels;
    double total_volume = 0.0;
    double total_bid_volume = 0.0;
    double total_ask_volume = 0.0;
    double poc_price = 0.0;         // Point of Control
    double vah_price = 0.0;         // Value Area High
    double val_price = 0.0;         // Value Area Low
    double max_volume = 0.0;
    bool is_valid = false;
};

/**
 * @brief Volume Profile Calculator
 */
class VolumeProfileCalculator {
public:
    VolumeProfileCalculator() = default;
    
    /**
     * @brief Calculate volume profile from OHLCV data
     * @param highs High prices
     * @param lows Low prices
     * @param volumes Volume data
     * @param closes Close prices (for determining buy/sell)
     * @param opens Open prices (for determining buy/sell)
     * @param min_price Minimum price for range
     * @param max_price Maximum price for range
     * @param config Configuration
     * @return Volume profile result
     */
    VolumeProfileResult calculate(
        const std::vector<double>& highs,
        const std::vector<double>& lows,
        const std::vector<double>& volumes,
        const std::vector<double>& closes,
        const std::vector<double>& opens,
        double min_price,
        double max_price,
        const VolumeProfileConfig& config = VolumeProfileConfig{})
    {
        VolumeProfileResult result;
        
        const size_t n = highs.size();
        if (n == 0 || n != lows.size() || n != volumes.size()) {
            return result;
        }
        
        // Calculate price step
        const double price_step = (max_price - min_price) / config.num_price_levels;
        
        // Initialize levels
        result.levels.resize(config.num_price_levels);
        for (int i = 0; i < config.num_price_levels; ++i) {
            result.levels[i].price = min_price + (i + 0.5) * price_step;
        }
        
        // Distribute volume across price levels
        for (size_t i = 0; i < n; ++i) {
            const double high = highs[i];
            const double low = lows[i];
            const double volume = volumes[i];
            const double close = closes[i];
            const double open = opens[i];
            
            // Determine if candle is bullish or bearish
            const bool is_bullish = close >= open;
            
            // Calculate price range for this candle
            const int low_level = static_cast<int>((low - min_price) / price_step);
            const int high_level = static_cast<int>((high - min_price) / price_step);
            
            // Distribute volume evenly across the candle's range
            const int num_levels = std::max(1, high_level - low_level + 1);
            const double volume_per_level = volume / num_levels;
            
            for (int level = std::max(0, low_level); 
                 level <= std::min(config.num_price_levels - 1, high_level); 
                 ++level) {
                result.levels[level].volume += volume_per_level;
                
                if (is_bullish) {
                    result.levels[level].bid_volume += volume_per_level;
                } else {
                    result.levels[level].ask_volume += volume_per_level;
                }
            }
            
            result.total_volume += volume;
            if (is_bullish) {
                result.total_bid_volume += volume;
            } else {
                result.total_ask_volume += volume;
            }
        }
        
        // Calculate delta for each level
        for (auto& level : result.levels) {
            level.delta = level.bid_volume - level.ask_volume;
        }
        
        // Find max volume and POC
        result.max_volume = 0.0;
        int poc_index = 0;
        for (int i = 0; i < config.num_price_levels; ++i) {
            if (result.levels[i].volume > result.max_volume) {
                result.max_volume = result.levels[i].volume;
                poc_index = i;
            }
        }
        
        result.levels[poc_index].is_poc = true;
        result.poc_price = result.levels[poc_index].price;
        
        // Calculate Value Area (70% of volume around POC)
        if (config.show_value_area) {
            calculateValueArea(result, config);
        }
        
        result.is_valid = true;
        return result;
    }
    
    /**
     * @brief Calculate volume profile from tick data
     */
    VolumeProfileResult calculateFromTicks(
        const std::vector<double>& prices,
        const std::vector<double>& volumes,
        const std::vector<bool>& is_buy,
        double min_price,
        double max_price,
        const VolumeProfileConfig& config = VolumeProfileConfig{})
    {
        VolumeProfileResult result;
        
        const size_t n = prices.size();
        if (n == 0 || n != volumes.size() || n != is_buy.size()) {
            return result;
        }
        
        const double price_step = (max_price - min_price) / config.num_price_levels;
        
        result.levels.resize(config.num_price_levels);
        for (int i = 0; i < config.num_price_levels; ++i) {
            result.levels[i].price = min_price + (i + 0.5) * price_step;
        }
        
        // Distribute tick volumes
        for (size_t i = 0; i < n; ++i) {
            const double price = prices[i];
            const double volume = volumes[i];
            
            int level = static_cast<int>((price - min_price) / price_step);
            level = std::clamp(level, 0, config.num_price_levels - 1);
            
            result.levels[level].volume += volume;
            
            if (is_buy[i]) {
                result.levels[level].bid_volume += volume;
                result.total_bid_volume += volume;
            } else {
                result.levels[level].ask_volume += volume;
                result.total_ask_volume += volume;
            }
            
            result.total_volume += volume;
        }
        
        // Calculate delta
        for (auto& level : result.levels) {
            level.delta = level.bid_volume - level.ask_volume;
        }
        
        // Find POC
        result.max_volume = 0.0;
        int poc_index = 0;
        for (int i = 0; i < config.num_price_levels; ++i) {
            if (result.levels[i].volume > result.max_volume) {
                result.max_volume = result.levels[i].volume;
                poc_index = i;
            }
        }
        
        result.levels[poc_index].is_poc = true;
        result.poc_price = result.levels[poc_index].price;
        
        if (config.show_value_area) {
            calculateValueArea(result, config);
        }
        
        result.is_valid = true;
        return result;
    }
    
    /**
     * @brief Calculate bar widths for rendering
     */
    void calculateBarWidths(
        VolumeProfileResult& result,
        float max_bar_width,
        bool split_bid_ask = true) const
    {
        if (result.max_volume == 0.0) return;
        
        for (auto& level : result.levels) {
            if (split_bid_ask) {
                level.bid_width = static_cast<float>(level.bid_volume / result.max_volume) * max_bar_width;
                level.ask_width = static_cast<float>(level.ask_volume / result.max_volume) * max_bar_width;
            } else {
                level.bar_width = static_cast<float>(level.volume / result.max_volume) * max_bar_width;
            }
        }
    }

private:
    void calculateValueArea(VolumeProfileResult& result, const VolumeProfileConfig& config) {
        const double target_volume = result.total_volume * config.value_area_percent;
        
        // Start from POC and expand
        int poc_index = 0;
        for (int i = 0; i < static_cast<int>(result.levels.size()); ++i) {
            if (result.levels[i].is_poc) {
                poc_index = i;
                break;
            }
        }
        
        double accumulated_volume = result.levels[poc_index].volume;
        int low_index = poc_index;
        int high_index = poc_index;
        
        result.levels[poc_index].in_value_area = true;
        
        while (accumulated_volume < target_volume) {
            // Check which direction has more volume
            double low_volume = (low_index > 0) ? result.levels[low_index - 1].volume : 0.0;
            double high_volume = (high_index < static_cast<int>(result.levels.size()) - 1) 
                ? result.levels[high_index + 1].volume : 0.0;
            
            if (low_volume >= high_volume && low_index > 0) {
                --low_index;
                accumulated_volume += result.levels[low_index].volume;
                result.levels[low_index].in_value_area = true;
            } else if (high_index < static_cast<int>(result.levels.size()) - 1) {
                ++high_index;
                accumulated_volume += result.levels[high_index].volume;
                result.levels[high_index].in_value_area = true;
            } else {
                break;
            }
        }
        
        result.vah_price = result.levels[high_index].price;
        result.val_price = result.levels[low_index].price;
    }
};

/**
 * @brief Volume Profile Renderer for Vulkan
 */
class VolumeProfileRenderer {
public:
    VolumeProfileRenderer() = default;
    
    /**
     * @brief Initialize the renderer
     */
    void initialize(const VolumeProfileConfig& config = VolumeProfileConfig{}) {
        config_ = config;
    }
    
    /**
     * @brief Generate vertex data for volume profile histogram
     */
    struct VertexData {
        glm::vec2 position;
        glm::vec4 color;
    };
    
    std::vector<VertexData> generateHistogramVertices(
        const VolumeProfileResult& profile,
        float x_offset,
        float chart_height,
        float chart_y_offset,
        float pixels_per_price,
        bool render_on_right = true)
    {
        std::vector<VertexData> vertices;
        
        if (!profile.is_valid) return vertices;
        
        const float row_height = config_.row_height;
        
        for (const auto& level : profile.levels) {
            if (level.volume == 0.0) continue;
            
            float y = chart_y_offset + chart_height - 
                static_cast<float>((level.price - profile.levels.front().price) * pixels_per_price);
            
            // Determine color based on delta
            glm::vec4 color;
            if (level.is_poc) {
                color = config_.poc_color;
            } else if (level.in_value_area) {
                color = config_.value_area_color;
            } else if (level.delta > 0) {
                color = config_.bid_volume_color;
            } else if (level.delta < 0) {
                color = config_.ask_volume_color;
            } else {
                color = config_.neutral_color;
            }
            
            // Generate quad for this level
            float bar_width = level.bar_width;
            if (bar_width > 0.0f) {
                float x_start = render_on_right ? x_offset : x_offset - bar_width;
                
                // Two triangles for quad
                // Triangle 1
                vertices.push_back({{x_start, y - row_height / 2}, color});
                vertices.push_back({{x_start + bar_width, y - row_height / 2}, color});
                vertices.push_back({{x_start, y + row_height / 2}, color});
                
                // Triangle 2
                vertices.push_back({{x_start + bar_width, y - row_height / 2}, color});
                vertices.push_back({{x_start + bar_width, y + row_height / 2}, color});
                vertices.push_back({{x_start, y + row_height / 2}, color});
            }
        }
        
        return vertices;
    }
    
    /**
     * @brief Generate split bid/ask histogram
     */
    std::vector<VertexData> generateSplitHistogramVertices(
        const VolumeProfileResult& profile,
        float x_center,
        float chart_height,
        float chart_y_offset,
        float pixels_per_price)
    {
        std::vector<VertexData> vertices;
        
        if (!profile.is_valid) return vertices;
        
        const float row_height = config_.row_height;
        
        for (const auto& level : profile.levels) {
            if (level.volume == 0.0) continue;
            
            float y = chart_y_offset + chart_height - 
                static_cast<float>((level.price - profile.levels.front().price) * pixels_per_price);
            
            // Bid bar (left side)
            if (level.bid_width > 0.0f) {
                float x_start = x_center - level.bid_width;
                
                vertices.push_back({{x_start, y - row_height / 2}, config_.bid_volume_color});
                vertices.push_back({{x_center, y - row_height / 2}, config_.bid_volume_color});
                vertices.push_back({{x_start, y + row_height / 2}, config_.bid_volume_color});
                
                vertices.push_back({{x_center, y - row_height / 2}, config_.bid_volume_color});
                vertices.push_back({{x_center, y + row_height / 2}, config_.bid_volume_color});
                vertices.push_back({{x_start, y + row_height / 2}, config_.bid_volume_color});
            }
            
            // Ask bar (right side)
            if (level.ask_width > 0.0f) {
                float x_end = x_center + level.ask_width;
                
                vertices.push_back({{x_center, y - row_height / 2}, config_.ask_volume_color});
                vertices.push_back({{x_end, y - row_height / 2}, config_.ask_volume_color});
                vertices.push_back({{x_center, y + row_height / 2}, config_.ask_volume_color});
                
                vertices.push_back({{x_end, y - row_height / 2}, config_.ask_volume_color});
                vertices.push_back({{x_end, y + row_height / 2}, config_.ask_volume_color});
                vertices.push_back({{x_center, y + row_height / 2}, config_.ask_volume_color});
            }
        }
        
        return vertices;
    }
    
    /**
     * @brief Generate POC line vertices
     */
    std::vector<VertexData> generatePOCLineVertices(
        const VolumeProfileResult& profile,
        float x_start,
        float x_end,
        float chart_height,
        float chart_y_offset,
        float pixels_per_price)
    {
        std::vector<VertexData> vertices;
        
        if (!profile.is_valid || profile.poc_price == 0.0) return vertices;
        
        float y = chart_y_offset + chart_height - 
            static_cast<float>((profile.poc_price - profile.levels.front().price) * pixels_per_price);
        
        vertices.push_back({{x_start, y}, config_.poc_color});
        vertices.push_back({{x_end, y}, config_.poc_color});
        
        return vertices;
    }
    
    /**
     * @brief Generate Value Area lines
     */
    std::vector<VertexData> generateValueAreaVertices(
        const VolumeProfileResult& profile,
        float x_start,
        float x_end,
        float chart_height,
        float chart_y_offset,
        float pixels_per_price)
    {
        std::vector<VertexData> vertices;
        
        if (!profile.is_valid) return vertices;
        
        const float price_offset = profile.levels.front().price;
        
        // VAH line
        float y_vah = chart_y_offset + chart_height - 
            static_cast<float>((profile.vah_price - price_offset) * pixels_per_price);
        
        vertices.push_back({{x_start, y_vah}, config_.value_area_color});
        vertices.push_back({{x_end, y_vah}, config_.value_area_color});
        
        // VAL line
        float y_val = chart_y_offset + chart_height - 
            static_cast<float>((profile.val_price - price_offset) * pixels_per_price);
        
        vertices.push_back({{x_start, y_val}, config_.value_area_color});
        vertices.push_back({{x_end, y_val}, config_.value_area_color});
        
        return vertices;
    }

private:
    VolumeProfileConfig config_;
};

/**
 * @brief Rolling Volume Profile for real-time updates
 */
class RollingVolumeProfile {
public:
    RollingVolumeProfile(size_t lookback_periods = 100)
        : lookback_periods_(lookback_periods) {}
    
    /**
     * @brief Add a new candle to the profile
     */
    void addCandle(double high, double low, double volume, double close, double open) {
        highs_.push_back(high);
        lows_.push_back(low);
        volumes_.push_back(volume);
        closes_.push_back(close);
        opens_.push_back(open);
        
        // Remove old data
        if (highs_.size() > lookback_periods_) {
            highs_.erase(highs_.begin());
            lows_.erase(lows_.begin());
            volumes_.erase(volumes_.begin());
            closes_.erase(closes_.begin());
            opens_.erase(opens_.begin());
        }
    }
    
    /**
     * @brief Calculate current volume profile
     */
    VolumeProfileResult calculate(
        double min_price,
        double max_price,
        const VolumeProfileConfig& config = VolumeProfileConfig{})
    {
        VolumeProfileCalculator calc;
        return calc.calculate(highs_, lows_, volumes_, closes_, opens_, 
                             min_price, max_price, config);
    }
    
    /**
     * @brief Clear all data
     */
    void clear() {
        highs_.clear();
        lows_.clear();
        volumes_.clear();
        closes_.clear();
        opens_.clear();
    }

private:
    size_t lookback_periods_;
    std::vector<double> highs_;
    std::vector<double> lows_;
    std::vector<double> volumes_;
    std::vector<double> closes_;
    std::vector<double> opens_;
};

} // namespace rendering
} // namespace btq
