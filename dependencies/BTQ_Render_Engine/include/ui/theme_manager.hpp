#pragma once

/**
 * @file theme_manager.hpp
 * @brief Dark/Light Theme Switching with Vulkan Push Constants
 * 
 * This implementation provides:
 * - Seamless theme switching without pipeline rebuilds
 * - Vulkan push constants for real-time color updates
 * - Predefined dark and light themes
 * - Custom theme support
 * - Smooth transitions between themes
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace btq {
namespace ui {

/**
 * @brief Theme color palette
 */
struct ThemeColors {
    // Background colors
    glm::vec4 background_primary{0.05f, 0.05f, 0.05f, 1.0f};
    glm::vec4 background_secondary{0.08f, 0.08f, 0.08f, 1.0f};
    glm::vec4 background_tertiary{0.12f, 0.12f, 0.12f, 1.0f};
    
    // Text colors
    glm::vec4 text_primary{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 text_secondary{0.7f, 0.7f, 0.7f, 1.0f};
    glm::vec4 text_muted{0.5f, 0.5f, 0.5f, 1.0f};
    
    // Chart colors
    glm::vec4 candle_bull{0.0f, 0.8f, 0.0f, 1.0f};
    glm::vec4 candle_bear{0.8f, 0.0f, 0.0f, 1.0f};
    glm::vec4 candle_wick{0.6f, 0.6f, 0.6f, 1.0f};
    
    // Grid colors
    glm::vec4 grid_major{0.3f, 0.3f, 0.3f, 0.8f};
    glm::vec4 grid_minor{0.2f, 0.2f, 0.2f, 0.4f};
    glm::vec4 axis_line{0.4f, 0.4f, 0.4f, 1.0f};
    
    // UI colors
    glm::vec4 accent_primary{0.2f, 0.5f, 0.8f, 1.0f};
    glm::vec4 accent_secondary{0.8f, 0.5f, 0.2f, 1.0f};
    glm::vec4 highlight{1.0f, 1.0f, 0.0f, 0.3f};
    glm::vec4 selection{0.3f, 0.5f, 0.8f, 0.5f};
    
    // Order colors
    glm::vec4 order_buy{0.0f, 0.8f, 0.0f, 0.8f};
    glm::vec4 order_sell{0.8f, 0.0f, 0.0f, 0.8f};
    glm::vec4 order_stop{1.0f, 0.5f, 0.0f, 0.8f};
    glm::vec4 order_take_profit{0.0f, 1.0f, 0.5f, 0.8f};
    
    // Indicator colors
    glm::vec4 indicator_ma{0.8f, 0.8f, 0.0f, 1.0f};
    glm::vec4 indicator_ema{0.0f, 0.8f, 0.8f, 1.0f};
    glm::vec4 indicator_rsi{0.8f, 0.4f, 0.8f, 1.0f};
    glm::vec4 indicator_macd{0.4f, 0.8f, 0.4f, 1.0f};
    glm::vec4 indicator_signal{0.8f, 0.4f, 0.4f, 1.0f};
    glm::vec4 indicator_histogram_pos{0.0f, 0.6f, 0.0f, 0.8f};
    glm::vec4 indicator_histogram_neg{0.6f, 0.0f, 0.0f, 0.8f};
    
    // Volume profile colors
    glm::vec4 volume_bid{0.0f, 0.6f, 0.0f, 0.6f};
    glm::vec4 volume_ask{0.6f, 0.0f, 0.0f, 0.6f};
    glm::vec4 volume_poc{1.0f, 1.0f, 0.0f, 1.0f};
    
    // Crosshair colors
    glm::vec4 crosshair_line{0.5f, 0.5f, 0.5f, 0.8f};
    glm::vec4 crosshair_label_bg{0.1f, 0.1f, 0.1f, 0.9f};
    glm::vec4 crosshair_label_text{1.0f, 1.0f, 1.0f, 1.0f};
    
    // Drawing tool colors
    glm::vec4 drawing_trendline{0.8f, 0.8f, 0.2f, 1.0f};
    glm::vec4 drawing_horizontal{0.2f, 0.8f, 0.8f, 1.0f};
    glm::vec4 drawing_fibonacci{0.8f, 0.4f, 0.8f, 1.0f};
    
    // Alert/notification colors
    glm::vec4 alert_info{0.2f, 0.5f, 0.8f, 1.0f};
    glm::vec4 alert_warning{0.8f, 0.6f, 0.0f, 1.0f};
    glm::vec4 alert_error{0.8f, 0.2f, 0.2f, 1.0f};
    glm::vec4 alert_success{0.2f, 0.8f, 0.2f, 1.0f};
};

/**
 * @brief Predefined dark theme
 */
inline ThemeColors getDarkTheme() {
    return ThemeColors{};
}

/**
 * @brief Predefined light theme
 */
inline ThemeColors getLightTheme() {
    ThemeColors theme;
    
    // Background colors
    theme.background_primary = {0.95f, 0.95f, 0.95f, 1.0f};
    theme.background_secondary = {0.92f, 0.92f, 0.92f, 1.0f};
    theme.background_tertiary = {0.88f, 0.88f, 0.88f, 1.0f};
    
    // Text colors
    theme.text_primary = {0.1f, 0.1f, 0.1f, 1.0f};
    theme.text_secondary = {0.3f, 0.3f, 0.3f, 1.0f};
    theme.text_muted = {0.5f, 0.5f, 0.5f, 1.0f};
    
    // Chart colors
    theme.candle_bull = {0.0f, 0.6f, 0.0f, 1.0f};
    theme.candle_bear = {0.8f, 0.0f, 0.0f, 1.0f};
    theme.candle_wick = {0.4f, 0.4f, 0.4f, 1.0f};
    
    // Grid colors
    theme.grid_major = {0.7f, 0.7f, 0.7f, 0.8f};
    theme.grid_minor = {0.8f, 0.8f, 0.8f, 0.4f};
    theme.axis_line = {0.6f, 0.6f, 0.6f, 1.0f};
    
    // UI colors
    theme.accent_primary = {0.2f, 0.4f, 0.7f, 1.0f};
    theme.accent_secondary = {0.7f, 0.4f, 0.2f, 1.0f};
    theme.highlight = {1.0f, 1.0f, 0.0f, 0.3f};
    theme.selection = {0.3f, 0.5f, 0.8f, 0.5f};
    
    // Crosshair colors
    theme.crosshair_line = {0.4f, 0.4f, 0.4f, 0.8f};
    theme.crosshair_label_bg = {0.9f, 0.9f, 0.9f, 0.9f};
    theme.crosshair_label_text = {0.1f, 0.1f, 0.1f, 1.0f};
    
    return theme;
}

/**
 * @brief Predefined classic trading theme (green/red)
 */
inline ThemeColors getClassicTheme() {
    ThemeColors theme = getDarkTheme();
    
    theme.candle_bull = {0.0f, 1.0f, 0.0f, 1.0f};
    theme.candle_bear = {1.0f, 0.0f, 0.0f, 1.0f};
    theme.volume_bid = {0.0f, 1.0f, 0.0f, 0.5f};
    theme.volume_ask = {1.0f, 0.0f, 0.0f, 0.5f};
    
    return theme;
}

/**
 * @brief Predefined monochrome theme
 */
inline ThemeColors getMonochromeTheme() {
    ThemeColors theme = getDarkTheme();
    
    theme.candle_bull = {0.8f, 0.8f, 0.8f, 1.0f};
    theme.candle_bear = {0.3f, 0.3f, 0.3f, 1.0f};
    theme.volume_bid = {0.6f, 0.6f, 0.6f, 0.5f};
    theme.volume_ask = {0.4f, 0.4f, 0.4f, 0.5f};
    
    return theme;
}

/**
 * @brief Theme push constants for Vulkan shaders
 * This structure is pushed to shaders for real-time theme updates
 */
struct ThemePushConstants {
    // Packed into 4 vec4 arrays for efficient GPU transfer
    glm::vec4 colors_0[8];   // 32 floats = 128 bytes
    glm::vec4 colors_1[8];   // 32 floats = 128 bytes
    
    /**
     * @brief Pack theme colors into push constants
     */
    static ThemePushConstants pack(const ThemeColors& colors) {
        ThemePushConstants pc;
        
        // First block
        pc.colors_0[0] = colors.background_primary;
        pc.colors_0[1] = colors.background_secondary;
        pc.colors_0[2] = colors.text_primary;
        pc.colors_0[3] = colors.text_secondary;
        pc.colors_0[4] = colors.candle_bull;
        pc.colors_0[5] = colors.candle_bear;
        pc.colors_0[6] = colors.candle_wick;
        pc.colors_0[7] = colors.grid_major;
        
        // Second block
        pc.colors_1[0] = colors.grid_minor;
        pc.colors_1[1] = colors.accent_primary;
        pc.colors_1[2] = colors.accent_secondary;
        pc.colors_1[3] = colors.order_buy;
        pc.colors_1[4] = colors.order_sell;
        pc.colors_1[5] = colors.crosshair_line;
        pc.colors_1[6] = colors.crosshair_label_bg;
        pc.colors_1[7] = colors.crosshair_label_text;
        
        return pc;
    }
};

/**
 * @brief Theme manager for seamless switching
 */
class ThemeManager {
public:
    enum class ThemeType : uint8_t {
        DARK,
        LIGHT,
        CLASSIC,
        MONOCHROME,
        CUSTOM
    };
    
    ThemeManager() {
        // Initialize with dark theme
        current_theme_ = getDarkTheme();
        target_theme_ = current_theme_;
    }
    
    /**
     * @brief Set the current theme
     */
    void setTheme(ThemeType type) {
        switch (type) {
            case ThemeType::DARK:
                target_theme_ = getDarkTheme();
                break;
            case ThemeType::LIGHT:
                target_theme_ = getLightTheme();
                break;
            case ThemeType::CLASSIC:
                target_theme_ = getClassicTheme();
                break;
            case ThemeType::MONOCHROME:
                target_theme_ = getMonochromeTheme();
                break;
            case ThemeType::CUSTOM:
                // Keep current custom theme
                break;
        }
        
        current_type_ = type;
        
        if (!transitioning_) {
            current_theme_ = target_theme_;
        }
    }
    
    /**
     * @brief Set a custom theme
     */
    void setCustomTheme(const ThemeColors& theme) {
        target_theme_ = theme;
        current_type_ = ThemeType::CUSTOM;
        
        if (!transitioning_) {
            current_theme_ = target_theme_;
        }
    }
    
    /**
     * @brief Start a smooth transition to the target theme
     */
    void beginTransition(float duration_seconds = 0.5f) {
        transitioning_ = true;
        transition_progress_ = 0.0f;
        transition_duration_ = duration_seconds;
        start_theme_ = current_theme_;
    }
    
    /**
     * @brief Update the transition (call each frame)
     */
    void updateTransition(float delta_seconds) {
        if (!transitioning_) return;
        
        transition_progress_ += delta_seconds / transition_duration_;
        
        if (transition_progress_ >= 1.0f) {
            transition_progress_ = 1.0f;
            transitioning_ = false;
            current_theme_ = target_theme_;
        } else {
            // Interpolate colors
            interpolateThemes(start_theme_, target_theme_, transition_progress_);
        }
    }
    
    /**
     * @brief Get the current theme colors
     */
    const ThemeColors& getCurrentTheme() const {
        return current_theme_;
    }
    
    /**
     * @brief Get push constants for current theme
     */
    ThemePushConstants getPushConstants() const {
        return ThemePushConstants::pack(current_theme_);
    }
    
    /**
     * @brief Get current theme type
     */
    ThemeType getThemeType() const {
        return current_type_;
    }
    
    /**
     * @brief Check if transitioning
     */
    bool isTransitioning() const {
        return transitioning_;
    }
    
    /**
     * @brief Get transition progress (0-1)
     */
    float getTransitionProgress() const {
        return transition_progress_;
    }
    
    /**
     * @brief Modify a single color in the current theme
     */
    void modifyColor(const std::string& color_name, const glm::vec4& color) {
        // Map color names to theme colors
        if (color_name == "background_primary") current_theme_.background_primary = color;
        else if (color_name == "background_secondary") current_theme_.background_secondary = color;
        else if (color_name == "text_primary") current_theme_.text_primary = color;
        else if (color_name == "text_secondary") current_theme_.text_secondary = color;
        else if (color_name == "candle_bull") current_theme_.candle_bull = color;
        else if (color_name == "candle_bear") current_theme_.candle_bear = color;
        else if (color_name == "candle_wick") current_theme_.candle_wick = color;
        else if (color_name == "grid_major") current_theme_.grid_major = color;
        else if (color_name == "grid_minor") current_theme_.grid_minor = color;
        else if (color_name == "accent_primary") current_theme_.accent_primary = color;
        else if (color_name == "order_buy") current_theme_.order_buy = color;
        else if (color_name == "order_sell") current_theme_.order_sell = color;
        else if (color_name == "crosshair_line") current_theme_.crosshair_line = color;
        
        current_type_ = ThemeType::CUSTOM;
    }

private:
    void interpolateThemes(const ThemeColors& start, const ThemeColors& end, float t) {
        // Smooth easing function
        float ease = t * t * (3.0f - 2.0f * t);
        
        auto lerp = [ease](const glm::vec4& a, const glm::vec4& b) {
            return a + (b - a) * ease;
        };
        
        current_theme_.background_primary = lerp(start.background_primary, end.background_primary);
        current_theme_.background_secondary = lerp(start.background_secondary, end.background_secondary);
        current_theme_.background_tertiary = lerp(start.background_tertiary, end.background_tertiary);
        
        current_theme_.text_primary = lerp(start.text_primary, end.text_primary);
        current_theme_.text_secondary = lerp(start.text_secondary, end.text_secondary);
        current_theme_.text_muted = lerp(start.text_muted, end.text_muted);
        
        current_theme_.candle_bull = lerp(start.candle_bull, end.candle_bull);
        current_theme_.candle_bear = lerp(start.candle_bear, end.candle_bear);
        current_theme_.candle_wick = lerp(start.candle_wick, end.candle_wick);
        
        current_theme_.grid_major = lerp(start.grid_major, end.grid_major);
        current_theme_.grid_minor = lerp(start.grid_minor, end.grid_minor);
        current_theme_.axis_line = lerp(start.axis_line, end.axis_line);
        
        current_theme_.accent_primary = lerp(start.accent_primary, end.accent_primary);
        current_theme_.accent_secondary = lerp(start.accent_secondary, end.accent_secondary);
        current_theme_.highlight = lerp(start.highlight, end.highlight);
        current_theme_.selection = lerp(start.selection, end.selection);
        
        current_theme_.order_buy = lerp(start.order_buy, end.order_buy);
        current_theme_.order_sell = lerp(start.order_sell, end.order_sell);
        current_theme_.order_stop = lerp(start.order_stop, end.order_stop);
        current_theme_.order_take_profit = lerp(start.order_take_profit, end.order_take_profit);
        
        current_theme_.crosshair_line = lerp(start.crosshair_line, end.crosshair_line);
        current_theme_.crosshair_label_bg = lerp(start.crosshair_label_bg, end.crosshair_label_bg);
        current_theme_.crosshair_label_text = lerp(start.crosshair_label_text, end.crosshair_label_text);
    }
    
    ThemeColors current_theme_;
    ThemeColors target_theme_;
    ThemeColors start_theme_;
    ThemeType current_type_ = ThemeType::DARK;
    
    bool transitioning_ = false;
    float transition_progress_ = 0.0f;
    float transition_duration_ = 0.5f;
};

/**
 * @brief ImGui theme adapter
 */
class ImGuiThemeAdapter {
public:
    /**
     * @brief Apply theme colors to ImGui
     */
    static void applyTheme(const ThemeColors& theme);
    
    /**
     * @brief Get ImGui style from theme
     */
    static struct ImGuiStyle getImGuiStyle(const ThemeColors& theme);
};

} // namespace ui
} // namespace btq
