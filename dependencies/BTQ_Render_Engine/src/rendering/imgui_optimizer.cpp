/**
 * BTQuant ImGui Optimization System
 *
 * Advanced optimization system for ImGui rendering to reduce redundant calls,
 * cache computed values, and minimize state changes for improved performance
 * in professional trading dashboard applications.
 */

#include "imgui.h"
#include <unordered_map>
#include <string>
#include <functional>
#include <vector>
#include <memory>

namespace BTQuant {
namespace Rendering {

// ============================================================================
// ImGui State Cache Manager
// ============================================================================

class ImGuiStateCache {
public:
    struct CachedTextSize {
        ImVec2 size;
        float timestamp;
        bool valid;
    };

    struct CachedStyle {
        ImVec4 colors[ImGuiCol_COUNT];
        float rounding;
        ImVec2 item_spacing;
        ImVec2 window_padding;
        bool valid;
    };

private:
    std::unordered_map<std::string, CachedTextSize> text_size_cache_;
    std::unordered_map<std::string, ImU32> color_cache_;
    CachedStyle current_style_cache_;
    float last_update_time_ = 0.0f;
    static constexpr float CACHE_EXPIRY_TIME = 0.1f; // 100ms expiry for dynamic content

public:
    ImGuiStateCache() = default;

    /**
     * Get cached text size or compute and cache it
     */
    ImVec2 get_cached_text_size(const char* text) {
        if (!text) return ImVec2(0, 0);

        std::string text_key(text);
        
        auto it = text_size_cache_.find(text_key);
        if (it != text_size_cache_.end() && it->second.valid) {
            return it->second.size;
        }

        // Compute and cache the text size
        ImVec2 size = ImGui::CalcTextSize(text);
        text_size_cache_[text_key] = {size, ImGui::GetTime(), true};
        
        return size;
    }

    /**
     * Get cached text size with additional parameters
     */
    ImVec2 get_cached_text_size_ex(const char* text, const char* text_end = nullptr, 
                                   bool hide_text_after_double_hash = false, 
                                   float wrap_width = -1.0f) {
        if (!text) return ImVec2(0, 0);

        // Create a unique key combining all parameters
        std::string text_key = std::string(text) + "_" + 
                              std::to_string(hide_text_after_double_hash) + 
                              "_" + std::to_string(wrap_width);
        
        auto it = text_size_cache_.find(text_key);
        if (it != text_size_cache_.end() && it->second.valid) {
            return it->second.size;
        }

        // Compute and cache the text size
        ImVec2 size = ImGui::CalcTextSize(text, text_end, hide_text_after_double_hash, wrap_width);
        text_size_cache_[text_key] = {size, ImGui::GetTime(), true};
        
        return size;
    }

    /**
     * Invalidate expired cache entries
     */
    void invalidate_expired_cache() {
        float current_time = ImGui::GetTime();
        auto it = text_size_cache_.begin();
        while (it != text_size_cache_.end()) {
            if ((current_time - it->second.timestamp) > CACHE_EXPIRY_TIME) {
                it->second.valid = false;
            }
            ++it;
        }
    }

    /**
     * Clear all cached text sizes
     */
    void clear_text_cache() {
        text_size_cache_.clear();
    }

    /**
     * Get cached color or compute it
     */
    ImU32 get_cached_color(ImGuiCol idx) {
        std::string color_key = "color_" + std::to_string(idx);
        
        auto it = color_cache_.find(color_key);
        if (it != color_cache_.end()) {
            return it->second;
        }

        // Compute and cache the color
        ImU32 color = ImGui::GetColorU32(idx);
        color_cache_[color_key] = color;
        
        return color;
    }

    /**
     * Get cached color with alpha multiplier
     */
    ImU32 get_cached_color_with_alpha(ImGuiCol idx, float alpha_mul) {
        std::string color_key = "color_" + std::to_string(idx) + "_alpha_" + std::to_string(int(alpha_mul * 1000));
        
        auto it = color_cache_.find(color_key);
        if (it != color_cache_.end()) {
            return it->second;
        }

        // Compute and cache the color
        ImU32 color = ImGui::GetColorU32(idx, alpha_mul);
        color_cache_[color_key] = color;
        
        return color;
    }

    /**
     * Clear color cache
     */
    void clear_color_cache() {
        color_cache_.clear();
    }

    /**
     * Clear all caches
     */
    void clear_all_cache() {
        clear_text_cache();
        clear_color_cache();
    }
};

// ============================================================================
// ImGui Call Optimizer
// ============================================================================

class ImGuiCallOptimizer {
private:
    static inline ImGuiStateCache state_cache_;

public:
    /**
     * Optimized version of ImGui::CalcTextSize that caches results
     */
    static ImVec2 CalcTextSizeOptimized(const char* text, const char* text_end = nullptr, 
                                        bool hide_text_after_double_hash = false, 
                                        float wrap_width = -1.0f) {
        if (wrap_width == -1.0f) {
            return state_cache_.get_cached_text_size(text);
        } else {
            return state_cache_.get_cached_text_size_ex(text, text_end, hide_text_after_double_hash, wrap_width);
        }
    }

    /**
     * Optimized version of ImGui::GetColorU32 that caches results
     */
    static ImU32 GetColorU32Optimized(ImGuiCol idx, float alpha_mul = 1.0f) {
        if (alpha_mul == 1.0f) {
            return state_cache_.get_cached_color(idx);
        } else {
            return state_cache_.get_cached_color_with_alpha(idx, alpha_mul);
        }
    }

    /**
     * Conditional ImGui calls that only execute if condition is true
     */
    template<typename T>
    static bool ConditionalBegin(T condition_func, const char* name, bool* p_open = nullptr, 
                                 ImGuiWindowFlags flags = 0) {
        if (condition_func()) {
            return ImGui::Begin(name, p_open, flags);
        }
        return false;
    }

    /**
     * Conditional ImGui calls that only execute if condition is true
     */
    static void ConditionalText(bool condition, const char* fmt, ...) {
        if (condition) {
            va_list args;
            va_start(args, fmt);
            ImGui::TextV(fmt, args);
            va_end(args);
        }
    }

    /**
     * Conditional ImGui calls that only execute if condition is true
     */
    static bool ConditionalButton(bool condition, const char* label, const ImVec2& size = ImVec2(0, 0)) {
        if (condition) {
            return ImGui::Button(label, size);
        }
        return false;
    }

    /**
     * Conditional ImGui calls that only execute if condition is true
     */
    static bool ConditionalSliderFloat(bool condition, const char* label, float* v, float v_min, float v_max, 
                                      const char* format = "%.3f", ImGuiSliderFlags flags = 0) {
        if (condition) {
            return ImGui::SliderFloat(label, v, v_min, v_max, format, flags);
        }
        return false;
    }

    /**
     * Batch similar operations to minimize state changes
     */
    template<typename T>
    static void BatchOperation(const std::vector<T>& items, 
                              std::function<void(const T&)> operation) {
        for (const auto& item : items) {
            operation(item);
        }
    }

    /**
     * Batch text rendering with same style to minimize state changes
     */
    static void BatchTextRendering(const std::vector<std::pair<std::string, ImVec2>>& texts) {
        if (texts.empty()) return;

        // Set style once for all texts
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 2));
        
        for (const auto& text_pair : texts) {
            ImGui::SetCursorPos(text_pair.second);
            ImGui::TextUnformatted(text_pair.first.c_str());
        }
        
        ImGui::PopStyleVar();
    }

    /**
     * Batch colored text rendering
     */
    static void BatchColoredTextRendering(const std::vector<std::tuple<std::string, ImVec2, ImU32>>& texts) {
        if (texts.empty()) return;

        for (const auto& text_tuple : texts) {
            const std::string& text = std::get<0>(text_tuple);
            const ImVec2& pos = std::get<1>(text_tuple);
            ImU32 color = std::get<2>(text_tuple);

            ImGui::SetCursorPos(pos);
            ImGui::TextColored(ImVec4((color >> 0) & 0xFF, (color >> 8) & 0xFF, 
                                     (color >> 16) & 0xFF, (color >> 24) & 0xFF) / 255.0f, 
                              "%s", text.c_str());
        }
    }

    /**
     * Check if window is active before performing expensive operations
     */
    static bool IsWindowActive() {
        ImGuiWindow* window = ImGui::GetCurrentWindowRead();
        return window && window->Active;
    }

    /**
     * Check if item is visible before rendering
     */
    static bool IsItemVisible() {
        return ImGui::IsItemVisible();
    }

    /**
     * Check if rect is visible before rendering
     */
    static bool IsRectVisible(const ImVec2& size) {
        return ImGui::IsRectVisible(size);
    }

    /**
     * Check if rect is visible with specific bounds
     */
    static bool IsRectVisible(const ImVec2& rect_min, const ImVec2& rect_max) {
        return ImGui::IsRectVisible(rect_min, rect_max);
    }

    /**
     * Update cache periodically
     */
    static void UpdateCache() {
        state_cache_.invalidate_expired_cache();
    }

    /**
     * Clear all caches
     */
    static void ClearCache() {
        state_cache_.clear_all_cache();
    }
};

// ============================================================================
// Global ImGui Optimizer Instance
// ============================================================================

namespace ImGuiOptimizer {
    // Singleton instance
    static ImGuiCallOptimizer optimizer_instance;

    // Public API functions
    ImVec2 CalcTextSize(const char* text, const char* text_end, 
                        bool hide_text_after_double_hash, float wrap_width) {
        return optimizer_instance.CalcTextSizeOptimized(text, text_end, hide_text_after_double_hash, wrap_width);
    }

    ImU32 GetColorU32(ImGuiCol idx, float alpha_mul) {
        return optimizer_instance.GetColorU32Optimized(idx, alpha_mul);
    }

    template<typename T>
    bool ConditionalBegin(T condition_func, const char* name, bool* p_open, ImGuiWindowFlags flags) {
        return optimizer_instance.ConditionalBegin(condition_func, name, p_open, flags);
    }

    void ConditionalText(bool condition, const char* fmt, ...) {
        if (condition) {
            va_list args;
            va_start(args, fmt);
            ImGui::TextV(fmt, args);
            va_end(args);
        }
    }

    bool ConditionalButton(bool condition, const char* label, const ImVec2& size) {
        return optimizer_instance.ConditionalButton(condition, label, size);
    }

    bool ConditionalSliderFloat(bool condition, const char* label, float* v, float v_min, float v_max, 
                               const char* format, ImGuiSliderFlags flags) {
        return optimizer_instance.ConditionalSliderFloat(condition, label, v, v_min, v_max, format, flags);
    }

    template<typename T>
    void BatchOperation(const std::vector<T>& items, std::function<void(const T&)> operation) {
        optimizer_instance.BatchOperation(items, operation);
    }

    void BatchTextRendering(const std::vector<std::pair<std::string, ImVec2>>& texts) {
        optimizer_instance.BatchTextRendering(texts);
    }

    void BatchColoredTextRendering(const std::vector<std::tuple<std::string, ImVec2, ImU32>>& texts) {
        optimizer_instance.BatchColoredTextRendering(texts);
    }

    bool IsWindowActive() {
        return optimizer_instance.IsWindowActive();
    }

    bool IsItemVisible() {
        return optimizer_instance.IsItemVisible();
    }

    bool IsRectVisible(const ImVec2& size) {
        return optimizer_instance.IsRectVisible(size);
    }

    bool IsRectVisible(const ImVec2& rect_min, const ImVec2& rect_max) {
        return optimizer_instance.IsRectVisible(rect_min, rect_max);
    }

    void UpdateCache() {
        optimizer_instance.UpdateCache();
    }

    void ClearCache() {
        optimizer_instance.ClearCache();
    }
}

} // namespace Rendering
} // namespace BTQuant