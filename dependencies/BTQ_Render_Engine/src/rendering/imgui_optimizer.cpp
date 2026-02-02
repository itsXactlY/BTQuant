/**
 * BTQuant ImGui Optimization System
 *
 * Advanced optimization system for ImGui rendering to reduce redundant calls,
 * cache computed values, and minimize state changes for improved performance
 * in professional trading dashboard applications.
 */

#include "imgui.h"
#include "imgui_internal.h"  // For access to ImGuiWindow and other internal structures
#include <unordered_map>
#include <string>
#include <functional>
#include <vector>
#include <memory>
#include <sstream>

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
    std::unordered_map<uint64_t, ImU32> color_cache_;  // Using uint64_t key for better performance
    std::unordered_map<uint64_t, CachedTextSize> text_size_cache_by_params_;
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

        // Use pointer as key for faster lookup
        uintptr_t text_ptr = reinterpret_cast<uintptr_t>(text);

        auto it = text_size_cache_.find(std::string(text));
        if (it != text_size_cache_.end() && it->second.valid) {
            return it->second.size;
        }

        // Compute and cache the text size
        ImVec2 size = ImGui::CalcTextSize(text);
        text_size_cache_[std::string(text)] = {size, static_cast<float>(ImGui::GetTime()), true};

        return size;
    }

    /**
     * Get cached text size with additional parameters
     */
    ImVec2 get_cached_text_size_ex(const char* text, const char* text_end = nullptr,
                                   bool hide_text_after_double_hash = false,
                                   float wrap_width = -1.0f) {
        if (!text) return ImVec2(0, 0);

        // Create a unique numeric key combining all parameters for better performance
        uint64_t params_key = (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(text)) << 32) |
                             (static_cast<uint32_t>(hide_text_after_double_hash) << 16) |
                             static_cast<uint16_t>(static_cast<int16_t>(wrap_width * 1000));

        auto it = text_size_cache_by_params_.find(params_key);
        if (it != text_size_cache_by_params_.end() && it->second.valid) {
            return it->second.size;
        }

        // Compute and cache the text size
        ImVec2 size = ImGui::CalcTextSize(text, text_end, hide_text_after_double_hash, wrap_width);
        text_size_cache_by_params_[params_key] = {size, static_cast<float>(ImGui::GetTime()), true};

        return size;
    }

    /**
     * Invalidate expired cache entries
     */
    void invalidate_expired_cache() {
        float current_time = ImGui::GetTime();

        // Clean up text size cache
        auto it = text_size_cache_.begin();
        while (it != text_size_cache_.end()) {
            if ((current_time - it->second.timestamp) > CACHE_EXPIRY_TIME) {
                it = text_size_cache_.erase(it);
            } else {
                ++it;
            }
        }

        // Clean up text size cache by params
        auto param_it = text_size_cache_by_params_.begin();
        while (param_it != text_size_cache_by_params_.end()) {
            if ((current_time - param_it->second.timestamp) > CACHE_EXPIRY_TIME) {
                param_it = text_size_cache_by_params_.erase(param_it);
            } else {
                ++param_it;
            }
        }
    }

    /**
     * Clear all cached text sizes
     */
    void clear_text_cache() {
        text_size_cache_.clear();
        text_size_cache_by_params_.clear();
    }

    /**
     * Get cached color or compute it
     */
    ImU32 get_cached_color(ImGuiCol idx) {
        uint64_t color_key = static_cast<uint64_t>(idx);

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
        // Create a unique key combining index and alpha multiplier
        uint64_t color_key = (static_cast<uint64_t>(idx) << 32) |
                            static_cast<uint32_t>(static_cast<int32_t>(alpha_mul * 1000000));

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
     * Conditional checkbox
     */
    static bool ConditionalCheckbox(bool condition, const char* label, bool* v) {
        if (condition) {
            return ImGui::Checkbox(label, v);
        }
        return false;
    }

    /**
     * Conditional combo box
     */
    static bool ConditionalCombo(bool condition, const char* label, int* current_item,
                                const char* const items[], int items_count, int popup_max_height_in_items = -1) {
        if (condition) {
            return ImGui::Combo(label, current_item, items, items_count, popup_max_height_in_items);
        }
        return false;
    }

    /**
     * Conditional combo box with items as string
     */
    static bool ConditionalComboStr(bool condition, const char* label, int* current_item,
                                   const char* items_separated_by_zeros, int popup_max_height_in_items = -1) {
        if (condition) {
            return ImGui::Combo(label, current_item, items_separated_by_zeros, popup_max_height_in_items);
        }
        return false;
    }

    /**
     * Batch similar operations to minimize state changes
     */
    template<typename T>
    static void BatchOperation(const std::vector<T>& items,
                              std::function<void(const T&)> operation) {
        if (items.empty()) return;

        for (const auto& item : items) {
            operation(item);
        }
    }

    /**
     * Batch text rendering with same style to minimize state changes
     */
    static void BatchTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts) {
        if (texts.empty()) return;

        // Set style once for all texts
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 2));

        for (const auto& text_pair : texts) {
            ImGui::SetCursorPos(text_pair.second);
            ImGui::TextUnformatted(text_pair.first);
        }

        ImGui::PopStyleVar();
    }

    /**
     * Batch colored text rendering with optimized performance
     */
    static void BatchColoredTextRendering(const std::vector<std::tuple<const char*, ImVec2, ImU32>>& texts) {
        if (texts.empty()) return;

        // Pre-cache common colors to minimize state changes
        for (const auto& text_tuple : texts) {
            const char* text = std::get<0>(text_tuple);
            const ImVec2& pos = std::get<1>(text_tuple);
            ImU32 color = std::get<2>(text_tuple);

            ImGui::SetCursorPos(pos);
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::TextUnformatted(text);
            ImGui::PopStyleColor();
        }
    }

    /**
     * Batch same-colored text rendering to minimize state changes
     */
    static void BatchSameColoredTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts, ImU32 color) {
        if (texts.empty()) return;

        // Apply color once for all texts
        ImGui::PushStyleColor(ImGuiCol_Text, color);

        for (const auto& text_pair : texts) {
            ImGui::SetCursorPos(text_pair.second);
            ImGui::TextUnformatted(text_pair.first);
        }

        ImGui::PopStyleColor();
    }

    /**
     * Batch same-styled text rendering to minimize state changes
     */
    static void BatchSameStyledTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts,
                                           ImGuiCol color_idx, float alpha_mul = 1.0f) {
        if (texts.empty()) return;

        // Apply color once for all texts
        ImU32 color = GetColorU32Optimized(color_idx, alpha_mul);
        ImGui::PushStyleColor(ImGuiCol_Text, color);

        for (const auto& text_pair : texts) {
            ImGui::SetCursorPos(text_pair.second);
            ImGui::TextUnformatted(text_pair.first);
        }

        ImGui::PopStyleColor();
    }

    /**
     * Check if window is active before performing expensive operations
     */
    static bool IsWindowActive() {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
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

    bool ConditionalCheckbox(bool condition, const char* label, bool* v) {
        return optimizer_instance.ConditionalCheckbox(condition, label, v);
    }

    bool ConditionalCombo(bool condition, const char* label, int* current_item,
                         const char* const items[], int items_count, int popup_max_height_in_items) {
        return optimizer_instance.ConditionalCombo(condition, label, current_item, items, items_count, popup_max_height_in_items);
    }

    bool ConditionalComboStr(bool condition, const char* label, int* current_item,
                            const char* items_separated_by_zeros) {
        return optimizer_instance.ConditionalComboStr(condition, label, current_item, items_separated_by_zeros);
    }

    template<typename T>
    void BatchOperation(const std::vector<T>& items, std::function<void(const T&)> operation) {
        optimizer_instance.BatchOperation(items, operation);
    }

    void BatchTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts) {
        optimizer_instance.BatchTextRendering(texts);
    }

    void BatchColoredTextRendering(const std::vector<std::tuple<const char*, ImVec2, ImU32>>& texts) {
        optimizer_instance.BatchColoredTextRendering(texts);
    }

    void BatchSameColoredTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts, ImU32 color) {
        optimizer_instance.BatchSameColoredTextRendering(texts, color);
    }

    void BatchSameStyledTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts,
                                    ImGuiCol color_idx, float alpha_mul) {
        optimizer_instance.BatchSameStyledTextRendering(texts, color_idx, alpha_mul);
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

    // Additional utility functions for performance optimization

    /**
     * Begin a child window only if it's visible
     */
    bool BeginChildConditional(const char* str_id, const ImVec2& size = ImVec2(0, 0), bool border = false, ImGuiWindowFlags flags = 0) {
        if (!IsWindowActive()) return false;
        return ImGui::BeginChild(str_id, size, border, flags);
    }

    /**
     * Render text only if it's going to be visible
     */
    void TextVisible(const char* fmt, ...) {
        if (!IsItemVisible()) return;

        va_list args;
        va_start(args, fmt);
        ImGui::TextV(fmt, args);
        va_end(args);
    }

    /**
     * Render text disabled (grayed out) based on condition
     */
    void TextDisabledConditional(bool condition, const char* fmt, ...) {
        if (condition) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);

            va_list args;
            va_start(args, fmt);
            ImGui::TextV(fmt, args);
            va_end(args);

            ImGui::PopStyleColor();
        }
    }

    /**
     * Push a style color only if condition is met
     */
    void PushStyleColorConditional(bool condition, ImGuiCol idx, ImU32 col) {
        if (condition) {
            ImGui::PushStyleColor(idx, col);
        }
    }

    /**
     * Pop a style color only if condition is met
     */
    void PopStyleColorConditional(bool condition, int count = 1) {
        if (condition) {
            ImGui::PopStyleColor(count);
        }
    }

    /**
     * Push a style var only if condition is met
     */
    void PushStyleVarConditional(bool condition, ImGuiStyleVar idx, float val) {
        if (condition) {
            ImGui::PushStyleVar(idx, val);
        }
    }

    /**
     * Push a style var only if condition is met
     */
    void PushStyleVarConditional(bool condition, ImGuiStyleVar idx, const ImVec2& val) {
        if (condition) {
            ImGui::PushStyleVar(idx, val);
        }
    }

    /**
     * Pop a style var only if condition is met
     */
    void PopStyleVarConditional(bool condition, int count = 1) {
        if (condition) {
            ImGui::PopStyleVar(count);
        }
    }
}

} // namespace Rendering
} // namespace BTQuant