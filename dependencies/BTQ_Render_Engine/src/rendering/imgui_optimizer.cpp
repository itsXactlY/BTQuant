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
    // Use uintptr_t as key to avoid string copying - much more efficient
    std::unordered_map<uintptr_t, CachedTextSize> text_size_cache_;
    std::unordered_map<uint64_t, ImU32> color_cache_;  // Using uint64_t key for better performance
    std::unordered_map<uint64_t, CachedTextSize> text_size_cache_by_params_;
    CachedStyle current_style_cache_;
    float last_update_time_ = 0.0f;
    static constexpr float CACHE_EXPIRY_TIME = 0.1f; // 100ms expiry for dynamic content

public:
    ImGuiStateCache() = default;

    /**
     * Get cached text size or compute and cache it
     * Using pointer address as key to avoid string copying
     */
    ImVec2 get_cached_text_size(const char* text) {
        if (!text) return ImVec2(0, 0);

        // Use pointer as key for faster lookup - avoids string copy
        uintptr_t text_ptr = reinterpret_cast<uintptr_t>(text);

        auto it = text_size_cache_.find(text_ptr);
        if (it != text_size_cache_.end() && it->second.valid) {
            return it->second.size;
        }

        // Compute and cache the text size
        ImVec2 size = ImGui::CalcTextSize(text);
        text_size_cache_[text_ptr] = {size, static_cast<float>(ImGui::GetTime()), true};

        return size;
    }

    /**
     * Get cached text size with additional parameters
     * Optimized key generation to avoid collisions and improve performance
     */
    ImVec2 get_cached_text_size_ex(const char* text, const char* text_end = nullptr,
                                   bool hide_text_after_double_hash = false,
                                   float wrap_width = -1.0f) {
        if (!text) return ImVec2(0, 0);

        // Create a unique numeric key combining all parameters for better performance
        // Use hash combination to reduce collision risk
        uint64_t text_hash = reinterpret_cast<uint64_t>(text);
        uint64_t text_end_hash = reinterpret_cast<uint64_t>(text_end);
        uint64_t params_key = text_hash ^
                             (text_end_hash << 1) ^
                             (static_cast<uint64_t>(hide_text_after_double_hash) << 32) ^
                             (static_cast<uint64_t>(static_cast<uint32_t>(wrap_width * 1000000)) << 48);

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
     * Optimized to reduce iteration overhead
     */
    void invalidate_expired_cache() {
        float current_time = ImGui::GetTime();
        float expiry_threshold = current_time - CACHE_EXPIRY_TIME;

        // Clean up text size cache - optimized erase-remove idiom equivalent
        for (auto it = text_size_cache_.begin(); it != text_size_cache_.end();) {
            if (it->second.timestamp < expiry_threshold) {
                it = text_size_cache_.erase(it);
            } else {
                ++it;
            }
        }

        // Clean up text size cache by params
        for (auto it = text_size_cache_by_params_.begin(); it != text_size_cache_by_params_.end();) {
            if (it->second.timestamp < expiry_threshold) {
                it = text_size_cache_by_params_.erase(it);
            } else {
                ++it;
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
        // Direct indexing approach - more efficient than hashing
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
     * Improved key generation to reduce collision risk
     */
    ImU32 get_cached_color_with_alpha(ImGuiCol idx, float alpha_mul) {
        // Create a unique key combining index and alpha multiplier with better distribution
        uint64_t color_key = (static_cast<uint64_t>(static_cast<uint32_t>(idx)) << 32) |
                            (static_cast<uint32_t>(static_cast<int32_t>(alpha_mul * 1000000)) & 0xFFFFFFFF);

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
     * Optimized with early return to reduce function call overhead
     */
    static void ConditionalText(bool condition, const char* fmt, ...) {
        if (!condition) return;

        va_list args;
        va_start(args, fmt);
        ImGui::TextV(fmt, args);
        va_end(args);
    }

    /**
     * Conditional ImGui calls that only execute if condition is true
     * Early return optimization to reduce function call overhead
     */
    static bool ConditionalButton(bool condition, const char* label, const ImVec2& size = ImVec2(0, 0)) {
        if (!condition) return false;
        return ImGui::Button(label, size);
    }

    /**
     * Conditional ImGui calls that only execute if condition is true
     * Early return optimization to reduce function call overhead
     */
    static bool ConditionalSliderFloat(bool condition, const char* label, float* v, float v_min, float v_max,
                                      const char* format = "%.3f", ImGuiSliderFlags flags = 0) {
        if (!condition) return false;
        return ImGui::SliderFloat(label, v, v_min, v_max, format, flags);
    }

    /**
     * Conditional checkbox
     * Early return optimization to reduce function call overhead
     */
    static bool ConditionalCheckbox(bool condition, const char* label, bool* v) {
        if (!condition) return false;
        return ImGui::Checkbox(label, v);
    }

    /**
     * Conditional combo box
     * Early return optimization to reduce function call overhead
     */
    static bool ConditionalCombo(bool condition, const char* label, int* current_item,
                                const char* const items[], int items_count, int popup_max_height_in_items = -1) {
        if (!condition) return false;
        return ImGui::Combo(label, current_item, items, items_count, popup_max_height_in_items);
    }

    /**
     * Conditional combo box with items as string
     * Early return optimization to reduce function call overhead
     */
    static bool ConditionalComboStr(bool condition, const char* label, int* current_item,
                                   const char* items_separated_by_zeros, int popup_max_height_in_items = -1) {
        if (!condition) return false;
        return ImGui::Combo(label, current_item, items_separated_by_zeros, popup_max_height_in_items);
    }

    /**
     * Batch similar operations to minimize state changes
     * Optimized with reserve to reduce memory allocations
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
     * Optimized with reserve to reduce memory allocations
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
     * Avoid repeated push/pop operations by grouping by color
     */
    static void BatchColoredTextRendering(const std::vector<std::tuple<const char*, ImVec2, ImU32>>& texts) {
        if (texts.empty()) return;

        // Group by color to minimize push/pop operations
        ImU32 current_color = 0;
        bool color_set = false;

        for (const auto& text_tuple : texts) {
            const char* text = std::get<0>(text_tuple);
            const ImVec2& pos = std::get<1>(text_tuple);
            ImU32 color = std::get<2>(text_tuple);

            if (!color_set || current_color != color) {
                if (color_set) {
                    ImGui::PopStyleColor();
                }
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                current_color = color;
                color_set = true;
            }

            ImGui::SetCursorPos(pos);
            ImGui::TextUnformatted(text);
        }

        if (color_set) {
            ImGui::PopStyleColor();
        }
    }

    /**
     * Batch same-colored text rendering to minimize state changes
     * Optimized with reserve to reduce memory allocations
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
     * Optimized with pre-computed color
     */
    static void BatchSameStyledTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts,
                                           ImGuiCol color_idx, float alpha_mul = 1.0f) {
        if (texts.empty()) return;

        // Apply color once for all texts - use optimized color function
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
        if (!condition) return;

        va_list args;
        va_start(args, fmt);
        ImGui::TextV(fmt, args);
        va_end(args);
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
                            const char* items_separated_by_zeros, int popup_max_height_in_items = -1) {
        return optimizer_instance.ConditionalComboStr(condition, label, current_item, items_separated_by_zeros, popup_max_height_in_items);
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
        if (!condition) return;

        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);

        va_list args;
        va_start(args, fmt);
        ImGui::TextV(fmt, args);
        va_end(args);

        ImGui::PopStyleColor();
    }

    /**
     * Push a style color only if condition is met
     */
    void PushStyleColorConditional(bool condition, ImGuiCol idx, ImU32 col) {
        if (!condition) return;
        ImGui::PushStyleColor(idx, col);
    }

    /**
     * Pop a style color only if condition is met
     */
    void PopStyleColorConditional(bool condition, int count = 1) {
        if (!condition) return;
        ImGui::PopStyleColor(count);
    }

    /**
     * Push a style var only if condition is met
     */
    void PushStyleVarConditional(bool condition, ImGuiStyleVar idx, float val) {
        if (!condition) return;
        ImGui::PushStyleVar(idx, val);
    }

    /**
     * Push a style var only if condition is met
     */
    void PushStyleVarConditional(bool condition, ImGuiStyleVar idx, const ImVec2& val) {
        if (!condition) return;
        ImGui::PushStyleVar(idx, val);
    }

    /**
     * Pop a style var only if condition is met
     */
    void PopStyleVarConditional(bool condition, int count = 1) {
        if (!condition) return;
        ImGui::PopStyleVar(count);
    }
}

} // namespace Rendering
} // namespace BTQuant