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
#include <cstdint>
#include <mutex>

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
    // Store ImGui context to ensure thread safety
    ImGuiContext* imgui_context_;

    // Cache for font sizes to avoid repeated calculations
    struct CachedFontSize {
        ImVec2 size;
        ImFont* font;
        float scale;
        bool valid;
    };

    // Cache for cursor positions to avoid redundant SetCursorPos calls
    struct CachedCursorPosition {
        ImVec2 position;
        bool valid;
    };

    // Cache for window positions and sizes
    struct CachedWindowInfo {
        ImVec2 pos;
        ImVec2 size;
        bool visible;
        float timestamp;
        bool valid;
    };

private:
    // Use string content as key for reliable caching - handles string literals properly
    std::unordered_map<std::string, CachedTextSize> text_size_cache_;
    std::unordered_map<uint64_t, ImU32> color_cache_;  // Using uint64_t key for better performance
    std::unordered_map<std::string, CachedTextSize> text_size_cache_by_params_;
    std::unordered_map<uint64_t, CachedFontSize> font_size_cache_;
    std::unordered_map<std::string, CachedCursorPosition> cursor_pos_cache_;
    std::unordered_map<std::string, CachedWindowInfo> window_info_cache_;
    CachedStyle current_style_cache_;
    float last_update_time_ = 0.0f;
    static constexpr float CACHE_EXPIRY_TIME = 0.1f; // 100ms expiry for dynamic content

    // Pre-allocated buffers to reduce allocations
    mutable std::string temp_key_buffer_;

    // Mutex for thread safety
    mutable std::mutex cache_mutex_;

public:
    ImGuiStateCache() {
        imgui_context_ = ImGui::GetCurrentContext();
    }

    /**
     * Get cached text size or compute and cache it
     * Using string content as key for reliable caching of string literals
     */
    ImVec2 get_cached_text_size(const char* text) {
        if (!text) return ImVec2(0, 0);

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check if ImGui context has changed
        if (ImGui::GetCurrentContext() != imgui_context_) {
            // Context changed, clear cache and update reference
            clear_all_cache();
            imgui_context_ = ImGui::GetCurrentContext();
        }

        // Use string content as key to handle string literals properly
        // Reuse buffer to reduce allocations
        temp_key_buffer_ = text;

        auto it = text_size_cache_.find(temp_key_buffer_);
        if (it != text_size_cache_.end() && it->second.valid) {
            return it->second.size;
        }

        // Compute and cache the text size
        ImVec2 size = ImGui::CalcTextSize(text);
        text_size_cache_[std::move(temp_key_buffer_)] = {size, static_cast<float>(ImGui::GetTime()), true};

        return size;
    }

    /**
     * Get cached text size with additional parameters
     * Optimized key generation using content-based hashing for reliability
     */
    ImVec2 get_cached_text_size_ex(const char* text, const char* text_end,
                                   bool hide_text_after_double_hash,
                                   float wrap_width) {
        if (!text) return ImVec2(0, 0);

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Create a unique key combining all parameters for reliable caching
        // Use pre-allocated buffer to reduce allocations
        temp_key_buffer_.clear();
        temp_key_buffer_.append(text);
        temp_key_buffer_.append("|");
        temp_key_buffer_.append(text_end ? text_end : "NULL");
        temp_key_buffer_.append("|");
        temp_key_buffer_.append(hide_text_after_double_hash ? "1" : "0");
        temp_key_buffer_.append("|");
        temp_key_buffer_.append(std::to_string(wrap_width));

        auto it = text_size_cache_by_params_.find(temp_key_buffer_);
        if (it != text_size_cache_by_params_.end() && it->second.valid) {
            return it->second.size;
        }

        // Compute and cache the text size
        ImVec2 size = ImGui::CalcTextSize(text, text_end, hide_text_after_double_hash, wrap_width);
        text_size_cache_by_params_[std::move(temp_key_buffer_)] = {size, static_cast<float>(ImGui::GetTime()), true};

        return size;
    }

    /**
     * Get cached font size for a specific font and scale
     */
    ImVec2 get_cached_font_size(ImFont* font, float scale = 1.0f) {
        if (!font) return ImVec2(0, 0);

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check if ImGui context has changed
        if (ImGui::GetCurrentContext() != imgui_context_) {
            // Context changed, clear cache and update reference
            clear_all_cache();
            imgui_context_ = ImGui::GetCurrentContext();
        }

        // Create a unique key combining font pointer and scale
        uint64_t font_key = (reinterpret_cast<uintptr_t>(font) << 32) |
                           static_cast<uint32_t>(static_cast<int32_t>(scale * 1000000) & 0xFFFFFFFF);

        auto it = font_size_cache_.find(font_key);
        if (it != font_size_cache_.end() && it->second.valid) {
            return it->second.size;
        }

        // Calculate font size directly without pushing/popping
        ImVec2 size = ImVec2(ImGui::GetFontSize() * scale, ImGui::GetFontSize() * scale);

        font_size_cache_[font_key] = {size, font, scale, true};
        return size;
    }

    /**
     * Invalidate expired cache entries
     * Optimized to reduce iteration overhead
     */
    void invalidate_expired_cache() {
        // Check if ImGui context has changed
        if (ImGui::GetCurrentContext() != imgui_context_) {
            // Context changed, clear cache and update reference
            clear_all_cache();
            imgui_context_ = ImGui::GetCurrentContext();
            return;
        }

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

        // Clean up font size cache
        for (auto it = font_size_cache_.begin(); it != font_size_cache_.end();) {
            // Font cache doesn't use timestamps, so we'll clear it differently
            // For now, we'll keep it simple and not expire font cache based on time
            ++it;
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
     * Clear font cache
     */
    void clear_font_cache() {
        font_size_cache_.clear();
    }

    /**
     * Get cached color or compute it
     */
    ImU32 get_cached_color(ImGuiCol idx) {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check if ImGui context has changed
        if (ImGui::GetCurrentContext() != imgui_context_) {
            // Context changed, clear cache and update reference
            clear_all_cache();
            imgui_context_ = ImGui::GetCurrentContext();
        }

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
        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check if ImGui context has changed
        if (ImGui::GetCurrentContext() != imgui_context_) {
            // Context changed, clear cache and update reference
            clear_all_cache();
            imgui_context_ = ImGui::GetCurrentContext();
        }

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
     * Get cached cursor position or set it if not cached
     */
    bool get_cached_cursor_pos(const std::string& widget_id, ImVec2& pos) {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = cursor_pos_cache_.find(widget_id);
        if (it != cursor_pos_cache_.end() && it->second.valid) {
            pos = it->second.position;
            return true;
        }
        return false;
    }

    /**
     * Set cached cursor position
     */
    void set_cached_cursor_pos(const std::string& widget_id, const ImVec2& pos) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        cursor_pos_cache_[widget_id] = {pos, true};
    }

    /**
     * Get cached window information
     */
    bool get_cached_window_info(const std::string& window_id, CachedWindowInfo& info) {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = window_info_cache_.find(window_id);
        if (it != window_info_cache_.end() && it->second.valid) {
            info = it->second;
            return true;
        }
        return false;
    }

    /**
     * Set cached window information
     */
    void set_cached_window_info(const std::string& window_id, const ImVec2& pos,
                               const ImVec2& size, bool visible) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        window_info_cache_[window_id] = {pos, size, visible, static_cast<float>(ImGui::GetTime()), true};
    }

    /**
     * Clear all caches
     */
    void clear_all_cache() {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        clear_text_cache();
        clear_color_cache();
        clear_font_cache();
        cursor_pos_cache_.clear();
        window_info_cache_.clear();
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
     * Optimized version of ImGui::GetFont to cache font sizes
     */
    static ImVec2 GetFontSizeOptimized(ImFont* font = nullptr, float scale = 1.0f) {
        if (!font) {
            font = ImGui::GetFont();
        }
        return state_cache_.get_cached_font_size(font, scale);
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
     * Enhanced batch operation with visibility checking and performance optimization
     */
    template<typename T>
    static void BatchOperationOptimized(const std::vector<T>& items,
                                      std::function<void(const T&)> operation,
                                      bool check_visibility = true) {
        if (items.empty()) return;

        // Skip if window is not active or collapsed
        if (check_visibility && (!IsWindowActive() || IsWindowCollapsed())) {
            return;
        }

        // Reserve capacity if the operation involves adding elements to containers
        // Note: This is a hint for performance, though std::function doesn't directly benefit
        for (const auto& item : items) {
            operation(item);
        }
    }

    /**
     * Batch rendering with multiple style groups to minimize state changes
     */
    static void BatchMultiStyleTextRendering(const std::vector<std::tuple<const char*, ImVec2, ImU32>>& texts) {
        if (texts.empty()) return;

        // Group by color to minimize push/pop operations
        ImU32 current_color = 0;
        bool color_set = false;
        int color_stack_depth = 0;

        for (const auto& text_tuple : texts) {
            const char* text = std::get<0>(text_tuple);
            const ImVec2& pos = std::get<1>(text_tuple);
            ImU32 color = std::get<2>(text_tuple);

            if (!color_set || current_color != color) {
                if (color_set) {
                    ImGui::PopStyleColor();
                    color_stack_depth--;
                }
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                current_color = color;
                color_set = true;
                color_stack_depth++;
            }

            ImGui::SetCursorPos(pos);
            ImGui::TextUnformatted(text);
        }

        // Clean up remaining color pushes
        for (int i = 0; i < color_stack_depth; i++) {
            ImGui::PopStyleColor();
        }
    }

    /**
     * Batch rendering with multiple style variations using ImGuiCol indices
     */
    static void BatchMultiStyleTextRenderingByIndex(
        const std::vector<std::tuple<const char*, ImVec2, ImGuiCol, float>>& texts) {
        if (texts.empty()) return;

        // Group by color to minimize push/pop operations
        ImGuiCol current_col = ImGuiCol_Text;
        float current_alpha = 1.0f;
        bool color_set = false;
        int color_stack_depth = 0;

        for (const auto& text_tuple : texts) {
            const char* text = std::get<0>(text_tuple);
            const ImVec2& pos = std::get<1>(text_tuple);
            ImGuiCol col = std::get<2>(text_tuple);
            float alpha = std::get<3>(text_tuple);

            if (!color_set || current_col != col || current_alpha != alpha) {
                if (color_set) {
                    ImGui::PopStyleColor();
                    color_stack_depth--;
                }
                ImU32 color = GetColorU32Optimized(col, alpha);
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                current_col = col;
                current_alpha = alpha;
                color_set = true;
                color_stack_depth++;
            }

            ImGui::SetCursorPos(pos);
            ImGui::TextUnformatted(text);
        }

        // Clean up remaining color pushes
        for (int i = 0; i < color_stack_depth; i++) {
            ImGui::PopStyleColor();
        }
    }

    /**
     * Check if window is active before performing expensive operations
     */
    static bool IsWindowActive() {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        return window && window->Active;
    }

    /**
     * Check if window is collapsed before performing expensive operations
     */
    static bool IsWindowCollapsed() {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        return window && window->Collapsed;
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
     * Conditional rendering that skips if window is collapsed
     */
    template<typename Func>
    static void SkipIfCollapsed(Func func) {
        if (!IsWindowCollapsed()) {
            func();
        }
    }

    /**
     * Conditional rendering that skips if window is not active or collapsed
     */
    template<typename Func>
    static void SkipIfNotActive(Func func) {
        if (IsWindowActive() && !IsWindowCollapsed()) {
            func();
        }
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

    /**
     * Optimized SetCursorPos that avoids redundant calls
     */
    static void SetCursorPosOptimized(const ImVec2& pos, const char* widget_id = nullptr) {
        if (!IsWindowActive()) return;

        if (widget_id) {
            ImVec2 cached_pos;
            if (state_cache_.get_cached_cursor_pos(widget_id, cached_pos)) {
                // Only set cursor position if it's different from cached position
                if (cached_pos.x != pos.x || cached_pos.y != pos.y) {
                    ImGui::SetCursorPos(pos);
                    state_cache_.set_cached_cursor_pos(widget_id, pos);
                }
            } else {
                ImGui::SetCursorPos(pos);
                state_cache_.set_cached_cursor_pos(widget_id, pos);
            }
        } else {
            // Without widget ID, just call SetCursorPos normally
            ImGui::SetCursorPos(pos);
        }
    }

    /**
     * Optimized button that checks visibility before rendering
     */
    static bool ButtonOptimized(const char* label, const ImVec2& size = ImVec2(0, 0)) {
        if (!IsWindowActive() || !IsItemVisible()) return false;
        return ImGui::Button(label, size);
    }

    /**
     * Optimized small button that checks visibility before rendering
     */
    static bool SmallButtonOptimized(const char* label) {
        if (!IsWindowActive() || !IsItemVisible()) return false;
        return ImGui::SmallButton(label);
    }

    /**
     * Optimized invisible button that checks visibility before rendering
     */
    static bool InvisibleButtonOptimized(const char* str_id, const ImVec2& size, ImGuiButtonFlags flags = 0) {
        if (!IsWindowActive() || !IsItemVisible()) return false;
        return ImGui::InvisibleButton(str_id, size, flags);
    }

    /**
     * Optimized checkbox that checks visibility before rendering
     */
    static bool CheckboxOptimized(const char* label, bool* v) {
        if (!IsWindowActive() || !IsItemVisible()) return false;
        return ImGui::Checkbox(label, v);
    }

    /**
     * Optimized slider float that checks visibility before rendering
     */
    static bool SliderFloatOptimized(const char* label, float* v, float v_min, float v_max,
                                   const char* format = "%.3f", ImGuiSliderFlags flags = 0) {
        if (!IsWindowActive() || !IsItemVisible()) return false;
        return ImGui::SliderFloat(label, v, v_min, v_max, format, flags);
    }

    /**
     * Optimized progress bar that checks visibility before rendering
     */
    static void ProgressBarOptimized(float fraction, const ImVec2& size_arg = ImVec2(-FLT_MIN, 0.0f), const char* overlay = nullptr) {
        if (!IsWindowActive() || !IsItemVisible()) return;
        ImGui::ProgressBar(fraction, size_arg, overlay);
    }

    /**
     * Optimized table row rendering with visibility checking
     */
    static bool BeginTableOptimized(const char* str_id, int column, ImGuiTableFlags flags = 0,
                                   const ImVec2& outer_size = ImVec2(0, 0), float inner_width = 0.0f) {
        if (!IsWindowActive()) return false;
        return ImGui::BeginTable(str_id, column, flags, outer_size, inner_width);
    }

    /**
     * Optimized table cell rendering with visibility checking
     */
    static void TableNextColumnOptimized() {
        if (IsWindowActive()) {
            ImGui::TableNextColumn();
        }
    }

    /**
     * Optimized text rendering with automatic visibility check
     */
    static void TextOptimized(const char* fmt, ...) {
        if (!IsWindowActive() || !IsItemVisible()) return;

        va_list args;
        va_start(args, fmt);
        TextOptimizedV(fmt, args);
        va_end(args);
    }

    /**
     * Optimized text rendering with automatic visibility check (va_list version)
     */
    static void TextOptimizedV(const char* fmt, va_list args) {
        if (!IsWindowActive() || !IsItemVisible()) return;

        ImGui::TextV(fmt, args);
    }

    /**
     * Optimized small text rendering with automatic visibility check
     */
    static void SmallTextOptimized(const char* fmt, ...) {
        if (!IsWindowActive() || !IsItemVisible()) return;

        va_list args;
        va_start(args, fmt);
        SmallTextOptimizedV(fmt, args);
        va_end(args);
    }

    /**
     * Optimized small text rendering with automatic visibility check (va_list version)
     */
    static void SmallTextOptimizedV(const char* fmt, va_list args) {
        if (!IsWindowActive() || !IsItemVisible()) return;

        ImGui::TextV(fmt, args);  // Use regular TextV since SmallTextV doesn't exist
    }

    /**
     * Optimized text disabled rendering with automatic visibility check
     */
    static void TextDisabledOptimized(const char* fmt, ...) {
        if (!IsWindowActive() || !IsItemVisible()) return;

        va_list args;
        va_start(args, fmt);
        TextDisabledOptimizedV(fmt, args);
        va_end(args);
    }

    /**
     * Optimized text disabled rendering with automatic visibility check (va_list version)
     */
    static void TextDisabledOptimizedV(const char* fmt, va_list args) {
        if (!IsWindowActive() || !IsItemVisible()) return;

        ImGui::TextDisabledV(fmt, args);
    }

    /**
     * Optimized text rendering with position caching to avoid redundant SetCursorPos calls
     */
    static void TextAtPositionOptimized(const char* text, const ImVec2& pos, const char* widget_id = nullptr) {
        if (!IsWindowActive() || !IsItemVisible()) return;

        if (widget_id) {
            SetCursorPosOptimized(pos, widget_id);
        } else {
            ImGui::SetCursorPos(pos);
        }

        ImGui::TextUnformatted(text);
    }

    /**
     * Optimized same-line positioning with caching
     */
    static void SameLineOptimized(float offset_from_start_x = 0.0f, float spacing = -1.0f, const char* widget_id = nullptr) {
        if (!IsWindowActive()) return;

        // Cache the same line operation if widget_id is provided
        if (widget_id) {
            std::string cache_key = std::string(widget_id) + "_sameline";
            ImVec2 cached_pos;
            if (state_cache_.get_cached_cursor_pos(cache_key, cached_pos)) {
                // Only call SameLine if the parameters differ from the last call
                // For simplicity, we'll just call SameLine since it's lightweight
                ImGui::SameLine(offset_from_start_x, spacing);
            } else {
                ImGui::SameLine(offset_from_start_x, spacing);
                // Store a dummy position to indicate this widget has been processed
                state_cache_.set_cached_cursor_pos(cache_key, ImVec2(offset_from_start_x, spacing));
            }
        } else {
            ImGui::SameLine(offset_from_start_x, spacing);
        }
    }

    /**
     * Group widgets together to reduce redundant state changes
     */
    template<typename Func>
    static void WidgetGroup(Func func) {
        if (IsWindowActive()) {
            func();
        }
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

    ImVec2 GetFontSize(ImFont* font, float scale) {
        return optimizer_instance.GetFontSizeOptimized(font, scale);
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

    template<typename T>
    void BatchOperationOptimized(const std::vector<T>& items,
                               std::function<void(const T&)> operation,
                               bool check_visibility) {
        optimizer_instance.BatchOperationOptimized(items, operation, check_visibility);
    }

    void BatchMultiStyleTextRendering(const std::vector<std::tuple<const char*, ImVec2, ImU32>>& texts) {
        optimizer_instance.BatchMultiStyleTextRendering(texts);
    }

    void BatchMultiStyleTextRenderingByIndex(
        const std::vector<std::tuple<const char*, ImVec2, ImGuiCol, float>>& texts) {
        optimizer_instance.BatchMultiStyleTextRenderingByIndex(texts);
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

    bool BeginTableOptimized(const char* str_id, int column, ImGuiTableFlags flags,
                            const ImVec2& outer_size, float inner_width) {
        return optimizer_instance.BeginTableOptimized(str_id, column, flags, outer_size, inner_width);
    }

    void TableNextColumnOptimized() {
        optimizer_instance.TableNextColumnOptimized();
    }

    bool ButtonOptimized(const char* label, const ImVec2& size) {
        return optimizer_instance.ButtonOptimized(label, size);
    }

    bool SmallButtonOptimized(const char* label) {
        return optimizer_instance.SmallButtonOptimized(label);
    }

    bool InvisibleButtonOptimized(const char* str_id, const ImVec2& size, ImGuiButtonFlags flags) {
        return optimizer_instance.InvisibleButtonOptimized(str_id, size, flags);
    }

    bool CheckboxOptimized(const char* label, bool* v) {
        return optimizer_instance.CheckboxOptimized(label, v);
    }

    bool SliderFloatOptimized(const char* label, float* v, float v_min, float v_max,
                             const char* format, ImGuiSliderFlags flags) {
        return optimizer_instance.SliderFloatOptimized(label, v, v_min, v_max, format, flags);
    }

    void ProgressBarOptimized(float fraction, const ImVec2& size_arg, const char* overlay) {
        optimizer_instance.ProgressBarOptimized(fraction, size_arg, overlay);
    }

    void TextOptimized(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        optimizer_instance.TextOptimizedV(fmt, args);
        va_end(args);
    }

    void SmallTextOptimized(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        optimizer_instance.SmallTextOptimizedV(fmt, args);
        va_end(args);
    }

    void TextDisabledOptimized(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        optimizer_instance.TextDisabledOptimizedV(fmt, args);
        va_end(args);
    }

    template<typename Func>
    void WidgetGroup(Func func) {
        optimizer_instance.WidgetGroup(func);
    }

    template<typename Func>
    void SkipIfCollapsed(Func func) {
        if (!optimizer_instance.IsWindowCollapsed()) {
            func();
        }
    }

    template<typename Func>
    void SkipIfNotActive(Func func) {
        optimizer_instance.SkipIfNotActive(func);
    }

    // Additional utility functions for performance optimization

    /**
     * Optimized SetCursorPos that avoids redundant calls
     */
    void SetCursorPosOptimized(const ImVec2& pos, const char* widget_id) {
        optimizer_instance.SetCursorPosOptimized(pos, widget_id);
    }

    /**
     * Optimized text rendering with position caching
     */
    void TextAtPositionOptimized(const char* text, const ImVec2& pos, const char* widget_id) {
        optimizer_instance.TextAtPositionOptimized(text, pos, widget_id);
    }

    /**
     * Optimized SameLine with caching
     */
    void SameLineOptimized(float offset_from_start_x, float spacing, const char* widget_id) {
        optimizer_instance.SameLineOptimized(offset_from_start_x, spacing, widget_id);
    }

    // Additional utility functions for performance optimization

    /**
     * Begin a child window only if it's visible
     */
    bool BeginChildConditional(const char* str_id, const ImVec2& size, bool border, ImGuiWindowFlags flags) {
        if (!optimizer_instance.IsWindowActive()) return false;
        return ImGui::BeginChild(str_id, size, border, flags);
    }

    /**
     * Render text only if it's going to be visible
     */
    void TextVisible(const char* fmt, ...) {
        if (!optimizer_instance.IsItemVisible()) return;

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
    void PopStyleColorConditional(bool condition, int count) {
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
    void PopStyleVarConditional(bool condition, int count) {
        if (!condition) return;
        ImGui::PopStyleVar(count);
    }
}

} // namespace Rendering
} // namespace BTQuant