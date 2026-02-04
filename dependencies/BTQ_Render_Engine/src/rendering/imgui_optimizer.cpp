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
        float timestamp;
        bool valid;
    };

    // Cache for cursor positions to avoid redundant SetCursorPos calls
    struct CachedCursorPosition {
        ImVec2 position;
        float timestamp;
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

    // Cache for widget bounds to avoid repeated calculations
    struct CachedWidgetBounds {
        ImVec2 min;
        ImVec2 max;
        float timestamp;
        bool valid;
    };

    // Cache for window scroll information
public:
    struct CachedScrollInfo {
        float scroll_x;
        float scroll_y;
        float scroll_max_x;
        float scroll_max_y;
        float timestamp;
        bool valid;
    };

    // Cache for window content region
    struct CachedContentRegion {
        ImVec2 min;
        ImVec2 max;
        ImVec2 size;
        float timestamp;
        bool valid;
    };
private:

    // Cache for item rectangles to avoid repeated calculations
    struct CachedItemRect {
        ImVec2 min;
        ImVec2 max;
        ImVec2 size;
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
    std::unordered_map<std::string, CachedWidgetBounds> widget_bounds_cache_;
    std::unordered_map<std::string, CachedScrollInfo> scroll_info_cache_;
    std::unordered_map<std::string, CachedContentRegion> content_region_cache_;
    CachedStyle current_style_cache_;
    float last_update_time_ = 0.0f;
    static constexpr float CACHE_EXPIRY_TIME = 0.1f; // 100ms expiry for dynamic content
    static constexpr float LONG_CACHE_EXPIRY_TIME = 1.0f; // 1s expiry for static content

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
            float current_time = ImGui::GetTime();
            // Check if cache entry is still valid (hasn't expired)
            if (current_time - it->second.timestamp < CACHE_EXPIRY_TIME) {
                return it->second.size;
            } else {
                // Entry has expired, remove it
                text_size_cache_.erase(it);
            }
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
            float current_time = ImGui::GetTime();
            // Check if cache entry is still valid (hasn't expired)
            if (current_time - it->second.timestamp < LONG_CACHE_EXPIRY_TIME) {
                return it->second.size;
            } else {
                // Entry has expired, remove it
                font_size_cache_.erase(it);
            }
        }

        // Calculate font size directly without pushing/popping
        ImVec2 size = ImVec2(ImGui::GetFontSize() * scale, ImGui::GetFontSize() * scale);

        font_size_cache_[font_key] = {size, font, scale, static_cast<float>(ImGui::GetTime()), true};
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
        float long_expiry_threshold = current_time - LONG_CACHE_EXPIRY_TIME;

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

        // Clean up font size cache - now with proper timestamp checking
        for (auto it = font_size_cache_.begin(); it != font_size_cache_.end();) {
            if (it->second.timestamp < long_expiry_threshold) {
                it = font_size_cache_.erase(it);
            } else {
                ++it;
            }
        }

        // Clean up cursor position cache
        for (auto it = cursor_pos_cache_.begin(); it != cursor_pos_cache_.end();) {
            if (it->second.timestamp < expiry_threshold) {
                it = cursor_pos_cache_.erase(it);
            } else {
                ++it;
            }
        }

        // Clean up window info cache
        for (auto it = window_info_cache_.begin(); it != window_info_cache_.end();) {
            if (it->second.timestamp < expiry_threshold) {
                it = window_info_cache_.erase(it);
            } else {
                ++it;
            }
        }

        // Clean up widget bounds cache
        for (auto it = widget_bounds_cache_.begin(); it != widget_bounds_cache_.end();) {
            if (it->second.timestamp < expiry_threshold) {
                it = widget_bounds_cache_.erase(it);
            } else {
                ++it;
            }
        }

        // Clean up scroll info cache
        for (auto it = scroll_info_cache_.begin(); it != scroll_info_cache_.end();) {
            if (it->second.timestamp < expiry_threshold) {
                it = scroll_info_cache_.erase(it);
            } else {
                ++it;
            }
        }

        // Clean up content region cache
        for (auto it = content_region_cache_.begin(); it != content_region_cache_.end();) {
            if (it->second.timestamp < expiry_threshold) {
                it = content_region_cache_.erase(it);
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
     * Get cached widget bounds or compute them
     */
    bool get_cached_widget_bounds(const std::string& widget_id, CachedWidgetBounds& bounds) {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = widget_bounds_cache_.find(widget_id);
        if (it != widget_bounds_cache_.end() && it->second.valid) {
            float current_time = ImGui::GetTime();
            // Check if cache entry is still valid (hasn't expired)
            if (current_time - it->second.timestamp < CACHE_EXPIRY_TIME) {
                bounds = it->second;
                return true;
            } else {
                // Entry has expired, remove it
                widget_bounds_cache_.erase(it);
            }
        }
        return false;
    }

    /**
     * Set cached widget bounds
     */
    void set_cached_widget_bounds(const std::string& widget_id, const ImVec2& min_bound, const ImVec2& max_bound) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        widget_bounds_cache_[widget_id] = {min_bound, max_bound, static_cast<float>(ImGui::GetTime()), true};
    }

    /**
     * Get cached scroll information or compute it
     */
    bool get_cached_scroll_info(const std::string& window_id, CachedScrollInfo& scroll_info) {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = scroll_info_cache_.find(window_id);
        if (it != scroll_info_cache_.end() && it->second.valid) {
            float current_time = ImGui::GetTime();
            // Check if cache entry is still valid (hasn't expired)
            if (current_time - it->second.timestamp < CACHE_EXPIRY_TIME) {
                scroll_info = it->second;
                return true;
            } else {
                // Entry has expired, remove it
                scroll_info_cache_.erase(it);
            }
        }
        return false;
    }

    /**
     * Set cached scroll information
     */
    void set_cached_scroll_info(const std::string& window_id, float scroll_x, float scroll_y,
                               float scroll_max_x, float scroll_max_y) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        scroll_info_cache_[window_id] = {scroll_x, scroll_y, scroll_max_x, scroll_max_y,
                                         static_cast<float>(ImGui::GetTime()), true};
    }

    /**
     * Get cached content region or compute it
     */
    bool get_cached_content_region(const std::string& window_id, CachedContentRegion& content_region) {
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = content_region_cache_.find(window_id);
        if (it != content_region_cache_.end() && it->second.valid) {
            float current_time = ImGui::GetTime();
            // Check if cache entry is still valid (hasn't expired)
            if (current_time - it->second.timestamp < CACHE_EXPIRY_TIME) {
                content_region = it->second;
                return true;
            } else {
                // Entry has expired, remove it
                content_region_cache_.erase(it);
            }
        }
        return false;
    }

    /**
     * Set cached content region
     */
    void set_cached_content_region(const std::string& window_id, const ImVec2& min,
                                  const ImVec2& max, const ImVec2& size) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        content_region_cache_[window_id] = {min, max, size, static_cast<float>(ImGui::GetTime()), true};
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
        widget_bounds_cache_.clear();
        scroll_info_cache_.clear();
        content_region_cache_.clear();
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
        int color_push_count = 0; // Track how many colors we've pushed

        for (const auto& text_tuple : texts) {
            const char* text = std::get<0>(text_tuple);
            const ImVec2& pos = std::get<1>(text_tuple);
            ImU32 color = std::get<2>(text_tuple);

            if (!color_set || current_color != color) {
                if (color_set) {
                    ImGui::PopStyleColor();
                    color_push_count--;
                }
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                current_color = color;
                color_set = true;
                color_push_count++;
            }

            ImGui::SetCursorPos(pos);
            ImGui::TextUnformatted(text);
        }

        if (color_push_count > 0) {
            ImGui::PopStyleColor(color_push_count); // Pop all remaining colors at once
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
     * Batch rendering with multiple attributes to minimize state changes
     * Combines color, font size, and positioning changes
     */
    static void BatchAdvancedTextRendering(const std::vector<std::tuple<const char*, ImVec2, ImU32, float>>& texts) {
        if (texts.empty()) return;

        // Group by color to minimize push/pop operations
        ImU32 current_color = 0;
        float current_font_scale = 1.0f;
        bool style_set = false;
        int color_stack_depth = 0;
        int font_stack_depth = 0;

        for (const auto& text_tuple : texts) {
            const char* text = std::get<0>(text_tuple);
            const ImVec2& pos = std::get<1>(text_tuple);
            ImU32 color = std::get<2>(text_tuple);
            float font_scale = std::get<3>(text_tuple);

            // Handle color changes
            if (!style_set || current_color != color) {
                if (style_set) {
                    ImGui::PopStyleColor();
                    color_stack_depth--;
                }
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                current_color = color;
                style_set = true;
                color_stack_depth++;
            }

            // Handle font scaling changes
            if (!style_set || current_font_scale != font_scale) {
                if (font_stack_depth > 0) {
                    ImGui::PopFont();
                    font_stack_depth--;
                }
                // Note: Actual font scaling would require font management
                // For now, we just track the change
                current_font_scale = font_scale;
            }

            ImGui::SetCursorPos(pos);
            ImGui::TextUnformatted(text);
        }

        // Clean up remaining pushes
        if (color_stack_depth > 0) {
            ImGui::PopStyleColor(color_stack_depth);
        }
        if (font_stack_depth > 0) {
            ImGui::PopFont();
        }
    }

    /**
     * Optimized group of widgets that share the same style properties
     */
    template<typename T>
    static void GroupStyledWidgets(const std::vector<T>& widgets,
                                  std::function<void(const T&, int)> render_func,
                                  ImGuiCol color_idx, float alpha_mul = 1.0f) {
        if (widgets.empty()) return;

        // Apply style once for all widgets
        ImU32 color = GetColorU32Optimized(color_idx, alpha_mul);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, color);

        for (size_t i = 0; i < widgets.size(); ++i) {
            render_func(widgets[i], static_cast<int>(i));
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

        // Clean up remaining color pushes - pop all at once for efficiency
        if (color_stack_depth > 0) {
            ImGui::PopStyleColor(color_stack_depth);
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

        // Clean up remaining color pushes - pop all at once for efficiency
        if (color_stack_depth > 0) {
            ImGui::PopStyleColor(color_stack_depth);
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
     * Conditional rendering that skips if item is not visible
     */
    template<typename Func>
    static void SkipIfNotVisible(Func func) {
        if (IsItemVisible()) {
            func();
        }
    }

    /**
     * Combined conditional rendering that skips if window is not active, collapsed, or item not visible
     */
    template<typename Func>
    static void SkipIfNotActiveOrVisible(Func func) {
        if (IsWindowActive() && !IsWindowCollapsed() && IsItemVisible()) {
            func();
        }
    }

    /**
     * Conditional rendering with bounds checking
     */
    template<typename Func>
    static bool ConditionalRenderWithBounds(const char* widget_id, const ImVec2& min_bound, const ImVec2& max_bound, Func func) {
        if (!IsWindowActive() || IsWindowCollapsed()) {
            return false;
        }

        // Check if bounds are visible
        if (!IsRectVisible(min_bound, max_bound)) {
            return false;
        }

        // Update cached bounds
        state_cache_.set_cached_widget_bounds(widget_id, min_bound, max_bound);
        func();
        return true;
    }

    /**
     * Get cached scroll information for a window
     */
    static bool GetCachedScrollInfo(const char* window_id, float& scroll_x, float& scroll_y,
                                   float& scroll_max_x, float& scroll_max_y) {
        ImGuiStateCache::CachedScrollInfo info;
        if (state_cache_.get_cached_scroll_info(std::string(window_id), info)) {
            scroll_x = info.scroll_x;
            scroll_y = info.scroll_y;
            scroll_max_x = info.scroll_max_x;
            scroll_max_y = info.scroll_max_y;
            return true;
        }
        return false;
    }

    /**
     * Get cached content region for a window
     */
    static bool GetCachedContentRegion(const char* window_id, ImVec2& min, ImVec2& max, ImVec2& size) {
        ImGuiStateCache::CachedContentRegion region;
        if (state_cache_.get_cached_content_region(std::string(window_id), region)) {
            min = region.min;
            max = region.max;
            size = region.size;
            return true;
        }
        return false;
    }

    /**
     * Optimized scroll information retrieval
     */
    static void GetScrollInfoOptimized(const char* window_id, float& scroll_x, float& scroll_y,
                                      float& scroll_max_x, float& scroll_max_y) {
        if (!GetCachedScrollInfo(window_id, scroll_x, scroll_y, scroll_max_x, scroll_max_y)) {
            // Get actual scroll information from ImGui
            ImGuiWindow* window = ImGui::GetCurrentWindow();
            if (window) {
                scroll_x = window->Scroll.x;
                scroll_y = window->Scroll.y;
                scroll_max_x = window->ScrollMax.x;
                scroll_max_y = window->ScrollMax.y;

                // Cache the values
                state_cache_.set_cached_scroll_info(std::string(window_id), scroll_x, scroll_y,
                                                   scroll_max_x, scroll_max_y);
            } else {
                scroll_x = scroll_y = scroll_max_x = scroll_max_y = 0.0f;
            }
        }
    }

    /**
     * Optimized content region retrieval
     */
    static void GetContentRegionOptimized(const char* window_id, ImVec2& min, ImVec2& max, ImVec2& size) {
        if (!GetCachedContentRegion(window_id, min, max, size)) {
            // Get actual content region from ImGui
            min = ImGui::GetContentRegionAvail();
            size = ImVec2(ImGui::GetWindowContentRegionMax().x - ImGui::GetWindowContentRegionMin().x,
                         ImGui::GetWindowContentRegionMax().y - ImGui::GetWindowContentRegionMin().y);
            max = ImVec2(min.x + size.x, min.y + size.y);

            // Cache the values
            state_cache_.set_cached_content_region(std::string(window_id), min, max, size);
        }
    }

    /**
     * Batch style variable changes to minimize push/pop operations
     */
    static void BatchStyleChanges(const std::vector<std::pair<ImGuiStyleVar, float>>& float_vars,
                                  const std::vector<std::pair<ImGuiStyleVar, ImVec2>>& vec2_vars,
                                  std::function<void()> render_func) {
        int push_count = 0;

        // Push float style variables
        for (const auto& var : float_vars) {
            ImGui::PushStyleVar(var.first, var.second);
            push_count++;
        }

        // Push ImVec2 style variables
        for (const auto& var : vec2_vars) {
            ImGui::PushStyleVar(var.first, var.second);
            push_count++;
        }

        // Execute rendering function
        render_func();

        // Pop all style variables at once
        if (push_count > 0) {
            ImGui::PopStyleVar(push_count);
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

    /**
     * Batch style changes to minimize push/pop operations
     */
    void BatchStyleChanges(const std::vector<std::pair<ImGuiStyleVar, float>>& float_vars,
                          const std::vector<std::pair<ImGuiStyleVar, ImVec2>>& vec2_vars,
                          std::function<void()> render_func) {
        optimizer_instance.BatchStyleChanges(float_vars, vec2_vars, render_func);
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

    /**
     * Conditional rendering that skips if item is not visible
     */
    template<typename Func>
    void SkipIfNotVisible(Func func) {
        optimizer_instance.SkipIfNotVisible(func);
    }

    /**
     * Combined conditional rendering that skips if window is not active, collapsed, or item not visible
     */
    template<typename Func>
    void SkipIfNotActiveOrVisible(Func func) {
        optimizer_instance.SkipIfNotActiveOrVisible(func);
    }

    /**
     * Conditional rendering with bounds checking
     */
    template<typename Func>
    bool ConditionalRenderWithBounds(const char* widget_id, const ImVec2& min_bound, const ImVec2& max_bound, Func func) {
        return optimizer_instance.ConditionalRenderWithBounds(widget_id, min_bound, max_bound, func);
    }

    /**
     * Optimized scroll information retrieval
     */
    void GetScrollInfoOptimized(const char* window_id, float& scroll_x, float& scroll_y,
                               float& scroll_max_x, float& scroll_max_y) {
        optimizer_instance.GetScrollInfoOptimized(window_id, scroll_x, scroll_y, scroll_max_x, scroll_max_y);
    }

    /**
     * Optimized content region retrieval
     */
    void GetContentRegionOptimized(const char* window_id, ImVec2& min, ImVec2& max, ImVec2& size) {
        optimizer_instance.GetContentRegionOptimized(window_id, min, max, size);
    }

    /**
     * Batch rendering with multiple attributes to minimize state changes
     */
    void BatchAdvancedTextRendering(const std::vector<std::tuple<const char*, ImVec2, ImU32, float>>& texts) {
        optimizer_instance.BatchAdvancedTextRendering(texts);
    }
}

} // namespace Rendering
} // namespace BTQuant