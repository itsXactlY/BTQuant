/**
 * BTQuant ImGui Optimization Header
 *
 * Header file for the ImGui optimization system that provides optimized
 * versions of common ImGui operations to improve performance in
 * professional trading dashboard applications.
 */

#pragma once

#include "imgui.h"
#include <string>
#include <vector>
#include <functional>
#include <tuple>

namespace BTQuant {
namespace Rendering {

namespace ImGuiOptimizer {

    /**
     * Optimized version of ImGui::CalcTextSize that caches results
     */
    ImVec2 CalcTextSize(const char* text, const char* text_end = nullptr,
                        bool hide_text_after_double_hash = false,
                        float wrap_width = -1.0f);

    /**
     * Optimized version of ImGui::GetFontSize that caches results
     */
    ImVec2 GetFontSize(ImFont* font = nullptr, float scale = 1.0f);

    /**
     * Optimized version of ImGui::GetColorU32 that caches results
     */
    ImU32 GetColorU32(ImGuiCol idx, float alpha_mul = 1.0f);

    /**
     * Conditional ImGui calls that only execute if condition is true
     */
    template<typename T>
    bool ConditionalBegin(T condition_func, const char* name, bool* p_open = nullptr,
                         ImGuiWindowFlags flags = 0);

    void ConditionalText(bool condition, const char* fmt, ...);

    bool ConditionalButton(bool condition, const char* label, const ImVec2& size = ImVec2(0, 0));

    bool ConditionalSliderFloat(bool condition, const char* label, float* v, float v_min, float v_max,
                               const char* format = "%.3f", ImGuiSliderFlags flags = 0);

    bool ConditionalCheckbox(bool condition, const char* label, bool* v);

    bool ConditionalCombo(bool condition, const char* label, int* current_item,
                         const char* const items[], int items_count, int popup_max_height_in_items = -1);

    bool ConditionalComboStr(bool condition, const char* label, int* current_item,
                            const char* items_separated_by_zeros, int popup_max_height_in_items = -1);

    /**
     * Batch similar operations to minimize state changes
     */
    template<typename T>
    void BatchOperation(const std::vector<T>& items,
                       std::function<void(const T&)> operation);

    /**
     * Batch text rendering with same style to minimize state changes
     */
    void BatchTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts);

    /**
     * Batch colored text rendering
     */
    void BatchColoredTextRendering(const std::vector<std::tuple<const char*, ImVec2, ImU32>>& texts);

    /**
     * Batch same-colored text rendering
     */
    void BatchSameColoredTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts, ImU32 color);

    /**
     * Batch same-styled text rendering
     */
    void BatchSameStyledTextRendering(const std::vector<std::pair<const char*, ImVec2>>& texts,
                                    ImGuiCol color_idx, float alpha_mul = 1.0f);

    /**
     * Enhanced batch operation with visibility checking and performance optimization
     */
    template<typename T>
    void BatchOperationOptimized(const std::vector<T>& items,
                               std::function<void(const T&)> operation,
                               bool check_visibility = true);

    /**
     * Batch rendering with multiple style groups to minimize state changes
     */
    void BatchMultiStyleTextRendering(const std::vector<std::tuple<const char*, ImVec2, ImU32>>& texts);

    /**
     * Batch rendering with multiple style variations using ImGuiCol indices
     */
    void BatchMultiStyleTextRenderingByIndex(
        const std::vector<std::tuple<const char*, ImVec2, ImGuiCol, float>>& texts);

    /**
     * Check if window is active before performing expensive operations
     */
    bool IsWindowActive();

    /**
     * Check if item is visible before rendering
     */
    bool IsItemVisible();

    /**
     * Check if rect is visible before rendering
     */
    bool IsRectVisible(const ImVec2& size);

    /**
     * Check if rect is visible with specific bounds
     */
    bool IsRectVisible(const ImVec2& rect_min, const ImVec2& rect_max);

    /**
     * Begin a child window only if it's visible
     */
    bool BeginChildConditional(const char* str_id, const ImVec2& size = ImVec2(0, 0), bool border = false, ImGuiWindowFlags flags = 0);

    /**
     * Render text only if it's going to be visible
     */
    void TextVisible(const char* fmt, ...);

    /**
     * Render text disabled (grayed out) based on condition
     */
    void TextDisabledConditional(bool condition, const char* fmt, ...);

    /**
     * Push a style color only if condition is met
     */
    void PushStyleColorConditional(bool condition, ImGuiCol idx, ImU32 col);

    /**
     * Pop a style color only if condition is met
     */
    void PopStyleColorConditional(bool condition, int count = 1);

    /**
     * Push a style var only if condition is met
     */
    void PushStyleVarConditional(bool condition, ImGuiStyleVar idx, float val);

    /**
     * Push a style var only if condition is met
     */
    void PushStyleVarConditional(bool condition, ImGuiStyleVar idx, const ImVec2& val);

    /**
     * Pop a style var only if condition is met
     */
    void PopStyleVarConditional(bool condition, int count = 1);

    /**
     * Update cache periodically
     */
    void UpdateCache();

    /**
     * Clear all caches
     */
    void ClearCache();

    /**
     * Optimized table row rendering with visibility checking
     */
    bool BeginTableOptimized(const char* str_id, int column, ImGuiTableFlags flags = 0,
                            const ImVec2& outer_size = ImVec2(0, 0), float inner_width = 0.0f);

    /**
     * Optimized table cell rendering with visibility checking
     */
    void TableNextColumnOptimized();

    /**
     * Optimized text rendering with automatic visibility check
     */
    void TextOptimized(const char* fmt, ...);

    /**
     * Optimized small text rendering with automatic visibility check
     */
    void SmallTextOptimized(const char* fmt, ...);

    /**
     * Optimized text disabled rendering with automatic visibility check
     */
    void TextDisabledOptimized(const char* fmt, ...);

    /**
     * Group widgets together to reduce redundant state changes
     */
    template<typename Func>
    void WidgetGroup(Func func);

    /**
     * Conditional rendering that skips if window is collapsed
     */
    template<typename Func>
    void SkipIfCollapsed(Func func);

    /**
     * Conditional rendering that skips if window is not active or collapsed
     */
    template<typename Func>
    void SkipIfNotActive(Func func);

    /**
     * Optimized button that checks visibility before rendering
     */
    bool ButtonOptimized(const char* label, const ImVec2& size = ImVec2(0, 0));

    /**
     * Optimized small button that checks visibility before rendering
     */
    bool SmallButtonOptimized(const char* label);

    /**
     * Optimized invisible button that checks visibility before rendering
     */
    bool InvisibleButtonOptimized(const char* str_id, const ImVec2& size, ImGuiButtonFlags flags = 0);

    /**
     * Optimized checkbox that checks visibility before rendering
     */
    bool CheckboxOptimized(const char* label, bool* v);

    /**
     * Optimized slider float that checks visibility before rendering
     */
    bool SliderFloatOptimized(const char* label, float* v, float v_min, float v_max,
                             const char* format = "%.3f", ImGuiSliderFlags flags = 0);

    /**
     * Optimized progress bar that checks visibility before rendering
     */
    void ProgressBarOptimized(float fraction, const ImVec2& size_arg = ImVec2(-FLT_MIN, 0.0f), const char* overlay = nullptr);

    /**
     * Optimized SetCursorPos that avoids redundant calls
     */
    void SetCursorPosOptimized(const ImVec2& pos, const char* widget_id = nullptr);

    /**
     * Optimized text rendering with position caching
     */
    void TextAtPositionOptimized(const char* text, const ImVec2& pos, const char* widget_id = nullptr);

    /**
     * Optimized SameLine with caching
     */
    void SameLineOptimized(float offset_from_start_x = 0.0f, float spacing = -1.0f, const char* widget_id = nullptr);

    /**
     * Batch style changes to minimize push/pop operations
     */
    void BatchStyleChanges(const std::vector<std::pair<ImGuiStyleVar, float>>& float_vars,
                          const std::vector<std::pair<ImGuiStyleVar, ImVec2>>& vec2_vars,
                          std::function<void()> render_func);

    /**
     * Conditional rendering that skips if item is not visible
     */
    template<typename Func>
    void SkipIfNotVisible(Func func);

    /**
     * Combined conditional rendering that skips if window is not active, collapsed, or item not visible
     */
    template<typename Func>
    void SkipIfNotActiveOrVisible(Func func);

    /**
     * Conditional rendering with bounds checking
     */
    template<typename Func>
    bool ConditionalRenderWithBounds(const char* widget_id, const ImVec2& min_bound, const ImVec2& max_bound, Func func);
}

} // namespace Rendering
} // namespace BTQuant