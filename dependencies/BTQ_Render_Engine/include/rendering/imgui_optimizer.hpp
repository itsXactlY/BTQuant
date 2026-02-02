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

    /**
     * Batch similar operations to minimize state changes
     */
    template<typename T>
    void BatchOperation(const std::vector<T>& items, 
                       std::function<void(const T&)> operation);

    /**
     * Batch text rendering with same style to minimize state changes
     */
    void BatchTextRendering(const std::vector<std::pair<std::string, ImVec2>>& texts);

    /**
     * Batch colored text rendering
     */
    void BatchColoredTextRendering(const std::vector<std::tuple<std::string, ImVec2, ImU32>>& texts);

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
     * Update cache periodically
     */
    void UpdateCache();

    /**
     * Clear all caches
     */
    void ClearCache();
}

} // namespace Rendering
} // namespace BTQuant