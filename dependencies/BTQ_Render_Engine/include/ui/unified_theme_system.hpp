#pragma once

#include "dashboard_config.hpp"

struct ImGuiStyle;

namespace BTQuant {

/**
 * @brief Unified Theme System for MMT Deep Void aesthetic
 * 
 * Centralized system for applying the MMT Dark Void theme across all ImGui elements.
 * Provides both configurable theming and hardcoded MMT preset.
 */
class UnifiedThemeSystem {
public:
    /**
     * @brief Apply theme from configuration
     * @param cfg Theme configuration to apply
     */
    static void apply(const RenderEngine::ThemeConfig& cfg);

    /**
     * @brief Apply hardcoded MMT Void theme preset
     */
    static void apply_mmt_void();

private:
    /**
     * @brief Apply border and rounding settings
     * @param s ImGui style to modify
     */
    static void apply_borders(ImGuiStyle& s);

    /**
     * @brief Apply color settings
     * @param s ImGui style to modify
     * @param cfg Theme configuration to source colors from
     */
    static void apply_colors(ImGuiStyle& s, const RenderEngine::ThemeConfig& cfg);

    /**
     * @brief Apply font settings
     */
    static void apply_fonts();
};

}  // namespace BTQuant