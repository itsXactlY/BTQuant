/// @file unified_theme_system.cpp
/// @brief Implements the unified theme system for MMT Deep Void aesthetic.

#include "ui/unified_theme_system.hpp"

#include <imgui.h>

#include "dashboard_config.hpp"

namespace BTQuant {

// Helper to convert ColorRGBA to ImVec4
static inline ImVec4 to_imvec4(const ColorRGBA& c) {
    return ImVec4(c.r, c.g, c.b, c.a);
}

void UnifiedThemeSystem::apply(const RenderEngine::ThemeConfig& cfg) {
    apply_colors(ImGui::GetStyle(), cfg);
    apply_borders(ImGui::GetStyle());
    apply_fonts();
}

void UnifiedThemeSystem::apply_mmt_void() {
    RenderEngine::ThemeConfig mmt_cfg;
    mmt_cfg.background_color = {0.043f, 0.055f, 0.067f, 1.0f};      // #0B0E11 - Deep Void
    mmt_cfg.child_bg_color = {0.082f, 0.098f, 0.118f, 1.0f};        // #15191E
    mmt_cfg.header_bg_color = {0.055f, 0.067f, 0.082f, 1.0f};       // #0E111A
    mmt_cfg.text_color = {0.9f, 0.9f, 0.9f, 1.0f};
    mmt_cfg.accent_color = {0.29f, 0.565f, 1.0f, 1.0f};             // #4A90FF
    mmt_cfg.positive_color = {0.0f, 0.9f, 0.4f, 1.0f};              // #00E566 - Neon Mint
    mmt_cfg.negative_color = {0.9f, 0.1f, 0.15f, 1.0f};             // #E61926 - Crimson
    mmt_cfg.neutral_color = {0.5f, 0.5f, 0.5f, 1.0f};
    mmt_cfg.grid_line_color = {0.15f, 0.18f, 0.22f, 1.0f};          // Subtle grid

    apply(mmt_cfg);
}

void UnifiedThemeSystem::apply_borders(ImGuiStyle& s) {
    s.WindowBorderSize  = 0.0f;
    s.ChildBorderSize   = 0.0f;
    s.FrameBorderSize   = 0.0f;
    s.WindowRounding    = 0.0f;
    s.FrameRounding     = 0.0f;
    s.PopupRounding     = 0.0f;
    s.TabRounding       = 0.0f;
    s.ScrollbarRounding = 0.0f;
    s.GrabRounding      = 0.0f;
}

void UnifiedThemeSystem::apply_colors(ImGuiStyle& s, const RenderEngine::ThemeConfig& cfg) {
    ImVec4* colors = s.Colors;

    colors[ImGuiCol_Text]                   = to_imvec4(cfg.text_color);
    colors[ImGuiCol_TextDisabled]           = ImVec4(cfg.text_color.r * 0.5f, cfg.text_color.g * 0.5f, cfg.text_color.b * 0.5f, cfg.text_color.a * 0.5f);
    colors[ImGuiCol_WindowBg]               = to_imvec4(cfg.background_color);
    colors[ImGuiCol_ChildBg]                = to_imvec4(cfg.child_bg_color);
    colors[ImGuiCol_PopupBg]                = ImVec4(cfg.background_color.r, cfg.background_color.g, cfg.background_color.b, 0.92f);
    colors[ImGuiCol_Border]                 = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);  // Transparent border
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_FrameBg]                = to_imvec4(cfg.child_bg_color);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.1f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.33f);
    colors[ImGuiCol_TitleBg]                = to_imvec4(cfg.header_bg_color);
    colors[ImGuiCol_TitleBgActive]          = to_imvec4(cfg.header_bg_color);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(cfg.header_bg_color.r, cfg.header_bg_color.g, cfg.header_bg_color.b, 0.75f);
    colors[ImGuiCol_MenuBarBg]              = to_imvec4(cfg.child_bg_color);
    colors[ImGuiCol_ScrollbarBg]            = to_imvec4(cfg.child_bg_color);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(cfg.neutral_color.r * 0.5f, cfg.neutral_color.g * 0.5f, cfg.neutral_color.b * 0.5f, 0.3f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(cfg.neutral_color.r * 0.5f, cfg.neutral_color.g * 0.5f, cfg.neutral_color.b * 0.5f, 0.78f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(cfg.neutral_color.r * 0.5f, cfg.neutral_color.g * 0.5f, cfg.neutral_color.b * 0.5f, 1.0f);
    colors[ImGuiCol_CheckMark]              = to_imvec4(cfg.accent_color);
    colors[ImGuiCol_SliderGrab]             = to_imvec4(cfg.accent_color);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.8f);
    colors[ImGuiCol_Button]                 = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.2f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.4f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.6f);
    colors[ImGuiCol_Header]                 = ImVec4(cfg.header_bg_color.r, cfg.header_bg_color.g, cfg.header_bg_color.b, 0.5f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.2f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.4f);
    colors[ImGuiCol_Separator]              = ImVec4(cfg.neutral_color.r * 0.5f, cfg.neutral_color.g * 0.5f, cfg.neutral_color.b * 0.5f, 0.3f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.4f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.6f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(cfg.neutral_color.r * 0.5f, cfg.neutral_color.g * 0.5f, cfg.neutral_color.b * 0.5f, 0.3f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.4f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.6f);
    colors[ImGuiCol_Tab]                    = ImVec4(cfg.header_bg_color.r, cfg.header_bg_color.g, cfg.header_bg_color.b, 0.86f);
    colors[ImGuiCol_TabHovered]             = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.6f);
    colors[ImGuiCol_TabActive]              = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.8f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(cfg.background_color.r, cfg.background_color.g, cfg.background_color.b, 0.9f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(cfg.header_bg_color.r, cfg.header_bg_color.g, cfg.header_bg_color.b, 1.0f);
    colors[ImGuiCol_PlotLines]              = to_imvec4(cfg.accent_color);
    colors[ImGuiCol_PlotLinesHovered]       = to_imvec4(cfg.positive_color);
    colors[ImGuiCol_PlotHistogram]          = to_imvec4(cfg.accent_color);
    colors[ImGuiCol_PlotHistogramHovered]   = to_imvec4(cfg.positive_color);
    colors[ImGuiCol_TableHeaderBg]          = to_imvec4(cfg.header_bg_color);
    colors[ImGuiCol_TableBorderStrong]      = ImVec4(cfg.neutral_color.r * 0.5f, cfg.neutral_color.g * 0.5f, cfg.neutral_color.b * 0.5f, 0.5f);
    colors[ImGuiCol_TableBorderLight]       = ImVec4(cfg.neutral_color.r * 0.5f, cfg.neutral_color.g * 0.5f, cfg.neutral_color.b * 0.5f, 0.3f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(cfg.background_color.r, cfg.background_color.g, cfg.background_color.b, 0.2f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(cfg.background_color.r, cfg.background_color.g, cfg.background_color.b, 0.4f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.2f);
    colors[ImGuiCol_DragDropTarget]         = ImVec4(cfg.positive_color.r, cfg.positive_color.g, cfg.positive_color.b, 0.3f);
    colors[ImGuiCol_NavHighlight]           = ImVec4(cfg.accent_color.r, cfg.accent_color.g, cfg.accent_color.b, 0.6f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(cfg.positive_color.r, cfg.positive_color.g, cfg.positive_color.b, 0.6f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(cfg.negative_color.r, cfg.negative_color.g, cfg.negative_color.b, 0.2f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(cfg.background_color.r, cfg.background_color.g, cfg.background_color.b, 0.66f);
}

void UnifiedThemeSystem::apply_fonts() {
    // This function would normally embed fonts like JetBrains Mono
    // For now, we'll just configure the default font appropriately
    ImFontConfig cfg;
    cfg.OversampleH = 4;
    cfg.OversampleV = 4;
    cfg.PixelSnapH = false;
    
    // Modify the default font to have better clarity
    ImGui::GetIO().Fonts->Build();
}

}  // namespace BTQuant