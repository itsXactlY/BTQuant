#pragma once

// ============================================================================
// MMT GENESIS - Deep Void Aesthetic Theme System
// Market Monkey Terminal Color Palette & Style Configuration
// ============================================================================

#include <cstdint>

#include "imgui.h"

namespace BTQuant {
namespace UI {
namespace MMT {

// ============================================================================
// MMT Color Palette - Professional Trading Terminal Colors
// ============================================================================

// Background Colors (Deep Void Aesthetic)
constexpr ImU32 COLOR_WINDOW_BG = IM_COL32(0x0B, 0x0E, 0x11, 0xFF);  // #0B0E11 - Main background
constexpr ImU32 COLOR_CHILD_BG = IM_COL32(0x15, 0x19, 0x1E, 0xFF);   // #15191E - Panel background
constexpr ImU32 COLOR_POPUP_BG = IM_COL32(0x1A, 0x1E, 0x24, 0xFF);   // #1A1E24 - Popup background
constexpr ImU32 COLOR_BORDER = IM_COL32(0x2A, 0x2E, 0x33, 0xFF);     // #2A2E33 - Border color

// Text Colors
constexpr ImU32 COLOR_TEXT_PRIMARY = IM_COL32(0xE8, 0xEA, 0xED, 0xFF);  // #E8EAED - Primary text
constexpr ImU32 COLOR_TEXT_SECONDARY =
    IM_COL32(0x9A, 0x9E, 0xA3, 0xFF);                                    // #9A9EA3 - Secondary text
constexpr ImU32 COLOR_TEXT_DISABLED = IM_COL32(0x5A, 0x5E, 0x63, 0xFF);  // #5A5E63 - Disabled text

// Accent Colors
constexpr ImU32 COLOR_ACCENT_PRIMARY =
    IM_COL32(0x4A, 0x90, 0xFF, 0xFF);  // #4A90FF - Primary accent (blue)
constexpr ImU32 COLOR_ACCENT_HOVER = IM_COL32(0x5A, 0xA0, 0xFF, 0xFF);   // #5AA0FF - Hover accent
constexpr ImU32 COLOR_ACCENT_ACTIVE = IM_COL32(0x3A, 0x80, 0xEF, 0xFF);  // #3A80EF - Active accent

// Trading Colors - Bullish (Green/Mint)
constexpr ImU32 COLOR_BULLISH = IM_COL32(0x00, 0xE6, 0x76, 0xFF);     // #00E676 - Neon Mint (Buy)
constexpr ImU32 COLOR_BULLISH_BG = IM_COL32(0x00, 0xE6, 0x76, 0x1A);  // #00E676 @ 10% alpha
constexpr ImU32 COLOR_BULLISH_STRONG = IM_COL32(0x00, 0xFF, 0x88, 0xFF);  // #00FF88 - Strong buy

// Trading Colors - Bearish (Red/Crimson)
constexpr ImU32 COLOR_BEARISH = IM_COL32(0xFF, 0x45, 0x45, 0xFF);     // #FF4545 - Crimson (Sell)
constexpr ImU32 COLOR_BEARISH_BG = IM_COL32(0xFF, 0x45, 0x45, 0x1A);  // #FF4545 @ 10% alpha
constexpr ImU32 COLOR_BEARISH_STRONG = IM_COL32(0xFF, 0x00, 0x00, 0xFF);  // #FF0000 - Strong sell

// Neutral Colors
constexpr ImU32 COLOR_NEUTRAL = IM_COL32(0x9A, 0x9E, 0xA3, 0xFF);  // #9A9EA3 - Neutral
constexpr ImU32 COLOR_NEUTRAL_BG =
    IM_COL32(0x2A, 0x2E, 0x33, 0xFF);  // #2A2E33 - Neutral background

// Heatmap Colors (Liquidity Gradient)
constexpr ImU32 COLOR_HEATMAP_VOID = IM_COL32(0x02, 0x02, 0x0F, 0xFF);  // Deep Blue/Black (empty)
constexpr ImU32 COLOR_HEATMAP_LOW = IM_COL32(0x00, 0x4D, 0xFF, 0xFF);   // Blue (low liquidity)
constexpr ImU32 COLOR_HEATMAP_MID = IM_COL32(0x00, 0xE6, 0x76, 0xFF);   // Green (medium liquidity)
constexpr ImU32 COLOR_HEATMAP_HIGH = IM_COL32(0xFF, 0xA5, 0x00, 0xFF);  // Orange (high liquidity)
constexpr ImU32 COLOR_HEATMAP_EXTREME = IM_COL32(0xFF, 0x0D, 0x00, 0xFF);  // Fire Red (extreme)

// Volume Profile Colors
constexpr ImU32 COLOR_VOLUME_BID = IM_COL32(0x00, 0xE6, 0x76, 0x80);  // Green @ 50% alpha
constexpr ImU32 COLOR_VOLUME_ASK = IM_COL32(0xFF, 0x45, 0x45, 0x80);  // Red @ 50% alpha
constexpr ImU32 COLOR_VOLUME_POC = IM_COL32(0x4A, 0x90, 0xFF, 0xFF);  // Blue (Point of Control)
constexpr ImU32 COLOR_VALUE_AREA = IM_COL32(0x4A, 0x90, 0xFF, 0x40);  // Blue @ 25% alpha

// TPO Colors
constexpr ImU32 COLOR_TPO_TEXT = IM_COL32(0x9A, 0x9E, 0xA3, 0xFF);     // Gray TPO letters
constexpr ImU32 COLOR_TPO_SINGLE = IM_COL32(0x4A, 0x90, 0xFF, 0xFF);   // Blue single print
constexpr ImU32 COLOR_TPO_INITIAL = IM_COL32(0xFF, 0xA5, 0x00, 0xFF);  // Orange initial balance

// ============================================================================
// MMT Style Configuration - Borderless, Zero-Rounding
// ============================================================================

inline void ApplyMMTStyle(ImGuiStyle* style = nullptr) {
  if (!style) {
    style = &ImGui::GetStyle();
  }

  // --- Borderless Aesthetic ---
  style->WindowBorderSize = 0.0f;
  style->ChildBorderSize = 0.0f;
  style->FrameBorderSize = 0.0f;
  style->PopupBorderSize = 0.0f;
  style->TabBorderSize = 0.0f;

  // --- Zero Rounding ---
  style->WindowRounding = 0.0f;
  style->ChildRounding = 0.0f;
  style->FrameRounding = 0.0f;
  style->PopupRounding = 0.0f;
  style->TabRounding = 0.0f;
  style->ScrollbarRounding = 0.0f;
  style->GrabRounding = 0.0f;
  style->LogSliderDeadzone = 0.0f;

  // --- Spacing & Padding ---
  style->WindowPadding = ImVec2(8.0f, 8.0f);
  style->FramePadding = ImVec2(4.0f, 4.0f);
  style->CellPadding = ImVec2(4.0f, 2.0f);
  style->ItemSpacing = ImVec2(8.0f, 4.0f);
  style->ItemInnerSpacing = ImVec2(4.0f, 4.0f);
  style->IndentSpacing = 16.0f;
  style->ScrollbarSize = 12.0f;
  style->GrabMinSize = 8.0f;

  // --- Apply Colors ---
  ImVec4* colors = style->Colors;

  // Background
  colors[ImGuiCol_WindowBg] = ImVec4(0.043f, 0.055f, 0.067f, 1.0f);
  colors[ImGuiCol_ChildBg] = ImColor(COLOR_CHILD_BG);
  colors[ImGuiCol_PopupBg] = ImColor(COLOR_POPUP_BG);
  colors[ImGuiCol_Border] = ImColor(COLOR_BORDER);
  colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

  // Text
  colors[ImGuiCol_Text] = ImColor(COLOR_TEXT_PRIMARY);
  colors[ImGuiCol_TextDisabled] = ImColor(COLOR_TEXT_DISABLED);
  colors[ImGuiCol_TextSelectedBg] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_TextSelectedBg].w = 0.35f;

  // Frames & Buttons
  colors[ImGuiCol_FrameBg] = ImColor(COLOR_CHILD_BG);
  colors[ImGuiCol_FrameBgHovered] = ImColor(COLOR_BORDER);
  colors[ImGuiCol_FrameBgActive] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_FrameBgActive].w = 0.3f;

  colors[ImGuiCol_Button] = ImColor(COLOR_CHILD_BG);
  colors[ImGuiCol_ButtonHovered] = ImColor(COLOR_ACCENT_HOVER);
  colors[ImGuiCol_ButtonHovered].w = 0.2f;
  colors[ImGuiCol_ButtonActive] = ImColor(COLOR_ACCENT_ACTIVE);
  colors[ImGuiCol_ButtonActive].w = 0.4f;

  // Headers & Tabs
  colors[ImGuiCol_Header] = ImColor(COLOR_CHILD_BG);
  colors[ImGuiCol_HeaderHovered] = ImColor(COLOR_ACCENT_HOVER);
  colors[ImGuiCol_HeaderHovered].w = 0.2f;
  colors[ImGuiCol_HeaderActive] = ImColor(COLOR_ACCENT_ACTIVE);
  colors[ImGuiCol_HeaderActive].w = 0.3f;

  colors[ImGuiCol_Tab] = ImColor(COLOR_CHILD_BG);
  colors[ImGuiCol_TabHovered] = ImColor(COLOR_ACCENT_HOVER);
  colors[ImGuiCol_TabActive] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_TabUnfocused] = ImColor(COLOR_WINDOW_BG);
  colors[ImGuiCol_TabUnfocusedActive] = ImColor(COLOR_CHILD_BG);

  // Title & Menu
  colors[ImGuiCol_TitleBg] = ImColor(COLOR_WINDOW_BG);
  colors[ImGuiCol_TitleBgActive] = ImColor(COLOR_CHILD_BG);
  colors[ImGuiCol_TitleBgCollapsed] = ImColor(COLOR_WINDOW_BG);
  colors[ImGuiCol_MenuBarBg] = ImColor(COLOR_WINDOW_BG);

  // Scrollbar
  colors[ImGuiCol_ScrollbarBg] = ImColor(COLOR_WINDOW_BG);
  colors[ImGuiCol_ScrollbarGrab] = ImColor(COLOR_BORDER);
  colors[ImGuiCol_ScrollbarGrabHovered] = ImColor(COLOR_ACCENT_HOVER);
  colors[ImGuiCol_ScrollbarGrabActive] = ImColor(COLOR_ACCENT_ACTIVE);

  // Slider & Grab
  colors[ImGuiCol_SliderGrab] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_SliderGrabActive] = ImColor(COLOR_ACCENT_ACTIVE);
  // Note: ImGui has no separate Grab color - SliderGrab* above covers this

  // Checkmark
  colors[ImGuiCol_CheckMark] = ImColor(COLOR_ACCENT_PRIMARY);

  // Separator
  colors[ImGuiCol_Separator] = ImColor(COLOR_BORDER);
  colors[ImGuiCol_SeparatorHovered] = ImColor(COLOR_ACCENT_HOVER);
  colors[ImGuiCol_SeparatorActive] = ImColor(COLOR_ACCENT_ACTIVE);

  // Resize Grip
  colors[ImGuiCol_ResizeGrip] = ImColor(COLOR_BORDER);
  colors[ImGuiCol_ResizeGripHovered] = ImColor(COLOR_ACCENT_HOVER);
  colors[ImGuiCol_ResizeGripActive] = ImColor(COLOR_ACCENT_ACTIVE);

  // Docking
  colors[ImGuiCol_DockingPreview] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_DockingPreview].w = 0.4f;
  colors[ImGuiCol_DockingEmptyBg] = ImColor(COLOR_WINDOW_BG);

  // Tables
  colors[ImGuiCol_TableHeaderBg] = ImColor(COLOR_CHILD_BG);
  colors[ImGuiCol_TableBorderStrong] = ImColor(COLOR_BORDER);
  colors[ImGuiCol_TableBorderLight] = ImColor(COLOR_BORDER);
  colors[ImGuiCol_TableBorderLight].w = 0.5f;
  colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
  colors[ImGuiCol_TableRowBgAlt] = ImColor(COLOR_CHILD_BG);
  colors[ImGuiCol_TableRowBgAlt].w = 0.3f;

  // Plot
  colors[ImGuiCol_PlotLines] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_PlotLinesHovered] = ImColor(COLOR_ACCENT_HOVER);
  colors[ImGuiCol_PlotHistogram] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_PlotHistogramHovered] = ImColor(COLOR_ACCENT_HOVER);

  // Drag & Drop
  colors[ImGuiCol_DragDropTarget] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_DragDropTarget].w = 0.9f;

  // Nav Highlight
  colors[ImGuiCol_NavHighlight] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_NavWindowingHighlight] = ImColor(COLOR_ACCENT_PRIMARY);
  colors[ImGuiCol_NavWindowingHighlight].w = 0.7f;
  colors[ImGuiCol_NavWindowingDimBg] = ImColor(COLOR_WINDOW_BG);
  colors[ImGuiCol_NavWindowingDimBg].w = 0.2f;

  // Modal
  colors[ImGuiCol_ModalWindowDimBg] = ImColor(COLOR_WINDOW_BG);
  colors[ImGuiCol_ModalWindowDimBg].w = 0.7f;
}

// ============================================================================
// MMT Font Configuration
// ============================================================================

struct FontConfig {
  const char* mainFontPath = "fonts/JetBrainsMono-Regular.ttf";
  const char* iconFontPath = "fonts/fa-solid-6.ttf";
  float mainFontSize = 13.0f;
  float iconFontSize = 13.0f;
  int oversampleH = 4;
  int oversampleV = 4;
  bool pixelSnapH = false;
  bool mergeIconFont = true;
  ImWchar iconFontRange[3] = {0xe005, 0xf8ff, 0};  // FontAwesome 6 range
};

inline ImFont* LoadMMTFont(ImGuiIO& io, const FontConfig& config) {
  ImFontConfig fontConfig;
  fontConfig.OversampleH = config.oversampleH;
  fontConfig.OversampleV = config.oversampleV;
  fontConfig.PixelSnapH = config.pixelSnapH;

  // Load main font
  ImFont* mainFont =
      io.Fonts->AddFontFromFileTTF(config.mainFontPath, config.mainFontSize, &fontConfig);

  if (config.mergeIconFont && mainFont) {
    // Merge icon font into main font
    ImFontConfig iconConfig;
    iconConfig.MergeMode = true;
    iconConfig.OversampleH = config.oversampleH;
    iconConfig.OversampleV = config.oversampleV;
    iconConfig.PixelSnapH = config.pixelSnapH;
    iconConfig.GlyphMinAdvanceX = config.iconFontSize;  // Monospace icons

    io.Fonts->AddFontFromFileTTF(config.iconFontPath, config.iconFontSize, &iconConfig,
                                 config.iconFontRange);
  }

  return mainFont;
}

// ============================================================================
// MMT Docking Layout
// ============================================================================

inline void SetupMMTDocking() {
  ImGuiID dockspace_id = ImGui::GetID("MMT_DockSpace");

  // Only setup once
  static bool initialized = false;
  if (initialized) return;

  ImGui::DockBuilderRemoveNode(dockspace_id);
  ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

  // Split into main regions
  ImGuiID left_id, center_id, right_id;
  ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.15f, &left_id, &center_id);
  ImGui::DockBuilderSplitNode(center_id, ImGuiDir_Right, 0.25f, &right_id, &center_id);

  // Split right panel into top (DOM) and bottom (Tape)
  ImGuiID right_top_id, right_bottom_id;
  ImGui::DockBuilderSplitNode(right_id, ImGuiDir_Up, 0.5f, &right_top_id, &right_bottom_id);

  // Dock windows to regions
  ImGui::DockBuilderDockWindow("Watchlist", left_id);
  ImGui::DockBuilderDockWindow("Chart", center_id);
  ImGui::DockBuilderDockWindow("DOM", right_top_id);
  ImGui::DockBuilderDockWindow("Tape", right_bottom_id);
  ImGui::DockBuilderDockWindow("Footprint", right_top_id);
  ImGui::DockBuilderDockWindow("TPO", right_bottom_id);

  ImGui::DockBuilderFinish(dockspace_id);
  initialized = true;
}

}  // namespace MMT
}  // namespace UI
}  // namespace BTQuant
