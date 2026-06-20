#include "theme_editor.hpp"

#include <cmath>
#include <cstdio>
#include <imgui.h>

namespace btquant::ui {

const char* ThemeEditor::colorName(int i) {
    switch (i) {
        case ImGuiCol_Text:                  return "Text";
        case ImGuiCol_TextDisabled:          return "Text (disabled)";
        case ImGuiCol_WindowBg:              return "Window background";
        case ImGuiCol_ChildBg:               return "Child background";
        case ImGuiCol_PopupBg:               return "Popup background";
        case ImGuiCol_Border:                return "Border";
        case ImGuiCol_BorderShadow:          return "Border shadow";
        case ImGuiCol_FrameBg:               return "Frame background";
        case ImGuiCol_FrameBgHovered:        return "Frame (hovered)";
        case ImGuiCol_FrameBgActive:         return "Frame (active)";
        case ImGuiCol_TitleBg:               return "Title background";
        case ImGuiCol_TitleBgActive:         return "Title (active)";
        case ImGuiCol_MenuBarBg:             return "Menu bar";
        case ImGuiCol_ScrollbarBg:           return "Scrollbar background";
        case ImGuiCol_ScrollbarGrab:         return "Scrollbar grab";
        case ImGuiCol_ScrollbarGrabHovered:  return "Scrollbar (hovered)";
        case ImGuiCol_ScrollbarGrabActive:   return "Scrollbar (active)";
        case ImGuiCol_CheckMark:             return "Checkmark";
        case ImGuiCol_SliderGrab:            return "Slider grab";
        case ImGuiCol_SliderGrabActive:      return "Slider (active)";
        case ImGuiCol_Button:                return "Button";
        case ImGuiCol_ButtonHovered:         return "Button (hovered)";
        case ImGuiCol_ButtonActive:          return "Button (active)";
        case ImGuiCol_Header:                return "Header";
        case ImGuiCol_HeaderHovered:         return "Header (hovered)";
        case ImGuiCol_HeaderActive:          return "Header (active)";
        case ImGuiCol_Separator:             return "Separator";
        case ImGuiCol_SeparatorHovered:      return "Separator (hovered)";
        case ImGuiCol_SeparatorActive:       return "Separator (active)";
        case ImGuiCol_ResizeGrip:            return "Resize grip";
        case ImGuiCol_ResizeGripHovered:     return "Resize (hovered)";
        case ImGuiCol_ResizeGripActive:      return "Resize (active)";
        case ImGuiCol_Tab:                   return "Tab";
        case ImGuiCol_TabHovered:            return "Tab (hovered)";
        case ImGuiCol_TabActive:             return "Tab (active)";
        case ImGuiCol_DockingPreview:        return "Docking preview";
        case ImGuiCol_DockingEmptyBg:        return "Docking empty bg";
        case ImGuiCol_PlotLines:             return "Plot lines";
        case ImGuiCol_PlotLinesHovered:      return "Plot lines (hovered)";
        case ImGuiCol_PlotHistogram:         return "Plot histogram";
        case ImGuiCol_PlotHistogramHovered:  return "Plot histogram (hovered)";
        case ImGuiCol_TableHeaderBg:         return "Table header";
        case ImGuiCol_TableBorderStrong:     return "Table border (strong)";
        case ImGuiCol_TableBorderLight:      return "Table border (light)";
        case ImGuiCol_TableRowBg:            return "Table row bg";
        case ImGuiCol_TableRowBgAlt:         return "Table row bg (alt)";
        case ImGuiCol_TextLink:              return "Text link";
        case ImGuiCol_TextSelectedBg:        return "Text selected";
        case ImGuiCol_TreeLines:             return "Tree lines";
        case ImGuiCol_DragDropTarget:        return "Drag/drop target";
        case ImGuiCol_DragDropTargetBg:      return "Drag/drop target (bg)";
        case ImGuiCol_UnsavedMarker:         return "Unsaved marker";
        case ImGuiCol_NavCursor:             return "Nav cursor";
        case ImGuiCol_NavWindowingHighlight: return "Nav windowing highlight";
        case ImGuiCol_NavWindowingDimBg:     return "Nav windowing dim";
        case ImGuiCol_ModalWindowDimBg:      return "Modal dim";
        default: {
            static thread_local char buf[24];
            std::snprintf(buf, sizeof(buf), "Color[%d]", i);
            return buf;
        }
    }
}

void ThemeEditor::render() {
    if (!m_open) return;

    ImGui::OpenPopup("Theme Editor");
    ImGui::SetNextWindowSize(ImVec2(540, 600), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Theme Editor", &m_open)) {
        return;
    }

    ImGuiStyle& st = ImGui::GetStyle();
    ImGui::Text("Edit colors and core style. Changes apply live to the running UI.");
    ImGui::Separator();

    // Core knobs.
    ImGui::SliderFloat("Window padding", &st.WindowPadding.x, 0.0f, 24.0f, "%.1f");
    ImGui::SliderFloat("Frame padding",  &st.FramePadding.x,  0.0f, 16.0f, "%.1f");
    ImGui::SliderFloat("Rounding",       &st.FrameRounding,   0.0f, 12.0f, "%.1f");
    ImGui::SliderFloat("Alpha",          &st.Alpha,           0.2f, 1.0f, "%.2f");

    ImGui::Separator();

    if (ImGui::BeginTable("colors", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("Color",  ImGuiTableColumnFlags_WidthFixed, 200.0f);
        ImGui::TableSetupColumn("Picker", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (int i = 0; i < kColorCount; ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(colorName(i));
            ImGui::TableNextColumn();
            ImGui::PushID(i);
            ImGui::ColorEdit4("##c", &st.Colors[i].x,
                              ImGuiColorEditFlags_AlphaBar |
                              ImGuiColorEditFlags_NoInputs |
                              ImGuiColorEditFlags_NoLabel);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    ImGui::Separator();
    ImGui::TextDisabled("Theme snapshot is persisted in Settings on exit "
                        "(if you checked \"Auto-save on close\" in Settings).");

    if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        m_open = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset to Dark")) ImGui::StyleColorsDark(&st);
    ImGui::SameLine();
    if (ImGui::Button("Reset to Light")) ImGui::StyleColorsLight(&st);

    ImGui::EndPopup();
}

void ThemeEditor::applySnapshot(ImGuiStyle& dst, const Snapshot& s) {
    for (int i = 0; i < kColorCount; ++i) {
        dst.Colors[i] = ImVec4(s.colors[i][0], s.colors[i][1],
                               s.colors[i][2], s.colors[i][3]);
    }
    dst.WindowPadding = ImVec2(s.windowPadding, s.windowPadding);
    dst.FramePadding  = ImVec2(s.framePadding,  s.framePadding);
    dst.FrameRounding = s.rounding;
    dst.Alpha         = s.alpha;
}

ThemeEditor::Snapshot ThemeEditor::capture(const ImGuiStyle& src) {
    Snapshot s;
    for (int i = 0; i < kColorCount; ++i) {
        s.colors[i] = {src.Colors[i].x, src.Colors[i].y,
                       src.Colors[i].z, src.Colors[i].w};
    }
    s.windowPadding = src.WindowPadding.x;
    s.framePadding  = src.FramePadding.x;
    s.rounding      = src.FrameRounding;
    s.alpha         = src.Alpha;
    s.dark          = true;  // We don't store the dark/light toggle in the
                             // snapshot — that's a separate Settings flag.
    return s;
}

bool ThemeEditor::equals(const Snapshot& a, const Snapshot& b, float tol) {
    for (int i = 0; i < kColorCount; ++i) {
        for (int k = 0; k < 4; ++k) {
            if (std::fabs(a.colors[i][k] - b.colors[i][k]) > tol) return false;
        }
    }
    if (std::fabs(a.windowPadding - b.windowPadding) > tol) return false;
    if (std::fabs(a.framePadding  - b.framePadding)  > tol) return false;
    if (std::fabs(a.rounding      - b.rounding)      > tol) return false;
    if (std::fabs(a.alpha         - b.alpha)         > tol) return false;
    return true;
}

} // namespace btquant::ui
