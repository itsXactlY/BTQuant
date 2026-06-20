#include "hotkey_help_overlay.hpp"

#include "../util/hotkey_config.hpp"

#include <algorithm>
#include <utility>
#include <vector>
#include <imgui.h>

namespace btquant::ui {

void HotkeyHelpOverlay::render() {
    if (!m_open) return;

    ImGui::OpenPopup("Hotkey Help");
    ImGui::SetNextWindowSize(ImVec2(520, 460), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Hotkey Help", &m_open)) {
        return;
    }

    if (!m_map) {
        ImGui::Text("Hotkey map not wired.");
        ImGui::TextDisabled("(WindowManager should call setHotkeyMap() "
                            "once on construction.)");
        ImGui::EndPopup();
        return;
    }

    ImGui::TextDisabled("Press Esc to close. Click Remap in Hotkey Editor "
                        "to change bindings.");
    ImGui::Separator();
    if (ImGui::BeginTable("hotkey_help_table", 2,
                          ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_BordersInnerH |
                          ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("Action",
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Binding",
                                ImGuiTableColumnFlags_WidthFixed, 160.0f);
        ImGui::TableHeadersRow();
        auto rows = m_map->enumerate();
        // Build a sorted copy by action name. enumerate() returns
        // a fresh vector so we could sort in place, but copy first
        // keeps the data flow obvious.
        std::vector<std::pair<util::HotkeyAction,
                              util::HotkeyBinding>> sorted;
        sorted.reserve(rows.size());
        for (const auto& r : rows) sorted.push_back(r);
        std::sort(sorted.begin(), sorted.end(),
                  [](const std::pair<util::HotkeyAction,
                                     util::HotkeyBinding>& a,
                     const std::pair<util::HotkeyAction,
                                     util::HotkeyBinding>& b) {
                      return util::HotkeyMap::actionName(a.first) <
                             util::HotkeyMap::actionName(b.first);
                  });
        for (const auto& r : sorted) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s",
                        util::HotkeyMap::actionName(r.first).c_str());
            ImGui::TableSetColumnIndex(1);
            if (r.second.glfwKey < 0) {
                ImGui::TextDisabled("(unbound)");
            } else {
                ImGui::Text("%s", r.second.label().c_str());
            }
        }
        ImGui::EndTable();
    }
    ImGui::EndPopup();

    // Esc closes (in addition to the X button).
    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        m_open = false;
    }
}

}  // namespace btquant::ui
