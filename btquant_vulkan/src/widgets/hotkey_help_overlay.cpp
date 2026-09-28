#include "hotkey_help_overlay.hpp"

#include "../util/hotkey_config.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <imgui.h>

namespace {

// Case-insensitive substring search. Cheap O(N) on short action
// names; the filter string itself is bounded by m_filter (64).
bool containsCi(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    if (needle.size() > haystack.size()) return false;
    auto it = std::search(haystack.begin(), haystack.end(),
                          needle.begin(), needle.end(),
                          [](unsigned char a, unsigned char b) {
                              return std::tolower(a) == std::tolower(b);
                          });
    return it != haystack.end();
}

}  // namespace

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
    // Sprint #66: filter box. 30+ rows in defaults; lets the trader
    // jump to a specific binding. Substring match, case-insensitive.
    ImGui::PushItemWidth(220.0f);
    ImGui::InputTextWithHint("##hkfilter", "Filter by action...",
                             m_filter, sizeof(m_filter));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) m_filter[0] = '\0';
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
        int visible = 0;
        const std::string filterStr(m_filter);
        for (const auto& r : sorted) {
            std::string name = util::HotkeyMap::actionName(r.first);
            if (!containsCi(name, filterStr)) continue;
            ++visible;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", name.c_str());
            ImGui::TableSetColumnIndex(1);
            if (r.second.glfwKey < 0) {
                ImGui::TextDisabled("(unbound)");
            } else {
                ImGui::Text("%s", r.second.label().c_str());
            }
        }
        ImGui::EndTable();
        // Footer shows visible/total when filter is active.
        if (!filterStr.empty()) {
            ImGui::TextDisabled("(%d / %zu shown)", visible, sorted.size());
        }
    }
    ImGui::EndPopup();

    if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        m_open = false;
    }
}

}  // namespace btquant::ui
