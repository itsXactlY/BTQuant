#include "hotkey_editor.hpp"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <cstdio>

namespace btquant::widgets {

namespace {

// One row of the table. We keep a parallel index alongside the binding
// so the capture flow can address rows by position.
struct Row {
    ::btquant::util::HotkeyAction action;
    ::btquant::util::HotkeyBinding binding;
};

} // namespace

void HotkeyEditor::render() {
    if (!m_open || !m_map) return;
    ImGui::SetNextWindowSize(ImVec2(560, 460), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Hotkey Editor", &m_open,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }
    if (isCapturing()) {
        ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                           "Press any key to bind to \"%s\"…",
                           ::btquant::util::HotkeyMap::actionName(
                               static_cast<::btquant::util::HotkeyAction>(
                                   m_capturing)).c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Cancel")) cancelCapture();
    } else {
        ImGui::TextDisabled("Click Remap, then press a key. Esc cancels.");
    }
    ImGui::Separator();

    // Table: Action | Current Binding | Remap button.
    if (ImGui::BeginTable("hotkey_table", 3,
                          ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Binding", ImGuiTableColumnFlags_WidthFixed, 140.0f);
        ImGui::TableSetupColumn("##",      ImGuiTableColumnFlags_WidthFixed, 110.0f);
        ImGui::TableHeadersRow();

        auto rows = m_map->enumerate();
        for (size_t i = 0; i < rows.size(); ++i) {
            ImGui::TableNextRow();
            ImGui::PushID(static_cast<int>(i));

            // Col 1 — action name.
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(::btquant::util::HotkeyMap::actionName(
                                       rows[i].first).c_str());

            // Col 2 — current binding (highlighted if capturing this row).
            ImGui::TableSetColumnIndex(1);
            if (static_cast<int>(i) == m_capturing) {
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f),
                                   "(waiting...)");
            } else {
                ImGui::TextUnformatted(rows[i].second.label().c_str());
            }

            // Col 3 — Remap + Reset buttons.
            ImGui::TableSetColumnIndex(2);
            char buf[16];
            std::snprintf(buf, sizeof(buf), "Remap##%zu", i);
            if (ImGui::SmallButton(buf)) {
                setCapturing(static_cast<int>(i));
            }
            ImGui::SameLine();
            std::snprintf(buf, sizeof(buf), "Reset##%zu", i);
            if (ImGui::SmallButton(buf)) {
                applyDefault(static_cast<int>(i));
            }

            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    ImGui::Separator();
    if (ImGui::Button("Reset all to defaults")) {
        *m_map = ::btquant::util::HotkeyMap::defaults();
        m_dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Button(m_dirty ? "Save* (auto-applied)" : "Saved")) {
        // No-op — WindowManager persists on shutdown. We just clear dirty.
        m_dirty = false;
    }

    ImGui::End();
}

void HotkeyEditor::applyDefault(int actionIndex) {
    if (!m_map) return;
    auto defaults = ::btquant::util::HotkeyMap::defaults();
    auto rows = defaults.enumerate();
    if (actionIndex < 0 || actionIndex >= static_cast<int>(rows.size())) return;
    m_map->set(rows[actionIndex].first, rows[actionIndex].second);
    m_dirty = true;
}

void HotkeyEditor::beginCapture(int actionIndex) {
    if (!m_map) return;
    auto rows = m_map->enumerate();
    if (actionIndex < 0 || actionIndex >= static_cast<int>(rows.size())) return;
    m_capturing = actionIndex;
}

void HotkeyEditor::injectCapture(int glfwKey,
                                 bool ctrlDown, bool altDown, bool shiftDown) {
    if (!isCapturing() || !m_map) return;
    auto rows = m_map->enumerate();
    if (m_capturing < 0 || m_capturing >= static_cast<int>(rows.size())) {
        cancelCapture();
        return;
    }
    // Esc (or glfwKey == -1 sentinel) cancels.
    if (glfwKey == -1 || glfwKey == GLFW_KEY_ESCAPE) {
        cancelCapture();
        return;
    }
    // Don't bind pure modifier keys as the trigger.
    if (glfwKey == GLFW_KEY_LEFT_CONTROL || glfwKey == GLFW_KEY_RIGHT_CONTROL ||
        glfwKey == GLFW_KEY_LEFT_SHIFT   || glfwKey == GLFW_KEY_RIGHT_SHIFT   ||
        glfwKey == GLFW_KEY_LEFT_ALT     || glfwKey == GLFW_KEY_RIGHT_ALT     ||
        glfwKey == GLFW_KEY_LEFT_SUPER   || glfwKey == GLFW_KEY_RIGHT_SUPER) {
        return;
    }
    ::btquant::util::HotkeyBinding b;
    b.glfwKey = glfwKey;
    b.ctrl    = ctrlDown;
    b.alt     = altDown;
    b.shift   = shiftDown;
    m_map->set(rows[m_capturing].first, b);
    m_dirty = true;
    cancelCapture();
}

} // namespace btquant::widgets
