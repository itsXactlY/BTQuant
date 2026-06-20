#include "symbol_picker.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <imgui.h>

namespace btquant::ui {

namespace {
std::string toLowerCopy(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}
} // namespace

void SymbolPicker::refresh() {
    m_filtered.clear();
    std::string f = toLowerCopy(m_filter);
    for (const auto& s : m_candidates) {
        if (f.empty() || toLowerCopy(s).find(f) != std::string::npos) {
            m_filtered.push_back(s);
        }
    }
    if (m_selected >= static_cast<int>(m_filtered.size())) m_selected = 0;
}

void SymbolPicker::render() {
    if (!m_open) return;

    ImGui::OpenPopup("Symbol Picker");
    ImGui::SetNextWindowSize(ImVec2(420, 360), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Symbol Picker", &m_open,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        return;
    }

    // Focus the input on first appear.
    static bool firstAppear = true;
    if (firstAppear) {
        ImGui::SetKeyboardFocusHere();
        firstAppear = false;
    }
    if (ImGui::InputText("##filter", m_input, sizeof(m_input),
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
        // Enter selects the currently highlighted row.
        if (m_select && !m_filtered.empty() &&
            m_selected >= 0 && m_selected < static_cast<int>(m_filtered.size())) {
            m_select(m_filtered[m_selected]);
            m_open = false;
        }
    }
    if (std::strcmp(m_input, m_filter.c_str()) != 0) {
        m_filter = m_input;
        refresh();
    }

    ImGui::Separator();

    if (ImGui::BeginListBox("##syms", ImVec2(-FLT_MIN, -FLT_MIN))) {
        for (size_t i = 0; i < m_filtered.size(); ++i) {
            const bool isSel = (static_cast<int>(i) == m_selected);
            if (ImGui::Selectable(m_filtered[i].c_str(), isSel)) {
                if (m_select) m_select(m_filtered[i]);
                m_open = false;
            }
            if (isSel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
    }

    ImGui::Separator();
    ImGui::TextDisabled("%zu / %zu symbols — type to filter, "
                        "Enter or click to select",
                        m_filtered.size(), m_candidates.size());

    if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        m_open = false;
    }

    // Arrow-key navigation in the list.
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
        ++m_selected;
        if (m_selected >= static_cast<int>(m_filtered.size())) m_selected = 0;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
        if (m_selected <= 0) m_selected = static_cast<int>(m_filtered.size()) - 1;
        else                 --m_selected;
    }

    ImGui::EndPopup();

    if (!m_open) {
        firstAppear = true;
        std::memset(m_input, 0, sizeof(m_input));
        m_filter.clear();
        m_selected = 0;
    }
}

} // namespace btquant::ui
