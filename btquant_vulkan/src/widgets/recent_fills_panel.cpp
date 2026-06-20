#include "recent_fills_panel.hpp"

#include "../data/trade_journal.hpp"
#include "log_panel.hpp"

#include <cstdio>
#include <ctime>
#include <imgui.h>

namespace btquant::ui {

RecentFillsPanel::RecentFillsPanel() = default;
RecentFillsPanel::~RecentFillsPanel() = default;

void RecentFillsPanel::addFill(const ::btquant::JournalFill& jf) {
    m_fills.push_front(jf);
    while (m_fills.size() > kMaxFills) m_fills.pop_back();
}

void RecentFillsPanel::clear() {
    m_fills.clear();
}

void RecentFillsPanel::render() {
    if (!m_open) return;

    ImGui::SetNextWindowSize(ImVec2(720, 360), ImGuiCond_Appearing);
    if (!ImGui::Begin("Recent Fills", &m_open,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    // Header row + controls.
    ImGui::Text("Last %zu fills (newest first, capped at %zu):",
                m_fills.size(), kMaxFills);
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) {
        clear();
        BTQ_LOG_INFO("RecentFills: cleared (visual only — "
                     "journal still on disk)");
    }
    ImGui::Separator();

    if (m_fills.empty()) {
        ImGui::TextDisabled("(no fills yet — submit an order to populate)");
        ImGui::End();
        return;
    }

    // Table: Time | Symbol | Side | Qty | Price | Realized | Tag.
    if (ImGui::BeginTable("RecentFillsTable", 7,
                          ImGuiTableFlags_BordersInnerH |
                          ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY,
                          ImVec2(0, 0))) {
        ImGui::TableSetupColumn("Time",     ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Symbol",   ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Side",     ImGuiTableColumnFlags_WidthFixed, 50.0f);
        ImGui::TableSetupColumn("Qty",      ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Price",    ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Realized", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Tag",      ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        for (const auto& jf : m_fills) {
            ImGui::TableNextRow();

            // Time: render as HH:MM:SS from timestamp_us.
            ImGui::TableSetColumnIndex(0);
            std::time_t tt = static_cast<std::time_t>(jf.timestamp_us / 1000000);
            std::tm tm{};
            ::localtime_r(&tt, &tm);
            char tbuf[16];
            std::snprintf(tbuf, sizeof(tbuf), "%02d:%02d:%02d",
                          tm.tm_hour, tm.tm_min, tm.tm_sec);
            ImGui::Text("%s", tbuf);

            // Symbol.
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%s", jf.symbol.c_str());

            // Side — colour-coded green/red.
            ImGui::TableSetColumnIndex(2);
            if (jf.isLong)
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 220, 120, 255));
            else
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
            ImGui::Text("%s", jf.isLong ? "BUY" : "SELL");
            ImGui::PopStyleColor();

            // Qty.
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.4f", jf.qty);

            // Price.
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("$%.2f", jf.price);

            // Realized — colour-coded green/red.
            ImGui::TableSetColumnIndex(5);
            if (jf.realizedDelta > 0)
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 220, 120, 255));
            else if (jf.realizedDelta < 0)
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 80, 80, 255));
            ImGui::Text("%s$%.2f",
                        jf.realizedDelta >= 0 ? "+" : "",
                        jf.realizedDelta);
            if (jf.realizedDelta != 0) ImGui::PopStyleColor();

            // Tag — "(untagged)" placeholder so the column isn't blank
            // for fills where the trader didn't set a strategy label.
            ImGui::TableSetColumnIndex(6);
            if (jf.tag.empty())
                ImGui::TextDisabled("(untagged)");
            else
                ImGui::Text("%s", jf.tag.c_str());
        }
        ImGui::EndTable();
    }

    ImGui::TextDisabled("(visual ring buffer — survives this session only. "
                        "Persistent history lives on disk via the journal.)");
    ImGui::End();
}

}  // namespace btquant::ui
