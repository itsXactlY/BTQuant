#include "journal_stats_panel.hpp"

#include <cmath>
#include <cstdio>

#include "../data/trade_journal.hpp"

#include "imgui.h"

namespace btquant::ui {

// Color rules for the breakdown rows:
//   green   — winner  (realized > 0)
//   red     — loser   (realized < 0)
//   dim     — exactly zero
// Using ImGui::PushStyleColor per-row keeps the formatting localized
// and avoids a full-window tinting for one cell.
static void colorizeRow(double realized) {
    if (realized > 1e-9) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
    } else if (realized < -1e-9) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
    }
}

void JournalStatsPanel::render() {
    if (!showWindow) return;
    if (!m_journal) {
        ImGui::Begin("Journal Stats", &showWindow);
        ImGui::TextDisabled("(no TradeJournal bound)");
        ImGui::End();
        return;
    }

    ImGui::Begin("Journal Stats", &showWindow);

    // ---- Total all-time P&L (Sprint #67) ----
    //
    // The headline number. Pulled directly from
    // TradeJournal::totalRealized() — single O(N) scan but the
    // symbol/tag tables also scan, so it's free to call here.
    double total = m_journal->totalRealized();
    ImGui::Text("All-time P&L:");
    ImGui::SameLine();
    colorizeRow(total);
    char totalBuf[64];
    std::snprintf(totalBuf, sizeof(totalBuf), "%+.2f", total);
    ImGui::TextUnformatted(totalBuf);
    ImGui::PopStyleColor();

    // Counts for context — cheap (line count) and helps the trader
    // distinguish "empty journal" from "small journal".
    size_t nFills = m_journal->count();
    ImGui::TextDisabled("(%zu fill%s on disk)",
                        nFills, nFills == 1 ? "" : "s");
    ImGui::Separator();

    // ---- By-symbol table (Sprint #72) ----
    //
    // TradeJournal::realizedBySymbol() returns abs-DESC-sorted pairs,
    // so we just take the first `m_maxRows` and render them. We
    // compute the running total in the same pass to show the "% of
    // total" column — useful when one symbol dominates.
    auto bySym = m_journal->realizedBySymbol();
    size_t rowsSym = std::min(m_maxRows, bySym.size());
    if (ImGui::CollapsingHeader("By symbol", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (bySym.empty()) {
            ImGui::TextDisabled("(empty)");
        } else if (ImGui::BeginTable("JournalStatsBySymbol",
                                     3,
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Symbol");
            ImGui::TableSetupColumn("Realized");
            ImGui::TableSetupColumn("% total");
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < rowsSym; ++i) {
                const auto& kv = bySym[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(kv.first.c_str());
                ImGui::TableSetColumnIndex(1);
                colorizeRow(kv.second);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%+.2f", kv.second);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
                ImGui::TableSetColumnIndex(2);
                if (std::fabs(total) > 1e-9) {
                    std::snprintf(buf, sizeof(buf),
                                  "%+.1f%%",
                                  100.0 * kv.second / total);
                    ImGui::TextUnformatted(buf);
                } else {
                    ImGui::TextUnformatted("-");
                }
            }
            ImGui::EndTable();
        }
        if (bySym.size() > m_maxRows) {
            ImGui::TextDisabled("(%zu more not shown)",
                                bySym.size() - m_maxRows);
        }
    }

    // ---- By-tag table (Sprint #73) ----
    //
    // Same shape as by-symbol, but reads from realizedByTag(). The
    // includeUntagged toggle controls whether untagged fills appear
    // under "__untagged__" — the Checkbox is right above the table so
    // the trader can flip it without scrolling.
    ImGui::Separator();
    bool incl = m_includeUntagged;
    if (ImGui::Checkbox("Include untagged (as __untagged__)", &incl)) {
        m_includeUntagged = incl;
    }
    auto byTag = m_journal->realizedByTag(m_includeUntagged);
    size_t rowsTag = std::min(m_maxRows, byTag.size());
    if (ImGui::CollapsingHeader("By tag", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (byTag.empty()) {
            ImGui::TextDisabled("(empty)");
        } else if (ImGui::BeginTable("JournalStatsByTag",
                                     3,
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Tag");
            ImGui::TableSetupColumn("Realized");
            ImGui::TableSetupColumn("% total");
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < rowsTag; ++i) {
                const auto& kv = byTag[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(kv.first.c_str());
                ImGui::TableSetColumnIndex(1);
                colorizeRow(kv.second);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%+.2f", kv.second);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
                ImGui::TableSetColumnIndex(2);
                if (std::fabs(total) > 1e-9) {
                    std::snprintf(buf, sizeof(buf),
                                  "%+.1f%%",
                                  100.0 * kv.second / total);
                    ImGui::TextUnformatted(buf);
                } else {
                    ImGui::TextUnformatted("-");
                }
            }
            ImGui::EndTable();
        }
        if (byTag.size() > m_maxRows) {
            ImGui::TextDisabled("(%zu more not shown)",
                                byTag.size() - m_maxRows);
        }
    }

    ImGui::End();
}

}  // namespace btquant::ui
