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

    // ---- Stats header (Sprint #76) ----
    //
    // Win rate + profit factor + expectancy + win/loss counts in a
    // compact row above the breakdowns. Sourced from
    // TradeJournal::stats() (#75) — same field semantics as
    // RiskMetrics, so a future Stats tab can render both without a
    // translation layer. The journal-wide view makes this the
    // persistent, all-time counterpart to RiskPanel's rolling
    // window.
    //
    // Rendered as a small table so columns line up regardless of
    // font / DPI. Profit factor uses the +inf sentinel from
    // TradeJournal::stats() (all wins, no losses) — formatted as
    // "∞" rather than "inf" to fit the visual style of the panel.
    auto st = m_journal->stats();
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Stats",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (st.roundTripCount == 0) {
            ImGui::TextDisabled("(no round-trip fills yet)");
        } else if (ImGui::BeginTable("JournalStatsHeader",
                                      6,
                                      ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Win rate");
            ImGui::TableSetupColumn("PF");
            ImGui::TableSetupColumn("Avg win");
            ImGui::TableSetupColumn("Avg loss");
            ImGui::TableSetupColumn("Expectancy");
            ImGui::TableSetupColumn("W / L");
            ImGui::TableHeadersRow();
            ImGui::TableNextRow();
            // Win rate: green when >= 50%, red when < 50%, dim at 0.
            ImGui::TableSetColumnIndex(0);
            if (st.winRate >= 0.5) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
            } else if (st.winRate > 0.0) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            }
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.1f%%", st.winRate * 100.0);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();

            // Profit factor: green when >= 1.5, red when < 1.0,
            // dim otherwise. "∞" when +inf (all wins, no losses).
            ImGui::TableSetColumnIndex(1);
            if (std::isinf(st.profitFactor)) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                ImGui::TextUnformatted("∞");
                ImGui::PopStyleColor();
            } else if (st.profitFactor >= 1.5) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                std::snprintf(buf, sizeof(buf), "%.2f", st.profitFactor);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            } else if (st.profitFactor < 1.0) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                std::snprintf(buf, sizeof(buf), "%.2f", st.profitFactor);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            } else {
                std::snprintf(buf, sizeof(buf), "%.2f", st.profitFactor);
                ImGui::TextUnformatted(buf);
            }

            // Avg winner: green positive.
            ImGui::TableSetColumnIndex(2);
            colorizeRow(st.avgWinner);
            std::snprintf(buf, sizeof(buf), "%+.2f", st.avgWinner);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();

            // Avg loser: red negative.
            ImGui::TableSetColumnIndex(3);
            colorizeRow(st.avgLoser);
            std::snprintf(buf, sizeof(buf), "%+.2f", st.avgLoser);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();

            // Expectancy: green when >= 0, red when < 0.
            ImGui::TableSetColumnIndex(4);
            colorizeRow(st.expectancy);
            std::snprintf(buf, sizeof(buf), "%+.2f", st.expectancy);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();

            // W / L counts.
            ImGui::TableSetColumnIndex(5);
            std::snprintf(buf, sizeof(buf), "%zu / %zu",
                          st.winCount, st.lossCount);
            ImGui::TextUnformatted(buf);

            ImGui::EndTable();
        }
    }

    // ---- Risk mini-section (Sprint #81) ----
    //
    // Drawdown summary sourced from TradeJournal::maxDrawdown()
    // (#80). Worst peak-to-trough decline + the dates that bracket
    // it, plus the current drawdown (== 0 when equity is at ATH).
    //
    // Two columns: maxDD on the left (always red — by definition
    // the worst), currentDD on the right (red when in DD, green
    // when at ATH, dim at 0). Dates shown in compact "YYYY-MM-DD"
    // format — same as the by-day table so the trader can read
    // them in one glance.
    auto dd = m_journal->maxDrawdown();
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Risk",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("JournalStatsRisk",
                              2,
                              ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Max drawdown");
            ImGui::TableSetupColumn("Current drawdown");
            ImGui::TableHeadersRow();
            ImGui::TableNextRow();
            // Max DD: always red (it's the worst by definition).
            ImGui::TableSetColumnIndex(0);
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
            char buf[96];
            if (dd.maxDrawdown > 1e-9) {
                std::snprintf(buf, sizeof(buf),
                              "-%.2f  (%s → %s)",
                              dd.maxDrawdown,
                              dd.peakDate.c_str(),
                              dd.troughDate.c_str());
            } else {
                std::snprintf(buf, sizeof(buf), "0.00  (no drawdown yet)");
            }
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();

            // Current DD: red when > 0, dim when 0 (at ATH).
            ImGui::TableSetColumnIndex(1);
            if (dd.currentDD > 1e-9) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                std::snprintf(buf, sizeof(buf), "-%.2f", dd.currentDD);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
                // Annotation when currently inside the worst DD.
                if (std::fabs(dd.currentDD - dd.maxDrawdown) < 1e-9 &&
                    !dd.troughDate.empty()) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("(in worst DD)");
                }
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                ImGui::TextUnformatted("0.00  (at ATH)");
                ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }
    }

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

    // ---- By-day table (Sprint #78) ----
    //
    // Surfaces TradeJournal::realizedByDay() (#77) — daily P&L
    // buckets in "YYYY-MM-DD" format. Sorted newest-first when a
    // lookback is set (default 30 days), since traders read this
    // as "what did I do recently?". When lookback=0, the entire
    // journal is shown oldest-first (matches realizedByDay's
    // natural order, and lets the trader see their full history).
    //
    // The lookback slider is right above the table so the trader
    // can flip between "last week / month / quarter / all" without
    // scrolling.
    ImGui::Separator();
    int lookback = static_cast<int>(m_dayLookback);
    int presets[] = { 7, 30, 90, 365, 0 };  // 0 = all-time
    const char* presetLabels[] = {
        "7d", "30d", "90d", "1y", "all"
    };
    ImGui::Text("Lookback:");
    ImGui::SameLine();
    for (int i = 0; i < 5; ++i) {
        if (i > 0) ImGui::SameLine();
        bool isSel = (lookback == presets[i]);
        if (isSel) ImGui::PushStyleColor(ImGuiCol_Button,
                                         ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
        if (ImGui::SmallButton(presetLabels[i])) {
            lookback = presets[i];
            m_dayLookback = static_cast<size_t>(lookback);
        }
        if (isSel) ImGui::PopStyleColor();
    }

    auto byDay = m_journal->realizedByDay();
    size_t rowsDay = 0;
    if (m_dayLookback > 0 && byDay.size() > m_dayLookback) {
        rowsDay = std::min(m_maxRows, m_dayLookback);
    } else {
        rowsDay = std::min(m_maxRows, byDay.size());
    }

    // The vector from realizedByDay is oldest-first. For the
    // "recent activity" view we want newest-first — reverse the
    // tail when a lookback is set. For all-time, leave it
    // oldest-first.
    std::vector<std::pair<std::string, double>> displayRows;
    if (m_dayLookback > 0 && byDay.size() > m_dayLookback) {
        // Last m_dayLookback entries, reversed to newest-first.
        size_t start = byDay.size() - m_dayLookback;
        for (size_t i = byDay.size(); i > start; --i) {
            displayRows.push_back(byDay[i - 1]);
        }
    } else {
        displayRows = byDay;
    }
    rowsDay = std::min(m_maxRows, displayRows.size());

    if (ImGui::CollapsingHeader("By day",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (displayRows.empty()) {
            ImGui::TextDisabled("(empty)");
        } else if (ImGui::BeginTable("JournalStatsByDay",
                                     2,
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Date");
            ImGui::TableSetupColumn("Realized");
            ImGui::TableHeadersRow();

            double sumWindow = 0.0;
            for (size_t i = 0; i < rowsDay; ++i) {
                const auto& kv = displayRows[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(kv.first.c_str());
                ImGui::TableSetColumnIndex(1);
                colorizeRow(kv.second);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%+.2f", kv.second);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
                sumWindow += kv.second;
            }
            ImGui::EndTable();

            // Window sum (sum of the displayed rows) — useful at a
            // glance: "the last 30 days netted $X". Colored to
            // match the row convention.
            ImGui::Text("Window total:");
            ImGui::SameLine();
            colorizeRow(sumWindow);
            char sumBuf[64];
            std::snprintf(sumBuf, sizeof(sumBuf), "%+.2f", sumWindow);
            ImGui::TextUnformatted(sumBuf);
            ImGui::PopStyleColor();
        }
        size_t hiddenCount = displayRows.size() > rowsDay
                                 ? displayRows.size() - rowsDay
                                 : 0;
        if (hiddenCount > 0) {
            ImGui::TextDisabled("(%zu more not shown)", hiddenCount);
        }

        // ---- Daily equity curve (Sprint #79) ----
        //
        // Cumulative P&L across the displayed days — same idea as
        // RiskPanel's session sparkline (#61) but sourced from the
        // persisted journal rather than the in-memory session.
        // Survives restarts and accumulates across days.
        //
        // The series is rebuilt from displayRows (newest-first
        // view → cumulative from oldest → newest = reverse the
        // accumulation order). Color: green if final equity >
        // 0, red if < 0, dim at zero — same convention as
        // RiskPanel.
        if (displayRows.size() >= 2) {
            // Build cumulative series in chronological order.
            // displayRows is newest-first when lookback>0, oldest-
            // first otherwise. The "all-time" branch keeps the
            // original order; the lookback branch reverses.
            std::vector<std::pair<std::string, double>> chrono;
            if (m_dayLookback > 0 && byDay.size() > m_dayLookback) {
                size_t start = byDay.size() - m_dayLookback;
                for (size_t i = start; i < byDay.size(); ++i) {
                    chrono.push_back(byDay[i]);
                }
            } else {
                chrono = byDay;
            }
            std::vector<float> scratch;
            scratch.reserve(chrono.size());
            double cum = 0.0;
            for (const auto& kv : chrono) {
                cum += kv.second;
                scratch.push_back(static_cast<float>(cum));
            }
            double mn = scratch.front();
            double mx = scratch.front();
            for (float v : scratch) {
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            double pad = (mx - mn) > 1e-9 ? (mx - mn) * 0.05 : 1.0;
            mn -= pad; mx += pad;
            char overlay[64];
            std::snprintf(overlay, sizeof(overlay),
                          "%zu days | %+.0f -> %+.0f",
                          scratch.size(),
                          static_cast<double>(scratch.front()),
                          static_cast<double>(scratch.back()));
            ImGui::PushStyleColor(ImGuiCol_PlotLines,
                scratch.back() >= 0
                    ? ImVec4(0.30f, 0.85f, 0.40f, 1.0f)
                    : ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
            ImGui::PlotLines("##daily_equity_curve",
                             scratch.data(),
                             static_cast<int>(scratch.size()),
                             0, overlay,
                             static_cast<float>(mn),
                             static_cast<float>(mx),
                             ImVec2(-1, 60));
            ImGui::PopStyleColor();
        } else {
            ImGui::TextDisabled("(equity curve: %zu day%s - "
                                "need >= 2 to plot)",
                                displayRows.size(),
                                displayRows.size() == 1 ? "" : "s");
        }
    }

    ImGui::End();
}

}  // namespace btquant::ui
