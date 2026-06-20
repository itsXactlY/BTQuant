#include "journal_stats_panel.hpp"

#include <cmath>
#include <cstdio>
#include <unordered_map>

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

    // ---- Headline summary (Sprint #100) ----
    //
    // One-line at-a-glance: "Best: BTC (Calmar=4.2, Sharpe=1.8)
    // Worst: SOL (Calmar=-0.5, Sharpe=-0.3)". The trader's eyes
    // land here first — answers "where am I winning, where am I
    // bleeding?" without scrolling through 12 sub-tables.
    //
    // Sort: by Calmar DESC (best strategy per unit of worst DD —
    // same metric the trader uses to size positions). Tie-break
    // by Sharpe. When only one symbol/tag exists, "Worst" is
    // suppressed (it's the same symbol).
    {
        auto perSymCl = m_journal->perSymbolCalmar();
        auto perSymSh = m_journal->perSymbolSharpe();
        if (!perSymCl.empty()) {
            // Calmar is sorted DESC by the method (#97); first
            // is best, last is worst. Filter out calmarRatio==0
            // (no DD yet) from the worst side — those aren't
            // really "best" or "worst", they're unrankable.
            const auto& best = perSymCl.front();
            const TradeJournal::PerSymbolCalmar* worst = nullptr;
            for (auto it = perSymCl.rbegin(); it != perSymCl.rend();
                 ++it) {
                if (it->calmarRatio < -1e-9 ||
                    it->maxDrawdown > 1e-9) {
                    worst = &(*it);
                    break;
                }
            }
            // Find matching Sharpe for "Best" and "Worst".
            auto findSharpe = [&](const std::string& sym) {
                for (const auto& s : perSymSh)
                    if (s.symbol == sym) return s.annualizedSharpe;
                return 0.0;
            };
            float bestSh = static_cast<float>(
                findSharpe(best.symbol));
            char headline[160];
            if (worst && worst->symbol != best.symbol) {
                float worstSh = static_cast<float>(
                    findSharpe(worst->symbol));
                std::snprintf(headline, sizeof(headline),
                    "Best: %s (Calmar=%.2f, Sharpe=%.2f)   "
                    "Worst: %s (Calmar=%.2f, Sharpe=%.2f)",
                    best.symbol.c_str(), best.calmarRatio, bestSh,
                    worst->symbol.c_str(), worst->calmarRatio,
                    worstSh);
            } else {
                std::snprintf(headline, sizeof(headline),
                    "Best: %s (Calmar=%.2f, Sharpe=%.2f)",
                    best.symbol.c_str(), best.calmarRatio, bestSh);
            }
            ImGui::TextUnformatted(headline);
        }
    }

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
                              3,
                              ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Max drawdown");
            ImGui::TableSetupColumn("Current drawdown");
            ImGui::TableSetupColumn("Recovery");   // Sprint #96
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

            // Recovery (Sprint #96): "recovered on YYYY-MM-DD
            // (N days)" when there's a recovery date, "— (not
            // recovered)" when still in DD, "(no DD)" when
            // maxDrawdown == 0. Dim informational.
            ImGui::TableSetColumnIndex(2);
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            if (dd.maxDrawdown < 1e-9) {
                std::snprintf(buf, sizeof(buf), "(no DD)");
            } else if (!dd.recoveryDate.empty()) {
                std::snprintf(buf, sizeof(buf),
                              "%s  (%zu day%s)",
                              dd.recoveryDate.c_str(),
                              dd.recoveryDays,
                              dd.recoveryDays == 1 ? "" : "s");
            } else {
                std::snprintf(buf, sizeof(buf),
                              "—  (not recovered)");
            }
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();
            ImGui::EndTable();
        }
    }

    // ---- Streaks mini-section (Sprint #83) ----
    //
    // Current + longest W/L streaks sourced from
    // TradeJournal::streaks() (#82). Two columns: current (the run
    // we're inside right now — green when winning, red when losing,
    // dim at 0) and longest (the all-time best run).
    //
    // Format: "3 wins" / "2 losses" / "—" — verbal so the trader
    // doesn't have to mentally decode a number. Combined with the
    // Risk section above, this answers "am I currently on a streak
    // and how does it compare to my best?".
    auto sk = m_journal->streaks();
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Streaks",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("JournalStatsStreaks",
                              4,
                              ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Current win");
            ImGui::TableSetupColumn("Current loss");
            ImGui::TableSetupColumn("Longest win");
            ImGui::TableSetupColumn("Longest loss");
            ImGui::TableHeadersRow();
            ImGui::TableNextRow();

            // Current win: green when > 0, dim when 0.
            ImGui::TableSetColumnIndex(0);
            if (sk.currentWinStreak > 0) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%zu wins",
                              sk.currentWinStreak);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                ImGui::TextUnformatted("—");
                ImGui::PopStyleColor();
            }

            // Current loss: red when > 0, dim when 0.
            ImGui::TableSetColumnIndex(1);
            if (sk.currentLossStreak > 0) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%zu losses",
                              sk.currentLossStreak);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                ImGui::TextUnformatted("—");
                ImGui::PopStyleColor();
            }

            // Longest win: dim (informational).
            ImGui::TableSetColumnIndex(2);
            char buf[32];
            if (sk.longestWinStreak > 0) {
                std::snprintf(buf, sizeof(buf), "%zu wins",
                              sk.longestWinStreak);
                ImGui::TextUnformatted(buf);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                ImGui::TextUnformatted("—");
                ImGui::PopStyleColor();
            }

            // Longest loss: dim (informational).
            ImGui::TableSetColumnIndex(3);
            if (sk.longestLossStreak > 0) {
                std::snprintf(buf, sizeof(buf), "%zu losses",
                              sk.longestLossStreak);
                ImGui::TextUnformatted(buf);
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                ImGui::TextUnformatted("—");
                ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }
    }

    // ---- Risk-Adjusted mini-section (Sprint #85) ----
    //
    // Sharpe ratio on the daily series, sourced from
    // TradeJournal::sharpe() (#84). Three columns: mean daily
    // return, daily Sharpe, annualized Sharpe. The annualized
    // column is the "headline" — comparable across strategies of
    // different frequencies.
    //
    // Color rules:
    //   - Mean daily: green when > 0, red when < 0 (matches the
    //     row convention; dim at 0).
    //   - Daily / annualized Sharpe: green when >= 1.0, red when
    //     < 0, dim otherwise. Threshold at 1.0 because the
    //     classic interpretation is "Sharpe > 1 = good, > 2 = very
    //     good, > 3 = excellent". A negative Sharpe is a losing
    //     strategy — colored red so it stands out.
    auto sh = m_journal->sharpe();
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Risk-Adjusted",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("JournalStatsSharpe",
                              3,
                              ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Mean / day");
            ImGui::TableSetupColumn("Daily Sharpe");
            ImGui::TableSetupColumn("Annualized");
            ImGui::TableHeadersRow();
            ImGui::TableNextRow();

            // Mean daily return: green/red/dim.
            ImGui::TableSetColumnIndex(0);
            colorizeRow(sh.meanDailyReturn);
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%+.2f", sh.meanDailyReturn);
            ImGui::TextUnformatted(buf);
            ImGui::SameLine();
            ImGui::TextDisabled("(N=%zu)", sh.sampleSize);
            ImGui::PopStyleColor();

            // Daily Sharpe: green >= 1, red < 0, dim otherwise.
            ImGui::TableSetColumnIndex(1);
            if (sh.dailySharpe >= 1.0) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                std::snprintf(buf, sizeof(buf), "%.2f", sh.dailySharpe);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            } else if (sh.dailySharpe < 0.0) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                std::snprintf(buf, sizeof(buf), "%.2f", sh.dailySharpe);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                std::snprintf(buf, sizeof(buf), "%.2f", sh.dailySharpe);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            }

            // Annualized Sharpe: same threshold rules as daily.
            ImGui::TableSetColumnIndex(2);
            if (sh.annualizedSharpe >= 1.0) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                std::snprintf(buf, sizeof(buf), "%.2f", sh.annualizedSharpe);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            } else if (sh.annualizedSharpe < 0.0) {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                std::snprintf(buf, sizeof(buf), "%.2f", sh.annualizedSharpe);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                std::snprintf(buf, sizeof(buf), "%.2f", sh.annualizedSharpe);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }

        // ---- Calmar mini-row (Sprint #96) ----
        //
        // Calmar ratio = annualized return / |max DD|. Sourced
        // from TradeJournal::calmar() (#95). Appended below the
        // Sharpe table rather than as a separate header — keeps
        // the panel compact while still giving the trader the
        // second risk-adjusted metric.
        //
        // Color rules: green >= 3.0 (very good risk-adjusted
        // return), red < 0 (stay away), dim otherwise. Calmar
        // sentinel of 0 (no DD yet) renders as "—" rather than
        // "0.00" so the trader knows the metric is undefined,
        // not zero.
        auto cm = m_journal->calmar();
        char buf[64];
        ImGui::Text("Calmar:");
        ImGui::SameLine();
        if (cm.calmarRatio == 0.0 && cm.maxDrawdown < 1e-9) {
            // No DD yet → metric undefined.
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            ImGui::TextUnformatted("—  (no DD yet)");
            ImGui::PopStyleColor();
        } else if (cm.calmarRatio >= 3.0) {
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
            std::snprintf(buf, sizeof(buf), "%.2f", cm.calmarRatio);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();
        } else if (cm.calmarRatio < 0.0) {
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
            std::snprintf(buf, sizeof(buf), "%.2f", cm.calmarRatio);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            std::snprintf(buf, sizeof(buf), "%.2f", cm.calmarRatio);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(annRet=$%.0f / maxDD=$%.0f)",
                            cm.annualizedReturn, cm.maxDrawdown);

        // ---- Sortino mini-row (Sprint #100) ----
        //
        // Sortino = mean(daily) / downsideDeviation × sqrt(252).
        // Companion to Calmar/Sharpe — penalizes only downside
        // vol instead of all vol. Rendered below Calmar as the
        // third risk-adjusted metric in the journal-wide
        // Risk-Adjusted section.
        //
        // Color rules: green >= 2.0 (excellent risk-adjusted
        // return), red < 0 (stay away), dim otherwise. The
        // "∞" sentinel (downsideDeviation == 0, all-positive
        // days) renders as such.
        auto so = m_journal->sortino();
        ImGui::Text("Sortino:");
        ImGui::SameLine();
        if (so.downsideDeviation < 1e-9 && so.sampleSize >= 1) {
            // All-positive days → no downside → "∞" (same
            // convention used by profit-factor when there are
            // no losses).
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
            ImGui::TextUnformatted("∞");
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::TextDisabled("(all-positive days, N=%zu)",
                                so.sampleSize);
        } else if (so.sampleSize < 1) {
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            ImGui::TextUnformatted("—");
            ImGui::PopStyleColor();
        } else if (so.annualizedSortino >= 2.0) {
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
            std::snprintf(buf, sizeof(buf), "%.2f",
                          so.annualizedSortino);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();
        } else if (so.annualizedSortino < 0.0) {
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
            std::snprintf(buf, sizeof(buf), "%.2f",
                          so.annualizedSortino);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text,
                ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            std::snprintf(buf, sizeof(buf), "%.2f",
                          so.annualizedSortino);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(daily=%.2f, dev=%.2f)",
                            so.dailySortino, so.downsideDeviation);
    }

    ImGui::Separator();

    // ---- By-symbol table (Sprint #72 + Sprint #87) ----
    //
    // Surfaces per-symbol performance. Sprint #72 used
    // realizedBySymbol() for just (symbol, realized); Sprint #87
    // upgrades to perSymbolStats() (#86) so the table also shows
    // win rate + profit factor + W/L counts. Six columns:
    //   symbol, realized, win rate, PF, W/L, % total.
    //
    // Sourced from perSymbolStats() — sorted by abs-realized
    // DESC, same as realizedBySymbol(). Sorted with a stable
    // secondary sort so two symbols with equal abs-realized keep
    // the same order across renders (UI doesn't flicker).
    auto bySym = m_journal->perSymbolStats();
    size_t rowsSym = std::min(m_maxRows, bySym.size());
    if (ImGui::CollapsingHeader("By symbol", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (bySym.empty()) {
            ImGui::TextDisabled("(empty)");
        } else if (ImGui::BeginTable("JournalStatsBySymbol",
                                     6,
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Symbol");
            ImGui::TableSetupColumn("Realized");
            ImGui::TableSetupColumn("Win rate");
            ImGui::TableSetupColumn("PF");
            ImGui::TableSetupColumn("W / L");
            ImGui::TableSetupColumn("% total");
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < rowsSym; ++i) {
                const auto& s = bySym[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(s.symbol.c_str());

                ImGui::TableSetColumnIndex(1);
                colorizeRow(s.realized);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%+.2f", s.realized);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();

                // Win rate: green >= 50%, red < 50%, dim at 0.
                ImGui::TableSetColumnIndex(2);
                if (s.winRate >= 0.5) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                } else if (s.winRate > 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                }
                std::snprintf(buf, sizeof(buf), "%.1f%%",
                              s.winRate * 100.0);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();

                // PF: green >= 1.5, red < 1.0, dim otherwise.
                // "∞" when +inf (all wins, no losses).
                ImGui::TableSetColumnIndex(3);
                if (std::isinf(s.profitFactor)) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    ImGui::TextUnformatted("∞");
                    ImGui::PopStyleColor();
                } else if (s.profitFactor >= 1.5) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f", s.profitFactor);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (s.profitFactor < 1.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f", s.profitFactor);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    std::snprintf(buf, sizeof(buf), "%.2f", s.profitFactor);
                    ImGui::TextUnformatted(buf);
                }

                // W / L counts (dim).
                ImGui::TableSetColumnIndex(4);
                std::snprintf(buf, sizeof(buf), "%zu / %zu",
                              s.winCount, s.lossCount);
                ImGui::TextUnformatted(buf);

                // % total (green if contribution is positive, red
                // if negative — a symbol that lost -50% of all-time
                // P&L gets a red percentage to match the realized
                // column coloring).
                ImGui::TableSetColumnIndex(5);
                if (std::fabs(total) > 1e-9) {
                    double pct = 100.0 * s.realized / total;
                    colorizeRow(pct);
                    std::snprintf(buf, sizeof(buf), "%+.1f%%", pct);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
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

    // ---- By-tag table (Sprint #73 + Sprint #89) ----
    //
    // Surfaces per-tag performance. Sprint #73 used realizedByTag()
    // for just (tag, realized); Sprint #89 upgrades to
    // perTagStats() (#88) so the table also shows win rate +
    // profit factor + W/L counts. Six columns: tag, realized,
    // win rate, PF, W/L, % total.
    //
    // Sourced from perTagStats() — sorted by abs-realized DESC,
    // same convention as realizedByTag() and perSymbolStats().
    // Color rules mirror the by-symbol table. The
    // "Include untagged" Checkbox is right above the table so
    // the trader can flip it without scrolling.
    ImGui::Separator();
    bool incl = m_includeUntagged;
    if (ImGui::Checkbox("Include untagged (as __untagged__)", &incl)) {
        m_includeUntagged = incl;
    }
    auto byTag = m_journal->perTagStats(m_includeUntagged);
    size_t rowsTag = std::min(m_maxRows, byTag.size());
    if (ImGui::CollapsingHeader("By tag", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (byTag.empty()) {
            ImGui::TextDisabled("(empty)");
        } else if (ImGui::BeginTable("JournalStatsByTag",
                                     6,
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Tag");
            ImGui::TableSetupColumn("Realized");
            ImGui::TableSetupColumn("Win rate");
            ImGui::TableSetupColumn("PF");
            ImGui::TableSetupColumn("W / L");
            ImGui::TableSetupColumn("% total");
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < rowsTag; ++i) {
                const auto& s = byTag[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(s.tag.c_str());

                ImGui::TableSetColumnIndex(1);
                colorizeRow(s.realized);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%+.2f", s.realized);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();

                // Win rate: green >= 50%, red < 50%, dim at 0.
                ImGui::TableSetColumnIndex(2);
                if (s.winRate >= 0.5) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                } else if (s.winRate > 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                }
                std::snprintf(buf, sizeof(buf), "%.1f%%",
                              s.winRate * 100.0);
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();

                // PF: green >= 1.5, red < 1.0, dim otherwise.
                ImGui::TableSetColumnIndex(3);
                if (std::isinf(s.profitFactor)) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    ImGui::TextUnformatted("∞");
                    ImGui::PopStyleColor();
                } else if (s.profitFactor >= 1.5) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f", s.profitFactor);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (s.profitFactor < 1.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f", s.profitFactor);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    std::snprintf(buf, sizeof(buf), "%.2f", s.profitFactor);
                    ImGui::TextUnformatted(buf);
                }

                // W / L counts (dim).
                ImGui::TableSetColumnIndex(4);
                std::snprintf(buf, sizeof(buf), "%zu / %zu",
                              s.winCount, s.lossCount);
                ImGui::TextUnformatted(buf);

                // % total (colorized to match realized).
                ImGui::TableSetColumnIndex(5);
                if (std::fabs(total) > 1e-9) {
                    double pct = 100.0 * s.realized / total;
                    colorizeRow(pct);
                    std::snprintf(buf, sizeof(buf), "%+.1f%%", pct);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
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

    // ---- Per-tag risk table (Sprint #94) ----
    //
    // Surfaces TradeJournal::perTagDrawdown() (#93) — worst
    // peak-to-trough decline per tag. Four columns:
    //   tag, max DD, peak→trough, current DD.
    //
    // Same column layout + color rules as the per-symbol risk
    // table (#92). Reuses m_includeUntagged so flipping the
    // checkbox above "By tag" also flips the rollup behavior here.
    auto perTagDD = m_journal->perTagDrawdown(m_includeUntagged);
    size_t rowsTagDD = std::min(m_maxRows, perTagDD.size());
    if (ImGui::CollapsingHeader("Per-tag risk",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (perTagDD.empty()) {
            ImGui::TextDisabled("(empty)");
        } else if (ImGui::BeginTable("JournalStatsPerTagRisk",
                                     5,
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Tag");
            ImGui::TableSetupColumn("Max DD");
            ImGui::TableSetupColumn("Peak → Trough");
            ImGui::TableSetupColumn("Current DD");
            ImGui::TableSetupColumn("Recovery");   // Sprint #96
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < rowsTagDD; ++i) {
                const auto& e = perTagDD[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(e.tag.c_str());

                // Max DD: always red.
                ImGui::TableSetColumnIndex(1);
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                char buf[96];
                if (e.maxDrawdown > 1e-9) {
                    std::snprintf(buf, sizeof(buf), "-%.2f",
                                  e.maxDrawdown);
                } else {
                    std::snprintf(buf, sizeof(buf), "0.00");
                }
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();

                // Peak → Trough: dim.
                ImGui::TableSetColumnIndex(2);
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                if (e.maxDrawdown > 1e-9 && !e.peakDate.empty() &&
                    !e.troughDate.empty()) {
                    std::snprintf(buf, sizeof(buf), "%s → %s",
                                  e.peakDate.c_str(),
                                  e.troughDate.c_str());
                } else {
                    std::snprintf(buf, sizeof(buf), "—");
                }
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();

                // Current DD: red when > 0, dim when 0.
                ImGui::TableSetColumnIndex(3);
                if (e.currentDD > 1e-9) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "-%.2f",
                                  e.currentDD);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                    if (std::fabs(e.currentDD - e.maxDrawdown) < 1e-9 &&
                        !e.troughDate.empty()) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("(in worst DD)");
                    }
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    ImGui::TextUnformatted("0.00 (at ATH)");
                    ImGui::PopStyleColor();
                }

                // Recovery (Sprint #96).
                ImGui::TableSetColumnIndex(4);
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                if (e.maxDrawdown < 1e-9) {
                    std::snprintf(buf, sizeof(buf), "(no DD)");
                } else if (!e.recoveryDate.empty()) {
                    std::snprintf(buf, sizeof(buf),
                                  "%s  (%zu day%s)",
                                  e.recoveryDate.c_str(),
                                  e.recoveryDays,
                                  e.recoveryDays == 1 ? "" : "s");
                } else {
                    std::snprintf(buf, sizeof(buf),
                                  "—  (not recovered)");
                }
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }
        if (perTagDD.size() > rowsTagDD) {
            ImGui::TextDisabled("(%zu more not shown)",
                                perTagDD.size() - rowsTagDD);
        }
    }

    // ---- Per-tag risk-adjusted table (Sprint #94) ----
    //
    // Surfaces TradeJournal::perTagSharpe() (#93). Four columns:
    //   tag, mean / day, daily Sharpe, annualized Sharpe.
    //
    // Best-first sort, same threshold rules + sample-size
    // annotation as the per-symbol risk-adjusted table (#92).
    auto perTagSh = m_journal->perTagSharpe(m_includeUntagged);
    size_t rowsTagSh = std::min(m_maxRows, perTagSh.size());
    // Calmar by-tag (#98) — same zip pattern as the per-symbol
    // table. Reuses m_includeUntagged so the rollup stays
    // consistent across the per-tag sub-tables.
    auto perTagCl = m_journal->perTagCalmar(m_includeUntagged);
    std::unordered_map<std::string, double> calmarByTag;
    calmarByTag.reserve(perTagCl.size());
    for (const auto& c : perTagCl) calmarByTag[c.tag] = c.calmarRatio;
    // Sortino by-tag (#100) — same pattern as per-symbol.
    auto perTagSo_ = m_journal->perTagSortino(m_includeUntagged);
    if (ImGui::CollapsingHeader("Per-tag risk-adjusted",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (perTagSh.empty()) {
            ImGui::TextDisabled("(empty)");
        } else if (ImGui::BeginTable("JournalStatsPerTagSharpe",
                                     6,
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Tag");
            ImGui::TableSetupColumn("Mean / day");
            ImGui::TableSetupColumn("Daily Sharpe");
            ImGui::TableSetupColumn("Annualized");
            ImGui::TableSetupColumn("Calmar");   // Sprint #98
            ImGui::TableSetupColumn("Sortino");  // Sprint #100
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < rowsTagSh; ++i) {
                const auto& e = perTagSh[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(e.tag.c_str());

                // Mean daily: green/red/dim with sample size.
                ImGui::TableSetColumnIndex(1);
                colorizeRow(e.meanDailyReturn);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%+.2f",
                              e.meanDailyReturn);
                ImGui::TextUnformatted(buf);
                ImGui::SameLine();
                ImGui::TextDisabled("(N=%zu)", e.sampleSize);
                ImGui::PopStyleColor();

                // Daily Sharpe: green >= 1, red < 0, dim otherwise.
                ImGui::TableSetColumnIndex(2);
                if (e.dailySharpe >= 1.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.dailySharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (e.dailySharpe < 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.dailySharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.dailySharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                }

                // Annualized Sharpe: same threshold rules.
                ImGui::TableSetColumnIndex(3);
                if (e.annualizedSharpe >= 1.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.annualizedSharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (e.annualizedSharpe < 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.annualizedSharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.annualizedSharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                }

                // Calmar (Sprint #98): same threshold rules as
                // the per-symbol table. Zipped from
                // perTagCalmar() by tag name.
                ImGui::TableSetColumnIndex(4);
                double tagCalmar = 0.0;
                bool   tagHasCalmar = false;
                auto it = calmarByTag.find(e.tag);
                if (it != calmarByTag.end()) {
                    tagCalmar = it->second;
                    tagHasCalmar = true;
                }
                if (tagHasCalmar && tagCalmar >= 3.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  tagCalmar);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (tagHasCalmar && tagCalmar < 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  tagCalmar);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (tagHasCalmar) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  tagCalmar);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "—");
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                }

                // Sortino (Sprint #100): same threshold rules as
                // the per-symbol Sortino column.
                ImGui::TableSetColumnIndex(5);
                double tagSo = 0.0;
                bool   tagHasSo = false;
                bool   tagSoNoDownside = false;
                for (const auto& so : perTagSo_) {
                    if (so.tag == e.tag) {
                        tagSo = so.annualizedSortino;
                        tagHasSo = true;
                        tagSoNoDownside = (so.downsideDeviation < 1e-9
                                           && so.sampleSize >= 1);
                        break;
                    }
                }
                if (tagHasSo && tagSoNoDownside) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    ImGui::TextUnformatted("∞");
                    ImGui::PopStyleColor();
                } else if (tagHasSo && tagSo >= 2.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f", tagSo);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (tagHasSo && tagSo < 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f", tagSo);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (tagHasSo) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "%.2f", tagSo);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "—");
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                }
            }
            ImGui::EndTable();
        }
        if (perTagSh.size() > rowsTagSh) {
            ImGui::TextDisabled("(%zu more not shown)",
                                perTagSh.size() - rowsTagSh);
        }
    }

    ImGui::Separator();

    // ---- Per-symbol risk table (Sprint #92) ----
    //
    // Surfaces TradeJournal::perSymbolDrawdown() (#91) — worst
    // peak-to-trough decline per symbol. Four columns:
    //   symbol, max DD, peak→trough, current DD.
    //
    // Sort order from the method is worst-first (biggest maxDD
    // first), so the trader reading the table top-to-bottom sees
    // the symbol that hurt them most at the top.
    //
    // Color rules:
    //   - Max DD: always red (it's the worst by definition).
    //   - Current DD: red when > 0, dim when 0 (at symbol-ATH),
    //     with an "(in worst DD)" annotation when the symbol is
    //     still inside its worst drop (matches the journal-wide
    //     Risk section, #81).
    //   - Peak→trough dates: dim (informational — the trader reads
    //     them as "when did this happen?" rather than a signal).
    auto perSymDD = m_journal->perSymbolDrawdown();
    size_t rowsDD = std::min(m_maxRows, perSymDD.size());
    if (ImGui::CollapsingHeader("Per-symbol risk",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (perSymDD.empty()) {
            ImGui::TextDisabled("(empty)");
        } else if (ImGui::BeginTable("JournalStatsPerSymbolRisk",
                                     5,
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Symbol");
            ImGui::TableSetupColumn("Max DD");
            ImGui::TableSetupColumn("Peak → Trough");
            ImGui::TableSetupColumn("Current DD");
            ImGui::TableSetupColumn("Recovery");   // Sprint #96
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < rowsDD; ++i) {
                const auto& e = perSymDD[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(e.symbol.c_str());

                // Max DD: always red (worst by definition).
                ImGui::TableSetColumnIndex(1);
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                char buf[96];
                if (e.maxDrawdown > 1e-9) {
                    std::snprintf(buf, sizeof(buf), "-%.2f",
                                  e.maxDrawdown);
                } else {
                    std::snprintf(buf, sizeof(buf), "0.00");
                }
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();

                // Peak → Trough: dim informational. Format
                // "YYYY-MM-DD → YYYY-MM-DD" or "-" when no DD yet.
                ImGui::TableSetColumnIndex(2);
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                if (e.maxDrawdown > 1e-9 && !e.peakDate.empty() &&
                    !e.troughDate.empty()) {
                    std::snprintf(buf, sizeof(buf), "%s → %s",
                                  e.peakDate.c_str(),
                                  e.troughDate.c_str());
                } else {
                    std::snprintf(buf, sizeof(buf), "—");
                }
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();

                // Current DD: red when > 0, dim when 0.
                ImGui::TableSetColumnIndex(3);
                if (e.currentDD > 1e-9) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "-%.2f",
                                  e.currentDD);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                    if (std::fabs(e.currentDD - e.maxDrawdown) < 1e-9 &&
                        !e.troughDate.empty()) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("(in worst DD)");
                    }
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    ImGui::TextUnformatted("0.00 (at ATH)");
                    ImGui::PopStyleColor();
                }

                // Recovery (Sprint #96). Same format as the
                // journal-wide Recovery column.
                ImGui::TableSetColumnIndex(4);
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                if (e.maxDrawdown < 1e-9) {
                    std::snprintf(buf, sizeof(buf), "(no DD)");
                } else if (!e.recoveryDate.empty()) {
                    std::snprintf(buf, sizeof(buf),
                                  "%s  (%zu day%s)",
                                  e.recoveryDate.c_str(),
                                  e.recoveryDays,
                                  e.recoveryDays == 1 ? "" : "s");
                } else {
                    std::snprintf(buf, sizeof(buf),
                                  "—  (not recovered)");
                }
                ImGui::TextUnformatted(buf);
                ImGui::PopStyleColor();
            }
            ImGui::EndTable();
        }
        if (perSymDD.size() > rowsDD) {
            ImGui::TextDisabled("(%zu more not shown)",
                                perSymDD.size() - rowsDD);
        }
    }

    // ---- Per-symbol risk-adjusted table (Sprint #92) ----
    //
    // Surfaces TradeJournal::perSymbolSharpe() (#91). Four
    // columns:
    //   symbol, mean / day, daily Sharpe, annualized Sharpe.
    //
    // Sort order from the method is best-first (highest
    // annualized Sharpe first) — answers "which symbol gives me
    // the best return per unit of risk?" at a glance.
    //
    // Color rules match the journal-wide Risk-Adjusted section
    // (#85):
    //   - Mean daily: green/red/dim (row convention).
    //   - Daily / annualized Sharpe: green >= 1.0, red < 0,
    //     dim otherwise. The classic threshold is Sharpe > 1 =
    //     good, > 2 = very good, > 3 = excellent; negative
    //     Sharpe is a losing strategy.
    auto perSymSh = m_journal->perSymbolSharpe();
    size_t rowsSh = std::min(m_maxRows, perSymSh.size());
    // Calmar by-symbol (#98) — fetched separately and zipped by
    // symbol name into a map. Avoids changing the PerSymbolSharpe
    // struct shape just to add a single Calmar column.
    auto perSymCl = m_journal->perSymbolCalmar();
    std::unordered_map<std::string, double> calmarBySymbol;
    calmarBySymbol.reserve(perSymCl.size());
    for (const auto& c : perSymCl) calmarBySymbol[c.symbol] = c.calmarRatio;
    // Sortino by-symbol (#100) — same pattern as Calmar. Stored
    // as a vector (not a map) since the per-row lookup is a
    // small linear scan over typically <10 symbols.
    auto perSymSo_ = m_journal->perSymbolSortino();
    if (ImGui::CollapsingHeader("Per-symbol risk-adjusted",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (perSymSh.empty()) {
            ImGui::TextDisabled("(empty)");
        } else if (ImGui::BeginTable("JournalStatsPerSymbolSharpe",
                                     6,
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_BordersH)) {
            ImGui::TableSetupColumn("Symbol");
            ImGui::TableSetupColumn("Mean / day");
            ImGui::TableSetupColumn("Daily Sharpe");
            ImGui::TableSetupColumn("Annualized");
            ImGui::TableSetupColumn("Calmar");   // Sprint #98
            ImGui::TableSetupColumn("Sortino");  // Sprint #100
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < rowsSh; ++i) {
                const auto& e = perSymSh[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(e.symbol.c_str());

                // Mean daily: green/red/dim, with sample size
                // annotation so the trader knows whether the
                // number is meaningful (a Sharpe on 2 days is
                // very different from one on 200 days).
                ImGui::TableSetColumnIndex(1);
                colorizeRow(e.meanDailyReturn);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%+.2f",
                              e.meanDailyReturn);
                ImGui::TextUnformatted(buf);
                ImGui::SameLine();
                ImGui::TextDisabled("(N=%zu)", e.sampleSize);
                ImGui::PopStyleColor();

                // Daily Sharpe: green >= 1, red < 0, dim otherwise.
                ImGui::TableSetColumnIndex(2);
                if (e.dailySharpe >= 1.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.dailySharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (e.dailySharpe < 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.dailySharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.dailySharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                }

                // Annualized Sharpe: same threshold rules as daily.
                ImGui::TableSetColumnIndex(3);
                if (e.annualizedSharpe >= 1.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.annualizedSharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (e.annualizedSharpe < 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.annualizedSharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  e.annualizedSharpe);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                }

                // Calmar (Sprint #98): green >= 3.0, red < 0,
                // dim otherwise. Sentinel 0 (no DD yet for this
                // symbol) renders as "—".
                ImGui::TableSetColumnIndex(4);
                double symCalmar = 0.0;
                bool   symHasCalmar = false;
                auto it = calmarBySymbol.find(e.symbol);
                if (it != calmarBySymbol.end()) {
                    symCalmar = it->second;
                    symHasCalmar = true;
                }
                if (symHasCalmar && symCalmar >= 3.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  symCalmar);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (symHasCalmar && symCalmar < 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  symCalmar);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (symHasCalmar) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "%.2f",
                                  symCalmar);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "—");
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                }

                // Sortino (Sprint #100): same threshold rules as
                // the journal-wide Sortino row.
                ImGui::TableSetColumnIndex(5);
                double symSo = 0.0;
                bool   symHasSo = false;
                size_t symSoN = 0;
                bool   symSoNoDownside = false;
                for (const auto& so : perSymSo_) {
                    if (so.symbol == e.symbol) {
                        symSo = so.annualizedSortino;
                        symHasSo = true;
                        symSoN = so.sampleSize;
                        symSoNoDownside = (so.downsideDeviation < 1e-9
                                           && so.sampleSize >= 1);
                        break;
                    }
                }
                if (symHasSo && symSoNoDownside) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    ImGui::TextUnformatted("∞");
                    ImGui::PopStyleColor();
                } else if (symHasSo && symSo >= 2.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.30f, 0.85f, 0.40f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f", symSo);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (symHasSo && symSo < 0.0) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImVec4(0.95f, 0.30f, 0.30f, 1.0f));
                    std::snprintf(buf, sizeof(buf), "%.2f", symSo);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else if (symHasSo) {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "%.2f", symSo);
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
                    std::snprintf(buf, sizeof(buf), "—");
                    ImGui::TextUnformatted(buf);
                    ImGui::PopStyleColor();
                }
            }
            ImGui::EndTable();
        }
        if (perSymSh.size() > rowsSh) {
            ImGui::TextDisabled("(%zu more not shown)",
                                perSymSh.size() - rowsSh);
        }
    }

    ImGui::End();
}

}  // namespace btquant::ui
