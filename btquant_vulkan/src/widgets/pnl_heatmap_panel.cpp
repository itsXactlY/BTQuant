#include "pnl_heatmap_panel.hpp"

#include "imgui.h"

#include "../data/trade_journal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace btquant::ui {

namespace {

// Color for a single cell. Realized P&L drives the saturation; sign
// drives the hue. Sparse cells (realized==0 AND no round-trips) get
// a dim neutral that the trader learns to read as "no activity".
//
// Color scale:
//   positive large:  bright green   (RGB 0.20, 0.85, 0.30)
//   positive small:  pale green     (interp toward 0.50,0.78,0.55)
//   zero/sparse:     dim gray       (RGB 0.30, 0.30, 0.30)
//   negative small:  pale red       (interp toward 0.78,0.50,0.55)
//   negative large:  bright red     (RGB 0.85, 0.20, 0.30)
//
// Intensity curve: sqrt(|realized|/maxAbs) — compresses the long tail
// so a single +$5k day doesn't bleach every other cell white-green.
ImU32 cellColor(double realized, double maxAbs, bool sparse) {
    if (sparse) return IM_COL32(45, 45, 45, 255);
    if (maxAbs < 1e-9) return IM_COL32(45, 45, 45, 255);
    double t = std::sqrt(std::fabs(realized) / maxAbs);
    if (t > 1.0) t = 1.0;
    if (realized >= 0.0) {
        // pale green  → bright green
        auto lerp = [](double a, double b, double x) {
            return a + (b - a) * x;
        };
        return IM_COL32(static_cast<int>(lerp(128, 51, t)),
                        static_cast<int>(lerp(199, 217, t)),
                        static_cast<int>(lerp(140, 77, t)), 255);
    }
    auto lerp = [](double a, double b, double x) {
        return a + (b - a) * x;
    };
    return IM_COL32(static_cast<int>(lerp(199, 217, t)),
                    static_cast<int>(lerp(128, 51, t)),
                    static_cast<int>(lerp(140, 77, t)), 255);
}

// Format realized for a cell. Three modes:
//   sparse    → "·"  (trader can scan for activity)
//   |v| < 1   → "+0.50" / "-0.20"  (2 decimals, signed)
//   |v| < 100 → "+12.3" / "-7.40"  (1 decimal, signed)
//   else      → "+1.2k" / "-450"   (compact)
const char* cellLabel(char* buf, size_t bufsz, double v, bool sparse) {
    if (sparse) {
        std::snprintf(buf, bufsz, "·");
        return buf;
    }
    double av = std::fabs(v);
    if (av < 1.0) {
        std::snprintf(buf, bufsz, "%+.2f", v);
    } else if (av < 100.0) {
        std::snprintf(buf, bufsz, "%+.1f", v);
    } else if (av < 10000.0) {
        std::snprintf(buf, bufsz, "%+.0f", v);
    } else {
        std::snprintf(buf, bufsz, "%+.1fk", v / 1000.0);
    }
    return buf;
}

// Compact date label: "MM-DD" (drop the year — column header bar gets
// crowded when the trader has many months of history).
const char* dateLabel(char* buf, size_t bufsz, const std::string& iso) {
    // iso = "YYYY-MM-DD" → "MM-DD"
    if (iso.size() >= 10) {
        std::snprintf(buf, bufsz, "%.5s", iso.c_str() + 5);
    } else {
        std::snprintf(buf, bufsz, "%s", iso.c_str());
    }
    return buf;
}

}  // namespace

void PnLHeatmapPanel::render() {
    if (!m_journal) {
        ImGui::TextDisabled("PnL Heatmap: journal not wired");
        return;
    }

    // ---- Fetch the grid ----
    TradeJournal::PerSymbolDayStats ps;  // (also used as Tag by alias)
    size_t nRows = 0;
    const std::vector<std::string>* rowLabels = nullptr;
    if (m_mode == Mode::Symbol) {
        ps = m_journal->perSymbolDayStats();
        nRows = ps.symbols.size();
        rowLabels = &ps.symbols;
    } else {
        auto pt = m_journal->perTagDayStats(m_includeUntagged);
        // Convert PerTagDayStats into the same shape via shallow
        // aliasing — same field names, same layout. We just borrow
        // the vectors and grid.
        ps.symbols = std::move(pt.tags);
        ps.dates   = std::move(pt.dates);
        ps.grid.assign(pt.tags.size() * pt.dates.size(),
                       TradeJournal::DayCell{});
        // Rebuild the grid from the per-tag variant.
        ps.grid = std::move(pt.grid);
        nRows = ps.symbols.size();
        rowLabels = &ps.symbols;
    }

    if (nRows == 0 || ps.dates.empty()) {
        ImGui::TextDisabled("PnL Heatmap: no fills yet");
        return;
    }

    // ---- Header: mode toggle + legend ----
    if (ImGui::RadioButton("Symbol", m_mode == Mode::Symbol)) {
        m_mode = Mode::Symbol;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Tag", m_mode == Mode::Tag)) {
        m_mode = Mode::Tag;
    }
    if (m_mode == Mode::Tag) {
        ImGui::SameLine();
        ImGui::Checkbox("Include untagged", &m_includeUntagged);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    // Mini legend — three swatches.
    ImGui::TextDisabled("Legend:");
    ImGui::SameLine();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = ImGui::GetTextLineHeight();
    auto swatch = [&](ImU32 c, const char* label) {
        ImVec2 sz(h * 0.85f, h * 0.85f);
        dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), c);
        ImGui::Dummy(sz);
        ImGui::SameLine();
        ImGui::TextUnformatted(label);
        ImGui::SameLine();
        p = ImGui::GetCursorScreenPos();
    };
    swatch(cellColor(-1000.0, 1000.0, false), "loss");
    swatch(cellColor(   0.0, 1000.0, true ),  "—");
    swatch(cellColor( 1000.0, 1000.0, false), "win");
    ImGui::NewLine();

    // ---- Cap row/col counts for viewport ----
    size_t showRows = std::min(nRows, m_maxRows);
    size_t nDates   = ps.dates.size();
    size_t showCols;
    if (m_maxDates > 0 && m_maxDates < nDates) {
        showCols = m_maxDates;
        // Show the most-recent N dates — slice from the end.
        // (Dates are already sorted ASC, so the last `showCols` are
        // the newest.)
    } else {
        showCols = nDates;
    }
    size_t dateOffset = (showCols < nDates) ? (nDates - showCols) : 0;

    // ---- Compute maxAbs for color scaling ----
    double maxAbs = 0.0;
    for (size_t r = 0; r < showRows; ++r) {
        for (size_t c = 0; c < showCols; ++c) {
            const auto& cell =
                ps.grid[r * nDates + (c + dateOffset)];
            double a = std::fabs(cell.realized);
            if (a > maxAbs) maxAbs = a;
        }
    }

    // ---- Render as a table ----
    // Column 0: row label (symbol/tag).
    // Columns 1..N: dates.
    // Row 0: header with date labels (rotated would be ideal but ImGui
    //         doesn't have first-class rotated text — keep horizontal,
    //         abbreviate to MM-DD).
    // Rows 1..M: row label + colored cells.
    const float cellW = 56.0f;
    const float cellH = 22.0f;
    const float labelW = 110.0f;
    const float headerH = 22.0f;

    // Column header row.
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        ImVec2(0.0f, 0.0f));
    {
        // Corner cell.
        ImGui::Dummy(ImVec2(labelW, headerH));
        ImGui::SameLine();
        char dbuf[16];
        for (size_t c = 0; c < showCols; ++c) {
            dateLabel(dbuf, sizeof(dbuf), ps.dates[c + dateOffset]);
            ImGui::PushID(static_cast<int>(c));
            ImGui::Button(dbuf, ImVec2(cellW, headerH));
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", ps.dates[c + dateOffset].c_str());
            }
            ImGui::PopID();
            if (c + 1 < showCols) ImGui::SameLine();
        }
        ImGui::NewLine();
    }

    // Data rows.
    for (size_t r = 0; r < showRows; ++r) {
        const std::string& row = (*rowLabels)[r];
        // Row label.
        ImGui::TextUnformatted(row.c_str());
        ImGui::SameLine();
        for (size_t c = 0; c < showCols; ++c) {
            const auto& cell =
                ps.grid[r * nDates + (c + dateOffset)];
            bool sparse = (cell.roundTrips == 0);
            ImU32 col = cellColor(cell.realized, maxAbs, sparse);
            char lbuf[32];
            cellLabel(lbuf, sizeof(lbuf), cell.realized, sparse);
            ImGui::PushID(static_cast<int>(r * showCols + c));
            ImGui::PushStyleColor(ImGuiCol_Button,        col);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, col);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,  col);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                                ImVec2(2.0f, 2.0f));
            ImGui::Button(lbuf, ImVec2(cellW, cellH));
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(3);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "%s on %s\n"
                    "  realized: %.2f\n"
                    "  round-trips: %zu  (W:%zu / L:%zu)",
                    row.c_str(),
                    ps.dates[c + dateOffset].c_str(),
                    cell.realized,
                    cell.roundTrips, cell.wins, cell.losses);
            }
            ImGui::PopID();
            if (c + 1 < showCols) ImGui::SameLine();
        }
        ImGui::NewLine();
    }
    ImGui::PopStyleVar();

    // ---- Footer: totals + truncation notice ----
    if (nRows > showRows || (m_maxDates > 0 && nDates > showCols)) {
        ImGui::TextDisabled(
            "(showing %zu/%zu rows × %zu/%zu dates — "
            "raise maxRows/maxDates in config to see more)",
            showRows, nRows, showCols, nDates);
    }
}

}  // namespace btquant::ui
