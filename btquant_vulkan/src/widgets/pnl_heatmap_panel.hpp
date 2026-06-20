#ifndef BTQUANT_PNL_HEATMAP_PANEL_HPP
#define BTQUANT_PNL_HEATMAP_PANEL_HPP

#include <cstddef>

namespace btquant { class TradeJournal; }

namespace btquant::ui {

// Calendar-style P&L heatmap (Sprint #103).
//
// Renders the grid produced by TradeJournal::perSymbolDayStats() or
// TradeJournal::perTagDayStats() as a 2-D color-coded table:
//
//     rows = symbols (or tags)  — sorted ASC
//     cols = dates              — sorted ASC, chronological
//     cell color = sign + intensity of realized P&L on that day
//                  for that symbol/tag.
//
// Sparse cells (no fills on that day for that row) render dim — the
// trader can see at a glance "BTC has zero fills on Tuesdays" or
// "SOL is consistently red on Mondays".
//
// Toggle row ↔ column overflow when the trader has hundreds of symbols
// or many months of history — `setMaxRows` / `setMaxDates` cap the
// viewport so the panel stays scannable.
//
// The widget does NOT own the journal — caller wires the pointer and
// it stays valid for the panel's lifetime.
class PnLHeatmapPanel {
public:
    void setJournal(::btquant::TradeJournal* j) { m_journal = j; }

    // Mode toggle — sprint #103.
    //   Symbol (default) — rows are distinct symbols.
    //   Tag              — rows are strategy tags (perTagDayStats).
    //                       Honors includeUntagged.
    enum class Mode { Symbol, Tag };
    void setMode(Mode m) { m_mode = m; }
    Mode mode() const    { return m_mode; }

    // When true (default), fills with an empty tag are aggregated
    // under "__untagged__" in Tag mode. Mirrors JournalStatsPanel
    // semantics. Ignored in Symbol mode.
    void setIncludeUntagged(bool v) { m_includeUntagged = v; }
    bool includeUntagged() const   { return m_includeUntagged; }

    // Cap the number of rows (symbols/tags) rendered. Default 24 —
    // fits the typical 24"-trader-monitor layout without scrolling.
    void  setMaxRows(size_t n) { m_maxRows = n; }
    size_t maxRows() const     { return m_maxRows; }

    // Cap the number of date columns rendered. When 0 (default), all
    // dates are shown. Useful when the trader has 12 months of history
    // but only wants to see the last 30 days.
    void  setMaxDates(size_t n) { m_maxDates = n; }
    size_t maxDates() const     { return m_maxDates; }

    void render();

    // Default closed — hotkey (Ctrl+Shift+H) and View menu toggle it.
    bool showWindow = false;

private:
    ::btquant::TradeJournal* m_journal         = nullptr;
    Mode    m_mode            = Mode::Symbol;
    bool    m_includeUntagged = true;
    size_t  m_maxRows         = 24;
    size_t  m_maxDates        = 0;     // 0 = unlimited
};

}  // namespace btquant::ui

#endif
