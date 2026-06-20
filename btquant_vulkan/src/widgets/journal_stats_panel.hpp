#ifndef BTQUANT_JOURNAL_STATS_PANEL_HPP
#define BTQUANT_JOURNAL_STATS_PANEL_HPP

#include <string>
#include <vector>

namespace btquant { class TradeJournal; }

namespace btquant::ui {

// All-time P&L statistics from the persisted TradeJournal. Surfaces
// three orthogonal breakdowns side-by-side:
//
//   * Total:    sum of realizedDelta across every persisted fill
//               (TradeJournal::totalRealized()).
//   * By symbol: largest gainers/losers first
//               (TradeJournal::realizedBySymbol()).
//   * By tag:    same, but grouped by JournalFill::tag
//               (TradeJournal::realizedByTag()).
//
// The widget does NOT own the journal — caller wires the pointer and
// it stays valid for the panel's lifetime (typically the program
// lifetime, since the journal is a singleton-style accessor in main).
//
// Reads are intentionally cheap: each render() call iterates the
// journal twice (once for by-symbol, once for by-tag) plus once for
// total. The journal is small enough (low-thousands of fills in
// typical use) that this is sub-millisecond and not worth caching.
//
// Sprint #74.
class JournalStatsPanel {
public:
    void setJournal(::btquant::TradeJournal* j) { m_journal = j; }

    // When true (default), fills with an empty tag are aggregated
    // under the synthetic key "__untagged__". When false, untagged
    // fills are skipped in the by-tag table. Toggling this at runtime
    // is harmless — it's just a re-read of the journal.
    void setIncludeUntagged(bool v) { m_includeUntagged = v; }
    bool includeUntagged() const   { return m_includeUntagged; }

    // Cap the number of rows rendered per breakdown. Defaults to 16
    // — fits comfortably in the panel and keeps the layout from
    // blowing up when the trader has hundreds of symbols.
    void setMaxRows(size_t n)    { m_maxRows = n; }
    size_t maxRows() const       { return m_maxRows; }

    // Lookback window for the by-day table (Sprint #78). When > 0,
    // only the last N days are shown (newest first). When 0 (the
    // default), the entire journal is shown — practical for traders
    // with < 1 year of history, but typically you'd set this to 30
    // or 90 for the "last month / quarter" view.
    void  setDayLookback(size_t n)   { m_dayLookback = n; }
    size_t dayLookback() const       { return m_dayLookback; }

    void render();

    // Default closed so the panel doesn't pop up uninvited when the
    // user adds the widget. Hotkey (Ctrl+J) and View menu toggle it.
    bool showWindow = false;

private:
    ::btquant::TradeJournal* m_journal = nullptr;
    bool  m_includeUntagged  = true;
    size_t m_maxRows         = 16;
    size_t m_dayLookback     = 30;   // Sprint #78 default
};

}  // namespace btquant::ui

#endif
