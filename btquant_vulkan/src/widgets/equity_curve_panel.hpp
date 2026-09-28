#ifndef BTQUANT_EQUITY_CURVE_PANEL_HPP
#define BTQUANT_EQUITY_CURVE_PANEL_HPP

#include <cstddef>

namespace btquant { class TradeJournal; }

namespace btquant::ui {

// Equity curve + drawdown overlay (Sprint #104).
//
// Renders TradeJournal::equityCurve() as a sparkline-style line
// graph, with the running peak drawn as a dim line and the area
// between equity and peak shaded red (the drawdown).
//
// Two visualizations stacked vertically:
//   TOP    — equity curve + drawdown shading
//   BOTTOM — drawdown series (negative bars below a zero axis)
//
// The widget does NOT own the journal — caller wires the pointer
// and it stays valid for the panel's lifetime.
class EquityCurvePanel {
public:
    void setJournal(::btquant::TradeJournal* j) { m_journal = j; }

    // Cap the number of points rendered. Default 0 = render all.
    // Useful for traders with many thousands of fills — keeps
    // the sparkline scannable.
    void  setMaxPoints(size_t n) { m_maxPoints = n; }
    size_t maxPoints() const     { return m_maxPoints; }

    // Rolling-Sharpe window size (Sprint #110). Default 30 days.
    // Used by the bottom chart to overlay the daily-Sharpe
    // time-series with mean=0 / stddev=1 normalization.
    void      setSharpeWindow(size_t n) { m_sharpeWindow = n; }
    size_t    sharpeWindow() const      { return m_sharpeWindow; }

    void render();

    bool showWindow = false;

private:
    ::btquant::TradeJournal* m_journal      = nullptr;
    size_t                   m_maxPoints    = 0;     // 0 = unlimited
    size_t                   m_sharpeWindow = 30;    // Sprint #110
};

}  // namespace btquant::ui

#endif
