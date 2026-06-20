#ifndef BTQUANT_RISK_LIMITS_PANEL_HPP
#define BTQUANT_RISK_LIMITS_PANEL_HPP

namespace btquant { class RiskGuard; }
namespace btquant { class PositionBook; }

namespace btquant::ui {

// Risk dashboard + limit editor. Always-visible style — shows live
// status (kill-tripped / armed / warning) plus current exposure and
// session P&L. Editable fields write back to the RiskGuard via Apply.
// Hotkey: Ctrl+R toggles visibility.
class RiskLimitsPanel {
public:
    void setRiskGuard(::btquant::RiskGuard* g) { m_guard = g; }
    void setPositionBook(::btquant::PositionBook* b) { m_book = b; }

    void render();

    bool isOpen() const    { return m_open; }
    void setOpen(bool v)   { m_open = v; }

    // Edit-buffer accessors — values are applied only on Apply click.
    double editedMaxPositionSizeUSD() const;
    double editedMaxLeverage() const;
    double editedKillOnDailyLossUSD() const;
    double editedEquityUSD() const;

private:
    void syncFromGuard();
    void applyToGuard();

    ::btquant::RiskGuard*    m_guard = nullptr;
    ::btquant::PositionBook* m_book  = nullptr;
    bool m_open = false;

    // Edit buffers — kept as char[] for ImGui InputText round-tripping.
    char m_maxPos[32]  = "100000";
    char m_maxLev[32]  = "10";
    char m_killUSD[32] = "5000";
    char m_equity[32]  = "10000";
};

} // namespace btquant::ui

#endif
