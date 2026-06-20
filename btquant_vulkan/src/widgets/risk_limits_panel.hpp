#ifndef BTQUANT_RISK_LIMITS_PANEL_HPP
#define BTQUANT_RISK_LIMITS_PANEL_HPP

#include <functional>

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

    // Persistence callback — invoked after the user clicks Apply (or a
    // preset button). WindowManager registers a closure that writes the
    // current guard config into Settings and saves state.ini, so the
    // edited values survive restart.
    using PersistFn = std::function<void(const ::btquant::RiskGuard&)>;
    void setPersistFn(PersistFn fn) { m_persistFn = std::move(fn); }

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
    void applyBufferToGuardField();   // re-reads a single buffer → guard

    ::btquant::RiskGuard*    m_guard = nullptr;
    ::btquant::PositionBook* m_book  = nullptr;
    PersistFn                m_persistFn;
    bool m_open = false;

    // Edit buffers — kept as char[] for ImGui InputText round-tripping.
    char m_maxPos[32]  = "100000";
    char m_maxLev[32]  = "10";
    char m_killUSD[32] = "5000";
    char m_equity[32]  = "10000";

    // Live-update mode: when true, every change to a limit field
    // immediately streams into the guard so the RiskPanel's progress
    // bar reflects the new threshold without waiting for the Apply
    // button. Default off so the existing Apply-button muscle memory
    // keeps working — toggle it in the panel header.
    bool m_liveUpdate = false;

    // Last value actually pushed to the guard — used to detect dirty
    // state for the unsaved-changes indicator. Initialized to the
    // default buffer's parsed value (5000.0) so a freshly-constructed
    // panel with the conservative config reads as clean.
    double m_lastAppliedKillUSD = 5000.0;

public:
    // True when the edit buffer differs from the last applied value.
    // Used by the render loop to draw the "* unsaved" hint.
    bool isKillDirty() const {
        return editedKillOnDailyLossUSD() != m_lastAppliedKillUSD;
    }
    // True if live-update mode is engaged.
    bool isLiveUpdate() const { return m_liveUpdate; }
    // Toggle live-update from outside (e.g. menu hotkey). No-op when
    // the panel isn't open or has no guard bound.
    void setLiveUpdate(bool v) { m_liveUpdate = v; }
};

} // namespace btquant::ui

#endif
