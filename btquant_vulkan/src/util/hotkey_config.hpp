#ifndef BTQUANT_HOTKEY_CONFIG_HPP
#define BTQUANT_HOTKEY_CONFIG_HPP

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace btquant::util {

// Logical action a hotkey can be bound to. Keep the enum in sync with
// actionName() and the WindowManager dispatch table.
enum class HotkeyAction : int {
    ToggleOrderBook      = 0,
    ToggleOrderBookDepth = 1,
    ToggleDOM            = 2,
    ToggleTrades         = 3,
    ToggleTPO            = 4,
    ToggleFootprint      = 5,
    ToggleVPVR           = 6,
    ToggleAlerts         = 7,
    ToggleMultiVWAP      = 8,
    ToggleRiskPanel      = 9,
    ToggleSettings       = 10,
    ResetLayout          = 11,
    OpenSymbolPicker     = 12,
    OpenThemeEditor      = 13,
    ToggleOrderTicket    = 14,
    TogglePositionPanel  = 15,
    ToggleRiskLimits     = 16,
    ToggleMiniPriceChart = 17,
    KillSwitch           = 18,
    ToggleStats          = 19,
    ToggleHotkeyHelp     = 20,
    ToggleHotkeyEditor   = 21,   // self-referential — opens this dialog
    // Quick-swap layout profiles. Bound to Ctrl+1..Ctrl+9 by default;
    // SwitchLayoutN applies the Nth .btqlayout file from
    // LayoutIO::list() (sorted). No-op if there are fewer than N
    // profiles — keeps the key useful as a "nothing to do" press.
    SwitchLayout1 = 22,
    SwitchLayout2 = 23,
    SwitchLayout3 = 24,
    SwitchLayout4 = 25,
    SwitchLayout5 = 26,
    SwitchLayout6 = 27,
    SwitchLayout7 = 28,
    SwitchLayout8 = 29,
    SwitchLayout9 = 30,
    // Submit hotkeys — when the Order Ticket is open, Alt+B / Alt+S
    // submit the current draft as a BUY / SELL respectively (flipping
    // the side first if needed). When the ticket is closed, the same
    // shortcuts open it in the matching side mode so the trader can
    // quickly reach the ticket pre-loaded for the action they want.
    // Bound to Alt+B / Alt+S by default.
    SubmitBuy  = 31,
    SubmitSell = 32,
    // Sprint #74: toggle the Journal Stats panel (all-time P&L by
    // total / symbol / tag from the persisted journal). Bound to
    // Ctrl+J by default — same modifier group as the other Ctrl
    // toggles (P/T/R/B/M/H).
    ToggleJournalStats = 33,
    // Sprint #103: toggle the P&L Heatmap panel (calendar-style
    // grid of realized P&L per symbol/tag × date). Bound to
    // Ctrl+Shift+H by default — Ctrl+H is the HotkeyHelp overlay,
    // and Ctrl+Shift+H keeps the same muscle memory as the
    // "history / heatmap" association.
    TogglePnLHeatmap = 34,
    // Sprint #104: toggle the Equity Curve panel (cumulative P&L
    // line graph + drawdown overlay). Bound to Ctrl+E.
    ToggleEquityCurve = 35,
    COUNT             = 36,
};

// One hotkey binding — a single GLFW key + optional modifier chord
// (Ctrl / Alt / Shift). Modifiers are independent flags, not a bitmask,
// so we can support any combination (e.g. Ctrl+Alt+K, Shift+Alt+F2).
// Alt is allowed despite the historical "WM key conflict" caveat —
// the order-ticket submit actions (Alt+B / Alt+S) use it deliberately
// because they don't conflict with any window-manager shortcut in
// the trading app's runtime posture.
struct HotkeyBinding {
    int  glfwKey = -1;   // GLFW_KEY_*, or -1 for unbound
    bool ctrl    = false;
    bool alt     = false;
    bool shift   = false;

    // Returns true if (glfwKey + ctrl + alt + shift) matches another binding.
    bool operator==(const HotkeyBinding& o) const {
        return glfwKey == o.glfwKey && ctrl == o.ctrl &&
               alt == o.alt && shift == o.shift;
    }
    bool operator!=(const HotkeyBinding& o) const { return !(*this == o); }

    // Human-readable label, e.g. "Ctrl+Alt+K" or "F2". Modifier order
    // is fixed (Ctrl → Alt → Shift) for stable diff/sort.
    std::string label() const;

    // Match against a runtime glfwGetKey snapshot. `ctrl`/`alt`/`shift`
    // here are the modifier states the caller observed.
    bool matches(int glfwKeyPressed,
                 bool ctrlDown, bool altDown, bool shiftDown) const {
        if (glfwKey != glfwKeyPressed) return false;
        if (ctrl  != ctrlDown)         return false;
        if (alt   != altDown)          return false;
        if (shift != shiftDown)        return false;
        return true;
    }
};

// Map: action → binding. Owns the default + persistence layer.
class HotkeyMap {
public:
    HotkeyMap() = default;

    // Construct from the built-in defaults — what the WindowManager was
    // hardcoded to before this layer existed.
    static HotkeyMap defaults();

    // Bind / lookup.
    void set(HotkeyAction a, const HotkeyBinding& b) { m_bind[a] = b; }
    HotkeyBinding get(HotkeyAction a) const;
    bool          has(HotkeyAction a) const;

    // Iterate in enum order — used by the editor UI to render the table.
    std::vector<std::pair<HotkeyAction, HotkeyBinding>>
        enumerate() const;

    // ---- Persistence ----
    // Format: key=value lines, one binding per line. Comments start with
    // '#'. Blank lines ignored. Keys are actionName(action).
    //   ToggleOrderBook=F2
    //   ToggleOrderTicket=Ctrl+Enter
    std::optional<HotkeyMap> static loadFromFile(const std::string& path);
    bool                      saveToFile(const std::string& path) const;

    // Find which action matches a runtime key+modifier snapshot.
    // Returns COUNT if no match (caller ignores).
    HotkeyAction match(int glfwKey,
                       bool ctrlDown, bool altDown, bool shiftDown) const;

    // ---- Pure helpers (test surface) ----
    static std::string actionName(HotkeyAction a);
    static std::string keyName(int glfwKey);
    static HotkeyBinding parseBinding(const std::string& token);

private:
    std::map<HotkeyAction, HotkeyBinding> m_bind;
};

} // namespace btquant::util

#endif
