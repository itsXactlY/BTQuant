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
    COUNT      = 33,
};

// One hotkey binding — a single GLFW key + optional Ctrl/Shift modifier.
// Modifiers are bit-style: ctrl XOR shift are supported; no Alt (would
// conflict with WM keys on Linux).
struct HotkeyBinding {
    int  glfwKey = -1;   // GLFW_KEY_*, or -1 for unbound
    bool ctrl    = false;
    bool shift   = false;

    // Returns true if (glfwKey + ctrl + shift) matches another binding.
    bool operator==(const HotkeyBinding& o) const {
        return glfwKey == o.glfwKey && ctrl == o.ctrl && shift == o.shift;
    }
    bool operator!=(const HotkeyBinding& o) const { return !(*this == o); }

    // Human-readable label, e.g. "Ctrl+K" or "F2".
    std::string label() const;

    // Match against a runtime glfwGetKey snapshot. `ctrl`/`shift` here
    // are the modifier states the caller observed.
    bool matches(int glfwKeyPressed, bool ctrlDown, bool shiftDown) const {
        if (glfwKey != glfwKeyPressed) return false;
        if (ctrl != ctrlDown)          return false;
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
    HotkeyAction match(int glfwKey, bool ctrlDown, bool shiftDown) const;

    // ---- Pure helpers (test surface) ----
    static std::string actionName(HotkeyAction a);
    static std::string keyName(int glfwKey);
    static HotkeyBinding parseBinding(const std::string& token);

private:
    std::map<HotkeyAction, HotkeyBinding> m_bind;
};

} // namespace btquant::util

#endif
