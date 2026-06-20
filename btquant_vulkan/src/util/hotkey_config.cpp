#include "hotkey_config.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

// GLFW is linked into both the main binary and the test binary
// (vulkan_context.cpp needs it), so we can always include the header
// here. If a future build target drops GLFW, this file would need a
// compile-time guard.
#include <GLFW/glfw3.h>

namespace btquant::util {

std::string HotkeyBinding::label() const {
    if (glfwKey < 0) return "(unbound)";
    std::string out;
    if (ctrl)  out += "Ctrl+";
    if (shift) out += "Shift+";
    out += HotkeyMap::keyName(glfwKey);
    return out;
}

HotkeyMap HotkeyMap::defaults() {
    HotkeyMap m;
    // F2..F12 — widget toggles.
    m.set(HotkeyAction::ToggleOrderBook,      { GLFW_KEY_F2,  false, false });
    m.set(HotkeyAction::ToggleOrderBookDepth, { GLFW_KEY_F3,  false, false });
    m.set(HotkeyAction::ToggleDOM,            { GLFW_KEY_F4,  false, false });
    m.set(HotkeyAction::ToggleTrades,         { GLFW_KEY_F5,  false, false });
    m.set(HotkeyAction::ToggleTPO,            { GLFW_KEY_F6,  false, false });
    m.set(HotkeyAction::ToggleFootprint,      { GLFW_KEY_F7,  false, false });
    m.set(HotkeyAction::ToggleVPVR,           { GLFW_KEY_F8,  false, false });
    m.set(HotkeyAction::ToggleAlerts,         { GLFW_KEY_F9,  false, false });
    m.set(HotkeyAction::ToggleMultiVWAP,      { GLFW_KEY_F11, false, false });
    m.set(HotkeyAction::ToggleRiskPanel,      { GLFW_KEY_F10, false, false });
    m.set(HotkeyAction::ToggleSettings,       { GLFW_KEY_F12, false, false });
    // Ctrl-prefixed actions.
    m.set(HotkeyAction::ResetLayout,          { GLFW_KEY_L,        true, false });
    m.set(HotkeyAction::OpenSymbolPicker,     { GLFW_KEY_P,        true, false });
    m.set(HotkeyAction::OpenThemeEditor,      { GLFW_KEY_T,        true, false });
    m.set(HotkeyAction::ToggleOrderTicket,    { GLFW_KEY_ENTER,    true, false });
    m.set(HotkeyAction::TogglePositionPanel,  { GLFW_KEY_B,        true, false });
    m.set(HotkeyAction::ToggleRiskLimits,     { GLFW_KEY_R,        true, false });
    m.set(HotkeyAction::ToggleMiniPriceChart, { GLFW_KEY_M,        true, false });
    m.set(HotkeyAction::KillSwitch,           { GLFW_KEY_K,        true, false });
    m.set(HotkeyAction::ToggleStats,          { GLFW_KEY_F1,      false, true  });
    m.set(HotkeyAction::ToggleHotkeyHelp,     { GLFW_KEY_SLASH,   false, true  });
    m.set(HotkeyAction::ToggleHotkeyEditor,   { GLFW_KEY_H,        true, false });
    m.set(HotkeyAction::SwitchLayout1,        { GLFW_KEY_1,        true, false });
    m.set(HotkeyAction::SwitchLayout2,        { GLFW_KEY_2,        true, false });
    m.set(HotkeyAction::SwitchLayout3,        { GLFW_KEY_3,        true, false });
    m.set(HotkeyAction::SwitchLayout4,        { GLFW_KEY_4,        true, false });
    m.set(HotkeyAction::SwitchLayout5,        { GLFW_KEY_5,        true, false });
    m.set(HotkeyAction::SwitchLayout6,        { GLFW_KEY_6,        true, false });
    m.set(HotkeyAction::SwitchLayout7,        { GLFW_KEY_7,        true, false });
    m.set(HotkeyAction::SwitchLayout8,        { GLFW_KEY_8,        true, false });
    m.set(HotkeyAction::SwitchLayout9,        { GLFW_KEY_9,        true, false });
    return m;
}

HotkeyBinding HotkeyMap::get(HotkeyAction a) const {
    auto it = m_bind.find(a);
    if (it == m_bind.end()) return HotkeyBinding{};
    return it->second;
}

bool HotkeyMap::has(HotkeyAction a) const {
    return m_bind.find(a) != m_bind.end();
}

std::vector<std::pair<HotkeyAction, HotkeyBinding>>
HotkeyMap::enumerate() const {
    std::vector<std::pair<HotkeyAction, HotkeyBinding>> out;
    out.reserve(static_cast<size_t>(HotkeyAction::COUNT));
    for (int i = 0; i < static_cast<int>(HotkeyAction::COUNT); ++i) {
        HotkeyAction a = static_cast<HotkeyAction>(i);
        out.push_back({a, get(a)});
    }
    return out;
}

std::string HotkeyMap::actionName(HotkeyAction a) {
    switch (a) {
        case HotkeyAction::ToggleOrderBook:      return "ToggleOrderBook";
        case HotkeyAction::ToggleOrderBookDepth: return "ToggleOrderBookDepth";
        case HotkeyAction::ToggleDOM:            return "ToggleDOM";
        case HotkeyAction::ToggleTrades:         return "ToggleTrades";
        case HotkeyAction::ToggleTPO:            return "ToggleTPO";
        case HotkeyAction::ToggleFootprint:      return "ToggleFootprint";
        case HotkeyAction::ToggleVPVR:           return "ToggleVPVR";
        case HotkeyAction::ToggleAlerts:         return "ToggleAlerts";
        case HotkeyAction::ToggleMultiVWAP:      return "ToggleMultiVWAP";
        case HotkeyAction::ToggleRiskPanel:      return "ToggleRiskPanel";
        case HotkeyAction::ToggleSettings:       return "ToggleSettings";
        case HotkeyAction::ResetLayout:          return "ResetLayout";
        case HotkeyAction::OpenSymbolPicker:     return "OpenSymbolPicker";
        case HotkeyAction::OpenThemeEditor:      return "OpenThemeEditor";
        case HotkeyAction::ToggleOrderTicket:    return "ToggleOrderTicket";
        case HotkeyAction::TogglePositionPanel:  return "TogglePositionPanel";
        case HotkeyAction::ToggleRiskLimits:     return "ToggleRiskLimits";
        case HotkeyAction::ToggleMiniPriceChart: return "ToggleMiniPriceChart";
        case HotkeyAction::KillSwitch:           return "KillSwitch";
        case HotkeyAction::ToggleStats:          return "ToggleStats";
        case HotkeyAction::ToggleHotkeyHelp:     return "ToggleHotkeyHelp";
        case HotkeyAction::ToggleHotkeyEditor:   return "ToggleHotkeyEditor";
        case HotkeyAction::SwitchLayout1:        return "SwitchLayout1";
        case HotkeyAction::SwitchLayout2:        return "SwitchLayout2";
        case HotkeyAction::SwitchLayout3:        return "SwitchLayout3";
        case HotkeyAction::SwitchLayout4:        return "SwitchLayout4";
        case HotkeyAction::SwitchLayout5:        return "SwitchLayout5";
        case HotkeyAction::SwitchLayout6:        return "SwitchLayout6";
        case HotkeyAction::SwitchLayout7:        return "SwitchLayout7";
        case HotkeyAction::SwitchLayout8:        return "SwitchLayout8";
        case HotkeyAction::SwitchLayout9:        return "SwitchLayout9";
        default: return "Unknown";
    }
}

std::string HotkeyMap::keyName(int glfwKey) {
#ifndef BTQUANT_HOTKEY_CONFIG_NO_GLFW
    switch (glfwKey) {
        case GLFW_KEY_SPACE:        return "Space";
        case GLFW_KEY_ENTER:        return "Enter";
        case GLFW_KEY_TAB:          return "Tab";
        case GLFW_KEY_ESCAPE:       return "Esc";
        case GLFW_KEY_BACKSPACE:    return "Backspace";
        case GLFW_KEY_SLASH:        return "?";
        case GLFW_KEY_SEMICOLON:    return ";";
        case GLFW_KEY_EQUAL:        return "=";
        case GLFW_KEY_LEFT_BRACKET: return "[";
        case GLFW_KEY_RIGHT_BRACKET:return "]";
        case GLFW_KEY_BACKSLASH:    return "\\";
        case GLFW_KEY_GRAVE_ACCENT: return "`";
        case GLFW_KEY_COMMA:        return ",";
        case GLFW_KEY_PERIOD:       return ".";
        case GLFW_KEY_MINUS:        return "-";
        case GLFW_KEY_APOSTROPHE:   return "'";
        default: break;
    }
    // F1..F25
    if (glfwKey >= GLFW_KEY_F1 && glfwKey <= GLFW_KEY_F25) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "F%d",
                      glfwKey - GLFW_KEY_F1 + 1);
        return std::string(buf);
    }
    // Printable ASCII range.
    if (glfwKey >= 32 && glfwKey < 127) {
        return std::string(1, static_cast<char>(glfwKey));
    }
#endif
    char buf[16];
    std::snprintf(buf, sizeof(buf), "Key%d", glfwKey);
    return std::string(buf);
}

HotkeyBinding HotkeyMap::parseBinding(const std::string& token) {
    HotkeyBinding b;
    b.glfwKey = -1;
    if (token.empty()) return b;
    std::string t = token;
    // Strip modifiers.
    if (t.size() > 5 && t.compare(0, 5, "Ctrl+") == 0) {
        b.ctrl = true;
        t = t.substr(5);
    }
    if (t.size() > 6 && t.compare(0, 6, "Shift+") == 0) {
        b.shift = true;
        t = t.substr(6);
    }
#ifndef BTQUANT_HOTKEY_CONFIG_NO_GLFW
    if (t == "Space")      b.glfwKey = GLFW_KEY_SPACE;
    else if (t == "Enter") b.glfwKey = GLFW_KEY_ENTER;
    else if (t == "Tab")   b.glfwKey = GLFW_KEY_TAB;
    else if (t == "Esc")   b.glfwKey = GLFW_KEY_ESCAPE;
    else if (t == "Backspace") b.glfwKey = GLFW_KEY_BACKSPACE;
    else if (t == "?")     b.glfwKey = GLFW_KEY_SLASH;
    else if (t == ";")     b.glfwKey = GLFW_KEY_SEMICOLON;
    else if (t == "=")     b.glfwKey = GLFW_KEY_EQUAL;
    else if (t == "[")     b.glfwKey = GLFW_KEY_LEFT_BRACKET;
    else if (t == "]")     b.glfwKey = GLFW_KEY_RIGHT_BRACKET;
    else if (t == "\\")    b.glfwKey = GLFW_KEY_BACKSLASH;
    else if (t == "`")     b.glfwKey = GLFW_KEY_GRAVE_ACCENT;
    else if (t == ",")     b.glfwKey = GLFW_KEY_COMMA;
    else if (t == ".")     b.glfwKey = GLFW_KEY_PERIOD;
    else if (t == "-")     b.glfwKey = GLFW_KEY_MINUS;
    else if (t == "'")     b.glfwKey = GLFW_KEY_APOSTROPHE;
    else if (!t.empty() && t[0] == 'F' && t.size() <= 4) {
        int n = std::atoi(t.c_str() + 1);
        if (n >= 1 && n <= 25) b.glfwKey = GLFW_KEY_F1 + (n - 1);
    } else if (t.size() == 1) {
        char c = t[0];
        if (c >= 'A' && c <= 'Z') b.glfwKey = c;        // uppercase letters
        else if (c >= 'a' && c <= 'z') b.glfwKey = c;   // lowercase too
        else if (c >= '0' && c <= '9') b.glfwKey = c;
    }
#endif
    return b;
}

std::optional<HotkeyMap> HotkeyMap::loadFromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) return std::nullopt;
    HotkeyMap m = defaults();
    std::string line;
    while (std::getline(in, line)) {
        // Trim.
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        std::string trimmed = line.substr(a, b - a + 1);
        if (trimmed.empty() || trimmed[0] == '#') continue;
        auto eq = trimmed.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trimmed.substr(0, eq);
        std::string val = trimmed.substr(eq + 1);
        // Match the key against every action name.
        for (int i = 0; i < static_cast<int>(HotkeyAction::COUNT); ++i) {
            HotkeyAction act = static_cast<HotkeyAction>(i);
            if (actionName(act) == key) {
                m.set(act, parseBinding(val));
                break;
            }
        }
    }
    return m;
}

bool HotkeyMap::saveToFile(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "# btquant_vulkan hotkey bindings\n";
    out << "# Format: ActionName=Ctrl+Shift+K (Ctrl/Shift optional; K = key)\n";
    out << "# Edit and reload — WindowManager reads on startup.\n\n";
    auto rows = enumerate();
    for (const auto& [a, b] : rows) {
        if (b.glfwKey < 0) continue;  // skip unbound
        out << actionName(a) << "=" << b.label() << "\n";
    }
    return out.good();
}

HotkeyAction HotkeyMap::match(int glfwKey, bool ctrlDown, bool shiftDown) const {
    for (int i = 0; i < static_cast<int>(HotkeyAction::COUNT); ++i) {
        HotkeyAction a = static_cast<HotkeyAction>(i);
        if (has(a) && get(a).matches(glfwKey, ctrlDown, shiftDown)) {
            return a;
        }
    }
    return HotkeyAction::COUNT;
}

} // namespace btquant::util
