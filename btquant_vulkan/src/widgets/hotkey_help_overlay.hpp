#ifndef BTQUANT_HOTKEY_HELP_OVERLAY_HPP
#define BTQUANT_HOTKEY_HELP_OVERLAY_HPP

namespace btquant::util { class HotkeyMap; }

namespace btquant::ui {

// Non-editable hotkey reference overlay (Sprint #64). Pops up on
// demand (Ctrl+? or Ctrl+/), lists every binding in the bound
// HotkeyMap as a sortable table, closes on Esc. Distinct from the
// HotkeyEditor (which is the edit surface): this is a quick
// "what was that shortcut again?" reference the trader summons
// mid-trade without leaving their current workflow.
//
// WindowManager owns the map and passes a non-owning pointer via
// setHotkeyMap(). The overlay reads it at render time so re-binding
// via HotkeyEditor is reflected here without extra plumbing.
class HotkeyHelpOverlay {
public:
    void render();

    // Bind the live map. Pass nullptr to disable (overlay shows a
    // "not wired" hint instead of an empty table).
    void setHotkeyMap(const ::btquant::util::HotkeyMap* m) { m_map = m; }

    // Open / close + visibility check.
    bool isOpen() const    { return m_open; }
    void setOpen(bool v)   { m_open = v; }
    void toggle()          { m_open = !m_open; }

private:
    bool m_open = false;
    const ::btquant::util::HotkeyMap* m_map = nullptr;
};

}  // namespace btquant::ui

#endif
