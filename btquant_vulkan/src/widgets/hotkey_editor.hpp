#ifndef BTQUANT_HOTKEY_EDITOR_HPP
#define BTQUANT_HOTKEY_EDITOR_HPP

#include <string>
#include "../util/hotkey_config.hpp"

namespace btquant::widgets {

// A docked/windowed panel that shows every HotkeyAction + its current
// binding, lets the user remap any action by pressing the next key, and
// writes changes back to the HotkeyMap (the WindowManager persists the
// map on shutdown).
//
// State machine:
//   IDLE          — table visible, [Remap] buttons are clickable
//   CAPTURING(i)  — row i is being rebound; next non-modifier key
//                   press becomes the new binding (Esc cancels)
//   MODIFIED      — m_dirty=true; WindowManager persists on next save
class HotkeyEditor {
public:
    HotkeyEditor() = default;

    void setHotkeyMap(::btquant::util::HotkeyMap* m) { m_map = m; }
    bool isOpen() const     { return m_open; }
    void setOpen(bool v)    { m_open = v; }
    void toggleOpen()       { m_open = !m_open; }

    // Has the user made any unsaved changes? (Caller persists via
    // saveToFile; we don't write here so the operation is debounced.)
    bool isDirty() const { return m_dirty; }
    void clearDirty()    { m_dirty = false; }

    // Called every frame by WindowManager; no-op when m_open is false.
    void render();

    // For tests — simulate the next keypress arriving while capturing.
    // glfwKey is a GLFW_KEY_* constant; ctrl/shift reflect live modifier
    // state at the moment of the press. Pass -1 for "Esc was pressed,
    // cancel capture".
    void injectCapture(int glfwKey, bool ctrlDown, bool shiftDown);

    // Enter capture mode for the action at the given enumerate() index.
    // Used by the [Remap] button and by tests. No-op if m_map is null.
    void beginCapture(int actionIndex);

    // Has the editor entered capture mode? (The currently-targeted row
    // index can be read with capturingAction(); -1 means idle.)
    bool isCapturing()       const { return m_capturing >= 0; }
    int  capturingAction()   const { return m_capturing; }

private:
    ::btquant::util::HotkeyMap* m_map = nullptr;
    bool   m_open    = false;
    bool   m_dirty   = false;
    int    m_capturing = -1;   // index into enumerate(), -1 = idle

    void applyDefault(int actionIndex);
    void cancelCapture()   { m_capturing = -1; }
    void setCapturing(int i) { m_capturing = i; }
};

} // namespace btquant::widgets

#endif
