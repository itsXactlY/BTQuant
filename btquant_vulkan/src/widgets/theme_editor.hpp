#ifndef BTQUANT_THEME_EDITOR_HPP
#define BTQUANT_THEME_EDITOR_HPP

#include <array>
#include <cstdint>

struct ImGuiStyle;

namespace btquant::ui {

// Live theme editor — exposes the 32 ImGui colors and the core style knobs
// (window padding, frame padding, rounding, alpha) via color pickers and
// sliders. Changes apply immediately on the running UI; on accept, the
// style is snapshotted into a small POD that the WindowManager persists.
class ThemeEditor {
public:
    // POD snapshot of the active theme (48 RGBA colors + 4 floats). Captured
    // by WindowManager on accept and saved alongside the rest of Settings.
    // Defined before the methods that reference it (see setOpen).
    struct Snapshot {
        std::array<float, 4> colors[48] = {};   // ImGui 1.90+ ImGuiCol_COUNT
        float windowPadding   = 8.0f;
        float framePadding    = 4.0f;
        float rounding        = 0.0f;
        float alpha           = 1.0f;
        bool  dark            = true;
    };
    static constexpr int kColorCount = 48;  // ImGui 1.90+ ImGuiCol_COUNT

    // Open / close the editor modal.
    void setOpen(bool v) {
        if (v && !m_open) {
            // First frame of opening — snapshot the style so "Discard changes"
            // can restore exactly what the trader had before editing. Lazy
            // capture (deferred to render()) so callers without an active
            // ImGui context can still flip the open flag.
            m_captured = false;
        }
        m_open = v;
    }
    bool isOpen() const   { return m_open; }

    // Render — call once per frame. Reads/writes the live ImGui style.
    void render();

    // True if the live style differs from the snapshot captured at open.
    // False when no ImGui context is alive or the editor isn't open.
    bool hasUnsavedChanges() const;

    // Restore the opening snapshot back into the live style. Returns true
    // on success (also returns true if the style was already at the opening
    // snapshot, since the call is idempotent). No-op when no context.
    bool discardChanges();

    static const char* colorName(int i);

    // Apply a snapshot to the running ImGui style.
    static void applySnapshot(ImGuiStyle& dst, const Snapshot& s);

    // Capture current style into a snapshot.
    static Snapshot capture(const ImGuiStyle& src);

    // Compare snapshots — used by the test suite.
    static bool equals(const Snapshot& a, const Snapshot& b, float tol = 1e-4f);

private:
    bool     m_open = false;
    bool     m_captured = false;        // true after first render-frame snapshot
    Snapshot m_openingSnapshot{};       // captured on first render frame
};

} // namespace btquant::ui

#endif
