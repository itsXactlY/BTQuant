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
    // Open / close the editor modal.
    void setOpen(bool v) { m_open = v; }
    bool isOpen() const   { return m_open; }

    // Render — call once per frame. Reads/writes the live ImGui style.
    void render();

    // POD snapshot of the active theme (32 RGBA colors + 4 floats). Captured
    // by WindowManager on accept and saved alongside the rest of Settings.
    struct Snapshot {
        std::array<float, 4> colors[48] = {};   // ImGui 1.90+ ImGuiCol_COUNT
        float windowPadding   = 8.0f;
        float framePadding    = 4.0f;
        float rounding        = 0.0f;
        float alpha           = 1.0f;
        bool  dark            = true;
    };

    static constexpr int kColorCount = 48;  // ImGui 1.90+ ImGuiCol_COUNT
    static const char* colorName(int i);

    // Apply a snapshot to the running ImGui style.
    static void applySnapshot(ImGuiStyle& dst, const Snapshot& s);

    // Capture current style into a snapshot.
    static Snapshot capture(const ImGuiStyle& src);

    // Compare snapshots — used by the test suite.
    static bool equals(const Snapshot& a, const Snapshot& b, float tol = 1e-4f);

private:
    bool m_open = false;
};

} // namespace btquant::ui

#endif
