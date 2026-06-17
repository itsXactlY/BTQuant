#pragma once

// ============================================================================
// Crosshair sync helper (TASK_ULTIMA_MMT_GENESIS_WIRED.md Phase 7.1)
// ============================================================================
// Atomic crosshair state is declared in <sync/global_state.hpp>. This header
// adds the small helper functions that the panels actually call:
//   - write_crosshair()       — called from a panel's mouse-hover handler
//   - render_dashed_hline()   — called from any panel that wants to draw the
//                                shared crosshair line on its own ImGui window
// The functions are header-only / inline; including this in a panel .cpp is
// enough to wire it up. No new build target needed.
// ============================================================================

#include <cstdint>
#include <algorithm>
#include <atomic>

#include "sync/global_state.hpp"

#include <imgui.h>

namespace BTQuant {

// Store the current mouse-hover price. Pass active_symbol_id so that only
// panels on the same symbol honour the crosshair (multi-chart dashboards).
inline void write_crosshair(double price, int32_t symbol_id) noexcept {
    g_crosshair_price.store(price, std::memory_order_release);
    g_crosshair_symbol_id.store(symbol_id, std::memory_order_release);
}

// Clear the crosshair (e.g. when the mouse leaves the panel).
inline void clear_crosshair() noexcept {
    g_crosshair_price.store(0.0, std::memory_order_release);
    g_crosshair_symbol_id.store(-1, std::memory_order_release);
}

// Returns true if a crosshair is active for this symbol. Caller then queries
// load_crosshair_price() and draws accordingly.
inline bool crosshair_active_for(int32_t active_symbol_id) noexcept {
    return g_crosshair_symbol_id.load(std::memory_order_acquire) == active_symbol_id
        && g_crosshair_price.load(std::memory_order_acquire) > 0.0;
}

// Draw a 1px dashed horizontal line spanning the panel's drawable area at
// the given screen Y coordinate. Caller is responsible for converting the
// crosshair price to a Y pixel using ChartMath::MapToScreen().
inline void render_dashed_hline(ImDrawList* dl,
                                float x_min, float x_max, float y,
                                ImU32 col = IM_COL32(255, 255, 255, 120),
                                float dash_px = 6.0f) noexcept {
    if (!dl) return;
    for (float x = x_min; x < x_max; x += dash_px * 2.0f) {
        float x2 = std::min(x + dash_px, x_max);
        dl->AddLine(ImVec2(x, y), ImVec2(x2, y), col, 1.0f);
    }
}

// Convenience: read the crosshair price, returning 0.0 if inactive for symbol.
inline double load_crosshair_price(int32_t active_symbol_id) noexcept {
    if (crosshair_active_for(active_symbol_id)) {
        return g_crosshair_price.load(std::memory_order_acquire);
    }
    return 0.0;
}

} // namespace BTQuant
