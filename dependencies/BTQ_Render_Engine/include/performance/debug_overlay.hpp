/**
 * Debug Overlay Header
 *
 * Defines the interface for the performance debug overlay
 */

#pragma once

#include <cstdint>
#include <string>

namespace BTQuant {

class DebugOverlay {
public:
    DebugOverlay();

    // Toggle visibility of the debug overlay
    void toggle_visibility();

    // Set/get visibility
    void set_visible(bool visible);
    bool is_visible() const;

    // Update position of the overlay
    void update_position(float x, float y);

    // Set refresh rate (updates per second)
    void set_refresh_rate(float hz);
    float get_refresh_rate() const;

    // Render the debug overlay
    void render();

    // Set active component counts
    void set_active_panels_count(size_t count);
    void set_active_indicators_count(size_t count);
    void set_active_alerts_count(size_t count);

    // Set renderer stats
    void set_renderer_stats(uint32_t frames_rendered, uint32_t lob_updates,
                          uint32_t trade_updates, uint32_t footprint_cells_rendered);

private:
    // Format bytes to human-readable string
    std::string format_bytes(size_t bytes) const;
    std::string format_large_number(size_t num) const;

private:
    bool visible_;
    float position_x_;
    float position_y_;
    float window_width_;
    float window_height_;
    float refresh_rate_;

    // Active component counts
    size_t active_panels_count_ = 0;
    size_t active_indicators_count_ = 0;
    size_t active_alerts_count_ = 0;

    // Renderer stats
    uint32_t frames_rendered_ = 0;
    uint32_t lob_updates_ = 0;
    uint32_t trade_updates_ = 0;
    uint32_t footprint_cells_rendered_ = 0;
};

// Global debug overlay instance
extern DebugOverlay g_debug_overlay;

}  // namespace BTQuant