#pragma once

#include <string>

namespace BTQuant {
namespace UI {

class SettingsManager;

class PerformanceSettings {
public:
    explicit PerformanceSettings(SettingsManager& settings_manager);

    // Initialize performance-related settings
    void initialize_performance_settings(SettingsManager& settings_manager);

    // Apply current performance settings to the system
    void apply_settings(SettingsManager& settings_manager);

    // Render performance-specific settings UI (if needed)
    void render_performance_settings(SettingsManager& settings_manager);

private:
    // Individual setting update methods
    void update_fps_limiter(int fps_limit);
    void update_vsync(bool enabled);
    void update_lod_settings(int low_distance, int medium_distance, int complexity_threshold);
    void update_caching_strategy(int strategy);
    void update_memory_limits(int limit_mb);
    void update_texture_streaming(bool enabled);
    void update_dynamic_batching(bool enabled);
    void update_shader_quality(int quality_level);
};

} // namespace UI
} // namespace BTQuant