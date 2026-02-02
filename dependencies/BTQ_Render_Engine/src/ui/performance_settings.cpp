#include "../../include/ui/performance_settings.hpp"
#include "../../include/ui/settings_manager.hpp"
#include "../../include/performance_monitor.hpp"
#include "../../include/dashboard_config.hpp"

#include <imgui.h>
#include <string>
#include <vector>

namespace BTQuant {
namespace UI {

PerformanceSettings::PerformanceSettings(SettingsManager& settings_manager) {
    initialize_performance_settings(settings_manager);
}

void PerformanceSettings::initialize_performance_settings(SettingsManager& settings_manager) {
    // FPS Limiter setting
    SettingInfo fps_limiter_setting;
    fps_limiter_setting.key = "performance.fps_limiter";
    fps_limiter_setting.display_name = "FPS Limiter";
    fps_limiter_setting.description = "Maximum frames per second (0 = unlimited)";
    fps_limiter_setting.type = SettingType::INTEGER;
    fps_limiter_setting.category = SettingCategory::PERFORMANCE;
    fps_limiter_setting.int_value = 60;
    fps_limiter_setting.min_int = 0;
    fps_limiter_setting.max_int = 240;
    settings_manager.register_setting(fps_limiter_setting);

    // V-Sync toggle setting
    SettingInfo vsync_setting;
    vsync_setting.key = "performance.vsync";
    vsync_setting.display_name = "Vertical Sync (V-Sync)";
    vsync_setting.description = "Synchronize frame rate with monitor refresh rate to prevent screen tearing";
    vsync_setting.type = SettingType::BOOLEAN;
    vsync_setting.category = SettingCategory::PERFORMANCE;
    vsync_setting.bool_value = true;
    settings_manager.register_setting(vsync_setting);

    // LOD Distance Threshold - Low setting
    SettingInfo lod_distance_low_setting;
    lod_distance_low_setting.key = "performance.lod_distance_low";
    lod_distance_low_setting.display_name = "LOD Distance - Low Quality";
    lod_distance_low_setting.description = "Distance threshold for low quality level of detail";
    lod_distance_low_setting.type = SettingType::INTEGER;
    lod_distance_low_setting.category = SettingCategory::PERFORMANCE;
    lod_distance_low_setting.int_value = 100;
    lod_distance_low_setting.min_int = 10;
    lod_distance_low_setting.max_int = 500;
    settings_manager.register_setting(lod_distance_low_setting);

    // LOD Distance Threshold - Medium setting
    SettingInfo lod_distance_medium_setting;
    lod_distance_medium_setting.key = "performance.lod_distance_medium";
    lod_distance_medium_setting.display_name = "LOD Distance - Medium Quality";
    lod_distance_medium_setting.description = "Distance threshold for medium quality level of detail";
    lod_distance_medium_setting.type = SettingType::INTEGER;
    lod_distance_medium_setting.category = SettingCategory::PERFORMANCE;
    lod_distance_medium_setting.int_value = 50;
    lod_distance_medium_setting.min_int = 5;
    lod_distance_medium_setting.max_int = 200;
    settings_manager.register_setting(lod_distance_medium_setting);

    // LOD Complexity Threshold setting
    SettingInfo lod_complexity_setting;
    lod_complexity_setting.key = "performance.lod_complexity_threshold";
    lod_complexity_setting.display_name = "LOD Complexity Threshold";
    lod_complexity_setting.description = "Threshold for determining when to reduce complexity of objects";
    lod_complexity_setting.type = SettingType::INTEGER;
    lod_complexity_setting.category = SettingCategory::PERFORMANCE;
    lod_complexity_setting.int_value = 75;
    lod_complexity_setting.min_int = 1;
    lod_complexity_setting.max_int = 100;
    settings_manager.register_setting(lod_complexity_setting);

    // Caching Strategy setting
    SettingInfo caching_strategy_setting;
    caching_strategy_setting.key = "performance.caching_strategy";
    caching_strategy_setting.display_name = "Caching Strategy";
    caching_strategy_setting.description = "Strategy for caching rendered elements and data";
    caching_strategy_setting.type = SettingType::ENUM;
    caching_strategy_setting.category = SettingCategory::PERFORMANCE;
    caching_strategy_setting.enum_options = {"Aggressive", "Balanced", "Conservative", "Off"};
    caching_strategy_setting.enum_selected_index = 1; // Balanced by default
    settings_manager.register_setting(caching_strategy_setting);

    // Memory Limit setting
    SettingInfo memory_limit_setting;
    memory_limit_setting.key = "performance.memory_limit_mb";
    memory_limit_setting.display_name = "Memory Limit (MB)";
    memory_limit_setting.description = "Maximum memory usage for performance-critical operations";
    memory_limit_setting.type = SettingType::INTEGER;
    memory_limit_setting.category = SettingCategory::PERFORMANCE;
    memory_limit_setting.int_value = 2048;
    memory_limit_setting.min_int = 512;
    memory_limit_setting.max_int = 8192;
    settings_manager.register_setting(memory_limit_setting);

    // Texture Streaming setting
    SettingInfo texture_streaming_setting;
    texture_streaming_setting.key = "performance.texture_streaming";
    texture_streaming_setting.display_name = "Texture Streaming";
    texture_streaming_setting.description = "Stream textures based on visibility and importance";
    texture_streaming_setting.type = SettingType::BOOLEAN;
    texture_streaming_setting.category = SettingCategory::PERFORMANCE;
    texture_streaming_setting.bool_value = true;
    settings_manager.register_setting(texture_streaming_setting);

    // Dynamic Batching setting
    SettingInfo dynamic_batching_setting;
    dynamic_batching_setting.key = "performance.dynamic_batching";
    dynamic_batching_setting.display_name = "Dynamic Batching";
    dynamic_batching_setting.description = "Combine similar objects for more efficient rendering";
    dynamic_batching_setting.type = SettingType::BOOLEAN;
    dynamic_batching_setting.category = SettingCategory::PERFORMANCE;
    dynamic_batching_setting.bool_value = true;
    settings_manager.register_setting(dynamic_batching_setting);

    // Shader Quality setting
    SettingInfo shader_quality_setting;
    shader_quality_setting.key = "performance.shader_quality";
    shader_quality_setting.display_name = "Shader Quality";
    shader_quality_setting.description = "Quality level for rendering shaders";
    shader_quality_setting.type = SettingType::ENUM;
    shader_quality_setting.category = SettingCategory::PERFORMANCE;
    shader_quality_setting.enum_options = {"Low", "Medium", "High", "Ultra"};
    shader_quality_setting.enum_selected_index = 2; // High by default
    settings_manager.register_setting(shader_quality_setting);
}

void PerformanceSettings::apply_settings(SettingsManager& settings_manager) {
    // Apply FPS limiter
    int fps_limit = settings_manager.get_int("performance.fps_limiter", 60);
    // In a real implementation, this would update the rendering loop
    
    // Apply V-Sync setting
    bool vsync_enabled = settings_manager.get_bool("performance.vsync", true);
    // In a real implementation, this would update the graphics API settings
    
    // Apply LOD settings
    int lod_low_distance = settings_manager.get_int("performance.lod_distance_low", 100);
    int lod_medium_distance = settings_manager.get_int("performance.lod_distance_medium", 50);
    int lod_complexity_threshold = settings_manager.get_int("performance.lod_complexity_threshold", 75);
    
    // Apply caching strategy
    int caching_strategy = settings_manager.get_enum("performance.caching_strategy", 1);
    
    // Apply memory limit
    int memory_limit = settings_manager.get_int("performance.memory_limit_mb", 2048);
    
    // Apply texture streaming
    bool texture_streaming = settings_manager.get_bool("performance.texture_streaming", true);
    
    // Apply dynamic batching
    bool dynamic_batching = settings_manager.get_bool("performance.dynamic_batching", true);
    
    // Apply shader quality
    int shader_quality = settings_manager.get_enum("performance.shader_quality", 2);
    
    // In a real implementation, these values would be applied to the appropriate systems
    // For now, we just store them or pass them to the relevant subsystems
}

void PerformanceSettings::render_performance_settings(SettingsManager& settings_manager) {
    // This method could be used to render specific performance settings UI
    // For now, the general settings UI in SettingsManager handles this
    // This method is kept for future expansion if specific performance UI is needed
}

} // namespace UI
} // namespace BTQuant