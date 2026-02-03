#pragma once

#include <vector>
#include <chrono>

namespace RenderEngine {

/**
 * @brief Structure defining rendering quality settings
 */
struct QualitySettings {
    float render_resolution_scale = 1.0f;        ///< Scale factor for rendering resolution (0.1 to 1.0)
    bool enable_antialiasing = true;             ///< Enable antialiasing
    bool enable_shadows = true;                  ///< Enable shadow rendering
    bool enable_reflections = true;              ///< Enable reflection rendering
    bool enable_post_processing = true;          ///< Enable post-processing effects
    bool enable_detailed_textures = true;        ///< Enable detailed textures
    bool enable_smooth_animations = true;        ///< Enable smooth animations
    int max_visible_elements = 10000;            ///< Maximum number of visible elements
    bool enable_fxaa = true;                     ///< Enable FXAA antialiasing
    bool enable_smaa = true;                     ///< Enable SMAA antialiasing
    bool texture_mipmaps = true;                 ///< Enable texture mipmaps
    int anisotropic_filtering = 16;              ///< Anisotropic filtering level (1, 2, 4, 8, 16)
    float particle_density = 1.0f;               ///< Particle density multiplier
    float lighting_quality = 1.0f;               ///< Lighting quality multiplier (0.0 to 1.0)
    float shadow_quality = 1.0f;                 ///< Shadow quality multiplier (0.0 to 1.0)
    float reflection_quality = 1.0f;             ///< Reflection quality multiplier (0.0 to 1.0)
    float post_process_quality = 1.0f;           ///< Post-process quality multiplier (0.0 to 1.0)
    int msaa_samples = 4;                        ///< MSAA sample count (1, 2, 4, 8, 16)
    bool enable_motion_blur = true;              ///< Enable motion blur effects
    bool enable_bloom = true;                    ///< Enable bloom effects
    bool enable_depth_of_field = true;           ///< Enable depth of field
    int max_lights = 8;                          ///< Maximum number of lights
    int max_particles = 10000;                   ///< Maximum number of particles
    bool enable_dynamic_lod = true;              ///< Enable dynamic level of detail
    float lod_bias = 1.0f;                       ///< Level of detail bias (higher = more detail)
    bool enable_occlusion_culling = true;        ///< Enable occlusion culling
    bool enable_frustum_culling = true;          ///< Enable frustum culling
    float shadow_map_resolution = 2048.0f;       ///< Shadow map resolution
    int max_shadow_cascades = 4;                 ///< Maximum shadow cascades
    bool enable_ssao = true;                     ///< Enable screen space ambient occlusion

    // Additional granular controls for fine-tuned quality management
    bool enable_transparency_aa = true;          ///< Enable transparency anti-aliasing
    bool enable_hdr_rendering = true;            ///< Enable HDR rendering pipeline
    bool enable_variable_rate_shading = false;   ///< Enable variable rate shading (if supported)
    float ui_scaling_factor = 1.0f;              ///< UI scaling factor to reduce UI rendering load
    bool enable_texture_compression = true;      ///< Enable texture compression
    int max_animated_objects = 1000;             ///< Maximum number of animated objects
    bool enable_gpu_skinning = true;             ///< Enable GPU-based skinning
    float shadow_distance = 100.0f;              ///< Maximum distance for shadow rendering
    bool enable_contact_hardening = true;        ///< Enable contact hardening for shadows
    float tessellation_factor = 1.0f;            ///< Tessellation level factor
    bool enable_ray_tracing_effects = false;     ///< Enable ray tracing effects (if supported)
    int max_draw_calls_per_frame = 10000;        ///< Maximum draw calls per frame
    bool enable_instancing = true;               ///< Enable geometry instancing
    float max_texture_memory_mb = 1024.0f;       ///< Maximum texture memory allocation in MB
    bool enable_async_compute = true;            ///< Enable asynchronous compute operations
    int max_buffer_updates_per_frame = 1000;     ///< Maximum buffer updates per frame
};

/**
 * @brief Configuration for the auto-quality controller
 */
struct AutoQualityConfig {
    uint32_t target_fps = 60;                    ///< Target frames per second
    double performance_threshold = 80.0;         ///< Performance threshold (0-100%) to trigger quality adjustments
    double variance_threshold = 5.0;             ///< Threshold for frame time variance (ms)
    int adjustment_cooldown_ms = 5000;           ///< Minimum time between quality adjustments (ms)
    int stability_window_ms = 10000;             ///< Time window to consider before increasing quality (ms)
    bool enable_logging = true;                  ///< Enable logging of quality adjustments
};

/**
 * @brief Auto-Quality Controller implementation to automatically adjust rendering quality
 * based on performance metrics to maintain responsive frame rates.
 */
class AutoQualityController {
public:
    static constexpr int QUALITY_LEVEL_COUNT = 5;
    static constexpr size_t FRAME_HISTORY_SIZE = 60;  // Keep history of 60 frames

    explicit AutoQualityController(const AutoQualityConfig& config);
    ~AutoQualityController();

    /**
     * @brief Record a frame time measurement for performance analysis
     */
    void recordFrameTime(double frame_time_ms);

    /**
     * @brief Get the current quality settings based on performance
     */
    const QualitySettings& getCurrentQualitySettings() const;

    /**
     * @brief Get the current quality index (0 = highest, QUALITY_LEVEL_COUNT-1 = lowest)
     */
    int getCurrentQualityIndex() const;

    /**
     * @brief Get the current performance score (0-100%)
     */
    double getPerformanceScore() const;

    /**
     * @brief Reset the controller to initial state
     */
    void reset();

    /**
     * @brief Update configuration at runtime
     */
    void updateConfig(const AutoQualityConfig& new_config);

    /**
     * @brief Force a specific quality level (0 to QUALITY_LEVEL_COUNT-1)
     */
    void forceQualityLevel(int level);

    /**
     * @brief Get recommended rendering settings based on current quality level
     * This method provides the actual rendering parameters that should be used
     */
    QualitySettings getRecommendedRenderingSettings() const;

    /**
     * @brief Predict future performance based on current trends
     */
    double predictFuturePerformance() const;

    /**
     * @brief Calculate the rate of performance degradation
     */
    double calculateDegradationRate() const;

    /**
     * @brief Calculate memory pressure score (0-100, higher is better)
     */
    double calculateMemoryPressureScore() const;

    /**
     * @brief Calculate thermal pressure score (0-100, higher is better)
     */
    double calculateThermalPressureScore() const;

private:
    AutoQualityConfig config_;
    std::vector<double> frame_times_;
    std::vector<QualitySettings> quality_levels_;
    int current_quality_index_;
    uint64_t frame_count_ = 0;
    double performance_score_;
    double target_performance_threshold_;
    
    std::chrono::high_resolution_clock::time_point last_adjustment_time_;
    std::chrono::milliseconds adjustment_cooldown_;

    // Adaptive threshold variables
    double adaptive_performance_threshold_;
    double performance_history_[10];  // Track recent performance scores
    int performance_history_index_ = 0;
    bool performance_history_full_ = false;

    // Private methods
    void initializeQualityLevels();
    void updatePerformanceScore();
    double calculateFrameTimeVariance() const;
    double calculateStabilityScore(double variance) const;
    double calculateResponsivenessScore() const;
    double calculateTrendPredictionScore() const;
    double calculateAdaptiveThreshold() const;
    void updatePerformanceHistory();
    void checkAndAdjustQuality();
    bool hasBeenStableAtCurrentLevel() const;
    int determineQualityReduction();
    int determineQualityIncrease();
    void logPerformanceStats() const;

    // Advanced performance detection methods
    bool detectPerformanceSpikes() const;
    double calculateJankPercentage() const;
    double calculatePerformanceConsistency() const;
    void updateAdvancedMetrics();

    // Methods moved to public section above
};

} // namespace RenderEngine