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

    // Private methods
    void initializeQualityLevels();
    void updatePerformanceScore();
    double calculateFrameTimeVariance() const;
    double calculateStabilityScore(double variance) const;
    double calculateResponsivenessScore() const;
    void checkAndAdjustQuality();
    bool hasBeenStableAtCurrentLevel() const;
    int determineQualityReduction();
    int determineQualityIncrease();
    void logPerformanceStats() const;
};

} // namespace RenderEngine