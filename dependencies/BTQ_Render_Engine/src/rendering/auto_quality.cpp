/**
 * BTQuant Auto-Quality Reduction System
 *
 * Advanced system to detect performance drops and automatically reduce visual quality
 * to maintain responsiveness in professional trading dashboard applications.
 */

#include "rendering/auto_quality.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace RenderEngine {

AutoQualityController::AutoQualityController(const AutoQualityConfig& config)
    : config_(config)
    , frame_times_(FRAME_HISTORY_SIZE, 1000.0 / config.target_fps)
    , quality_levels_(QUALITY_LEVEL_COUNT)
    , current_quality_index_(0)  // Start at highest quality
    , last_adjustment_time_(std::chrono::high_resolution_clock::now())
    , adjustment_cooldown_(std::chrono::milliseconds(static_cast<int>(config.adjustment_cooldown_ms)))
    , performance_score_(100.0)
    , target_performance_threshold_(config.performance_threshold)
{
    initializeQualityLevels();
}

AutoQualityController::~AutoQualityController() = default;

void AutoQualityController::initializeQualityLevels() {
    // Define quality levels from highest to lowest
    // Each level represents progressively reduced visual fidelity

    // Highest Quality (index 0)
    quality_levels_[0] = {
        .render_resolution_scale = 1.0f,
        .enable_antialiasing = true,
        .enable_shadows = true,
        .enable_reflections = true,
        .enable_post_processing = true,
        .enable_detailed_textures = true,
        .enable_smooth_animations = true,
        .max_visible_elements = 10000,
        .enable_fxaa = true,
        .enable_smaa = true,
        .texture_mipmaps = true,
        .anisotropic_filtering = 16,
        .particle_density = 1.0f,
        .lighting_quality = 1.0f,
        .shadow_quality = 1.0f,
        .reflection_quality = 1.0f,
        .post_process_quality = 1.0f,
        .msaa_samples = 4,
        .enable_motion_blur = true,
        .enable_bloom = true,
        .enable_depth_of_field = true,
        .max_lights = 8,
        .max_particles = 10000,
        .enable_dynamic_lod = true,
        .lod_bias = 1.0f,
        .enable_occlusion_culling = true,
        .enable_frustum_culling = true,
        .shadow_map_resolution = 2048.0f,
        .max_shadow_cascades = 4,
        .enable_ssao = true
    };

    // High Quality (index 1)
    quality_levels_[1] = {
        .render_resolution_scale = 0.9f,
        .enable_antialiasing = true,
        .enable_shadows = true,
        .enable_reflections = true,
        .enable_post_processing = true,
        .enable_detailed_textures = true,
        .enable_smooth_animations = true,
        .max_visible_elements = 8000,
        .enable_fxaa = true,
        .enable_smaa = false,
        .texture_mipmaps = true,
        .anisotropic_filtering = 8,
        .particle_density = 0.8f,
        .lighting_quality = 0.9f,
        .shadow_quality = 0.9f,
        .reflection_quality = 0.9f,
        .post_process_quality = 0.9f,
        .msaa_samples = 2,
        .enable_motion_blur = true,
        .enable_bloom = true,
        .enable_depth_of_field = false,
        .max_lights = 6,
        .max_particles = 8000,
        .enable_dynamic_lod = true,
        .lod_bias = 0.9f,
        .enable_occlusion_culling = true,
        .enable_frustum_culling = true,
        .shadow_map_resolution = 1024.0f,
        .max_shadow_cascades = 3,
        .enable_ssao = true
    };

    // Medium Quality (index 2)
    quality_levels_[2] = {
        .render_resolution_scale = 0.8f,
        .enable_antialiasing = true,
        .enable_shadows = true,
        .enable_reflections = false,
        .enable_post_processing = true,
        .enable_detailed_textures = true,
        .enable_smooth_animations = false,
        .max_visible_elements = 6000,
        .enable_fxaa = true,
        .enable_smaa = false,
        .texture_mipmaps = true,
        .anisotropic_filtering = 4,
        .particle_density = 0.6f,
        .lighting_quality = 0.7f,
        .shadow_quality = 0.7f,
        .reflection_quality = 0.0f,
        .post_process_quality = 0.7f,
        .msaa_samples = 2,
        .enable_motion_blur = false,
        .enable_bloom = true,
        .enable_depth_of_field = false,
        .max_lights = 4,
        .max_particles = 6000,
        .enable_dynamic_lod = false,
        .lod_bias = 0.8f,
        .enable_occlusion_culling = true,
        .enable_frustum_culling = true,
        .shadow_map_resolution = 512.0f,
        .max_shadow_cascades = 2,
        .enable_ssao = false
    };

    // Low Quality (index 3)
    quality_levels_[3] = {
        .render_resolution_scale = 0.7f,
        .enable_antialiasing = false,
        .enable_shadows = false,
        .enable_reflections = false,
        .enable_post_processing = false,
        .enable_detailed_textures = false,
        .enable_smooth_animations = false,
        .max_visible_elements = 4000,
        .enable_fxaa = false,
        .enable_smaa = false,
        .texture_mipmaps = false,
        .anisotropic_filtering = 1,
        .particle_density = 0.4f,
        .lighting_quality = 0.5f,
        .shadow_quality = 0.3f,
        .reflection_quality = 0.0f,
        .post_process_quality = 0.3f,
        .msaa_samples = 1,
        .enable_motion_blur = false,
        .enable_bloom = false,
        .enable_depth_of_field = false,
        .max_lights = 2,
        .max_particles = 4000,
        .enable_dynamic_lod = false,
        .lod_bias = 0.6f,
        .enable_occlusion_culling = false,
        .enable_frustum_culling = true,
        .shadow_map_resolution = 256.0f,
        .max_shadow_cascades = 1,
        .enable_ssao = false
    };

    // Lowest Quality (index 4)
    quality_levels_[4] = {
        .render_resolution_scale = 0.6f,
        .enable_antialiasing = false,
        .enable_shadows = false,
        .enable_reflections = false,
        .enable_post_processing = false,
        .enable_detailed_textures = false,
        .enable_smooth_animations = false,
        .max_visible_elements = 2000,
        .enable_fxaa = false,
        .enable_smaa = false,
        .texture_mipmaps = false,
        .anisotropic_filtering = 1,
        .particle_density = 0.2f,
        .lighting_quality = 0.2f,
        .shadow_quality = 0.1f,
        .reflection_quality = 0.0f,
        .post_process_quality = 0.1f,
        .msaa_samples = 1,
        .enable_motion_blur = false,
        .enable_bloom = false,
        .enable_depth_of_field = false,
        .max_lights = 1,
        .max_particles = 2000,
        .enable_dynamic_lod = false,
        .lod_bias = 0.4f,
        .enable_occlusion_culling = false,
        .enable_frustum_culling = true,
        .shadow_map_resolution = 128.0f,
        .max_shadow_cascades = 1,
        .enable_ssao = false
    };
}

void AutoQualityController::recordFrameTime(double frame_time_ms) {
    // Add frame time to history
    frame_times_[frame_count_ % FRAME_HISTORY_SIZE] = frame_time_ms;
    frame_count_++;

    // Calculate current performance score based on frame times
    updatePerformanceScore();

    // Check if we need to adjust quality
    checkAndAdjustQuality();

    // Log performance stats periodically
    if (config_.enable_logging && frame_count_ % 60 == 0) {  // Every 60 frames
        logPerformanceStats();
    }
}

void AutoQualityController::logPerformanceStats() const {
    if (!config_.enable_logging) return;

    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count == 0) return;

    // Calculate min, max, and average frame times
    double min_frame_time = *std::min_element(frame_times_.begin(), frame_times_.begin() + sample_count);
    double max_frame_time = *std::max_element(frame_times_.begin(), frame_times_.begin() + sample_count);

    double sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        sum += frame_times_[i];
    }
    double avg_frame_time = sum / sample_count;

    // Convert to FPS
    double avg_fps = avg_frame_time > 0 ? 1000.0 / avg_frame_time : 0;
    double min_fps = min_frame_time > 0 ? 1000.0 / min_frame_time : 0;
    double max_fps = max_frame_time > 0 ? 1000.0 / max_frame_time : 0;

    printf("AutoQuality Stats - FPS: Avg=%.1f Min=%.1f Max=%.1f | Quality: %d/%d | Perf Score: %.1f%%\n",
           avg_fps, min_fps, max_fps, current_quality_index_, QUALITY_LEVEL_COUNT - 1, performance_score_);
}

void AutoQualityController::updatePerformanceScore() {
    if (frame_count_ < 10) {
        // Not enough data yet, keep current score
        return;
    }

    // Calculate average frame time over recent history
    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    double sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        sum += frame_times_[i];
    }
    double avg_frame_time = sum / sample_count;

    // Calculate target frame time
    double target_frame_time = 1000.0 / config_.target_fps;

    // Calculate performance score based on frame time efficiency
    double fps_score;
    if (avg_frame_time <= target_frame_time) {
        // Perfect or better than target performance
        fps_score = 100.0;
    } else {
        // Performance degrades as frame time increases
        double ratio = target_frame_time / avg_frame_time;
        fps_score = std::max(0.0, ratio * 100.0);
    }

    // Calculate frame time variance
    double variance = calculateFrameTimeVariance();
    double stability_score = calculateStabilityScore(variance);

    // Calculate responsiveness score based on worst frame times
    double responsiveness_score = calculateResponsivenessScore();

    // Weighted combination of all performance factors
    // FPS efficiency: 50%, Stability: 30%, Responsiveness: 20%
    performance_score_ = (fps_score * 0.5) + (stability_score * 0.3) + (responsiveness_score * 0.2);
}

double AutoQualityController::calculateStabilityScore(double variance) const {
    // Calculate stability score based on frame time variance
    // Lower variance means higher stability
    double normalized_variance = variance / config_.variance_threshold;
    double stability_penalty = std::min(normalized_variance * 20.0, 50.0); // Cap penalty at 50 points
    return std::max(0.0, 100.0 - stability_penalty);
}

double AutoQualityController::calculateResponsivenessScore() const {
    // Calculate score based on worst frame times (jank detection)
    // Find the maximum frame time in the recent history
    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count == 0) return 100.0;

    double max_frame_time = *std::max_element(frame_times_.begin(),
                                             frame_times_.begin() + sample_count);

    // Calculate how much worse the worst frame is compared to target
    double target_frame_time = 1000.0 / config_.target_fps;
    double worst_ratio = target_frame_time / max_frame_time;

    // Responsiveness score based on worst-case performance
    double responsiveness_score = std::min(100.0, worst_ratio * 100.0);

    // Also penalize if too many frames exceed the target significantly
    int poor_frames = 0;
    for (size_t i = 0; i < sample_count; ++i) {
        if (frame_times_[i] > target_frame_time * 2.0) { // More than 2x target time
            poor_frames++;
        }
    }

    double poor_frame_ratio = static_cast<double>(poor_frames) / sample_count;
    double poor_frame_penalty = poor_frame_ratio * 50.0; // Up to 50 point penalty

    return std::max(0.0, responsiveness_score - poor_frame_penalty);
}

double AutoQualityController::calculateFrameTimeVariance() const {
    if (frame_count_ < 2) {
        return 0.0;
    }
    
    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count < 2) {
        return 0.0;
    }
    
    // Calculate mean
    double sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        sum += frame_times_[i];
    }
    double mean = sum / sample_count;
    
    // Calculate variance
    double variance_sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        double diff = frame_times_[i] - mean;
        variance_sum += diff * diff;
    }
    return variance_sum / sample_count;
}

void AutoQualityController::checkAndAdjustQuality() {
    auto now = std::chrono::high_resolution_clock::now();

    // Check if we're past the cooldown period
    if (now - last_adjustment_time_ < adjustment_cooldown_) {
        return;
    }

    // Use hysteresis to prevent oscillation
    // Different thresholds for reducing vs increasing quality
    double lower_threshold = target_performance_threshold_ * (0.9 - (current_quality_index_ * 0.02)); // Lower threshold becomes stricter at lower quality levels
    double upper_threshold = target_performance_threshold_ * (1.1 + (current_quality_index_ * 0.03)); // Upper threshold becomes more lenient at lower quality levels

    // Determine if we need to adjust quality based on performance
    if (performance_score_ < lower_threshold) {
        // Performance is below threshold, reduce quality
        if (current_quality_index_ < QUALITY_LEVEL_COUNT - 1) {
            // Adjust quality gradually based on performance drop severity
            int new_quality_index = determineQualityReduction();

            if (new_quality_index != current_quality_index_) {
                current_quality_index_ = new_quality_index;
                last_adjustment_time_ = now;

                // Log the quality reduction
                if (config_.enable_logging) {
                    printf("AutoQuality: Reduced quality to level %d (Performance: %.2f%%, Lower Threshold: %.2f%%)\n",
                           current_quality_index_, performance_score_, lower_threshold);
                }
            }
        }
    } else if (performance_score_ > upper_threshold && current_quality_index_ > 0) {
        // Performance is above threshold, try to increase quality
        // But only if we've been stable at current level for a while
        if (hasBeenStableAtCurrentLevel()) {
            // Gradually increase quality if performance is consistently good
            int new_quality_index = determineQualityIncrease();

            if (new_quality_index != current_quality_index_) {
                current_quality_index_ = new_quality_index;
                last_adjustment_time_ = now;

                // Log the quality increase
                if (config_.enable_logging) {
                    printf("AutoQuality: Increased quality to level %d (Performance: %.2f%%, Upper Threshold: %.2f%%)\n",
                           current_quality_index_, performance_score_, upper_threshold);
                }
            }
        }
    }
}

int AutoQualityController::determineQualityReduction() {
    // Determine how much to reduce quality based on performance severity
    double performance_deficit = target_performance_threshold_ - performance_score_;

    // If performance is extremely poor, reduce quality more aggressively
    if (performance_deficit > 40.0) {
        // Very poor performance - jump 2 levels down if possible
        return std::min(current_quality_index_ + 2, QUALITY_LEVEL_COUNT - 1);
    } else if (performance_deficit > 20.0) {
        // Poor performance - jump 1-2 levels down
        return std::min(current_quality_index_ + 1, QUALITY_LEVEL_COUNT - 1);
    } else {
        // Moderate performance issues - reduce by 1 level
        return std::min(current_quality_index_ + 1, QUALITY_LEVEL_COUNT - 1);
    }
}

int AutoQualityController::determineQualityIncrease() {
    // Determine how much to increase quality based on performance surplus
    // Only increase quality gradually to avoid oscillation
    double performance_surplus = performance_score_ - target_performance_threshold_;

    if (performance_surplus > 30.0) {
        // Excellent performance - could increase by 2 levels if stable
        return std::max(current_quality_index_ - 2, 0);
    } else {
        // Good performance - increase by 1 level
        return std::max(current_quality_index_ - 1, 0);
    }
}

bool AutoQualityController::hasBeenStableAtCurrentLevel() const {
    // Check if we've been at the current quality level for a sufficient time
    // This prevents rapid oscillation between quality levels
    auto now = std::chrono::high_resolution_clock::now();
    auto time_at_current_level = std::chrono::duration<double, std::milli>(
        now - last_adjustment_time_).count();
    
    return time_at_current_level > config_.stability_window_ms;
}

const QualitySettings& AutoQualityController::getCurrentQualitySettings() const {
    return quality_levels_[current_quality_index_];
}

int AutoQualityController::getCurrentQualityIndex() const {
    return current_quality_index_;
}

double AutoQualityController::getPerformanceScore() const {
    return performance_score_;
}

void AutoQualityController::reset() {
    current_quality_index_ = 0;  // Start at highest quality
    performance_score_ = 100.0;
    frame_count_ = 0;
    last_adjustment_time_ = std::chrono::high_resolution_clock::now();

    // Reset frame time history
    std::fill(frame_times_.begin(), frame_times_.end(), 1000.0 / config_.target_fps);
}

void AutoQualityController::updateConfig(const AutoQualityConfig& new_config) {
    config_ = new_config;
    target_performance_threshold_ = new_config.performance_threshold;
    adjustment_cooldown_ = std::chrono::milliseconds(
        static_cast<int>(new_config.adjustment_cooldown_ms));
    
    // Reinitialize quality levels if needed
    initializeQualityLevels();
}

void AutoQualityController::forceQualityLevel(int level) {
    if (level >= 0 && level < QUALITY_LEVEL_COUNT) {
        current_quality_index_ = level;
        last_adjustment_time_ = std::chrono::high_resolution_clock::now();
        
        if (config_.enable_logging) {
            printf("AutoQuality: Forced quality to level %d\n", level);
        }
    }
}

} // namespace RenderEngine