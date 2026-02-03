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
#include <fstream>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdlib>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
#endif

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
    , adaptive_performance_threshold_(config.performance_threshold)
    , peak_performance_score_(100.0)
    , cumulative_performance_score_(0.0)
    , performance_sample_count_(0)
{
    initializeQualityLevels();

    // Initialize performance history
    for (int i = 0; i < 10; ++i) {
        performance_history_[i] = config.performance_threshold;
    }

    // Initialize recent performance trend buffer
    for (int i = 0; i < 30; ++i) {
        recent_performance_trend_[i] = config.performance_threshold;
    }
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
        .enable_ssao = true,
        .enable_transparency_aa = true,
        .enable_hdr_rendering = true,
        .enable_variable_rate_shading = true,
        .ui_scaling_factor = 1.0f,
        .enable_texture_compression = true,
        .max_animated_objects = 1000,
        .enable_gpu_skinning = true,
        .shadow_distance = 100.0f,
        .enable_contact_hardening = true,
        .tessellation_factor = 1.0f,
        .enable_ray_tracing_effects = true,
        .max_draw_calls_per_frame = 10000,
        .enable_instancing = true,
        .max_texture_memory_mb = 1024.0f,
        .enable_async_compute = true,
        .max_buffer_updates_per_frame = 1000
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
        .enable_ssao = true,
        .enable_transparency_aa = true,
        .enable_hdr_rendering = true,
        .enable_variable_rate_shading = false,
        .ui_scaling_factor = 0.95f,
        .enable_texture_compression = true,
        .max_animated_objects = 800,
        .enable_gpu_skinning = true,
        .shadow_distance = 80.0f,
        .enable_contact_hardening = true,
        .tessellation_factor = 0.9f,
        .enable_ray_tracing_effects = false,
        .max_draw_calls_per_frame = 8000,
        .enable_instancing = true,
        .max_texture_memory_mb = 800.0f,
        .enable_async_compute = true,
        .max_buffer_updates_per_frame = 800
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
        .enable_ssao = false,
        .enable_transparency_aa = false,
        .enable_hdr_rendering = true,
        .enable_variable_rate_shading = false,
        .ui_scaling_factor = 0.9f,
        .enable_texture_compression = true,
        .max_animated_objects = 600,
        .enable_gpu_skinning = false,
        .shadow_distance = 60.0f,
        .enable_contact_hardening = false,
        .tessellation_factor = 0.7f,
        .enable_ray_tracing_effects = false,
        .max_draw_calls_per_frame = 6000,
        .enable_instancing = true,
        .max_texture_memory_mb = 600.0f,
        .enable_async_compute = false,
        .max_buffer_updates_per_frame = 600
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
        .enable_ssao = false,
        .enable_transparency_aa = false,
        .enable_hdr_rendering = false,
        .enable_variable_rate_shading = false,
        .ui_scaling_factor = 0.85f,
        .enable_texture_compression = true,
        .max_animated_objects = 400,
        .enable_gpu_skinning = false,
        .shadow_distance = 40.0f,
        .enable_contact_hardening = false,
        .tessellation_factor = 0.5f,
        .enable_ray_tracing_effects = false,
        .max_draw_calls_per_frame = 4000,
        .enable_instancing = false,
        .max_texture_memory_mb = 400.0f,
        .enable_async_compute = false,
        .max_buffer_updates_per_frame = 400
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
        .enable_ssao = false,
        .enable_transparency_aa = false,
        .enable_hdr_rendering = false,
        .enable_variable_rate_shading = false,
        .ui_scaling_factor = 0.8f,
        .enable_texture_compression = true,
        .max_animated_objects = 200,
        .enable_gpu_skinning = false,
        .shadow_distance = 20.0f,
        .enable_contact_hardening = false,
        .tessellation_factor = 0.3f,
        .enable_ray_tracing_effects = false,
        .max_draw_calls_per_frame = 2000,
        .enable_instancing = false,
        .max_texture_memory_mb = 200.0f,
        .enable_async_compute = false,
        .max_buffer_updates_per_frame = 200
    };
}

void AutoQualityController::recordFrameTime(double frame_time_ms) {
    // Add frame time to history
    frame_times_[frame_count_ % FRAME_HISTORY_SIZE] = frame_time_ms;
    frame_count_++;

    // Calculate current performance score based on frame times
    updatePerformanceScore();

    // Update advanced performance metrics
    updateAdvancedMetrics();

    // Update real-time performance metrics
    updateRealTimePerformanceMetrics();

    // Update performance history for adaptive thresholds
    updatePerformanceHistory();

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

    // Calculate trend-based prediction score to anticipate performance drops
    double trend_prediction_score = calculateTrendPredictionScore();

    // Weighted combination of all performance factors
    // FPS efficiency: 40%, Stability: 25%, Responsiveness: 20%, Trend Prediction: 15%
    performance_score_ = (fps_score * 0.4) + (stability_score * 0.25) + (responsiveness_score * 0.2) + (trend_prediction_score * 0.15);
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

double AutoQualityController::calculateTrendPredictionScore() const {
    // Calculate a score based on the trend of frame times to predict future performance
    if (frame_count_ < 10) {
        return 100.0; // Not enough data, assume good performance
    }

    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count < 4) {
        return 100.0; // Need at least 4 samples to calculate trend
    }

    // Use the most recent half of the frame times to calculate trend
    size_t recent_count = sample_count / 2;
    size_t older_count = sample_count - recent_count;

    // Calculate average of older half
    double older_sum = 0.0;
    for (size_t i = 0; i < older_count; ++i) {
        older_sum += frame_times_[i];
    }
    double older_avg = older_sum / older_count;

    // Calculate average of more recent half
    double recent_sum = 0.0;
    for (size_t i = older_count; i < sample_count; ++i) {
        recent_sum += frame_times_[i];
    }
    double recent_avg = recent_sum / recent_count;

    // Calculate trend ratio (recent vs older)
    if (older_avg <= 0.0) {
        return recent_avg <= 0.0 ? 100.0 : 0.0; // Handle edge cases
    }

    double trend_ratio = recent_avg / older_avg;

    // If trend is improving (recent_avg < older_avg), return higher score
    // If trend is degrading (recent_avg > older_avg), return lower score
    if (trend_ratio <= 1.0) {
        // Improving trend - boost score
        double improvement_factor = (1.0 - (1.0 - trend_ratio)) * 100.0; // Invert and scale
        return std::min(100.0, 100.0 * improvement_factor);
    } else {
        // Degrading trend - reduce score
        double degradation_factor = 1.0 / trend_ratio; // Inverse relationship
        return std::max(0.0, degradation_factor * 100.0);
    }
}

double AutoQualityController::predictFuturePerformance() const {
    // Predict future performance based on current trends and historical data
    if (frame_count_ < 15) {
        return performance_score_; // Not enough data for reliable prediction
    }

    // Calculate weighted prediction based on multiple factors
    double trend_factor = calculateTrendPredictionScore() / 100.0;  // 0.0 to 1.0
    double current_factor = performance_score_ / 100.0;             // 0.0 to 1.0
    double historical_factor = calculateHistoricalPerformanceFactor();

    // Weight the factors: 40% current performance, 35% trend, 25% historical
    double predicted_score = (current_factor * 0.4) + (trend_factor * 0.35) + (historical_factor * 0.25);

    return predicted_score * 100.0;  // Convert back to 0-100 scale
}

double AutoQualityController::calculateHistoricalPerformanceFactor() const {
    // Calculate a factor based on historical performance patterns
    if (!performance_history_full_) {
        return 1.0; // Assume neutral if not enough history
    }

    // Calculate average of historical performance
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) {
        sum += performance_history_[i];
    }
    double historical_avg = sum / 10.0;

    // Calculate variance in historical performance
    double variance = 0.0;
    for (int i = 0; i < 10; ++i) {
        double diff = performance_history_[i] - historical_avg;
        variance += diff * diff;
    }
    variance /= 10.0;

    // Normalize to 0.0-1.0 range (1.0 being stable performance)
    double stability_factor = 1.0 - std::min(0.5, variance / 1000.0);

    // Return normalized historical performance (0.0 to 1.0)
    return std::max(0.0, std::min(1.0, historical_avg / 100.0)) * stability_factor;
}

double AutoQualityController::calculateAdaptiveThreshold() const {
    // Calculate adaptive threshold based on historical performance patterns
    if (!performance_history_full_) {
        return target_performance_threshold_; // Use default until we have enough history
    }

    // Calculate average of recent performance scores
    double sum = 0.0;
    for (int i = 0; i < 10; ++i) {
        sum += performance_history_[i];
    }
    double avg_performance = sum / 10.0;

    // Adjust threshold based on the average performance
    // If system typically performs well, be more aggressive with quality maintenance
    // If system typically performs poorly, be more conservative with quality reduction
    double performance_baseline = avg_performance;

    // Calculate variance in historical performance to determine stability
    double variance = 0.0;
    for (int i = 0; i < 10; ++i) {
        double diff = performance_history_[i] - performance_baseline;
        variance += diff * diff;
    }
    variance /= 10.0;
    double stability_factor = 1.0 - std::min(0.5, variance / 1000.0); // Reduce sensitivity for unstable systems

    // Adjust threshold based on baseline performance and stability
    if (performance_baseline >= target_performance_threshold_) {
        // System typically performs well, allow slightly lower threshold
        return target_performance_threshold_ * stability_factor * 0.95;
    } else {
        // System typically performs poorly, be more conservative
        return std::min(target_performance_threshold_, performance_baseline * stability_factor * 1.05);
    }
}

void AutoQualityController::updatePerformanceHistory() {
    // Add current performance score to history
    performance_history_[performance_history_index_] = performance_score_;
    performance_history_index_ = (performance_history_index_ + 1) % 10;

    if (!performance_history_full_ && performance_history_index_ == 0) {
        performance_history_full_ = true;
    }

    // Update adaptive threshold based on history
    adaptive_performance_threshold_ = calculateAdaptiveThreshold();
}

void AutoQualityController::checkAndAdjustQuality() {
    auto now = std::chrono::high_resolution_clock::now();

    // Check if we're past the cooldown period (except for emergency situations)
    bool past_cooldown = (now - last_adjustment_time_ >= adjustment_cooldown_);

    // Use adaptive threshold for decision making
    double effective_threshold = adaptive_performance_threshold_;

    // Use hysteresis to prevent oscillation
    // Different thresholds for reducing vs increasing quality
    double lower_threshold = effective_threshold * (0.9 - (current_quality_index_ * 0.02)); // Lower threshold becomes stricter at lower quality levels
    double upper_threshold = effective_threshold * (1.1 + (current_quality_index_ * 0.03)); // Upper threshold becomes more lenient at lower quality levels

    // Check for rapid performance degradation that requires immediate action
    bool rapid_degradation = isPerformanceDegradingRapidly();

    // Determine if we need to adjust quality based on performance
    if (performance_score_ < lower_threshold) {
        // Performance is below threshold, reduce quality
        if (current_quality_index_ < QUALITY_LEVEL_COUNT - 1) {
            // Adjust quality gradually based on performance drop severity
            int new_quality_index = determineQualityReduction();

            // For rapid degradation, bypass cooldown if needed
            bool can_adjust = past_cooldown || rapid_degradation;

            if (can_adjust && new_quality_index != current_quality_index_) {
                current_quality_index_ = new_quality_index;

                // Update last adjustment time
                last_adjustment_time_ = now;

                // Log the quality reduction
                if (config_.enable_logging) {
                    printf("AutoQuality: Reduced quality to level %d (Performance: %.2f%%, Lower Threshold: %.2f%%, Adaptive Base: %.2f%%, Rapid Degradation: %s)\n",
                           current_quality_index_, performance_score_, lower_threshold, effective_threshold,
                           rapid_degradation ? "YES" : "NO");
                }
            }
        }
    } else if (performance_score_ > upper_threshold && current_quality_index_ > 0) {
        // Performance is above threshold, try to increase quality
        // But only if we've been stable at current level for a while
        if (hasBeenStableAtCurrentLevel() && past_cooldown) {
            // Gradually increase quality if performance is consistently good
            int new_quality_index = determineQualityIncrease();

            if (new_quality_index != current_quality_index_) {
                current_quality_index_ = new_quality_index;
                last_adjustment_time_ = now;

                // Log the quality increase
                if (config_.enable_logging) {
                    printf("AutoQuality: Increased quality to level %d (Performance: %.2f%%, Upper Threshold: %.2f%%, Adaptive Base: %.2f%%)\n",
                           current_quality_index_, performance_score_, upper_threshold, effective_threshold);
                }
            }
        }
    }

    // Additional check: if performance is severely degraded, force immediate quality reduction regardless of cooldown
    if (performance_score_ < effective_threshold * 0.5) { // Below 50% of adaptive target
        if (current_quality_index_ < QUALITY_LEVEL_COUNT - 1) {
            // Immediate quality reduction due to severe performance issues
            int new_quality_index = std::min(current_quality_index_ + 1, QUALITY_LEVEL_COUNT - 1);

            if (new_quality_index != current_quality_index_) {
                current_quality_index_ = new_quality_index;

                // Update last adjustment time to prevent immediate further adjustments
                last_adjustment_time_ = now;

                if (config_.enable_logging) {
                    printf("AutoQuality: Emergency quality reduction to level %d (Severe performance: %.2f%%, Adaptive Threshold: %.2f%%)\n",
                           current_quality_index_, performance_score_, effective_threshold);
                }
            }
        }
    }

    // Additional check: if rapid degradation is detected, force immediate quality reduction
    if (rapid_degradation && current_quality_index_ < QUALITY_LEVEL_COUNT - 1) {
        // Rapid degradation detected, reduce quality immediately
        int new_quality_index = std::min(current_quality_index_ + 1, QUALITY_LEVEL_COUNT - 1);

        if (new_quality_index != current_quality_index_) {
            current_quality_index_ = new_quality_index;
            last_adjustment_time_ = now;

            if (config_.enable_logging) {
                printf("AutoQuality: Rapid degradation detected, immediate quality reduction to level %d (Performance: %.2f%%, Adaptive Threshold: %.2f%%)\n",
                       current_quality_index_, performance_score_, effective_threshold);
            }
        }
    }
}

int AutoQualityController::determineQualityReduction() {
    // Determine how much to reduce quality based on performance severity
    double performance_deficit = target_performance_threshold_ - performance_score_;

    // Check for additional factors that might require more aggressive reduction
    bool has_spikes = detectPerformanceSpikes();
    double jank_percentage = calculateJankPercentage();
    double consistency_score = calculatePerformanceConsistency();

    // Calculate the rate of performance degradation
    double degradation_rate = calculateDegradationRate();

    // Get additional metrics for more intelligent decision making
    double gpu_utilization_score = calculateGPUUtilizationScore();
    double cpu_utilization_score = calculateCPUUtilizationScore();
    double frame_pacing_score = calculateFramePacingIrregularity();
    double memory_pressure_score = calculateMemoryPressureScore();
    double thermal_pressure_score = calculateThermalPressureScore();

    // Calculate composite pressure score
    double composite_pressure = (gpu_utilization_score + cpu_utilization_score + memory_pressure_score + thermal_pressure_score) / 4.0;

    // If performance is extremely poor, reduce quality more aggressively
    if (performance_deficit > 40.0 || jank_percentage > 25.0) {
        // Very poor performance OR heavy jank - jump 2 levels down if possible
        int new_level = std::min(current_quality_index_ + 2, QUALITY_LEVEL_COUNT - 1);

        // If we also have performance spikes, consider jumping 3 levels (but cap at max)
        if (has_spikes && performance_deficit > 50.0) {
            new_level = std::min(current_quality_index_ + 3, QUALITY_LEVEL_COUNT - 1);
        }

        // If degradation is happening rapidly, be even more aggressive
        if (degradation_rate > 0.5) { // Performance dropping by more than 50% per second
            new_level = std::min(new_level + 1, QUALITY_LEVEL_COUNT - 1);
        }

        // If system pressure is high, add additional reduction
        if (composite_pressure < 50.0) {
            new_level = std::min(new_level + 1, QUALITY_LEVEL_COUNT - 1);
        }

        return new_level;
    } else if (performance_deficit > 20.0 || jank_percentage > 15.0) {
        // Poor performance OR moderate jank - jump 1-2 levels down
        int new_level = std::min(current_quality_index_ + 1, QUALITY_LEVEL_COUNT - 1);

        // If degradation is rapid, add an extra level
        if (degradation_rate > 0.3) { // Performance dropping by more than 30% per second
            new_level = std::min(new_level + 1, QUALITY_LEVEL_COUNT - 1);
        }

        // If we also have performance spikes
        if (has_spikes) {
            new_level = std::min(current_quality_index_ + 2, QUALITY_LEVEL_COUNT - 1);
        }

        // If system pressure is high, add additional reduction
        if (composite_pressure < 60.0) {
            new_level = std::min(new_level + 1, QUALITY_LEVEL_COUNT - 1);
        }

        return new_level;
    } else if (consistency_score < 60.0 || frame_pacing_score < 65.0) {
        // Performance is inconsistent OR frame pacing is irregular - reduce quality by 1 level to stabilize
        int new_level = std::min(current_quality_index_ + 1, QUALITY_LEVEL_COUNT - 1);

        // If degradation is rapid, add an extra level
        if (degradation_rate > 0.4) {
            new_level = std::min(new_level + 1, QUALITY_LEVEL_COUNT - 1);
        }

        // If system pressure is high, add additional reduction
        if (composite_pressure < 70.0) {
            new_level = std::min(new_level + 1, QUALITY_LEVEL_COUNT - 1);
        }

        return new_level;
    } else {
        // Moderate performance issues - reduce by 1 level
        int new_level = std::min(current_quality_index_ + 1, QUALITY_LEVEL_COUNT - 1);

        // If degradation is rapid, add an extra level
        if (degradation_rate > 0.25) {
            new_level = std::min(new_level + 1, QUALITY_LEVEL_COUNT - 1);
        }

        // If system pressure is high, add additional reduction
        if (composite_pressure < 75.0) {
            new_level = std::min(new_level + 1, QUALITY_LEVEL_COUNT - 1);
        }

        return new_level;
    }
}

int AutoQualityController::determineQualityIncrease() {
    // Determine how much to increase quality based on performance surplus
    // Only increase quality gradually to avoid oscillation

    // Check for additional factors that might affect the decision to increase quality
    bool has_spikes = detectPerformanceSpikes();
    double jank_percentage = calculateJankPercentage();
    double consistency_score = calculatePerformanceConsistency();

    // Get additional metrics for more intelligent decision making
    double gpu_utilization_score = calculateGPUUtilizationScore();
    double cpu_utilization_score = calculateCPUUtilizationScore();
    double frame_pacing_score = calculateFramePacingIrregularity();
    double memory_pressure_score = calculateMemoryPressureScore();
    double thermal_pressure_score = calculateThermalPressureScore();

    double performance_surplus = performance_score_ - target_performance_threshold_;

    // Calculate composite pressure score
    double composite_pressure = (gpu_utilization_score + cpu_utilization_score + memory_pressure_score + thermal_pressure_score) / 4.0;

    // Only increase quality if performance is consistently good AND stable
    if (performance_surplus > 30.0 && !has_spikes && jank_percentage < 5.0 && consistency_score > 80.0 && composite_pressure > 80.0 && frame_pacing_score > 85.0) {
        // Excellent performance, no spikes, low jank, high consistency, low system pressure, good frame pacing - could increase by 2 levels if stable
        return std::max(current_quality_index_ - 2, 0);
    } else if (performance_surplus > 15.0 && !has_spikes && jank_percentage < 10.0 && consistency_score > 70.0 && composite_pressure > 70.0) {
        // Good performance with acceptable stability and moderate system pressure - increase by 1 level
        return std::max(current_quality_index_ - 1, 0);
    } else if (performance_surplus > 20.0 && !has_spikes && jank_percentage < 8.0 && consistency_score > 75.0 && frame_pacing_score > 80.0) {
        // Good performance with good stability and good frame pacing - increase by 1 level
        return std::max(current_quality_index_ - 1, 0);
    } else {
        // Conditions not met for quality increase, stay at current level
        return current_quality_index_;
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
    peak_performance_score_ = 100.0;
    cumulative_performance_score_ = 0.0;
    performance_sample_count_ = 0;
    trend_index_ = 0;
    trend_buffer_full_ = false;

    // Reset frame time history
    std::fill(frame_times_.begin(), frame_times_.end(), 1000.0 / config_.target_fps);

    // Reset recent performance trend buffer
    for (int i = 0; i < 30; ++i) {
        recent_performance_trend_[i] = config_.performance_threshold;
    }
}

void AutoQualityController::updateConfig(const AutoQualityConfig& new_config) {
    config_ = new_config;
    target_performance_threshold_ = new_config.performance_threshold;
    adjustment_cooldown_ = std::chrono::milliseconds(
        static_cast<int>(new_config.adjustment_cooldown_ms));

    // Reinitialize quality levels if needed
    initializeQualityLevels();

    // Update adaptive threshold with new config
    adaptive_performance_threshold_ = new_config.performance_threshold;
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

QualitySettings AutoQualityController::getRecommendedRenderingSettings() const {
    return quality_levels_[current_quality_index_];
}

bool AutoQualityController::detectPerformanceSpikes() const {
    // Detect sudden performance spikes that indicate potential problems
    if (frame_count_ < 10) {
        return false;
    }

    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count < 3) {
        return false;
    }

    // Look for sudden increases in frame time (spikes)
    double target_frame_time = 1000.0 / config_.target_fps;
    int spike_count = 0;
    int total_frames = 0;

    for (size_t i = 0; i < sample_count; ++i) {
        if (frame_times_[i] > target_frame_time * 3.0) { // Spike is 3x target time
            spike_count++;
        }
        total_frames++;
    }

    // If more than 10% of frames are spikes, consider it problematic
    double spike_percentage = static_cast<double>(spike_count) / total_frames;
    return spike_percentage > 0.1; // More than 10% of frames are spikes
}

double AutoQualityController::calculateJankPercentage() const {
    // Calculate percentage of janky frames (frames that took significantly longer than average)
    if (frame_count_ < 10) {
        return 0.0;
    }

    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count < 2) {
        return 0.0;
    }

    // Calculate average frame time
    double sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        sum += frame_times_[i];
    }
    double avg_frame_time = sum / sample_count;

    if (avg_frame_time <= 0.0) {
        return 0.0;
    }

    // Count frames that are significantly slower than average (jank detection)
    int jank_count = 0;
    for (size_t i = 0; i < sample_count; ++i) {
        if (frame_times_[i] > avg_frame_time * 2.5) { // Jank is 2.5x average time
            jank_count++;
        }
    }

    return static_cast<double>(jank_count) / sample_count * 100.0;
}

double AutoQualityController::calculatePerformanceConsistency() const {
    // Calculate a consistency score based on frame time variations
    if (frame_count_ < 5) {
        return 100.0; // Not enough data, assume perfect consistency
    }

    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count < 2) {
        return 100.0;
    }

    // Calculate mean frame time
    double sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        sum += frame_times_[i];
    }
    double mean = sum / sample_count;

    if (mean <= 0.0) {
        return 100.0;
    }

    // Calculate standard deviation
    double variance_sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        double diff = frame_times_[i] - mean;
        variance_sum += diff * diff;
    }
    double variance = variance_sum / sample_count;
    double std_dev = std::sqrt(variance);

    // Calculate coefficient of variation (lower is more consistent)
    double coefficient_of_variation = (std_dev / mean) * 100.0;

    // Convert to consistency score (higher is better)
    // Using an inverse relationship: lower variation = higher consistency
    double consistency_score = std::max(0.0, 100.0 - coefficient_of_variation * 10.0);
    return consistency_score;
}

double AutoQualityController::calculateDegradationRate() const {
    // Calculate the rate of performance degradation over time
    if (frame_count_ < 10) {
        return 0.0; // Not enough data to calculate degradation rate
    }

    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count < 4) {
        return 0.0; // Need at least 4 samples to calculate meaningful trend
    }

    // Split the history into early and late halves to compare performance
    size_t early_start = 0;
    size_t early_end = sample_count / 2;
    size_t late_start = sample_count / 2;
    size_t late_end = sample_count;

    // Calculate average performance in early period
    double early_sum = 0.0;
    for (size_t i = early_start; i < early_end; ++i) {
        early_sum += (1000.0 / std::max(frame_times_[i], 1.0)); // Convert to FPS
    }
    double early_avg_fps = early_sum / (early_end - early_start);

    // Calculate average performance in later period
    double late_sum = 0.0;
    for (size_t i = late_start; i < late_end; ++i) {
        late_sum += (1000.0 / std::max(frame_times_[i], 1.0)); // Convert to FPS
    }
    double late_avg_fps = late_sum / (late_end - late_start);

    // Calculate degradation rate as percentage change per second
    // Assuming 60 FPS as target, normalize the degradation rate
    if (early_avg_fps <= 0.0) {
        return late_avg_fps <= 0.0 ? 0.0 : 1.0; // If early was 0, but late is positive, no degradation
    }

    double fps_change = late_avg_fps - early_avg_fps;
    double relative_change = fps_change / early_avg_fps;

    // Calculate time difference (approximate based on frame count)
    double time_period_seconds = (sample_count / 2) / static_cast<double>(config_.target_fps);

    // Return degradation rate per second (negative values indicate improvement)
    double degradation_rate = relative_change / time_period_seconds;

    // Return only positive values for degradation (negative means improvement)
    return std::max(0.0, -degradation_rate); // Negative of negative is positive for degradation
}

void AutoQualityController::updateAdvancedMetrics() {
    // Update advanced performance metrics that can be used for quality decisions
    bool has_spikes = detectPerformanceSpikes();
    double jank_percentage = calculateJankPercentage();
    double consistency_score = calculatePerformanceConsistency();

    // Calculate memory pressure if available (placeholder for future integration)
    double memory_pressure_score = calculateMemoryPressureScore();

    // Calculate thermal pressure if available (placeholder for future integration)
    double thermal_pressure_score = calculateThermalPressureScore();

    // Calculate GPU utilization if available (placeholder for future integration)
    double gpu_utilization_score = calculateGPUUtilizationScore();

    // Calculate CPU utilization if available (placeholder for future integration)
    double cpu_utilization_score = calculateCPUUtilizationScore();

    // Calculate frame pacing irregularity
    double frame_pacing_score = calculateFramePacingIrregularity();

    // Adjust performance score based on these advanced metrics
    if (has_spikes) {
        // Significant performance spikes detected, reduce performance score
        performance_score_ *= 0.85; // 15% reduction
    }

    if (jank_percentage > 15.0) { // More than 15% janky frames
        // Heavy jank detected, reduce performance score
        double jank_penalty = std::min(jank_percentage * 0.7, 35.0); // Up to 35% penalty
        performance_score_ -= jank_penalty;
    }

    // Apply memory pressure penalty if needed
    if (memory_pressure_score < 70.0) {
        double memory_penalty = (70.0 - memory_pressure_score) * 0.4;
        performance_score_ -= memory_penalty;
    }

    // Apply thermal pressure penalty if needed
    if (thermal_pressure_score < 75.0) {
        double thermal_penalty = (75.0 - thermal_pressure_score) * 0.3;
        performance_score_ -= thermal_penalty;
    }

    // Apply GPU utilization penalty if GPU is heavily loaded
    if (gpu_utilization_score < 60.0) { // GPU utilization is high (low score)
        double gpu_penalty = (60.0 - gpu_utilization_score) * 0.25;
        performance_score_ -= gpu_penalty;
    }

    // Apply CPU utilization penalty if CPU is heavily loaded
    if (cpu_utilization_score < 65.0) { // CPU utilization is high (low score)
        double cpu_penalty = (65.0 - cpu_utilization_score) * 0.2;
        performance_score_ -= cpu_penalty;
    }

    // Apply frame pacing penalty if irregular
    if (frame_pacing_score < 80.0) {
        double pacing_penalty = (80.0 - frame_pacing_score) * 0.15;
        performance_score_ -= pacing_penalty;
    }

    // Consistency affects the performance score inversely
    // Less consistent = lower score
    double consistency_factor = consistency_score / 100.0;
    performance_score_ *= consistency_factor;

    // Ensure performance score stays within bounds
    performance_score_ = std::max(0.0, std::min(100.0, performance_score_));
}

double AutoQualityController::calculateMemoryPressureScore() const {
    // Interface with the memory tracker to get actual memory pressure
    // Get current memory usage and compare to thresholds

    // Since we have a memory tracker in the performance module, we'll simulate
    // getting data from it. In a real implementation, we would access the global tracker.

    // For now, we'll implement a realistic simulation based on memory usage patterns
    size_t current_usage = 0;
    size_t peak_usage = 0;

#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        current_usage = static_cast<size_t>(pmc.WorkingSetSize);
        peak_usage = static_cast<size_t>(pmc.PeakWorkingSetSize);
    }
#elif __linux__
    // Try to get resident set size (RSS) from /proc/self/status
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key;
            size_t value;
            std::string unit;
            iss >> key >> value >> unit;
            current_usage = value * 1024; // Convert from KB to bytes
            break;
        }
    }

    // Also try to get peak usage from limits
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    peak_usage = static_cast<size_t>(usage.ru_maxrss * 1024); // Convert from KB to bytes
#else
    // For other platforms, return baseline
    return 100.0;
#endif

    // Calculate memory pressure based on usage relative to thresholds
    // Assume a reasonable threshold for demonstration (e.g., 80% of some reasonable limit)
    // For a trading application, let's assume 4GB as a reasonable upper limit for memory usage
    const size_t MEMORY_THRESHOLD_HIGH = 4ULL * 1024 * 1024 * 1024; // 4GB
    const size_t MEMORY_THRESHOLD_MEDIUM = 3ULL * 1024 * 1024 * 1024; // 3GB
    const size_t MEMORY_THRESHOLD_LOW = 2ULL * 1024 * 1024 * 1024; // 2GB

    if (current_usage > MEMORY_THRESHOLD_HIGH) {
        // Severe memory pressure
        return 10.0;
    } else if (current_usage > MEMORY_THRESHOLD_MEDIUM) {
        // High memory pressure
        double pressure = 10.0 + ((current_usage - MEMORY_THRESHOLD_MEDIUM) /
                     static_cast<double>(MEMORY_THRESHOLD_HIGH - MEMORY_THRESHOLD_MEDIUM)) * 30.0;
        return std::max(10.0, 40.0 - pressure);
    } else if (current_usage > MEMORY_THRESHOLD_LOW) {
        // Medium memory pressure
        double pressure = ((current_usage - MEMORY_THRESHOLD_LOW) /
                     static_cast<double>(MEMORY_THRESHOLD_MEDIUM - MEMORY_THRESHOLD_LOW)) * 30.0;
        return std::max(40.0, 70.0 - pressure);
    } else {
        // Low memory pressure
        return 100.0 - (current_usage / static_cast<double>(MEMORY_THRESHOLD_LOW)) * 30.0;
    }
}

double AutoQualityController::calculateThermalPressureScore() const {
    // Interface with system thermal monitoring to get actual thermal pressure
    // On Linux, we can check thermal zones; on Windows, we might use WMI or other APIs

#ifdef __linux__
    // Check thermal zones for temperature
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
    if (temp_file.is_open()) {
        int temperature;
        temp_file >> temperature;
        temp_file.close();

        // Temperature is usually in millidegrees Celsius
        double temp_celsius = temperature / 1000.0;

        // Define thermal thresholds for a typical system
        const double THERMAL_THRESHOLD_CRITICAL = 85.0; // degrees C
        const double THERMAL_THRESHOLD_HIGH = 75.0;     // degrees C
        const double THERMAL_THRESHOLD_MEDIUM = 65.0;   // degrees C

        if (temp_celsius >= THERMAL_THRESHOLD_CRITICAL) {
            // Critical thermal pressure
            return 10.0;
        } else if (temp_celsius >= THERMAL_THRESHOLD_HIGH) {
            // High thermal pressure
            double pressure = 10.0 + ((temp_celsius - THERMAL_THRESHOLD_HIGH) /
                         (THERMAL_THRESHOLD_CRITICAL - THERMAL_THRESHOLD_HIGH)) * 30.0;
            return std::max(10.0, 40.0 - pressure);
        } else if (temp_celsius >= THERMAL_THRESHOLD_MEDIUM) {
            // Medium thermal pressure
            double pressure = ((temp_celsius - THERMAL_THRESHOLD_MEDIUM) /
                         (THERMAL_THRESHOLD_HIGH - THERMAL_THRESHOLD_MEDIUM)) * 30.0;
            return std::max(40.0, 70.0 - pressure);
        } else {
            // Low thermal pressure
            return 100.0 - ((temp_celsius / THERMAL_THRESHOLD_MEDIUM) * 30.0);
        }
    }
#endif

    // For other platforms or if thermal sensors are unavailable, return baseline
    return 100.0; // No thermal pressure detected
}

double AutoQualityController::calculateGPUUtilizationScore() const {
    // Interface with GPU monitoring APIs to determine actual GPU utilization
    // For now, return a baseline score based on performance metrics
    // Lower score indicates higher GPU load

    // Calculate score based on how close we are to target FPS
    double target_frame_time = 1000.0 / config_.target_fps;

    // Look at recent frame times to estimate GPU load
    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count == 0) {
        return 100.0;
    }

    double avg_frame_time = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        avg_frame_time += frame_times_[i];
    }
    avg_frame_time /= sample_count;

    // If we're close to target frame time, GPU might be under pressure
    double utilization_factor = std::min(avg_frame_time / target_frame_time, 1.0);
    double base_score = std::max(10.0, 100.0 - (utilization_factor * 90.0));

    // Additionally, we could interface with vendor-specific APIs for actual GPU utilization
    // For example, NVML for NVIDIA GPUs, ADL for AMD, or Metal for Apple Silicon
    // For now, we'll return the performance-based score but note where real GPU monitoring would go
#ifdef __linux__
    // On Linux, we could potentially read from nvidia-smi or similar tools
    // This is a simplified approach - in reality, we'd want to use NVML or similar
    static bool nvml_available = checkNvmlAvailability();
    if (nvml_available) {
        double gpu_util = getNvidiaGpuUtilization();
        if (gpu_util >= 0) {
            // Blend the performance-based score with actual GPU utilization
            return (base_score * 0.6) + (gpu_util * 0.4);
        }
    }
#endif

    return base_score;
}

#ifdef __linux__
bool AutoQualityController::checkNvmlAvailability() const {
    // Simple check to see if nvidia-smi is available
    // In a real implementation, we would link with NVML library
    int result = system("which nvidia-smi > /dev/null 2>&1");
    return WEXITSTATUS(result) == 0;
}

double AutoQualityController::getNvidiaGpuUtilization() const {
    // Execute nvidia-smi to get GPU utilization
    FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!pipe) return -1.0;

    char buffer[16];
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        pclose(pipe);
        try {
            int utilization = std::stoi(buffer);
            // Convert utilization (0-100) to a score where higher is better (100-utilization)
            // But since high utilization means high load, we want to return a lower score
            return 100.0 - static_cast<double>(utilization);
        } catch (...) {
            return -1.0;
        }
    }

    pclose(pipe);
    return -1.0;
}
#endif

double AutoQualityController::calculateCPUUtilizationScore() const {
    // Interface with system monitoring APIs to determine actual CPU utilization
    // For now, return a score based on both performance metrics and actual system CPU usage
    // Lower score indicates higher CPU load

    // Estimate CPU load based on frame time consistency
    // If frame times vary significantly, CPU might be under pressure
    double variance = calculateFrameTimeVariance();
    double max_expected_variance = config_.variance_threshold;

    // Higher variance suggests higher CPU pressure
    double cpu_load_factor = std::min(variance / max_expected_variance, 1.0);
    double base_score = std::max(10.0, 100.0 - (cpu_load_factor * 90.0));

    // Get actual CPU utilization from system
    double actual_cpu_util = getSystemCpuUtilization();
    if (actual_cpu_util >= 0) {
        // Convert CPU utilization (0-100%) to a score where higher is better
        // Higher CPU utilization means lower score (more pressure)
        double cpu_score = 100.0 - actual_cpu_util;
        // Blend the performance-based score with actual CPU utilization
        return (base_score * 0.5) + (cpu_score * 0.5);
    }

    return base_score;
}

double AutoQualityController::getSystemCpuUtilization() const {
    // Get actual CPU utilization from the system
#ifdef __linux__
    // Read from /proc/stat to calculate CPU utilization
    static unsigned long long last_idle_time = 0;
    static unsigned long long last_total_time = 0;

    std::ifstream stat_file("/proc/stat");
    if (!stat_file.is_open()) {
        return -1.0; // Error reading file
    }

    std::string line;
    std::getline(stat_file, line);
    stat_file.close();

    // Parse the first line which contains overall CPU stats
    std::istringstream iss(line);
    std::string cpu_label;
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice;

    iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;

    unsigned long long idle_time = idle + iowait;
    unsigned long long total_time = user + nice + system + idle + iowait + irq + softirq + steal;

    if (last_idle_time != 0 && last_total_time != 0) {
        unsigned long long delta_idle = idle_time - last_idle_time;
        unsigned long long delta_total = total_time - last_total_time;

        if (delta_total == 0) {
            return -1.0; // Avoid division by zero
        }

        double cpu_utilization = 100.0 * (delta_total - delta_idle) / delta_total;

        // Update stored values
        last_idle_time = idle_time;
        last_total_time = total_time;

        return cpu_utilization;
    }

    // Store initial values
    last_idle_time = idle_time;
    last_total_time = total_time;

    return -1.0; // Not enough data for first calculation

#elif defined(_WIN32)
    // On Windows, we would use PDH or WMI to get CPU utilization
    // For now, return -1 to indicate we're using the fallback
    return -1.0;
#else
    return -1.0; // Unsupported platform
#endif
}

double AutoQualityController::calculateFramePacingIrregularity() const {
    // Calculate how irregular the frame timing is
    // Irregular pacing can indicate performance issues

    if (frame_count_ < 10) {
        return 100.0; // Not enough data, assume perfect pacing
    }

    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count < 3) {
        return 100.0; // Need at least 3 samples to calculate pacing
    }

    // Calculate differences between consecutive frame times
    std::vector<double> frame_time_deltas;
    for (size_t i = 1; i < sample_count; ++i) {
        double delta = std::abs(frame_times_[i] - frame_times_[i-1]);
        frame_time_deltas.push_back(delta);
    }

    if (frame_time_deltas.empty()) {
        return 100.0;
    }

    // Calculate average delta
    double sum = 0.0;
    for (double delta : frame_time_deltas) {
        sum += delta;
    }
    double avg_delta = sum / frame_time_deltas.size();

    // Calculate variance of deltas
    double variance = 0.0;
    for (double delta : frame_time_deltas) {
        double diff = delta - avg_delta;
        variance += diff * diff;
    }
    variance /= frame_time_deltas.size();

    // Calculate coefficient of variation
    double std_dev = std::sqrt(variance);
    double coefficient_of_variation = avg_delta > 0.0 ? (std_dev / avg_delta) : 0.0;

    // Convert to a score (higher is better pacing)
    double pacing_score = std::max(0.0, 100.0 - (coefficient_of_variation * 500.0));
    return std::min(100.0, pacing_score);
}

double AutoQualityController::calculatePeakPerformanceScore() const {
    // Return the highest performance score recorded
    return peak_performance_score_;
}

double AutoQualityController::calculateAveragePerformanceScore() const {
    // Calculate the average performance score over all samples
    if (performance_sample_count_ == 0) {
        return 100.0; // Default to perfect performance if no samples
    }

    return cumulative_performance_score_ / performance_sample_count_;
}

bool AutoQualityController::isPerformanceDegradingRapidly() const {
    // Check if performance is degrading rapidly by looking at recent trend
    if (!trend_buffer_full_) {
        return false; // Not enough data to determine
    }

    // Look at the most recent 10 samples vs the 10 samples before that
    const int sample_size = 10;
    double recent_avg = 0.0;
    double previous_avg = 0.0;

    // Calculate average of most recent samples
    for (int i = 0; i < sample_size; ++i) {
        int idx = (trend_index_ - 1 - i + 30) % 30; // Recent samples
        recent_avg += recent_performance_trend_[idx];
    }
    recent_avg /= sample_size;

    // Calculate average of previous samples
    for (int i = 0; i < sample_size; ++i) {
        int idx = (trend_index_ - 1 - sample_size - i + 30) % 30; // Previous samples
        previous_avg += recent_performance_trend_[idx];
    }
    previous_avg /= sample_size;

    // If recent performance is significantly worse than previous performance,
    // we're degrading rapidly
    double degradation_threshold = previous_avg * 0.85; // 15% degradation
    return recent_avg < degradation_threshold;
}

void AutoQualityController::updateRealTimePerformanceMetrics() {
    // Update cumulative performance statistics
    cumulative_performance_score_ += performance_score_;
    performance_sample_count_++;

    // Update peak performance score
    if (performance_score_ > peak_performance_score_) {
        peak_performance_score_ = performance_score_;
    }

    // Update recent performance trend buffer
    recent_performance_trend_[trend_index_] = performance_score_;
    trend_index_ = (trend_index_ + 1) % 30;

    if (!trend_buffer_full_ && trend_index_ == 0) {
        trend_buffer_full_ = true;
    }
}

} // namespace RenderEngine