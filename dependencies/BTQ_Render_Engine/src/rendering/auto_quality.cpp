/**
 * BTQuant Auto-Quality Reduction System
 *
 * Advanced system to detect performance drops and automatically reduce visual quality
 * to maintain responsiveness in professional trading dashboard applications.
 */

#include "rendering/auto_quality.hpp"
#include <algorithm>
#include <cmath>

namespace RenderEngine {

AutoQualityController::AutoQualityController(const AutoQualityConfig& config)
    : config_(config)
    , frame_times_(FRAME_HISTORY_SIZE, 1000.0 / config.target_fps)
    , quality_levels_(QUALITY_LEVEL_COUNT)
    , current_quality_index_(QUALITY_LEVEL_COUNT - 1)  // Start at highest quality
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
        .post_process_quality = 1.0f
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
        .post_process_quality = 0.9f
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
        .post_process_quality = 0.7f
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
        .post_process_quality = 0.3f
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
        .post_process_quality = 0.1f
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
    
    // Calculate performance score (higher is better)
    // Score is based on how close we are to target FPS
    if (avg_frame_time <= target_frame_time) {
        // Perfect performance
        performance_score_ = 100.0;
    } else {
        // Performance degrades as frame time increases
        double ratio = target_frame_time / avg_frame_time;
        performance_score_ = std::max(0.0, ratio * 100.0);
    }
    
    // Also consider frame time variance
    double variance = calculateFrameTimeVariance();
    if (variance > config_.variance_threshold) {
        // Penalize for inconsistent frame times
        double penalty = (variance - config_.variance_threshold) * 10.0;
        performance_score_ = std::max(0.0, performance_score_ - penalty);
    }
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
    
    // Determine if we need to adjust quality based on performance
    if (performance_score_ < target_performance_threshold_ * 0.9) {
        // Performance is significantly below threshold, reduce quality
        if (current_quality_index_ < QUALITY_LEVEL_COUNT - 1) {
            // Move to lower quality level
            current_quality_index_++;
            last_adjustment_time_ = now;
            
            // Log the quality reduction
            if (config_.enable_logging) {
                printf("AutoQuality: Reduced quality to level %d (Performance: %.2f%%)\n", 
                       current_quality_index_, performance_score_);
            }
        }
    } else if (performance_score_ > target_performance_threshold_ * 1.1 && 
               current_quality_index_ > 0) {
        // Performance is significantly above threshold, try to increase quality
        // But only if we've been stable at current level for a while
        if (hasBeenStableAtCurrentLevel()) {
            // Move to higher quality level
            current_quality_index_--;
            last_adjustment_time_ = now;
            
            // Log the quality increase
            if (config_.enable_logging) {
                printf("AutoQuality: Increased quality to level %d (Performance: %.2f%%)\n", 
                       current_quality_index_, performance_score_);
            }
        }
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
    current_quality_index_ = QUALITY_LEVEL_COUNT - 1;  // Start at highest quality
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