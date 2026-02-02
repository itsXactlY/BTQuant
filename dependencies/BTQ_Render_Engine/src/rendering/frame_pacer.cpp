#include "rendering/frame_pacer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace RenderEngine {

FramePacer::FramePacer(const Config& config)
    : config_(config)
    , frame_time_history_(FRAME_HISTORY_SIZE, 1000.0 / config.target_fps)  // Initialize with target frame time
    , smoothed_frame_times_(SMOOTHING_WINDOW, 1000.0 / config.target_fps)
    , last_frame_time_(std::chrono::high_resolution_clock::now())
{
    // Initialize stats
    stats_.avg_frame_time_ms = 1000.0 / config.target_fps;
    stats_.current_fps = config.target_fps;
    stats_.smoothed_fps = config.target_fps;
}

FramePacer::FramePacer()
    : FramePacer(Config{})  // Call the other constructor with default config
{
}

FramePacer::~FramePacer() = default;

void FramePacer::begin_frame() {
    frame_start_time_ = std::chrono::high_resolution_clock::now();
}

void FramePacer::end_frame() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto frame_duration = current_time - last_frame_time_;
    double frame_time_ms = std::chrono::duration<double, std::milli>(frame_duration).count();
    
    // Store frame time in history
    frame_time_history_[frame_count_ % FRAME_HISTORY_SIZE] = frame_time_ms;
    
    // Update smoothed frame times
    if (config_.enable_frame_smoothing) {
        smoothed_frame_times_[frame_count_ % SMOOTHING_WINDOW] = frame_time_ms;
    }
    
    // Update statistics
    update_statistics();
    
    // Update frame variance
    update_frame_variance();
    
    // Update recent average for adaptive sync
    if (config_.enable_adaptive_sync) {
        recent_avg_frame_time_ = (recent_avg_frame_time_ * recent_frame_count_ + frame_time_ms) / 
                                (recent_frame_count_ + 1);
        recent_frame_count_++;
        
        // Reset recent average periodically to avoid drift
        if (recent_frame_count_ > 30) {
            recent_frame_count_ = 0;
            recent_avg_frame_time_ = 0.0;
        }
    }
    
    last_frame_time_ = current_time;
    frame_count_++;
    stats_.total_frames++;
}

void FramePacer::wait_for_next_frame() {
    if (config_.target_fps == 0) {
        return; // Unlimited FPS
    }

    double sleep_duration = calculate_sleep_duration();
    
    if (sleep_duration > 0.0) {
        // Sleep for most of the remaining time, then busy wait for precision
        auto sleep_ms = static_cast<int>(sleep_duration);
        if (sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
        
        // Busy wait for remaining time for higher precision
        auto target_time = frame_start_time_ + std::chrono::milliseconds(1000 / config_.target_fps);
        while (std::chrono::high_resolution_clock::now() < target_time) {
            std::this_thread::yield();
        }
    }
}

FramePacer::Stats FramePacer::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void FramePacer::update_config(const Config& new_config) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    config_ = new_config;
    
    // Adjust frame time history if target FPS changed significantly
    if (new_config.target_fps != 0) {
        double new_target_time = 1000.0 / new_config.target_fps;
        std::fill(frame_time_history_.begin(), frame_time_history_.end(), new_target_time);
        std::fill(smoothed_frame_times_.begin(), smoothed_frame_times_.end(), new_target_time);
    }
}

void FramePacer::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    stats_ = Stats{};
    stats_.avg_frame_time_ms = 1000.0 / config_.target_fps;
    stats_.current_fps = config_.target_fps;
    stats_.smoothed_fps = config_.target_fps;
    
    std::fill(frame_time_history_.begin(), frame_time_history_.end(), 1000.0 / config_.target_fps);
    std::fill(smoothed_frame_times_.begin(), smoothed_frame_times_.end(), 1000.0 / config_.target_fps);
    
    accumulated_frame_time_ = 0.0;
    frame_count_ = 0;
    recent_avg_frame_time_ = 0.0;
    recent_frame_count_ = 0;
}

double FramePacer::calculate_sleep_duration() const {
    if (config_.target_fps == 0) {
        return 0.0; // Unlimited FPS
    }
    
    double target_frame_time = 1000.0 / config_.target_fps;
    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - frame_start_time_).count();
    
    double remaining_time = target_frame_time - elapsed;
    
    // Apply adaptive sync if enabled and frame time is too variable
    if (config_.enable_adaptive_sync && stats_.frame_time_variance > config_.frame_time_variance_threshold * 1000.0) {
        // Reduce target frame time slightly to compensate for variance
        target_frame_time *= 0.95;
        remaining_time = target_frame_time - elapsed;
    }
    
    // Apply burst reduction if enabled
    if (config_.enable_burst_reduction) {
        // If we're ahead of schedule, add a small buffer to prevent bursts
        if (remaining_time > 0 && remaining_time < 1.0) {  // Less than 1ms remaining
            remaining_time = std::max(0.0, remaining_time - 0.2);  // Add 0.2ms buffer
        }
    }
    
    return std::max(0.0, remaining_time);
}

void FramePacer::update_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Calculate current frame time
    auto frame_duration = std::chrono::high_resolution_clock::now() - frame_start_time_;
    double current_frame_time = std::chrono::duration<double, std::milli>(frame_duration).count();
    
    // Update average frame time
    accumulated_frame_time_ += current_frame_time;
    stats_.avg_frame_time_ms = accumulated_frame_time_ / frame_count_;
    
    // Update min/max frame times
    if (frame_count_ == 1) {
        stats_.min_frame_time_ms = current_frame_time;
        stats_.max_frame_time_ms = current_frame_time;
    } else {
        stats_.min_frame_time_ms = std::min(stats_.min_frame_time_ms, current_frame_time);
        stats_.max_frame_time_ms = std::max(stats_.max_frame_time_ms, current_frame_time);
    }
    
    // Calculate current FPS
    if (current_frame_time > 0) {
        stats_.current_fps = 1000.0 / current_frame_time;
    }
    
    // Calculate smoothed FPS if smoothing is enabled
    if (config_.enable_frame_smoothing && frame_count_ >= SMOOTHING_WINDOW) {
        double sum = 0.0;
        size_t count = 0;
        for (size_t i = 0; i < SMOOTHING_WINDOW; ++i) {
            size_t idx = (frame_count_ - 1 - i) % SMOOTHING_WINDOW;
            if (smoothed_frame_times_[idx] > 0) {
                sum += 1000.0 / smoothed_frame_times_[idx];
                count++;
            }
        }
        if (count > 0) {
            stats_.smoothed_fps = sum / count;
        }
    } else {
        stats_.smoothed_fps = stats_.current_fps;
    }
}

void FramePacer::update_frame_variance() {
    if (frame_count_ < 2) {
        stats_.frame_time_variance = 0.0;
        return;
    }
    
    // Calculate variance over recent frame times
    size_t sample_count = std::min(static_cast<size_t>(frame_count_), FRAME_HISTORY_SIZE);
    if (sample_count < 2) {
        stats_.frame_time_variance = 0.0;
        return;
    }
    
    // Calculate mean
    double sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        sum += frame_time_history_[i];
    }
    double mean = sum / sample_count;
    
    // Calculate variance
    double variance_sum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
        double diff = frame_time_history_[i] - mean;
        variance_sum += diff * diff;
    }
    stats_.frame_time_variance = variance_sum / sample_count;
}

} // namespace RenderEngine