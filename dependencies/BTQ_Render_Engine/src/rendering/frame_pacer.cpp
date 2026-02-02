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
    , frame_timer_()
    , spike_detector_()
    , dropped_frame_counter_(0)
    , adaptive_target_fps_(config.target_fps)
    , last_adaptive_update_(std::chrono::high_resolution_clock::now())
{
    // Initialize stats
    stats_.avg_frame_time_ms = 1000.0 / config.target_fps;
    stats_.current_fps = config.target_fps;
    stats_.smoothed_fps = config.target_fps;

    // Initialize frame timer for precise timing
    frame_timer_.last_frame_time = std::chrono::high_resolution_clock::now();
    frame_timer_.target_frame_time_us = 1000000.0 / config.target_fps; // Convert to microseconds

    // Initialize spike detector
    spike_detector_.baseline_frame_time = 1000.0 / config.target_fps;
    spike_detector_.spike_threshold_multiplier = 2.0;
    spike_detector_.spike_history.resize(SPIKE_DETECTION_WINDOW, 0.0);
    spike_detector_.spike_index = 0;
    spike_detector_.spike_count_recent = 0;
}

FramePacer::FramePacer()
    : FramePacer(Config{})  // Call the other constructor with default config
{
}

FramePacer::~FramePacer() = default;

void FramePacer::begin_frame() {
    frame_start_time_ = std::chrono::high_resolution_clock::now();

    // Check if we need to adapt the target FPS based on recent performance
    adapt_target_fps();
}

void FramePacer::end_frame() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto frame_duration = current_time - last_frame_time_;
    double frame_time_ms = std::chrono::duration<double, std::milli>(frame_duration).count();
    double frame_time_us = frame_time_ms * 1000.0; // Convert to microseconds

    // Store frame time in history
    frame_time_history_[frame_count_ % FRAME_HISTORY_SIZE] = frame_time_ms;

    // Update smoothed frame times
    if (config_.enable_frame_smoothing) {
        smoothed_frame_times_[frame_count_ % SMOOTHING_WINDOW] = frame_time_ms;
    }

    // Detect frame spikes
    detect_spikes(frame_time_ms);

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

    // Check for dropped frames
    check_dropped_frames(frame_time_us);

    // Increment frame counters before updating statistics
    frame_count_++;
    stats_.total_frames++;

    // Update statistics after incrementing counters
    update_statistics(frame_time_ms);

    // Update frame variance
    update_frame_variance();

    last_frame_time_ = current_time;
}

void FramePacer::wait_for_next_frame() {
    if (adaptive_target_fps_ == 0) {
        return; // Unlimited FPS
    }

    // Calculate precise sleep duration
    double sleep_duration = calculate_sleep_duration();

    if (sleep_duration > 0.0) {
        // Use a more sophisticated sleep strategy for better precision
        precise_sleep(sleep_duration);
    }

    // Update frame timer
    frame_timer_.last_frame_time = std::chrono::high_resolution_clock::now();
}

FramePacer::Stats FramePacer::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void FramePacer::update_config(const Config& new_config) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    config_ = new_config;
    adaptive_target_fps_ = new_config.target_fps;

    // Adjust frame time history if target FPS changed significantly
    if (new_config.target_fps != 0) {
        double new_target_time = 1000.0 / new_config.target_fps;
        std::fill(frame_time_history_.begin(), frame_time_history_.end(), new_target_time);
        std::fill(smoothed_frame_times_.begin(), smoothed_frame_times_.end(), new_target_time);

        // Update frame timer target
        frame_timer_.target_frame_time_us = 1000000.0 / new_config.target_fps;
        spike_detector_.baseline_frame_time = new_target_time;
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

    // Reset frame timer
    frame_timer_.last_frame_time = std::chrono::high_resolution_clock::now();
    frame_timer_.target_frame_time_us = 1000000.0 / config_.target_fps;

    // Reset spike detector
    std::fill(spike_detector_.spike_history.begin(), spike_detector_.spike_history.end(), 0.0);
    spike_detector_.spike_index = 0;
    spike_detector_.spike_count_recent = 0;

    // Reset dropped frame counter
    dropped_frame_counter_ = 0;

    // Reset adaptive FPS
    adaptive_target_fps_ = config_.target_fps;
    last_adaptive_update_ = std::chrono::high_resolution_clock::now();
}

double FramePacer::calculate_sleep_duration() const {
    if (adaptive_target_fps_ == 0) {
        return 0.0; // Unlimited FPS
    }

    double target_frame_time = 1000.0 / adaptive_target_fps_;
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

void FramePacer::update_statistics(double current_frame_time) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    // Update average frame time - frame_count_ has already been incremented
    accumulated_frame_time_ += current_frame_time;
    if (frame_count_ > 0) {
        stats_.avg_frame_time_ms = accumulated_frame_time_ / frame_count_;
    } else {
        stats_.avg_frame_time_ms = 0.0;
    }

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

    // Add dropped frame count to stats
    stats_.dropped_frames = dropped_frame_counter_;
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

void FramePacer::precise_sleep(double sleep_duration_ms) const {
    // More precise sleep implementation
    auto start = std::chrono::high_resolution_clock::now();
    auto target_time = start + std::chrono::duration<double, std::milli>(sleep_duration_ms);

    // Use nanosleep for higher precision when possible
    auto remaining_ms = sleep_duration_ms;
    if (remaining_ms > 2.0) {  // If more than 2ms to sleep, use standard sleep
        auto sleep_ms = static_cast<int>(remaining_ms - 1.0);  // Sleep slightly less
        if (sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }

    // Busy wait for the final precise timing
    while (std::chrono::high_resolution_clock::now() < target_time) {
        // Use a more efficient busy-wait approach
        std::this_thread::yield();
        // Small pause to reduce CPU usage during busy wait
        std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }
}

void FramePacer::detect_spikes(double frame_time_ms) {
    // Add current frame time to spike detection history
    spike_detector_.spike_history[spike_detector_.spike_index] = frame_time_ms;
    spike_detector_.spike_index = (spike_detector_.spike_index + 1) % SPIKE_DETECTION_WINDOW;

    // Check if this frame is a spike (significantly longer than baseline)
    double threshold = spike_detector_.baseline_frame_time * spike_detector_.spike_threshold_multiplier;
    if (frame_time_ms > threshold) {
        spike_detector_.spike_count_recent++;

        // Adjust baseline if we're consistently seeing higher frame times
        if (frame_time_ms > spike_detector_.baseline_frame_time * 3.0) {
            spike_detector_.baseline_frame_time = frame_time_ms * 0.8; // Adjust baseline upward
        }
    }

    // Decrement recent spike count periodically to decay old spikes
    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - spike_detector_.last_spike_decay).count() > 1000) {
        spike_detector_.spike_count_recent = std::max(0, spike_detector_.spike_count_recent - 1);
        spike_detector_.last_spike_decay = now;
    }

    // Update stats with spike information
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.spike_count_recent = spike_detector_.spike_count_recent;
    }
}

void FramePacer::check_dropped_frames(double frame_time_us) {
    // Check if the current frame took significantly longer than expected
    double expected_frame_time_us = 1000000.0 / adaptive_target_fps_;
    double threshold = expected_frame_time_us * 1.8; // 80% over expected time

    if (frame_time_us > threshold) {
        dropped_frame_counter_++;
    }
}

void FramePacer::adapt_target_fps() {
    auto now = std::chrono::high_resolution_clock::now();
    auto time_since_update = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_adaptive_update_).count();

    // Only adapt every 500ms to prevent constant fluctuations
    if (time_since_update < 500) {
        return;
    }

    // Check if we need to adjust target FPS based on recent performance
    if (config_.enable_adaptive_sync && frame_count_ > 10) {
        // Calculate average frame time over recent frames
        size_t sample_count = std::min(static_cast<size_t>(frame_count_),
                                     static_cast<size_t>(FRAME_HISTORY_SIZE / 4)); // Use 1/4 of history
        if (sample_count > 5) {
            double sum = 0.0;
            for (size_t i = 0; i < sample_count; ++i) {
                size_t idx = (frame_count_ - 1 - i) % FRAME_HISTORY_SIZE;
                sum += frame_time_history_[idx];
            }
            double avg_recent_frame_time = sum / sample_count;
            double avg_recent_fps = 1000.0 / avg_recent_frame_time;

            // Adjust target FPS if we're consistently underperforming
            if (avg_recent_fps < adaptive_target_fps_ * 0.8) { // If we're at 80% of target
                adaptive_target_fps_ = static_cast<uint32_t>(avg_recent_fps * 0.95); // Lower target slightly
                adaptive_target_fps_ = std::max(adaptive_target_fps_,
                                              static_cast<uint32_t>(config_.target_fps * 0.5)); // Don't go too low
            } else if (avg_recent_fps > adaptive_target_fps_ * 1.1 &&
                      adaptive_target_fps_ < config_.target_fps) { // If we can handle more
                adaptive_target_fps_ = std::min(config_.target_fps,
                                              adaptive_target_fps_ + 5); // Gradually increase
            }
        }
    }

    last_adaptive_update_ = now;
}

} // namespace RenderEngine