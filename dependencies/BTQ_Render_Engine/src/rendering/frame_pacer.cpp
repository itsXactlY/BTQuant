#include "rendering/frame_pacer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <ratio>

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
    , frame_prediction_error_(0.0)
    , prediction_integral_(0.0)
    , prediction_derivative_(0.0)
    , last_prediction_error_(0.0)
    , frame_budget_tracker_(1000.0 / config.target_fps)
    , frame_jitter_compensator_(0.0)
    , frame_stability_score_(1.0)
    , frame_phase_lock_(false)
    , phase_reference_time_(std::chrono::high_resolution_clock::now())
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

    // Initialize PID controller parameters for frame prediction
    pid_controller_.kp = 0.1;  // Proportional gain
    pid_controller_.ki = 0.01; // Integral gain
    pid_controller_.kd = 0.001; // Derivative gain

    // Initialize frame budget tracker with target frame time
    frame_budget_tracker_ = 1000.0 / config.target_fps;

    // Initialize phase reference time for frame synchronization
    phase_reference_time_ = std::chrono::high_resolution_clock::now();
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

    // Perform frame synchronization if enabled
    if (config_.enable_frame_smoothing && frame_phase_lock_) {
        synchronize_frame_phase();
    }
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

    // Update advanced frame pacing components
    if (config_.enable_frame_smoothing) {
        update_frame_stability();
        apply_jitter_compensation();
        update_frame_budget();
        synchronize_frame_phase();
    }

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

    // Reset PID controller when config changes to prevent instability
    frame_prediction_error_ = 0.0;
    prediction_integral_ = 0.0;
    prediction_derivative_ = 0.0;
    last_prediction_error_ = 0.0;

    // Reset advanced frame pacing components when config changes
    frame_budget_tracker_ = 1000.0 / new_config.target_fps;
    frame_jitter_compensator_ = 0.0;
    frame_stability_score_ = 1.0;
    frame_phase_lock_ = false;
    phase_reference_time_ = std::chrono::high_resolution_clock::now();
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

    // Reset PID controller values
    frame_prediction_error_ = 0.0;
    prediction_integral_ = 0.0;
    prediction_derivative_ = 0.0;
    last_prediction_error_ = 0.0;

    // Reset advanced frame pacing components
    frame_budget_tracker_ = 1000.0 / config_.target_fps;
    frame_jitter_compensator_ = 0.0;
    frame_stability_score_ = 1.0;
    frame_phase_lock_ = false;
    phase_reference_time_ = std::chrono::high_resolution_clock::now();
}

double FramePacer::calculate_sleep_duration() const {
    if (adaptive_target_fps_ == 0) {
        return 0.0; // Unlimited FPS
    }

    double target_frame_time = 1000.0 / adaptive_target_fps_;

    // Predict the next frame time based on recent performance
    double predicted_frame_time = predict_frame_time();

    // Adjust target based on prediction to maintain consistency
    if (config_.enable_frame_smoothing) {
        target_frame_time = (target_frame_time * 0.7) + (predicted_frame_time * 0.3);
    }

    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - frame_start_time_).count();

    double remaining_time = target_frame_time - elapsed;

    // Apply PID-like control to smooth out timing variations
    if (config_.enable_frame_smoothing && frame_count_ > 2) {
        double error = target_frame_time - predicted_frame_time;

        // Since we can't modify state in a const method, use a simplified approach
        // Apply proportional control only to avoid state changes
        double p_correction = 0.1 * error;  // Proportional term only for const method

        // Apply correction but limit it to prevent over-correction
        double max_correction = target_frame_time * 0.2; // Limit to 20% of target time
        p_correction = std::clamp(p_correction, -max_correction, max_correction);
        remaining_time += p_correction;
    }

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

    // Apply advanced frame pacing enhancements
    if (config_.enable_frame_smoothing) {
        // Apply jitter compensation if available
        remaining_time += frame_jitter_compensator_;

        // Adjust based on frame stability score
        remaining_time *= frame_stability_score_;

        // Apply frame budget adjustments
        remaining_time = (remaining_time * 0.8) + (frame_budget_tracker_ * 0.2);
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
    // More precise sleep implementation using hybrid approach
    auto start = std::chrono::high_resolution_clock::now();
    auto target_time = start + std::chrono::duration<double, std::milli>(sleep_duration_ms);

    // Use different strategies based on sleep duration
    if (sleep_duration_ms > 10.0) {
        // For longer sleeps, use standard sleep with margin
        auto sleep_ms = static_cast<int>(sleep_duration_ms * 0.8);  // Sleep 80% of the time
        if (sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    } else if (sleep_duration_ms > 2.0) {
        // For medium sleeps, sleep most of the way
        auto sleep_ms = static_cast<int>(sleep_duration_ms - 0.5);  // Sleep all but 0.5ms
        if (sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
    }
    // For very short sleeps (< 2ms), skip the sleep and go directly to busy wait
    // as the OS sleep functions are not precise enough

    // Busy wait for the final precise timing using a more efficient approach
    auto remaining_time = target_time - std::chrono::high_resolution_clock::now();
    while (remaining_time.count() > 0) {
        // For very short waits, use CPU yield to minimize power usage
        if (remaining_time.count() < 100000) { // Less than 100 microseconds remaining
            std::this_thread::yield();
        } else {
            // For longer busy waits, use a short sleep to reduce CPU usage
            std::this_thread::sleep_for(std::chrono::microseconds(
                static_cast<int>(std::min(static_cast<long long>(remaining_time.count() / 10), 500LL)))); // Max 500us sleep
        }

        remaining_time = target_time - std::chrono::high_resolution_clock::now();
    }
}

void FramePacer::detect_spikes(double frame_time_ms) {
    // Add current frame time to spike detection history
    spike_detector_.spike_history[spike_detector_.spike_index] = frame_time_ms;
    spike_detector_.spike_index = (spike_detector_.spike_index + 1) % SPIKE_DETECTION_WINDOW;

    // Calculate dynamic threshold based on recent frame time statistics
    double avg_frame_time = 0.0;
    size_t valid_samples = 0;

    // Calculate average of recent frame times for better baseline
    for (size_t i = 0; i < SPIKE_DETECTION_WINDOW; ++i) {
        if (spike_detector_.spike_history[i] > 0) {
            avg_frame_time += spike_detector_.spike_history[i];
            valid_samples++;
        }
    }

    if (valid_samples > 0) {
        avg_frame_time /= valid_samples;
        // Set threshold as 2.5x the recent average frame time, with minimum based on target FPS
        double dynamic_threshold = std::max(avg_frame_time * 2.5, spike_detector_.baseline_frame_time * 2.0);

        // Check if this frame is a spike (significantly longer than dynamic threshold)
        if (frame_time_ms > dynamic_threshold) {
            spike_detector_.spike_count_recent++;

            // Trigger frame pacing adjustments when spikes are detected
            if (config_.enable_frame_smoothing) {
                // Temporarily reduce target FPS to accommodate the spike
                adaptive_target_fps_ = static_cast<uint32_t>(std::max(
                    static_cast<double>(config_.target_fps) * 0.8,  // Don't go below 80% of target
                    static_cast<double>(adaptive_target_fps_) * 0.95)); // Reduce by 5%
            }
        } else if (frame_time_ms < avg_frame_time * 0.7) {
            // If we're seeing consistently faster frames, gradually increase baseline
            spike_detector_.baseline_frame_time = avg_frame_time * 0.9; // Adjust baseline downward
        }
    }

    // Decrement recent spike count periodically to decay old spikes
    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - spike_detector_.last_spike_decay).count() > 1000) {
        // Decay spike count more gradually to maintain awareness of recent performance
        spike_detector_.spike_count_recent = static_cast<int>(spike_detector_.spike_count_recent * 0.8);
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

    // Calculate adaptive threshold based on multiple factors
    double adaptive_threshold = expected_frame_time_us * 1.8; // Base threshold

    // Increase threshold if we're seeing high variance to avoid false positives
    if (stats_.frame_time_variance > config_.frame_time_variance_threshold * 1000000.0) { // Convert to microseconds^2
        adaptive_threshold *= 1.5; // Higher tolerance during high variance
    }

    // Also check against recent frame time history for more accurate threshold
    if (frame_count_ > 10) {
        // Calculate average of recent frame times
        size_t sample_count = std::min(static_cast<size_t>(frame_count_),
                                     static_cast<size_t>(FRAME_HISTORY_SIZE / 4));
        double recent_avg = 0.0;
        size_t valid_samples = 0;

        for (size_t i = 0; i < sample_count; ++i) {
            size_t idx = (frame_count_ - 1 - i) % FRAME_HISTORY_SIZE;
            if (frame_time_history_[idx] > 0) {
                recent_avg += frame_time_history_[idx];
                valid_samples++;
            }
        }

        if (valid_samples > 0) {
            recent_avg = (recent_avg / valid_samples) * 1000.0; // Convert to microseconds

            // Use the higher of the two thresholds to be more accurate
            adaptive_threshold = std::max(adaptive_threshold, recent_avg * 2.0);
        }
    }

    // Check if frame time exceeds adaptive threshold
    if (frame_time_us > adaptive_threshold) {
        dropped_frame_counter_++;

        // When a frame is detected as dropped, trigger adaptive FPS adjustment
        if (config_.enable_adaptive_sync) {
            // Reduce the adaptive target FPS to account for performance issues
            adaptive_target_fps_ = static_cast<uint32_t>(std::max(
                static_cast<double>(config_.target_fps) * 0.6,  // Don't go below 60% of target
                static_cast<double>(adaptive_target_fps_) * 0.85)); // Reduce by 15% for more aggressive response
        }
    } else if (frame_time_us < expected_frame_time_us * 0.7 && adaptive_target_fps_ < config_.target_fps) {
        // If we're consistently under budget, gradually increase target FPS
        // Only do this if we're not already at the target FPS
        adaptive_target_fps_ = std::min(config_.target_fps,
                                      adaptive_target_fps_ + 1); // Conservative increase
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
                // Gradually increase FPS when performance is good
                adaptive_target_fps_ = std::min(config_.target_fps,
                                              adaptive_target_fps_ + 5); // Gradually increase
            }

            // Recovery logic: if we've been performing well for a sustained period, gradually increase target
            if (avg_recent_fps > config_.target_fps * 0.95 && adaptive_target_fps_ < config_.target_fps) {
                // If we're consistently hitting close to our target FPS, gradually increase toward the original target
                adaptive_target_fps_ = std::min(config_.target_fps,
                                              adaptive_target_fps_ + 2); // Slow recovery
            }
        }
    }

    last_adaptive_update_ = now;
}

double FramePacer::predict_frame_time() const {
    if (frame_count_ < 3) {
        return 1000.0 / config_.target_fps;
    }

    // Use a weighted average of recent frame times to predict the next frame time
    size_t sample_count = std::min(static_cast<size_t>(frame_count_),
                                 static_cast<size_t>(SMOOTHING_WINDOW));

    double weighted_sum = 0.0;
    double weight_sum = 0.0;

    for (size_t i = 0; i < sample_count; ++i) {
        // More recent frames have higher weights
        double weight = 1.0 + (static_cast<double>(i) / sample_count);
        size_t idx = (frame_count_ - 1 - i) % SMOOTHING_WINDOW;
        weighted_sum += smoothed_frame_times_[idx] * weight;
        weight_sum += weight;
    }

    return weighted_sum / weight_sum;
}

// Note: This method is intentionally not const to allow updating internal state
double FramePacer::apply_pid_control(double error) {
    // Update integral term
    prediction_integral_ += error;

    // Apply anti-windup protection to prevent integral windup
    const double integral_limit = 100.0; // Limit the integral term to prevent windup
    prediction_integral_ = std::clamp(prediction_integral_, -integral_limit, integral_limit);

    // Calculate derivative term
    prediction_derivative_ = error - last_prediction_error_;

    // Apply PID formula
    double output = (pid_controller_.kp * error) +
                   (pid_controller_.ki * prediction_integral_) +
                   (pid_controller_.kd * prediction_derivative_);

    // Store current error for next derivative calculation
    last_prediction_error_ = error;

    return output;
}

void FramePacer::update_frame_stability() {
    if (frame_count_ < 10) {
        frame_stability_score_ = 1.0;
        return;
    }

    // Calculate stability based on frame time variance and consistency
    double stability_factor = 1.0 - std::min(stats_.frame_time_variance / 100.0, 0.9); // Normalize variance impact

    // Factor in recent frame spikes
    double spike_impact = std::min(static_cast<double>(spike_detector_.spike_count_recent) / 10.0, 0.5);
    stability_factor -= spike_impact;

    // Factor in dropped frames
    if (stats_.total_frames > 0) {
        double drop_rate = static_cast<double>(dropped_frame_counter_) / stats_.total_frames;
        stability_factor -= std::min(drop_rate * 10.0, 0.3); // Weight dropped frames heavily
    }

    // Clamp stability score between 0.1 and 1.0
    frame_stability_score_ = std::clamp(stability_factor, 0.1, 1.0);
}

void FramePacer::apply_jitter_compensation() {
    if (frame_count_ < 5) {
        frame_jitter_compensator_ = 0.0;
        return;
    }

    // Calculate jitter as deviation from ideal frame timing
    double target_frame_time = 1000.0 / adaptive_target_fps_;
    double actual_frame_time = 0.0;

    if (frame_count_ > 0) {
        actual_frame_time = frame_time_history_[(frame_count_ - 1) % FRAME_HISTORY_SIZE];
    }

    double jitter = actual_frame_time - target_frame_time;

    // Apply exponential moving average to smooth jitter compensation
    const double JITTER_SMOOTHING_FACTOR = 0.1;
    frame_jitter_compensator_ = (frame_jitter_compensator_ * (1.0 - JITTER_SMOOTHING_FACTOR)) +
                               (jitter * JITTER_SMOOTHING_FACTOR);

    // Limit compensation to prevent overcorrection
    const double MAX_JITTER_COMPENSATION = target_frame_time * 0.3; // 30% of target time
    frame_jitter_compensator_ = std::clamp(frame_jitter_compensator_,
                                         -MAX_JITTER_COMPENSATION,
                                         MAX_JITTER_COMPENSATION);
}

void FramePacer::update_frame_budget() {
    if (!config_.enable_frame_smoothing) {
        frame_budget_tracker_ = 1000.0 / config_.target_fps;
        return;
    }

    // Calculate how much time we have left in our frame budget
    double target_frame_time = 1000.0 / config_.target_fps;
    double current_frame_time = 0.0;

    if (frame_count_ > 0) {
        current_frame_time = frame_time_history_[(frame_count_ - 1) % FRAME_HISTORY_SIZE];
    }

    // Update budget based on actual vs expected frame time
    double budget_delta = target_frame_time - current_frame_time;

    // Apply smoothing to prevent sudden budget changes
    const double BUDGET_SMOOTHING = 0.05;
    frame_budget_tracker_ = (frame_budget_tracker_ * (1.0 - BUDGET_SMOOTHING)) +
                           (target_frame_time * BUDGET_SMOOTHING);

    // Adjust budget based on stability and recent performance
    frame_budget_tracker_ *= frame_stability_score_;

    // If we're consistently under budget, gradually increase budget to maintain target FPS
    if (current_frame_time < target_frame_time * 0.8 && frame_stability_score_ > 0.8) {
        frame_budget_tracker_ = std::min(frame_budget_tracker_ * 1.001, target_frame_time);
    }

    // If we're consistently over budget, reduce budget to maintain stability
    if (current_frame_time > target_frame_time * 1.2) {
        frame_budget_tracker_ = std::max(frame_budget_tracker_ * 0.999, target_frame_time * 0.7);
    }
}

void FramePacer::synchronize_frame_phase() {
    if (!config_.enable_frame_smoothing) {
        frame_phase_lock_ = false;
        return;
    }

    // Calculate phase difference between expected and actual frame timing
    auto now = std::chrono::high_resolution_clock::now();
    auto expected_frame_interval = std::chrono::duration<double, std::milli>(
        std::chrono::milliseconds(static_cast<int>(1000 / adaptive_target_fps_)));

    // Calculate how many intervals have passed since reference time
    auto time_since_ref = now - phase_reference_time_;
    double time_since_ref_ms = std::chrono::duration<double, std::milli>(time_since_ref).count();
    double expected_interval_ms = expected_frame_interval.count();
    double intervals_passed = time_since_ref_ms / expected_interval_ms;

    // Calculate phase error (how far we are from ideal frame boundary)
    double phase_fraction = intervals_passed - std::floor(intervals_passed);
    double phase_error = (phase_fraction > 0.5) ? (1.0 - phase_fraction) : phase_fraction;

    // If phase error is small, we're synchronized
    frame_phase_lock_ = (phase_error < 0.1); // Within 10% of frame interval

    // If we're out of phase, adjust timing to get back in sync
    if (!frame_phase_lock_ && phase_error > 0.2) { // Significant phase error
        // Apply gentle correction to get back in phase
        double phase_correction = (expected_interval_ms * 0.1) *
                                 ((phase_fraction > 0.5) ? -1.0 : 1.0);

        // Apply correction gradually to avoid jarring transitions
        frame_jitter_compensator_ += phase_correction * 0.1;
    }
}

} // namespace RenderEngine