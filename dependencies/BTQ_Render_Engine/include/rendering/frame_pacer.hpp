#pragma once

#include <chrono>
#include <thread>
#include <vector>
#include <mutex>

namespace RenderEngine {

/**
 * @brief Frame Pacer implementation to maintain consistent frame rates,
 * smooth out rendering spikes, and prevent dropped frames.
 */
class FramePacer {
public:
    /**
     * @brief Configuration options for the frame pacer
     */
    struct Config {
        uint32_t target_fps = 60;           ///< Target frames per second
        bool enable_adaptive_sync = true;   ///< Enable adaptive sync to adjust to workload
        bool enable_frame_smoothing = true; ///< Enable frame time smoothing to reduce spikes
        bool enable_burst_reduction = true; ///< Enable burst reduction to prevent frame drops
        double frame_time_variance_threshold = 0.016;  ///< Threshold for frame time variance (in seconds)
    };

    /**
     * @brief Statistics about frame pacing performance
     */
    struct Stats {
        double avg_frame_time_ms = 0.0;     ///< Average frame time in milliseconds
        double min_frame_time_ms = 0.0;     ///< Minimum frame time in milliseconds
        double max_frame_time_ms = 0.0;     ///< Maximum frame time in milliseconds
        double current_fps = 0.0;           ///< Current frames per second
        double smoothed_fps = 0.0;          ///< Smoothed frames per second
        uint64_t total_frames = 0;          ///< Total frames rendered
        double frame_time_variance = 0.0;   ///< Variance in frame times
    };

    explicit FramePacer(const Config& config);
    FramePacer();  // Default constructor
    ~FramePacer();

    /**
     * @brief Begin a frame - call at the start of each render cycle
     */
    void begin_frame();

    /**
     * @brief End a frame - call at the end of each render cycle
     */
    void end_frame();

    /**
     * @brief Wait for the next frame to maintain target FPS
     */
    void wait_for_next_frame();

    /**
     * @brief Get current frame pacing statistics
     */
    Stats get_stats() const;

    /**
     * @brief Update configuration at runtime
     */
    void update_config(const Config& new_config);

    /**
     * @brief Reset frame statistics
     */
    void reset_stats();

private:
    Config config_;
    Stats stats_;
    
    mutable std::mutex stats_mutex_;
    
    std::chrono::high_resolution_clock::time_point frame_start_time_;
    std::chrono::high_resolution_clock::time_point last_frame_time_;
    
    std::vector<double> frame_time_history_;
    static constexpr size_t FRAME_HISTORY_SIZE = 120;  // Keep history of 120 frames
    
    double accumulated_frame_time_ = 0.0;
    uint64_t frame_count_ = 0;
    
    // For adaptive sync
    double recent_avg_frame_time_ = 0.0;
    size_t recent_frame_count_ = 0;
    
    // For frame smoothing
    std::vector<double> smoothed_frame_times_;
    static constexpr size_t SMOOTHING_WINDOW = 5;
    
    // Timing helpers
    double calculate_sleep_duration() const;
    void update_statistics();
    void update_frame_variance();
};

} // namespace RenderEngine