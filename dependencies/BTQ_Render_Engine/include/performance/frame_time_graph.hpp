#pragma once

#include <chrono>
#include <vector>
#include <mutex>
#include <string>

namespace BTQuant {

class FrameTimeGraph {
public:
    FrameTimeGraph(size_t max_samples = 200);  // Default to showing 200 samples

    // Start timing a frame
    void start_frame();

    // End timing a frame and record the frame time
    void end_frame();

    // Get the current frame time in milliseconds
    double get_current_frame_time_ms() const;

    // Get the average frame time over the recorded samples
    double get_average_frame_time_ms() const;

    // Get the minimum frame time over the recorded samples
    double get_min_frame_time_ms() const;

    // Get the maximum frame time over the recorded samples
    double get_max_frame_time_ms() const;

    // Get the current FPS
    double get_current_fps() const;

    // Get the average FPS over the recorded samples
    double get_average_fps() const;

    // Get frame time history for visualization
    std::vector<double> get_frame_time_history() const;

    // Get frame times as a vector of indices for plotting
    std::vector<double> get_time_indices() const;

    // Reset all recorded data
    void reset();

    // Enable/disable frame time recording
    void set_enabled(bool enabled);
    bool is_enabled() const;

    // Threshold management for performance warnings
    void set_warning_threshold(double ms);
    void set_critical_threshold(double ms);
    double get_warning_threshold() const;
    double get_critical_threshold() const;

    // Check for performance issues
    bool is_performance_issue_detected() const;
    double get_last_frame_time_deviation() const;

    // Render the frame time graph using ImGui/ImPlot
    void render(const char* title = "Frame Time Graph", float width = 0.0f, float height = 200.0f);

    // Get performance analysis data
    double get_variance() const;
    double get_standard_deviation() const;
    double get_percentile(double percentile) const;  // e.g., 95th percentile
    std::pair<size_t, size_t> get_frames_outside_thresholds() const;  // {warning_count, critical_count}
    double get_smoothed_frame_time(int window_size = 5) const;  // Moving average
    double get_median_frame_time() const;  // Median frame time
    double get_frame_time_at_percentile(double percentile) const;  // Frame time at specific percentile
    std::vector<std::pair<size_t, double>> get_spike_frames(double threshold_multiplier = 2.0) const;  // Frames that are spikes
    size_t get_consecutive_frame_drops(size_t min_drop_count = 3, double threshold_ms = 33.33) const;  // Consecutive slow frames

private:
    std::chrono::high_resolution_clock::time_point frame_start_time_;
    std::vector<double> frame_times_;
    size_t max_samples_;
    mutable std::mutex data_mutex_;
    bool enabled_;

    // Statistics
    double current_frame_time_ms_;
    double current_fps_;
    double average_frame_time_ms_;
    double min_frame_time_ms_;
    double max_frame_time_ms_;
    double average_fps_;

    // Performance thresholds
    double frame_time_threshold_warning_;   // Warning level (e.g., 16.67ms for 60fps)
    double frame_time_threshold_critical_;  // Critical level (e.g., 33.33ms for 30fps)
    bool performance_issue_detected_;
    std::chrono::high_resolution_clock::time_point last_performance_issue_time_;
    double last_frame_time_deviation_;

    // Update statistics
    void update_statistics();
};

// Global instance
extern FrameTimeGraph g_frame_time_graph;

} // namespace BTQuant