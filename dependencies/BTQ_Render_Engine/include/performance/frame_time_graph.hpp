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
    
    // Render the frame time graph using ImGui/ImPlot
    void render(const char* title = "Frame Time Graph", float width = 0.0f, float height = 200.0f);

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
    
    // Update statistics
    void update_statistics();
};

// Global instance
extern FrameTimeGraph g_frame_time_graph;

} // namespace BTQuant