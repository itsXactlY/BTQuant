#include "../include/performance/frame_time_graph.hpp"
#include "../src/imgui/implot.h"
#include "../src/imgui/imgui.h"

#include <algorithm>
#include <numeric>
#include <cmath>

namespace BTQuant {

FrameTimeGraph::FrameTimeGraph(size_t max_samples) 
    : max_samples_(max_samples)
    , enabled_(true)
    , current_frame_time_ms_(0.0)
    , current_fps_(0.0)
    , average_frame_time_ms_(0.0)
    , min_frame_time_ms_(std::numeric_limits<double>::max())
    , max_frame_time_ms_(0.0)
    , average_fps_(0.0) {
    frame_times_.reserve(max_samples_);
}

void FrameTimeGraph::start_frame() {
    if (!enabled_) return;
    
    frame_start_time_ = std::chrono::high_resolution_clock::now();
}

void FrameTimeGraph::end_frame() {
    if (!enabled_) return;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - frame_start_time_);
    
    double frame_time_ms = static_cast<double>(duration.count()) / 1000.0;
    
    // Update current values
    current_frame_time_ms_ = frame_time_ms;
    current_fps_ = frame_time_ms > 0 ? 1000.0 / frame_time_ms : 0.0;
    
    // Add to history
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        frame_times_.push_back(frame_time_ms);
        
        // Maintain buffer size
        if (frame_times_.size() > max_samples_) {
            frame_times_.erase(frame_times_.begin());
        }
    }
    
    update_statistics();
}

double FrameTimeGraph::get_current_frame_time_ms() const {
    return current_frame_time_ms_;
}

double FrameTimeGraph::get_average_frame_time_ms() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.empty()) return 0.0;
    
    double sum = std::accumulate(frame_times_.begin(), frame_times_.end(), 0.0);
    return sum / frame_times_.size();
}

double FrameTimeGraph::get_min_frame_time_ms() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.empty()) return 0.0;
    
    auto min_it = std::min_element(frame_times_.begin(), frame_times_.end());
    return *min_it;
}

double FrameTimeGraph::get_max_frame_time_ms() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.empty()) return 0.0;
    
    auto max_it = std::max_element(frame_times_.begin(), frame_times_.end());
    return *max_it;
}

double FrameTimeGraph::get_current_fps() const {
    return current_fps_;
}

double FrameTimeGraph::get_average_fps() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.empty()) return 0.0;
    
    double total_time = std::accumulate(frame_times_.begin(), frame_times_.end(), 0.0);
    if (total_time > 0.0) {
        // Calculate average FPS as total frames / total time in seconds
        return (frame_times_.size() * 1000.0) / total_time;
    }
    return 0.0;
}

std::vector<double> FrameTimeGraph::get_frame_time_history() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return frame_times_;
}

std::vector<double> FrameTimeGraph::get_time_indices() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::vector<double> indices(frame_times_.size());
    for (size_t i = 0; i < frame_times_.size(); ++i) {
        indices[i] = static_cast<double>(i);
    }
    return indices;
}

void FrameTimeGraph::reset() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    frame_times_.clear();
    
    // Reset statistics
    current_frame_time_ms_ = 0.0;
    current_fps_ = 0.0;
    average_frame_time_ms_ = 0.0;
    min_frame_time_ms_ = std::numeric_limits<double>::max();
    max_frame_time_ms_ = 0.0;
    average_fps_ = 0.0;
}

void FrameTimeGraph::set_enabled(bool enabled) {
    enabled_ = enabled;
    if (!enabled) {
        reset();
    }
}

bool FrameTimeGraph::is_enabled() const {
    return enabled_;
}

void FrameTimeGraph::update_statistics() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (frame_times_.empty()) {
        average_frame_time_ms_ = 0.0;
        min_frame_time_ms_ = std::numeric_limits<double>::max();
        max_frame_time_ms_ = 0.0;
        average_fps_ = 0.0;
        return;
    }
    
    // Calculate average frame time
    double sum = std::accumulate(frame_times_.begin(), frame_times_.end(), 0.0);
    average_frame_time_ms_ = sum / frame_times_.size();
    
    // Find min and max frame times
    auto minmax = std::minmax_element(frame_times_.begin(), frame_times_.end());
    min_frame_time_ms_ = *minmax.first;
    max_frame_time_ms_ = *minmax.second;
    
    // Calculate average FPS
    double total_time = sum;
    if (total_time > 0.0) {
        average_fps_ = (frame_times_.size() * 1000.0) / total_time;
    } else {
        average_fps_ = 0.0;
    }
}

void FrameTimeGraph::render(const char* title, float width, float height) {
    if (!enabled_) return;

    // Lock data access
    std::lock_guard<std::mutex> lock(data_mutex_);

    if (frame_times_.empty()) {
        // Show empty plot with some info
        ImGui::Text("Frame time data: No samples collected");
        return;
    }

    // Prepare data for plotting
    std::vector<double> x_values(frame_times_.size());
    for (size_t i = 0; i < frame_times_.size(); ++i) {
        x_values[i] = static_cast<double>(i);
    }

    // Create the plot
    ImVec2 plotSize = ImVec2(width, height);
    if (ImPlot::BeginPlot(title, plotSize)) {
        ImPlot::SetupAxes("Frame", "Time (ms)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);

        // Plot frame times
        ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), 2.0f); // Green line
        ImPlot::PlotLine("Frame Time", x_values.data(), frame_times_.data(), frame_times_.size());

        // Draw horizontal lines for min, max, and average
        if (!frame_times_.empty()) {
            double avg = average_frame_time_ms_;
            double min_val = min_frame_time_ms_;
            double max_val = max_frame_time_ms_;

            // Instead, let's plot a flat line using the same Y value across the range
            std::vector<double> avg_values(frame_times_.size(), avg);
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), 1.0f); // Yellow solid
            ImPlot::PlotLine("Average", x_values.data(), avg_values.data(), frame_times_.size());

            // Min line
            std::vector<double> min_values(frame_times_.size(), min_val);
            ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), 1.0f); // Cyan solid
            ImPlot::PlotLine("Min", x_values.data(), min_values.data(), frame_times_.size());

            // Max line
            std::vector<double> max_values(frame_times_.size(), max_val);
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f); // Red solid
            ImPlot::PlotLine("Max", x_values.data(), max_values.data(), frame_times_.size());
        }

        ImPlot::EndPlot();
    }

    // Display statistics
    ImGui::Text("Current: %.2f ms (%.1f FPS)", current_frame_time_ms_, current_fps_);
    ImGui::SameLine();
    ImGui::Text("Avg: %.2f ms (%.1f FPS)", average_frame_time_ms_, average_fps_);
    ImGui::SameLine();
    ImGui::Text("Min: %.2f ms | Max: %.2f ms", min_frame_time_ms_, max_frame_time_ms_);
}

// Global instance
FrameTimeGraph g_frame_time_graph;

} // namespace BTQuant