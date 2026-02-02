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
    , average_fps_(0.0)
    , frame_time_threshold_warning_(16.67)  // ~60 FPS threshold
    , frame_time_threshold_critical_(33.33) // ~30 FPS threshold
    , performance_issue_detected_(false)
    , last_performance_issue_time_(std::chrono::high_resolution_clock::now()) {
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

    // Check for performance issues
    if (current_frame_time_ms_ > frame_time_threshold_warning_) {
        performance_issue_detected_ = true;
        last_performance_issue_time_ = std::chrono::high_resolution_clock::now();
        last_frame_time_deviation_ = current_frame_time_ms_ - average_frame_time_ms_;
    } else {
        performance_issue_detected_ = false;
    }
}

void FrameTimeGraph::set_warning_threshold(double ms) {
    frame_time_threshold_warning_ = ms;
}

void FrameTimeGraph::set_critical_threshold(double ms) {
    frame_time_threshold_critical_ = ms;
}

double FrameTimeGraph::get_warning_threshold() const {
    return frame_time_threshold_warning_;
}

double FrameTimeGraph::get_critical_threshold() const {
    return frame_time_threshold_critical_;
}

bool FrameTimeGraph::is_performance_issue_detected() const {
    return performance_issue_detected_;
}

double FrameTimeGraph::get_last_frame_time_deviation() const {
    return last_frame_time_deviation_;
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

        // Draw threshold lines
        std::vector<double> warning_values(frame_times_.size(), frame_time_threshold_warning_);
        std::vector<double> critical_values(frame_times_.size(), frame_time_threshold_critical_);

        // Critical threshold line (red dashed)
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 0.7f), 1.5f); // Red line
        ImPlot::PlotLine("Critical Threshold", x_values.data(), critical_values.data(), frame_times_.size());

        // Warning threshold line (yellow dashed)
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 0.0f, 0.7f), 1.5f); // Yellow line
        ImPlot::PlotLine("Warning Threshold", x_values.data(), warning_values.data(), frame_times_.size());

        // Plot frame times
        ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), 2.0f); // Green line
        ImPlot::PlotLine("Frame Time", x_values.data(), frame_times_.data(), frame_times_.size());

        // Highlight performance issues
        for (size_t i = 0; i < frame_times_.size(); ++i) {
            if (frame_times_[i] > frame_time_threshold_warning_) {
                // Draw a vertical line to highlight performance issue
                double x_pos = x_values[i];
                double y_values[] = {0.0, frame_times_[i]};
                double x_coords[] = {x_pos, x_pos};

                if (frame_times_[i] > frame_time_threshold_critical_) {
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5, ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f, ImVec4(1.0f, 0.0f, 0.0f, 0.5f)); // Red marker for critical
                    ImPlot::PlotScatter("Critical Issue", &x_pos, &frame_times_[i], 1);
                } else {
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 4, ImVec4(1.0f, 1.0f, 0.0f, 1.0f), 1.0f, ImVec4(1.0f, 1.0f, 0.0f, 0.5f)); // Yellow marker for warning
                    ImPlot::PlotScatter("Warning Issue", &x_pos, &frame_times_[i], 1);
                }
            }
        }

        // Draw horizontal lines for min, max, and average
        if (!frame_times_.empty()) {
            double avg = average_frame_time_ms_;
            double min_val = min_frame_time_ms_;
            double max_val = max_frame_time_ms_;

            // Average line (yellow)
            std::vector<double> avg_values(frame_times_.size(), avg);
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), 1.0f); // Yellow solid
            ImPlot::PlotLine("Average", x_values.data(), avg_values.data(), frame_times_.size());

            // Min line (cyan)
            std::vector<double> min_values(frame_times_.size(), min_val);
            ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), 1.0f); // Cyan solid
            ImPlot::PlotLine("Min", x_values.data(), min_values.data(), frame_times_.size());

            // Max line (red)
            std::vector<double> max_values(frame_times_.size(), max_val);
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f); // Red solid
            ImPlot::PlotLine("Max", x_values.data(), max_values.data(), frame_times_.size());
        }

        ImPlot::EndPlot();
    }

    // Display statistics with performance issue indicators
    ImGui::Text("Current: %.2f ms (%.1f FPS)", current_frame_time_ms_, current_fps_);
    ImGui::SameLine();
    ImGui::Text("Avg: %.2f ms (%.1f FPS)", average_frame_time_ms_, average_fps_);
    ImGui::SameLine();
    ImGui::Text("Min: %.2f ms | Max: %.2f ms", min_frame_time_ms_, max_frame_time_ms_);

    // Performance issue status
    if (performance_issue_detected_) {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_performance_issue_time_);

        ImVec4 issue_color = current_frame_time_ms_ > frame_time_threshold_critical_ ?
                            ImVec4(1.0f, 0.0f, 0.0f, 1.0f) : ImVec4(1.0f, 1.0f, 0.0f, 1.0f);

        ImGui::TextColored(issue_color, "PERFORMANCE ISSUE DETECTED!");
        ImGui::SameLine();
        ImGui::Text("Deviation: %.2f ms from average", last_frame_time_deviation_);
    } else {
        ImGui::Text("Status: OK");
    }

    // Threshold controls
    ImGui::Separator();
    ImGui::Text("Thresholds:");
    ImGui::SameLine();
    ImGui::PushItemWidth(100);
    double warning_min = 1.0;
    double warning_max = 50.0;
    double critical_min = 1.0;
    double critical_max = 100.0;
    ImGui::DragScalar("Warning (ms)", ImGuiDataType_Double, &frame_time_threshold_warning_, 0.1f, &warning_min, &warning_max, "%.2f");
    ImGui::SameLine();
    ImGui::DragScalar("Critical (ms)", ImGuiDataType_Double, &frame_time_threshold_critical_, 0.1f, &critical_min, &critical_max, "%.2f");
    ImGui::PopItemWidth();
}

// Global instance
FrameTimeGraph g_frame_time_graph;

} // namespace BTQuant