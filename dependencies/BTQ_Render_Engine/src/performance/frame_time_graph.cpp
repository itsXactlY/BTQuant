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

double FrameTimeGraph::get_variance() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.size() < 2) return 0.0;

    double mean = average_frame_time_ms_;
    double sum_sq_diff = 0.0;
    for (double ft : frame_times_) {
        double diff = ft - mean;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / frame_times_.size();
}

double FrameTimeGraph::get_standard_deviation() const {
    return std::sqrt(get_variance());
}

double FrameTimeGraph::get_percentile(double percentile) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.empty()) return 0.0;

    if (percentile < 0.0 || percentile > 100.0) return 0.0;

    std::vector<double> sorted_times = frame_times_;
    std::sort(sorted_times.begin(), sorted_times.end());

    double index = (percentile / 100.0) * (sorted_times.size() - 1);
    size_t lower_idx = static_cast<size_t>(std::floor(index));
    size_t upper_idx = static_cast<size_t>(std::ceil(index));

    if (lower_idx == upper_idx) {
        return sorted_times[lower_idx];
    }

    // Linear interpolation between adjacent values
    double fraction = index - lower_idx;
    return sorted_times[lower_idx] + fraction * (sorted_times[upper_idx] - sorted_times[lower_idx]);
}

std::pair<size_t, size_t> FrameTimeGraph::get_frames_outside_thresholds() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    size_t warning_count = 0;
    size_t critical_count = 0;

    for (double ft : frame_times_) {
        if (ft > frame_time_threshold_critical_) {
            critical_count++;
        } else if (ft > frame_time_threshold_warning_) {
            warning_count++;
        }
    }

    return {warning_count, critical_count};
}

double FrameTimeGraph::get_smoothed_frame_time(int window_size) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.empty()) return 0.0;

    // Use the most recent frame times up to the window size
    size_t count = std::min(static_cast<size_t>(window_size), frame_times_.size());
    double sum = 0.0;
    for (size_t i = frame_times_.size() - count; i < frame_times_.size(); ++i) {
        sum += frame_times_[i];
    }

    return sum / count;
}

double FrameTimeGraph::get_median_frame_time() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.empty()) return 0.0;

    std::vector<double> sorted_times = frame_times_;
    std::sort(sorted_times.begin(), sorted_times.end());

    size_t size = sorted_times.size();
    if (size % 2 == 0) {
        return (sorted_times[size/2 - 1] + sorted_times[size/2]) / 2.0;
    } else {
        return sorted_times[size/2];
    }
}

double FrameTimeGraph::get_frame_time_at_percentile(double percentile) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.empty()) return 0.0;

    if (percentile < 0.0 || percentile > 100.0) return 0.0;

    std::vector<double> sorted_times = frame_times_;
    std::sort(sorted_times.begin(), sorted_times.end());

    double index = (percentile / 100.0) * (sorted_times.size() - 1);
    size_t lower_idx = static_cast<size_t>(std::floor(index));
    size_t upper_idx = static_cast<size_t>(std::ceil(index));

    if (lower_idx == upper_idx) {
        return sorted_times[lower_idx];
    }

    // Linear interpolation between adjacent values
    double fraction = index - lower_idx;
    return sorted_times[lower_idx] + fraction * (sorted_times[upper_idx] - sorted_times[lower_idx]);
}

std::vector<std::pair<size_t, double>> FrameTimeGraph::get_spike_frames(double threshold_multiplier) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::vector<std::pair<size_t, double>> spikes;

    if (frame_times_.size() < 3) return spikes; // Need at least 3 frames to detect spikes

    // Calculate moving average to detect outliers
    for (size_t i = 1; i < frame_times_.size() - 1; ++i) {
        double prev_frame = frame_times_[i - 1];
        double curr_frame = frame_times_[i];
        double next_frame = frame_times_[i + 1];

        // Calculate local average of surrounding frames
        double local_avg = (prev_frame + next_frame) / 2.0;

        // If current frame is significantly higher than local average, it's a spike
        if (curr_frame > local_avg * threshold_multiplier) {
            spikes.push_back({i, curr_frame});
        }
    }

    return spikes;
}

size_t FrameTimeGraph::get_consecutive_frame_drops(size_t min_drop_count, double threshold_ms) const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (frame_times_.size() < min_drop_count) return 0;

    size_t max_consecutive = 0;
    size_t current_consecutive = 0;

    for (double frame_time : frame_times_) {
        if (frame_time > threshold_ms) {
            current_consecutive++;
            max_consecutive = std::max(max_consecutive, current_consecutive);
        } else {
            current_consecutive = 0;
        }
    }

    // Only return if we have at least the minimum drop count
    return max_consecutive >= min_drop_count ? max_consecutive : 0;
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

        // Critical threshold line (red)
        ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.0f, 0.0f, 0.8f), 2.0f); // Thicker red line
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
        ImPlot::PlotLine("Critical Threshold", x_values.data(), critical_values.data(), frame_times_.size());
        ImPlot::PopStyleVar();

        // Warning threshold line (yellow/orange)
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.6f, 0.0f, 0.8f), 1.5f); // Orange line
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f);
        ImPlot::PlotLine("Warning Threshold", x_values.data(), warning_values.data(), frame_times_.size());
        ImPlot::PopStyleVar();

        // Plot frame times with enhanced visualization
        ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), 2.0f); // Light green line
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
        ImPlot::PlotLine("Frame Time", x_values.data(), frame_times_.data(), frame_times_.size());
        ImPlot::PopStyleVar();

        // Highlight performance issues with filled areas
        std::vector<double> warning_x, warning_y, critical_x, critical_y;
        for (size_t i = 0; i < frame_times_.size(); ++i) {
            if (frame_times_[i] > frame_time_threshold_warning_) {
                if (frame_times_[i] > frame_time_threshold_critical_) {
                    // Critical issue - red
                    critical_x.push_back(x_values[i]);
                    critical_y.push_back(frame_times_[i]);
                } else {
                    // Warning issue - orange
                    warning_x.push_back(x_values[i]);
                    warning_y.push_back(frame_times_[i]);
                }
            }
        }

        // Fill areas for warning and critical issues separately
        if (!warning_x.empty()) {
            ImPlot::SetNextFillStyle(ImVec4(1.0f, 0.6f, 0.0f, 0.2f)); // Semi-transparent orange
            ImPlot::PlotShaded("Warning Spikes", warning_x.data(), warning_y.data(), warning_x.size(), frame_time_threshold_warning_);
        }

        if (!critical_x.empty()) {
            ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.0f, 0.0f, 0.3f)); // Semi-transparent red
            ImPlot::PlotShaded("Critical Spikes", critical_x.data(), critical_y.data(), critical_x.size(), frame_time_threshold_critical_);
        }

        // Draw markers for performance issues
        for (size_t i = 0; i < frame_times_.size(); ++i) {
            if (frame_times_[i] > frame_time_threshold_warning_) {
                double x_pos = x_values[i];

                if (frame_times_[i] > frame_time_threshold_critical_) {
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 8, ImVec4(0.8f, 0.0f, 0.0f, 1.0f), 3.0f, ImVec4(1.0f, 1.0f, 1.0f, 0.8f)); // Large red circle for critical
                    ImPlot::PlotScatter("Critical", &x_pos, &frame_times_[i], 1);
                } else {
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 7, ImVec4(1.0f, 0.6f, 0.0f, 1.0f), 2.5f, ImVec4(0.0f, 0.0f, 0.0f, 0.8f)); // Orange diamond for warning
                    ImPlot::PlotScatter("Warning", &x_pos, &frame_times_[i], 1);
                }
            }
        }

        // Draw horizontal lines for min, max, and average with better visibility
        if (!frame_times_.empty()) {
            double avg = average_frame_time_ms_;
            double min_val = min_frame_time_ms_;
            double max_val = max_frame_time_ms_;

            // Average line (blue)
            std::vector<double> avg_values(frame_times_.size(), avg);
            ImPlot::SetNextLineStyle(ImVec4(0.0f, 0.5f, 1.0f, 0.9f), 1.5f); // Blue solid
            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f);
            ImPlot::PlotLine("Average", x_values.data(), avg_values.data(), frame_times_.size());
            ImPlot::PopStyleVar();

            // Min line (light blue) - using stipple pattern to simulate dashed
            std::vector<double> min_values(frame_times_.size(), min_val);
            ImPlot::SetNextLineStyle(ImVec4(0.2f, 0.8f, 1.0f, 0.7f), 1.0f); // Light blue
            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.0f);
            ImPlot::PlotLine("Min", x_values.data(), min_values.data(), frame_times_.size());
            ImPlot::PopStyleVar();

            // Max line (purple) - using stipple pattern to simulate dashed
            std::vector<double> max_values(frame_times_.size(), max_val);
            ImPlot::SetNextLineStyle(ImVec4(0.8f, 0.4f, 1.0f, 0.7f), 1.0f); // Purple
            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.0f);
            ImPlot::PlotLine("Max", x_values.data(), max_values.data(), frame_times_.size());
            ImPlot::PopStyleVar();
        }

        ImPlot::EndPlot();
    }

    // Display enhanced statistics with performance issue indicators
    ImGui::Text("Current: %.2f ms (%.1f FPS)", current_frame_time_ms_, current_fps_);
    ImGui::SameLine();
    ImGui::Text("Avg: %.2f ms (%.1f FPS)", average_frame_time_ms_, average_fps_);
    ImGui::SameLine();
    ImGui::Text("Min: %.2f ms | Max: %.2f ms", min_frame_time_ms_, max_frame_time_ms_);

    // Additional performance metrics
    ImGui::Text("Sample Count: %zu | Range: %.2f ms", frame_times_.size(), max_frame_time_ms_ - min_frame_time_ms_);

    // Performance issue status with more detail
    if (performance_issue_detected_) {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_performance_issue_time_);

        ImVec4 issue_color = current_frame_time_ms_ > frame_time_threshold_critical_ ?
                            ImVec4(0.8f, 0.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.6f, 0.0f, 1.0f);

        ImGui::TextColored(issue_color, "PERFORMANCE ISSUE DETECTED!");
        ImGui::SameLine();
        ImGui::Text("Deviation: %.2f ms from average", last_frame_time_deviation_);

        // Show additional performance insights
        if (current_frame_time_ms_ > frame_time_threshold_critical_) {
            ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "CRITICAL: Frame time exceeds 30 FPS threshold!");
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "WARNING: Frame time exceeds 60 FPS threshold!");
        }
    } else {
        ImGui::Text("Status: OK");
    }

    // Threshold controls with better layout
    ImGui::Separator();
    ImGui::Text("Performance Thresholds:");
    ImGui::Columns(2, "thresholds"); // Create columns for better layout

    ImGui::SetColumnWidth(0, 150);
    ImGui::SetColumnWidth(1, 150);

    ImGui::Text("Warning Level:");
    ImGui::NextColumn();
    ImGui::PushItemWidth(-1);
    double warning_min = 1.0;
    double warning_max = 50.0;
    ImGui::DragScalar("##Warning", ImGuiDataType_Double, &frame_time_threshold_warning_, 0.1f, &warning_min, &warning_max, "%.2f ms (~%.1f FPS)");
    ImGui::PopItemWidth();
    ImGui::NextColumn();

    ImGui::Text("Critical Level:");
    ImGui::NextColumn();
    ImGui::PushItemWidth(-1);
    double critical_min = 1.0;
    double critical_max = 100.0;
    ImGui::DragScalar("##Critical", ImGuiDataType_Double, &frame_time_threshold_critical_, 0.1f, &critical_min, &critical_max, "%.2f ms (~%.1f FPS)");
    ImGui::PopItemWidth();
    ImGui::NextColumn();

    ImGui::Columns(1); // Close columns

    // Additional performance analysis controls
    if (ImGui::CollapsingHeader("Performance Analysis")) {
        ImGui::Text("Frame Time Distribution:");

        // Calculate distribution bins
        if (!frame_times_.empty()) {
            double min_time = *std::min_element(frame_times_.begin(), frame_times_.end());
            double max_time = *std::max_element(frame_times_.begin(), frame_times_.end());

            // Create histogram
            int bins = 10;
            std::vector<int> counts(bins, 0);
            double bin_size = (max_time - min_time) / bins;

            for (double ft : frame_times_) {
                int bin_idx = static_cast<int>((ft - min_time) / bin_size);
                if (bin_idx >= bins) bin_idx = bins - 1;
                counts[bin_idx]++;
            }

            // Display histogram with color coding
            for (int i = 0; i < bins; ++i) {
                double range_start = min_time + i * bin_size;
                double range_end = min_time + (i + 1) * bin_size;

                // Color code based on performance
                ImVec4 bin_color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f); // Green for good
                if (range_start >= frame_time_threshold_critical_) {
                    bin_color = ImVec4(0.8f, 0.0f, 0.0f, 1.0f); // Red for critical
                } else if (range_start >= frame_time_threshold_warning_) {
                    bin_color = ImVec4(1.0f, 0.6f, 0.0f, 1.0f); // Orange for warning
                }

                ImGui::TextColored(bin_color, "%.2f-%.2f ms: %d frames", range_start, range_end, counts[i]);
            }
        }

        ImGui::Separator();
        ImGui::Text("Statistical Analysis:");

        // Show variance and standard deviation
        double variance = get_variance();
        double std_dev = get_standard_deviation();
        ImGui::Text("Variance: %.4f ms²", variance);
        ImGui::Text("Standard Deviation: %.4f ms", std_dev);

        // Show percentiles
        ImGui::Text("Percentiles:");
        ImGui::Indent();
        ImGui::Text("50th (Median): %.2f ms", get_percentile(50.0));
        ImGui::Text("90th: %.2f ms", get_percentile(90.0));
        ImGui::Text("95th: %.2f ms", get_percentile(95.0));
        ImGui::Text("99th: %.2f ms", get_percentile(99.0));
        ImGui::Unindent();

        // Show frames outside thresholds
        auto [warning_count, critical_count] = get_frames_outside_thresholds();
        size_t total_frames = frame_times_.size();
        ImGui::Text("Performance Issues:");
        ImGui::Indent();
        ImGui::Text("Warning (>%.2f ms): %zu frames (%.2f%%)", frame_time_threshold_warning_, warning_count,
                   total_frames > 0 ? (warning_count * 100.0) / total_frames : 0.0);
        ImGui::Text("Critical (>%.2f ms): %zu frames (%.2f%%)", frame_time_threshold_critical_, critical_count,
                   total_frames > 0 ? (critical_count * 100.0) / total_frames : 0.0);
        ImGui::Unindent();

        // Show smoothed frame time
        ImGui::Text("Smoothed Frame Time (last 5): %.2f ms", get_smoothed_frame_time(5));

        // Show median frame time
        ImGui::Text("Median Frame Time: %.2f ms", get_median_frame_time());

        // Add a performance health indicator
        ImGui::Separator();
        ImGui::Text("Performance Health:");

        // Calculate performance health score (0-100)
        double health_score = 100.0;
        if (!frame_times_.empty()) {
            // Lower scores for high average frame times
            double avg_frame_time_penalty = std::min(50.0, (average_frame_time_ms_ / 33.33) * 50.0);

            // Penalty for high variance
            double variance_penalty = std::min(30.0, (get_variance() / 100.0) * 30.0);

            // Penalty for percentage of frames exceeding thresholds
            auto [warn_count, crit_count] = get_frames_outside_thresholds();
            double issue_percentage = ((warn_count + crit_count) * 100.0) / frame_times_.size();
            double issue_penalty = std::min(20.0, issue_percentage * 0.2);

            health_score = 100.0 - avg_frame_time_penalty - variance_penalty - issue_penalty;
            health_score = std::max(0.0, health_score);
        }

        // Color code the health score
        ImVec4 health_color;
        if (health_score >= 80) {
            health_color = ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green
        } else if (health_score >= 60) {
            health_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow
        } else {
            health_color = ImVec4(0.8f, 0.0f, 0.0f, 1.0f); // Red
        }

        ImGui::TextColored(health_color, "Health Score: %.1f/100", health_score);

        // Show spike analysis
        ImGui::Separator();
        ImGui::Text("Spike Analysis:");
        auto spikes = get_spike_frames(2.0); // Frames that are 2x the local average
        ImGui::Text("Spikes detected: %zu", spikes.size());
        if (!spikes.empty() && ImGui::TreeNode("View Spike Details")) {
            for (const auto& spike : spikes) {
                ImVec4 spike_color = spike.second > frame_time_threshold_critical_ ?
                                   ImVec4(0.8f, 0.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.6f, 0.0f, 1.0f);
                ImGui::TextColored(spike_color, "Frame #%zu: %.2f ms", spike.first, spike.second);
            }
            ImGui::TreePop();
        }

        // Show consecutive frame drops
        size_t consecutive_drops = get_consecutive_frame_drops(3, 33.33); // 3+ consecutive frames > 33.33ms
        ImGui::Text("Longest sequence of slow frames: %zu", consecutive_drops);

        // Add a visual representation of frame time stability
        ImGui::Separator();
        ImGui::Text("Stability Indicator:");
        if (std_dev < 2.0) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Very Stable");
        } else if (std_dev < 5.0) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Stable");
        } else if (std_dev < 10.0) {
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Moderately Stable");
        } else {
            ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Unstable");
        }
        ImGui::SameLine();
        ImGui::Text("(Std Dev: %.2f ms)", std_dev);
    }

    // Add a button to reset statistics
    if (ImGui::Button("Reset Statistics")) {
        reset();
    }
}

// Global instance
FrameTimeGraph g_frame_time_graph;

} // namespace BTQuant