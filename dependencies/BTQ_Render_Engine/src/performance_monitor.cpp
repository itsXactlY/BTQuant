/**
 * Performance Monitor Implementation
 *
 * Tracks and reports performance metrics for the trading terminal
 */

#include "performance_monitor.hpp"
#include "../include/performance/frame_time_graph.hpp"
#include "../src/imgui/implot.h"
#include "../src/imgui/imgui.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace BTQuant {

PerformanceMonitor::PerformanceMonitor()
    : frame_start_(std::chrono::high_resolution_clock::now()),
      current_frame_time_ms_(0.0),
      frame_time_history_(),
      used_memory_bytes_(0),
      total_memory_bytes_(0),
      data_processed_count_(0),
      indicators_calculated_count_(0),
      data_processing_time_ms_(0.0),
      min_fps_(std::numeric_limits<double>::max()),
      max_fps_(0.0),
      enabled_(true) {
    frame_time_history_.reserve(MAX_HISTORY_SIZE);
}

PerformanceMonitor::~PerformanceMonitor() = default;

void PerformanceMonitor::start_frame() {
    if (!enabled_) return;

    frame_start_ = std::chrono::high_resolution_clock::now();

    // Also start the global frame time graph
    g_frame_time_graph.start_frame();
}

void PerformanceMonitor::end_frame() {
    if (!enabled_) return;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - frame_start_);

    double frame_time_ms = static_cast<double>(duration.count()) / 1000.0;

    // Update current frame time
    current_frame_time_ms_ = frame_time_ms;

    // Add to history
    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        frame_time_history_.push_back(frame_time_ms);

        // Maintain buffer size
        if (frame_time_history_.size() > MAX_HISTORY_SIZE) {
            frame_time_history_.erase(frame_time_history_.begin());
        }
    }

    // Update FPS
    double current_fps = frame_time_ms > 0 ? 1000.0 / frame_time_ms : 0.0;

    // Update min/max FPS
    if (current_fps > max_fps_) {
        max_fps_ = current_fps;
    }
    if (current_fps < min_fps_ || min_fps_ == std::numeric_limits<double>::max()) {
        min_fps_ = current_fps;
    }

    // Also end the frame for the global frame time graph
    g_frame_time_graph.end_frame();
}

double PerformanceMonitor::get_frame_time_ms() const {
    return current_frame_time_ms_;
}

double PerformanceMonitor::get_avg_frame_time_ms(size_t window_size) const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    size_t count = std::min(window_size, frame_time_history_.size());
    if (count == 0) return 0.0;

    double sum = 0.0;
    for (size_t i = frame_time_history_.size() - count; i < frame_time_history_.size(); ++i) {
        sum += frame_time_history_[i];
    }

    return sum / count;
}

double PerformanceMonitor::get_fps() const {
    return current_frame_time_ms_ > 0 ? 1000.0 / current_frame_time_ms_ : 0.0;
}

double PerformanceMonitor::get_avg_fps(size_t window_size) const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    size_t count = std::min(window_size, frame_time_history_.size());
    if (count == 0) return 0.0;

    double total_time = 0.0;
    for (size_t i = frame_time_history_.size() - count; i < frame_time_history_.size(); ++i) {
        total_time += frame_time_history_[i];
    }

    if (total_time > 0.0) {
        return (count * 1000.0) / total_time;
    }
    return 0.0;
}

double PerformanceMonitor::get_max_fps() const {
    return max_fps_;
}

double PerformanceMonitor::get_min_fps() const {
    return min_fps_ != std::numeric_limits<double>::max() ? min_fps_ : 0.0;
}

void PerformanceMonitor::set_memory_usage(size_t used_bytes, size_t total_bytes) {
    used_memory_bytes_ = used_bytes;
    total_memory_bytes_ = total_bytes;
}

double PerformanceMonitor::get_memory_usage_percent() const {
    if (total_memory_bytes_ == 0) return 0.0;
    return static_cast<double>(used_memory_bytes_) / static_cast<double>(total_memory_bytes_) * 100.0;
}

size_t PerformanceMonitor::get_used_memory_bytes() const {
    return used_memory_bytes_;
}

size_t PerformanceMonitor::get_total_memory_bytes() const {
    return total_memory_bytes_;
}

void PerformanceMonitor::increment_data_processed(size_t count) {
    data_processed_count_ += count;
}

void PerformanceMonitor::increment_indicators_calculated(size_t count) {
    indicators_calculated_count_ += count;
}

void PerformanceMonitor::set_data_processing_time(double ms) {
    data_processing_time_ms_ = ms;
}

size_t PerformanceMonitor::get_data_processed_count() const {
    return data_processed_count_;
}

size_t PerformanceMonitor::get_indicators_calculated_count() const {
    return indicators_calculated_count_;
}

double PerformanceMonitor::get_data_processing_time_ms() const {
    return data_processing_time_ms_;
}

void PerformanceMonitor::reset() {
    std::lock_guard<std::mutex> lock(history_mutex_);
    frame_time_history_.clear();

    // Reset statistics
    current_frame_time_ms_ = 0.0;
    used_memory_bytes_ = 0;
    total_memory_bytes_ = 0;
    data_processed_count_ = 0;
    indicators_calculated_count_ = 0;
    data_processing_time_ms_ = 0.0;
    min_fps_ = std::numeric_limits<double>::max();
    max_fps_ = 0.0;

    // Also reset the global frame time graph
    g_frame_time_graph.reset();
}

std::vector<PerformanceMetric> PerformanceMonitor::get_metrics() const {
    std::vector<PerformanceMetric> metrics;

    // Add frame time metrics
    PerformanceMetric frame_time_metric;
    frame_time_metric.name = "Current Frame Time";
    frame_time_metric.value = get_frame_time_ms();
    frame_time_metric.unit = "ms";
    frame_time_metric.avg_value = get_avg_frame_time_ms();
    frame_time_metric.min_value = get_min_frame_time();
    frame_time_metric.max_value = get_max_frame_time();
    frame_time_metric.timestamp = std::chrono::high_resolution_clock::now();
    metrics.push_back(frame_time_metric);

    // Add FPS metrics
    PerformanceMetric fps_metric;
    fps_metric.name = "Current FPS";
    fps_metric.value = get_fps();
    fps_metric.unit = "fps";
    fps_metric.avg_value = get_avg_fps();
    fps_metric.min_value = get_min_fps();
    fps_metric.max_value = get_max_fps();
    fps_metric.timestamp = std::chrono::high_resolution_clock::now();
    metrics.push_back(fps_metric);

    // Add memory usage metrics
    PerformanceMetric memory_metric;
    memory_metric.name = "Memory Usage";
    memory_metric.value = get_memory_usage_percent();
    memory_metric.unit = "%";
    memory_metric.avg_value = 0.0; // No average for this metric
    memory_metric.min_value = 0.0;
    memory_metric.max_value = 100.0;
    memory_metric.timestamp = std::chrono::high_resolution_clock::now();
    metrics.push_back(memory_metric);

    return metrics;
}

void PerformanceMonitor::set_enabled(bool enabled) {
    enabled_ = enabled;
    if (!enabled) {
        reset();
    }
}

bool PerformanceMonitor::is_enabled() const {
    return enabled_;
}

double PerformanceMonitor::get_min_frame_time() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    if (frame_time_history_.empty()) return 0.0;

    auto min_it = std::min_element(frame_time_history_.begin(), frame_time_history_.end());
    return *min_it;
}

double PerformanceMonitor::get_max_frame_time() const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    if (frame_time_history_.empty()) return 0.0;

    auto max_it = std::max_element(frame_time_history_.begin(), frame_time_history_.end());
    return *max_it;
}

void PerformanceMonitor::render_frame_time_graph(const char* title, float width, float height) {
    g_frame_time_graph.render(title, width, height);
}

}  // namespace BTQuant
