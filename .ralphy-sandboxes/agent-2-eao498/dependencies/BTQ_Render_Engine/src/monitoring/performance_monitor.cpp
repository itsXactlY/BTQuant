#include "../../include/performance_monitor.hpp"
#include <numeric>
#include <algorithm>
#include <iostream>

namespace BTQuant {

PerformanceMonitor::PerformanceMonitor()
    : current_frame_time_ms_(0.0),
      used_memory_bytes_(0),
      total_memory_bytes_(0),
      data_processed_count_(0),
      indicators_calculated_count_(0),
      data_processing_time_ms_(0.0),
      min_fps_(1000.0),
      max_fps_(0.0),
      enabled_(true) {}

void PerformanceMonitor::start_frame() {
  if (!enabled_) return;
  
  frame_start_ = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::end_frame() {
  if (!enabled_) return;
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - frame_start_);
  
  current_frame_time_ms_ = duration.count() / 1000.0;
  update_frame_time(current_frame_time_ms_);
}

double PerformanceMonitor::get_frame_time_ms() const {
  return current_frame_time_ms_;
}

double PerformanceMonitor::get_avg_frame_time_ms(size_t window_size) const {
  std::lock_guard<std::mutex> lock(history_mutex_);
  
  if (frame_time_history_.empty()) {
    return 0.0;
  }
  
  size_t actual_window = std::min(window_size, frame_time_history_.size());
  auto start = frame_time_history_.end() - actual_window;
  double sum = std::accumulate(start, frame_time_history_.end(), 0.0);
  
  return sum / actual_window;
}

double PerformanceMonitor::get_fps() const {
  if (current_frame_time_ms_ <= 0.0) {
    return 0.0;
  }
  
  return 1000.0 / current_frame_time_ms_;
}

double PerformanceMonitor::get_avg_fps(size_t window_size) const {
  double avg_frame_time = get_avg_frame_time_ms(window_size);
  if (avg_frame_time <= 0.0) {
    return 0.0;
  }
  
  return 1000.0 / avg_frame_time;
}

double PerformanceMonitor::get_max_fps() const {
  return max_fps_;
}

double PerformanceMonitor::get_min_fps() const {
  return min_fps_;
}

void PerformanceMonitor::set_memory_usage(size_t used_bytes, size_t total_bytes) {
  used_memory_bytes_ = used_bytes;
  total_memory_bytes_ = total_bytes;
}

double PerformanceMonitor::get_memory_usage_percent() const {
  if (total_memory_bytes_ == 0) {
    return 0.0;
  }
  
  return static_cast<double>(used_memory_bytes_) / total_memory_bytes_ * 100.0;
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
  return data_processed_count_.load();
}

size_t PerformanceMonitor::get_indicators_calculated_count() const {
  return indicators_calculated_count_.load();
}

double PerformanceMonitor::get_data_processing_time_ms() const {
  return data_processing_time_ms_;
}

void PerformanceMonitor::reset() {
  std::lock_guard<std::mutex> lock(history_mutex_);
  
  current_frame_time_ms_ = 0.0;
  frame_time_history_.clear();
  used_memory_bytes_ = 0;
  total_memory_bytes_ = 0;
  data_processed_count_.store(0);
  indicators_calculated_count_.store(0);
  data_processing_time_ms_ = 0.0;
  min_fps_ = 1000.0;
  max_fps_ = 0.0;
}

std::vector<PerformanceMetric> PerformanceMonitor::get_metrics() const {
  std::vector<PerformanceMetric> metrics;
  
  // Frame time metrics
  metrics.push_back({
    "Frame Time",
    current_frame_time_ms_,
    "ms",
    get_min_frame_time(),
    get_max_frame_time(),
    get_avg_frame_time_ms(),
    std::chrono::high_resolution_clock::now()
  });
  
  metrics.push_back({
    "FPS",
    get_fps(),
    "fps",
    min_fps_,
    max_fps_,
    get_avg_fps(),
    std::chrono::high_resolution_clock::now()
  });
  
  // Memory metrics
  metrics.push_back({
    "Memory Usage",
    get_memory_usage_percent(),
    "%",
    0.0,
    100.0,
    get_memory_usage_percent(),
    std::chrono::high_resolution_clock::now()
  });
  
  // Data processing metrics
  metrics.push_back({
    "Data Processed",
    static_cast<double>(get_data_processed_count()),
    "count",
    0.0,
    static_cast<double>(get_data_processed_count()),
    static_cast<double>(get_data_processed_count()),
    std::chrono::high_resolution_clock::now()
  });
  
  metrics.push_back({
    "Indicators Calculated",
    static_cast<double>(get_indicators_calculated_count()),
    "count",
    0.0,
    static_cast<double>(get_indicators_calculated_count()),
    static_cast<double>(get_indicators_calculated_count()),
    std::chrono::high_resolution_clock::now()
  });
  
  metrics.push_back({
    "Data Processing Time",
    data_processing_time_ms_,
    "ms",
    0.0,
    data_processing_time_ms_,
    data_processing_time_ms_,
    std::chrono::high_resolution_clock::now()
  });
  
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

void PerformanceMonitor::update_frame_time(double ms) {
  std::lock_guard<std::mutex> lock(history_mutex_);
  
  frame_time_history_.push_back(ms);
  
  if (frame_time_history_.size() > MAX_HISTORY_SIZE) {
    frame_time_history_.erase(frame_time_history_.begin());
  }
  
  double current_fps = 1000.0 / ms;
  if (current_fps > max_fps_) {
    max_fps_ = current_fps;
  }
  if (current_fps < min_fps_) {
    min_fps_ = current_fps;
  }
}

double PerformanceMonitor::get_min_frame_time() const {
  std::lock_guard<std::mutex> lock(history_mutex_);
  
  if (frame_time_history_.empty()) {
    return 0.0;
  }
  
  return *std::min_element(frame_time_history_.begin(), frame_time_history_.end());
}

double PerformanceMonitor::get_max_frame_time() const {
  std::lock_guard<std::mutex> lock(history_mutex_);
  
  if (frame_time_history_.empty()) {
    return 0.0;
  }
  
  return *std::max_element(frame_time_history_.begin(), frame_time_history_.end());
}

// Global instance
PerformanceMonitor g_performance_monitor;

} // namespace BTQuant
