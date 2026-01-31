#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace BTQuant {

struct PerformanceMetric {
  std::string name;
  double value;
  std::string unit;
  double min_value;
  double max_value;
  double avg_value;
  std::chrono::high_resolution_clock::time_point timestamp;
};

class PerformanceMonitor {
 public:
  PerformanceMonitor();
  ~PerformanceMonitor() = default;

  // Start/Stop measuring frame time
  void start_frame();
  void end_frame();

  // Get current frame time in milliseconds
  double get_frame_time_ms() const;

  // Get average frame time over specified window
  double get_avg_frame_time_ms(size_t window_size = 60) const;

  // Get current FPS
  double get_fps() const;

  // Get average FPS over specified window
  double get_avg_fps(size_t window_size = 60) const;

  // Get maximum FPS recorded
  double get_max_fps() const;

  // Get minimum FPS recorded
  double get_min_fps() const;

  // Memory usage tracking
  void set_memory_usage(size_t used_bytes, size_t total_bytes);
  double get_memory_usage_percent() const;
  size_t get_used_memory_bytes() const;
  size_t get_total_memory_bytes() const;

  // Data processing metrics
  void increment_data_processed(size_t count);
  void increment_indicators_calculated(size_t count);
  void set_data_processing_time(double ms);

  // Get performance metrics
  size_t get_data_processed_count() const;
  size_t get_indicators_calculated_count() const;
  double get_data_processing_time_ms() const;

  // Reset all metrics
  void reset();

  // Get all recorded metrics
  std::vector<PerformanceMetric> get_metrics() const;

  // Enable/disable performance monitoring
  void set_enabled(bool enabled);
  bool is_enabled() const;

  // Helper methods (added for get_metrics)
  double get_min_frame_time() const;
  double get_max_frame_time() const;

 private:
  // Frame timing
  std::chrono::high_resolution_clock::time_point frame_start_;
  double current_frame_time_ms_;
  std::vector<double> frame_time_history_;
  mutable std::mutex history_mutex_;
  static constexpr size_t MAX_HISTORY_SIZE = 1000;

  // Memory usage
  size_t used_memory_bytes_;
  size_t total_memory_bytes_;

  // Data processing metrics
  std::atomic<size_t> data_processed_count_;
  std::atomic<size_t> indicators_calculated_count_;
  double data_processing_time_ms_;

  // Performance statistics
  double min_fps_;
  double max_fps_;

  // Enable flag
  std::atomic<bool> enabled_;

  // Helper methods
  void update_frame_time(double ms);
  void calculate_statistics() const;
};

// Global performance monitor instance
extern PerformanceMonitor g_performance_monitor;

}  // namespace BTQuant
