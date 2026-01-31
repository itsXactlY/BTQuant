#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace BTQuant::Monitoring {

/**
 * ShmHealthMonitor - Monitors shared memory health and integrity
 *
 * This class provides continuous monitoring of shared memory segments
 * to detect deletion, corruption, or permission issues.
 */
class ShmHealthMonitor {
public:
  enum class HealthStatus {
    HEALTHY = 0,  // All checks passed
    DEGRADED = 1, // Some non-critical issues detected
    CRITICAL = 2, // Critical issues requiring immediate attention
    FAILED = 3    // Complete failure - no data available
  };

  struct HealthReport {
    HealthStatus status = HealthStatus::FAILED;
    std::array<char, 256>
        shm_path{}; // Fixed-size buffer for atomic compatibility
    uint64_t last_check_us = 0;
    uint64_t uptime_us = 0;
    size_t buffer_capacity = 0;
    size_t buffer_used = 0;
    uint64_t read_errors = 0;
    uint64_t write_errors = 0;
    std::array<char, 512>
        error_message{}; // Fixed-size buffer for atomic compatibility
    bool file_exists = false;
    bool has_valid_header = false;
    bool permissions_ok = false;

    void set_shm_path(const std::string &path);
    void set_error_message(const std::string &msg);

    std::string status_string() const {
      switch (status) {
      case HealthStatus::HEALTHY:
        return "HEALTHY";
      case HealthStatus::DEGRADED:
        return "DEGRADED";
      case HealthStatus::CRITICAL:
        return "CRITICAL";
      case HealthStatus::FAILED:
        return "FAILED";
      default:
        return "UNKNOWN";
      }
    }

    double utilization_percent() const {
      if (buffer_capacity == 0)
        return 0.0;
      return (static_cast<double>(buffer_used) / buffer_capacity) * 100.0;
    }
  };

  struct MonitorConfig {
    uint64_t check_interval_us = 1'000'000;   // 1 second
    uint64_t startup_timeout_us = 10'000'000; // 10 seconds
    bool auto_recovery = false;
    uint32_t max_consecutive_failures = 3;
  };

  explicit ShmHealthMonitor(const std::string &shm_path,
                            const MonitorConfig &config);
  ~ShmHealthMonitor();

  // Non-copyable, non-movable
  ShmHealthMonitor(const ShmHealthMonitor &) = delete;
  ShmHealthMonitor &operator=(const ShmHealthMonitor &) = delete;
  ShmHealthMonitor(ShmHealthMonitor &&) = delete;
  ShmHealthMonitor &operator=(ShmHealthMonitor &&) = delete;

  // Health check operations
  HealthReport check_health();
  bool is_healthy() const {
    std::lock_guard<std::mutex> lock(report_mutex_);
    return current_report_.status == HealthStatus::HEALTHY;
  }
  bool is_attached() const {
    std::lock_guard<std::mutex> lock(report_mutex_);
    return current_report_.file_exists && current_report_.has_valid_header;
  }

  // Monitoring control
  void start_monitoring();
  void stop_monitoring();
  bool is_monitoring() const { return monitoring_.load(); }

  // Statistics
  uint64_t get_total_checks() const { return total_checks_.load(); }
  uint64_t get_failed_checks() const { return failed_checks_.load(); }
  uint32_t get_consecutive_failures() const {
    return consecutive_failures_.load();
  }

  // Configuration
  const MonitorConfig &get_config() const { return config_; }
  void set_config(const MonitorConfig &config);

private:
  std::string shm_path_;
  MonitorConfig config_;

  // Use atomic variables for thread-safe access instead of atomic<HealthReport>
  std::atomic<bool> monitoring_{false};
  std::thread monitor_thread_;

  std::atomic<uint64_t> total_checks_{0};
  std::atomic<uint64_t> failed_checks_{0};
  std::atomic<uint32_t> consecutive_failures_{0};
  std::atomic<uint64_t> start_time_us_{0};

  // Individual atomics for health report fields (trivially copyable)
  std::atomic<HealthStatus> report_status{HealthStatus::FAILED};
  std::atomic<uint64_t> report_last_check_us{0};
  std::atomic<uint64_t> report_uptime_us{0};
  std::atomic<size_t> report_buffer_capacity{0};
  std::atomic<size_t> report_buffer_used{0};
  std::atomic<uint64_t> report_read_errors{0};
  std::atomic<uint64_t> report_write_errors{0};
  std::atomic<bool> report_file_exists{false};
  std::atomic<bool> report_has_valid_header{false};
  std::atomic<bool> report_permissions_ok{false};

  // Thread-safe string storage using fixed-size arrays
  std::array<char, 256> report_shm_path{};
  std::array<char, 512> report_error_message{};
  std::mutex report_path_mutex_;
  std::mutex report_error_mutex_;

  HealthReport current_report_;
  mutable std::mutex report_mutex_; // Protects current_report_ access
  mutable std::mutex
      current_report_mutex_; // For atomic-like access to current_report_

  // Atomic accessors for current_report_
  void set_current_report(const HealthReport &report) {
    std::lock_guard<std::mutex> lock(report_mutex_);
    current_report_ = report;
  }

  HealthReport get_current_report() const {
    std::lock_guard<std::mutex> lock(report_mutex_);
    return current_report_;
  }

  // Internal check methods
  HealthReport perform_full_check();
  bool check_file_exists();
  bool check_permissions();
  bool check_header_integrity(size_t &capacity, size_t &used);
  void update_status(const HealthReport &report);
  void monitor_loop();
};

/**
 * Shared memory utilities for health checking
 */
class ShmUtils {
public:
  static bool file_exists(const std::string &path);
  static bool is_shm_file(const std::string &path);
  static bool check_read_permission(const std::string &path);
  static bool check_write_permission(const std::string &path);
  static size_t get_file_size(const std::string &path);
  static uint64_t get_file_mtime(const std::string &path);
  static std::string read_file_header(const std::string &path,
                                      size_t bytes = 256);
};

} // namespace BTQuant::Monitoring
