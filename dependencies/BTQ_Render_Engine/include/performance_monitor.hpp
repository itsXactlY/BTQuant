#pragma once

#include <chrono>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <atomic>

namespace BTQuant {
namespace RenderEngine {

// System metrics structure
struct SystemMetrics {
    // Rendering metrics
    double fps = 0.0;
    double frame_time_ms = 0.0;
    std::chrono::high_resolution_clock::time_point last_frame_time;
    
    // Data processing metrics
    double data_to_display_latency_us = 0.0;
    
    // System resource metrics
    double memory_usage_mb = 0.0;
    double cpu_usage_percent = 0.0;
    double gpu_usage_percent = 0.0;
    double temperature_celsius = 0.0;
    
    // Network metrics
    bool network_connected = false;
    double network_latency_ms = 0.0;
    double network_throughput_mbps = 0.0;
    
    // Timestamps
    std::chrono::high_resolution_clock::time_point last_update;
};

// Performance statistics
struct PerformanceStatistics {
    // FPS statistics
    double avg_fps = 0.0;
    double min_fps = 0.0;
    double max_fps = 0.0;
    double fps_std_dev = 0.0;
    
    // Latency statistics
    double avg_latency_ms = 0.0;
    double min_latency_ms = 0.0;
    double max_latency_ms = 0.0;
    double latency_p95_ms = 0.0;  // 95th percentile
    double latency_p99_ms = 0.0;  // 99th percentile
    
    // Memory statistics
    double avg_memory_mb = 0.0;
    double peak_memory_mb = 0.0;
    
    // Uptime
    uint64_t uptime_seconds = 0;
};

// Metric sample for history tracking
struct MetricSample {
    std::chrono::high_resolution_clock::time_point timestamp;
    double value;
};

// Alert types
enum class AlertType {
    LOW_FPS,
    HIGH_LATENCY,
    HIGH_MEMORY,
    HIGH_CPU,
    HIGH_GPU,
    HIGH_TEMPERATURE,
    CONNECTION_LOST
};

// Performance alert
struct PerformanceAlert {
    AlertType type;
    std::string message;
    std::chrono::high_resolution_clock::time_point timestamp;
};

/**
 * PerformanceMonitor - Comprehensive system performance monitoring
 * 
 * This class provides real-time monitoring of system performance metrics
 * including:
 * - FPS and frame timing analysis
 * - Data-to-display latency measurement
 * - System resource usage (CPU, GPU, memory, temperature)
 * - Network connection status and performance
 * - Performance alerts and thresholds
 * - Historical data tracking and statistics
 * - Performance reporting and logging
 */
class PerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    // Non-copyable, non-movable
    PerformanceMonitor(const PerformanceMonitor&) = delete;
    PerformanceMonitor& operator=(const PerformanceMonitor&) = delete;
    PerformanceMonitor(PerformanceMonitor&&) = delete;
    PerformanceMonitor& operator=(PerformanceMonitor&&) = delete;
    
    /**
     * Start/stop monitoring
     */
    bool startMonitoring();
    void stopMonitoring();
    
    /**
     * Update metrics from external sources
     */
    void updateFrameMetrics(double fps, double frame_time_ms);
    void updateDataLatency(double data_to_display_latency_us);
    void updateMemoryUsage(double memory_mb);
    void updateNetworkStatus(bool connected, double latency_ms, double throughput_mbps);
    void updateSystemHealth(double cpu_usage_percent, double gpu_usage_percent, double temperature_celsius);
    
    /**
     * Get current metrics and statistics
     */
    SystemMetrics getCurrentMetrics() const;
    PerformanceStatistics getStatistics() const;
    
    /**
     * Get historical data
     */
    std::vector<MetricSample> getFPSHistory() const;
    std::vector<MetricSample> getLatencyHistory() const;
    std::vector<MetricSample> getMemoryHistory() const;
    
    /**
     * Alert management
     */
    std::vector<PerformanceAlert> getRecentAlerts(size_t max_count = 50) const;
    void clearAlerts();
    
    /**
     * Reporting
     */
    std::string generateReport() const;
    bool saveReport(const std::string& filename) const;
    
    /**
     * Configuration
     */
    void setAlertThresholds(double fps_threshold, double latency_ms_threshold, double memory_mb_threshold);
    void setMonitoringInterval(uint32_t interval_ms);
    void setHistorySize(size_t size);
    void enableMonitoring(bool enabled) { monitoring_enabled_ = enabled; }

private:
    // Configuration
    std::atomic<bool> monitoring_enabled_;
    uint32_t monitoring_interval_ms_;
    size_t history_size_;
    
    // Alert thresholds
    double alert_threshold_fps_;
    double alert_threshold_latency_ms_;
    double alert_threshold_memory_mb_;
    
    // Monitoring thread
    std::atomic<bool> monitoring_running_{false};
    std::thread monitoring_thread_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // Current metrics
    mutable std::mutex metrics_mutex_;
    SystemMetrics current_metrics_;
    
    // Historical data
    std::vector<MetricSample> fps_history_;
    std::vector<MetricSample> latency_history_;
    std::vector<MetricSample> memory_history_;
    
    // Alerts
    mutable std::mutex alerts_mutex_;
    std::vector<PerformanceAlert> alerts_;
    
    // Private methods
    void monitoringLoop();
    void collectSystemMetrics();
    void triggerAlert(AlertType type, const std::string& message);
    void printStatus() const;
    
    // Statistics calculation helpers
    double calculateAverage(const std::vector<double>& values) const;
    double calculateStandardDeviation(const std::vector<double>& values) const;
    double calculatePercentile(std::vector<double> values, double percentile) const;
    
    // Utility methods
    std::string alertTypeToString(AlertType type) const;
    std::string getCurrentTimestamp() const;
    std::string formatTimestamp(const std::chrono::high_resolution_clock::time_point& tp) const;
};

} // namespace RenderEngine
} // namespace BTQuant