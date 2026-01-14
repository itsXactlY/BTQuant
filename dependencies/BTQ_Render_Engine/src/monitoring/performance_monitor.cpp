#include "performance_monitor.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>

namespace BTQuant {
namespace RenderEngine {

PerformanceMonitor::PerformanceMonitor()
    : monitoring_enabled_(true)
    , monitoring_interval_ms_(1000)
    , history_size_(300)  // 5 minutes at 1 second intervals
    , alert_threshold_fps_(30.0)
    , alert_threshold_latency_ms_(10.0)
    , alert_threshold_memory_mb_(1024.0)
{
    // Initialize metrics
    current_metrics_ = SystemMetrics{};
    
    // Pre-allocate history vectors
    fps_history_.reserve(history_size_);
    latency_history_.reserve(history_size_);
    memory_history_.reserve(history_size_);
    
    std::cout << "[PerformanceMonitor] Initialized with " << history_size_ << " sample history" << std::endl;
}

PerformanceMonitor::~PerformanceMonitor() {
    stopMonitoring();
}

bool PerformanceMonitor::startMonitoring() {
    if (monitoring_thread_.joinable()) {
        return true;  // Already running
    }
    
    monitoring_running_ = true;
    monitoring_thread_ = std::thread(&PerformanceMonitor::monitoringLoop, this);
    
    std::cout << "[PerformanceMonitor] Started monitoring thread" << std::endl;
    return true;
}

void PerformanceMonitor::stopMonitoring() {
    if (monitoring_thread_.joinable()) {
        monitoring_running_ = false;
        monitoring_thread_.join();
        std::cout << "[PerformanceMonitor] Stopped monitoring thread" << std::endl;
    }
}

void PerformanceMonitor::updateFrameMetrics(double fps, double frame_time_ms) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.fps = fps;
    current_metrics_.frame_time_ms = frame_time_ms;
    current_metrics_.last_frame_time = std::chrono::high_resolution_clock::now();
    
    // Update history
    fps_history_.push_back({std::chrono::high_resolution_clock::now(), fps});
    if (fps_history_.size() > history_size_) {
        fps_history_.erase(fps_history_.begin());
    }
    
    // Check for alerts
    if (fps < alert_threshold_fps_) {
        triggerAlert(AlertType::LOW_FPS, "Low FPS detected: " + std::to_string(fps));
    }
    
    if (frame_time_ms > alert_threshold_latency_ms_) {
        triggerAlert(AlertType::HIGH_LATENCY, "High frame time: " + std::to_string(frame_time_ms) + "ms");
    }
}

void PerformanceMonitor::updateDataLatency(double data_to_display_latency_us) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.data_to_display_latency_us = data_to_display_latency_us;
    double latency_ms = data_to_display_latency_us / 1000.0;
    
    // Update history
    latency_history_.push_back({std::chrono::high_resolution_clock::now(), latency_ms});
    if (latency_history_.size() > history_size_) {
        latency_history_.erase(latency_history_.begin());
    }
    
    // Check for alerts
    if (latency_ms > alert_threshold_latency_ms_) {
        triggerAlert(AlertType::HIGH_LATENCY, "High data latency: " + std::to_string(latency_ms) + "ms");
    }
}

void PerformanceMonitor::updateMemoryUsage(double memory_mb) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.memory_usage_mb = memory_mb;
    
    // Update history
    memory_history_.push_back({std::chrono::high_resolution_clock::now(), memory_mb});
    if (memory_history_.size() > history_size_) {
        memory_history_.erase(memory_history_.begin());
    }
    
    // Check for alerts
    if (memory_mb > alert_threshold_memory_mb_) {
        triggerAlert(AlertType::HIGH_MEMORY, "High memory usage: " + std::to_string(memory_mb) + "MB");
    }
}

void PerformanceMonitor::updateNetworkStatus(bool connected, double latency_ms, double throughput_mbps) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.network_connected = connected;
    current_metrics_.network_latency_ms = latency_ms;
    current_metrics_.network_throughput_mbps = throughput_mbps;
    
    // Check for alerts
    if (!connected) {
        triggerAlert(AlertType::CONNECTION_LOST, "Network connection lost");
    } else if (latency_ms > 100.0) {  // High network latency threshold
        triggerAlert(AlertType::HIGH_LATENCY, "High network latency: " + std::to_string(latency_ms) + "ms");
    }
}

void PerformanceMonitor::updateSystemHealth(double cpu_usage_percent, double gpu_usage_percent, 
                                           double temperature_celsius) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.cpu_usage_percent = cpu_usage_percent;
    current_metrics_.gpu_usage_percent = gpu_usage_percent;
    current_metrics_.temperature_celsius = temperature_celsius;
    
    // Check for alerts
    if (cpu_usage_percent > 90.0) {
        triggerAlert(AlertType::HIGH_CPU, "High CPU usage: " + std::to_string(cpu_usage_percent) + "%");
    }
    
    if (gpu_usage_percent > 95.0) {
        triggerAlert(AlertType::HIGH_GPU, "High GPU usage: " + std::to_string(gpu_usage_percent) + "%");
    }
    
    if (temperature_celsius > 80.0) {
        triggerAlert(AlertType::HIGH_TEMPERATURE, "High temperature: " + std::to_string(temperature_celsius) + "°C");
    }
}

SystemMetrics PerformanceMonitor::getCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return current_metrics_;
}

PerformanceStatistics PerformanceMonitor::getStatistics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    PerformanceStatistics stats;
    
    // Calculate FPS statistics
    if (!fps_history_.empty()) {
        std::vector<double> fps_values;
        for (const auto& sample : fps_history_) {
            fps_values.push_back(sample.value);
        }
        
        stats.avg_fps = calculateAverage(fps_values);
        stats.min_fps = *std::min_element(fps_values.begin(), fps_values.end());
        stats.max_fps = *std::max_element(fps_values.begin(), fps_values.end());
        stats.fps_std_dev = calculateStandardDeviation(fps_values);
    }
    
    // Calculate latency statistics
    if (!latency_history_.empty()) {
        std::vector<double> latency_values;
        for (const auto& sample : latency_history_) {
            latency_values.push_back(sample.value);
        }
        
        stats.avg_latency_ms = calculateAverage(latency_values);
        stats.min_latency_ms = *std::min_element(latency_values.begin(), latency_values.end());
        stats.max_latency_ms = *std::max_element(latency_values.begin(), latency_values.end());
        stats.latency_p95_ms = calculatePercentile(latency_values, 95.0);
        stats.latency_p99_ms = calculatePercentile(latency_values, 99.0);
    }
    
    // Calculate memory statistics
    if (!memory_history_.empty()) {
        std::vector<double> memory_values;
        for (const auto& sample : memory_history_) {
            memory_values.push_back(sample.value);
        }
        
        stats.avg_memory_mb = calculateAverage(memory_values);
        stats.peak_memory_mb = *std::max_element(memory_values.begin(), memory_values.end());
    }
    
    stats.uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start_time_).count();
    
    return stats;
}

std::vector<MetricSample> PerformanceMonitor::getFPSHistory() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return fps_history_;
}

std::vector<MetricSample> PerformanceMonitor::getLatencyHistory() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return latency_history_;
}

std::vector<MetricSample> PerformanceMonitor::getMemoryHistory() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return memory_history_;
}

std::vector<PerformanceAlert> PerformanceMonitor::getRecentAlerts(size_t max_count) const {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    
    std::vector<PerformanceAlert> recent_alerts;
    size_t start_idx = alerts_.size() > max_count ? alerts_.size() - max_count : 0;
    
    for (size_t i = start_idx; i < alerts_.size(); ++i) {
        recent_alerts.push_back(alerts_[i]);
    }
    
    return recent_alerts;
}

void PerformanceMonitor::clearAlerts() {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    alerts_.clear();
}

std::string PerformanceMonitor::generateReport() const {
    std::ostringstream report;
    
    auto stats = getStatistics();
    auto metrics = getCurrentMetrics();
    
    report << "=== Performance Monitor Report ===" << std::endl;
    report << "Generated: " << getCurrentTimestamp() << std::endl;
    report << "Uptime: " << stats.uptime_seconds << " seconds" << std::endl;
    report << std::endl;
    
    // Current metrics
    report << "Current Metrics:" << std::endl;
    report << "  FPS: " << std::fixed << std::setprecision(2) << metrics.fps << std::endl;
    report << "  Frame Time: " << metrics.frame_time_ms << " ms" << std::endl;
    report << "  Data Latency: " << (metrics.data_to_display_latency_us / 1000.0) << " ms" << std::endl;
    report << "  Memory Usage: " << metrics.memory_usage_mb << " MB" << std::endl;
    report << "  CPU Usage: " << metrics.cpu_usage_percent << "%" << std::endl;
    report << "  GPU Usage: " << metrics.gpu_usage_percent << "%" << std::endl;
    report << "  Temperature: " << metrics.temperature_celsius << "°C" << std::endl;
    report << "  Network: " << (metrics.network_connected ? "Connected" : "Disconnected") << std::endl;
    if (metrics.network_connected) {
        report << "    Latency: " << metrics.network_latency_ms << " ms" << std::endl;
        report << "    Throughput: " << metrics.network_throughput_mbps << " Mbps" << std::endl;
    }
    report << std::endl;
    
    // Statistics
    report << "Performance Statistics:" << std::endl;
    report << "  FPS - Avg: " << stats.avg_fps << ", Min: " << stats.min_fps 
           << ", Max: " << stats.max_fps << ", StdDev: " << stats.fps_std_dev << std::endl;
    report << "  Latency - Avg: " << stats.avg_latency_ms << " ms, Min: " << stats.min_latency_ms 
           << " ms, Max: " << stats.max_latency_ms << " ms" << std::endl;
    report << "  Latency - P95: " << stats.latency_p95_ms << " ms, P99: " << stats.latency_p99_ms << " ms" << std::endl;
    report << "  Memory - Avg: " << stats.avg_memory_mb << " MB, Peak: " << stats.peak_memory_mb << " MB" << std::endl;
    report << std::endl;
    
    // Recent alerts
    auto recent_alerts = getRecentAlerts(10);
    if (!recent_alerts.empty()) {
        report << "Recent Alerts:" << std::endl;
        for (const auto& alert : recent_alerts) {
            report << "  [" << formatTimestamp(alert.timestamp) << "] " 
                   << alertTypeToString(alert.type) << ": " << alert.message << std::endl;
        }
    } else {
        report << "No recent alerts" << std::endl;
    }
    
    return report.str();
}

bool PerformanceMonitor::saveReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[PerformanceMonitor] Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    file << generateReport();
    file.close();
    
    std::cout << "[PerformanceMonitor] Report saved to: " << filename << std::endl;
    return true;
}

void PerformanceMonitor::setAlertThresholds(double fps_threshold, double latency_ms_threshold, 
                                           double memory_mb_threshold) {
    alert_threshold_fps_ = fps_threshold;
    alert_threshold_latency_ms_ = latency_ms_threshold;
    alert_threshold_memory_mb_ = memory_mb_threshold;
    
    std::cout << "[PerformanceMonitor] Updated alert thresholds - FPS: " << fps_threshold 
              << ", Latency: " << latency_ms_threshold << "ms, Memory: " << memory_mb_threshold << "MB" << std::endl;
}

void PerformanceMonitor::setMonitoringInterval(uint32_t interval_ms) {
    monitoring_interval_ms_ = std::max(100u, interval_ms);  // Minimum 100ms
    std::cout << "[PerformanceMonitor] Set monitoring interval to " << monitoring_interval_ms_ << "ms" << std::endl;
}

void PerformanceMonitor::setHistorySize(size_t size) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    history_size_ = std::max(size_t(10), std::min(size, size_t(3600)));  // 10 to 3600 samples
    
    // Resize existing histories
    if (fps_history_.size() > history_size_) {
        fps_history_.erase(fps_history_.begin(), fps_history_.end() - history_size_);
    }
    if (latency_history_.size() > history_size_) {
        latency_history_.erase(latency_history_.begin(), latency_history_.end() - history_size_);
    }
    if (memory_history_.size() > history_size_) {
        memory_history_.erase(memory_history_.begin(), memory_history_.end() - history_size_);
    }
    
    std::cout << "[PerformanceMonitor] Set history size to " << history_size_ << " samples" << std::endl;
}

void PerformanceMonitor::monitoringLoop() {
    std::cout << "[PerformanceMonitor] Monitoring loop started" << std::endl;
    start_time_ = std::chrono::high_resolution_clock::now();
    
    while (monitoring_running_) {
        try {
            // Collect system metrics
            collectSystemMetrics();
            
            // Print periodic status
            static int status_counter = 0;
            if (++status_counter % 60 == 0) {  // Every 60 iterations (1 minute at 1s intervals)
                printStatus();
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[PerformanceMonitor] Monitoring error: " << e.what() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(monitoring_interval_ms_));
    }
    
    std::cout << "[PerformanceMonitor] Monitoring loop stopped" << std::endl;
}

void PerformanceMonitor::collectSystemMetrics() {
    // In a real implementation, this would collect actual system metrics
    // For now, we'll update the timestamp to show the monitor is active
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_.last_update = std::chrono::high_resolution_clock::now();
    
    // TODO: Implement actual system metric collection
    // - CPU usage via /proc/stat or system APIs
    // - Memory usage via /proc/meminfo or system APIs
    // - GPU usage via vendor-specific APIs (NVIDIA-ML, etc.)
    // - Temperature via sensors or system APIs
    // - Network statistics via /proc/net/dev or system APIs
}

void PerformanceMonitor::triggerAlert(AlertType type, const std::string& message) {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    
    PerformanceAlert alert;
    alert.type = type;
    alert.message = message;
    alert.timestamp = std::chrono::high_resolution_clock::now();
    
    alerts_.push_back(alert);
    
    // Limit alert history
    if (alerts_.size() > 1000) {
        alerts_.erase(alerts_.begin(), alerts_.begin() + 100);  // Remove oldest 100
    }
    
    // Log alert
    std::cout << "[PerformanceMonitor] ALERT [" << alertTypeToString(type) << "]: " << message << std::endl;
}

void PerformanceMonitor::printStatus() const {
    auto metrics = getCurrentMetrics();
    auto stats = getStatistics();
    
    std::cout << "[PerformanceMonitor] Status - FPS: " << std::fixed << std::setprecision(1) 
              << metrics.fps << ", Latency: " << (metrics.data_to_display_latency_us / 1000.0) 
              << "ms, Memory: " << metrics.memory_usage_mb << "MB, Uptime: " 
              << stats.uptime_seconds << "s" << std::endl;
}

double PerformanceMonitor::calculateAverage(const std::vector<double>& values) const {
    if (values.empty()) return 0.0;
    
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum / values.size();
}

double PerformanceMonitor::calculateStandardDeviation(const std::vector<double>& values) const {
    if (values.size() < 2) return 0.0;
    
    double mean = calculateAverage(values);
    double variance = 0.0;
    
    for (double value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= (values.size() - 1);
    
    return std::sqrt(variance);
}

double PerformanceMonitor::calculatePercentile(std::vector<double> values, double percentile) const {
    if (values.empty()) return 0.0;
    
    std::sort(values.begin(), values.end());
    
    double index = (percentile / 100.0) * (values.size() - 1);
    size_t lower_index = static_cast<size_t>(std::floor(index));
    size_t upper_index = static_cast<size_t>(std::ceil(index));
    
    if (lower_index == upper_index) {
        return values[lower_index];
    }
    
    double weight = index - lower_index;
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight;
}

std::string PerformanceMonitor::alertTypeToString(AlertType type) const {
    switch (type) {
        case AlertType::LOW_FPS: return "LOW_FPS";
        case AlertType::HIGH_LATENCY: return "HIGH_LATENCY";
        case AlertType::HIGH_MEMORY: return "HIGH_MEMORY";
        case AlertType::HIGH_CPU: return "HIGH_CPU";
        case AlertType::HIGH_GPU: return "HIGH_GPU";
        case AlertType::HIGH_TEMPERATURE: return "HIGH_TEMPERATURE";
        case AlertType::CONNECTION_LOST: return "CONNECTION_LOST";
        default: return "UNKNOWN";
    }
}

std::string PerformanceMonitor::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string PerformanceMonitor::formatTimestamp(const std::chrono::high_resolution_clock::time_point& tp) const {
    auto system_tp = std::chrono::system_clock::now() + 
                    std::chrono::duration_cast<std::chrono::system_clock::duration>(
                        tp - std::chrono::high_resolution_clock::now());
    auto time_t = std::chrono::system_clock::to_time_t(system_tp);
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    return oss.str();
}

} // namespace RenderEngine
} // namespace BTQuant