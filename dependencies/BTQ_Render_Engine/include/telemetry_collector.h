#ifndef BTQ_TELEMETRY_COLLECTOR_H
#define BTQ_TELEMETRY_COLLECTOR_H

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>

namespace btq {

struct PerformanceMeasurement {
    double value;
    std::string unit;
    std::chrono::steady_clock::time_point timestamp;
};

struct EventRecord {
    std::string event_type;
    std::map<std::string, std::string> properties;
    std::chrono::steady_clock::time_point timestamp;
    std::string session_id;
};

struct TelemetryPayload {
    std::string client_id;
    std::string session_id;
    long long timestamp;
    std::map<std::string, int> feature_usage;
    std::map<std::string, std::vector<PerformanceMeasurement>> performance_metrics;
    std::vector<EventRecord> events;
};

class TelemetryCollector {
public:
    // Singleton access
    static TelemetryCollector& getInstance();
    
    // Initialize and start telemetry collection
    void initialize();
    
    // Stop telemetry collection
    void stop();
    
    // Enable/disable telemetry collection
    void setEnabled(bool enabled);
    bool isEnabled() const;
    
    // Record feature usage
    void recordFeatureUsage(const std::string& feature_name);
    
    // Record performance metrics
    void recordPerformanceMetric(const std::string& metric_name, 
                                double value, 
                                const std::string& unit = "");
    
    // Record custom events
    void recordEvent(const std::string& event_type, 
                    const std::map<std::string, std::string>& properties = {});
    
    // Record user actions
    void recordUserAction(const std::string& action, 
                         const std::string& context = "");
    
    // Get average performance metric
    double getAveragePerformanceMetric(const std::string& metric_name);
    
    // Get feature usage count
    int getFeatureUsageCount(const std::string& feature_name);
    
    // Reset data
    void resetFeatureUsage();
    void resetPerformanceMetrics();

private:
    explicit TelemetryCollector();

    friend std::default_delete<TelemetryCollector>;

    // Main collection loop
    void collectLoop();

    // Flush collected data
    void flushData();

    // Send data to telemetry endpoint
    void sendData(const TelemetryPayload& payload);

    // Save data to local file (for demonstration)
    void saveToFile(const TelemetryPayload& payload);

    // Generate unique session ID
    std::string generateSessionId();

    // Get or create client ID
    std::string getClientId();

    // Load configuration
    void loadConfiguration();

    // Singleton instance
    static std::unique_ptr<btq::TelemetryCollector> instance_;
    static std::mutex mutex_;

    // Configuration
    bool enabled_;
    bool initialized_;
    std::atomic<bool> should_stop_{false};

    // Data storage
    std::string session_id_;
    std::string client_id_;
    std::map<std::string, int> feature_usage_;
    std::map<std::string, std::vector<PerformanceMeasurement>> performance_metrics_;
    std::vector<EventRecord> events_;

    // Threading
    std::thread telemetry_thread_;
    std::mutex data_mutex_;

    // Time tracking
    std::chrono::steady_clock::time_point last_feature_usage_time_;
};

} // namespace btq

#endif // BTQ_TELEMETRY_COLLECTOR_H