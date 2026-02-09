#include "telemetry_collector.h"
#include "../include/structured_logger.hpp"
#include <chrono>
#include <thread>
#include <random>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <sys/stat.h>
#include <uuid/uuid.h>

namespace btq {

// Static member initialization
std::unique_ptr<TelemetryCollector> TelemetryCollector::instance_ = nullptr;
std::recursive_mutex TelemetryCollector::mutex_;  // Changed to recursive_mutex for consistency

TelemetryCollector::TelemetryCollector() : enabled_(true), initialized_(false), 
                                          session_id_(generateSessionId()),
                                          client_id_(getClientId()) {
    // Initialize random seed
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
}


TelemetryCollector& TelemetryCollector::getInstance() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);  // Already using recursive_mutex, keeping consistent
    if (!instance_) {
        instance_ = std::unique_ptr<TelemetryCollector>(new TelemetryCollector());
    }
    return *instance_;
}

void TelemetryCollector::initialize() {
    if (initialized_) return;
    
    // Load configuration
    loadConfiguration();
    
    // Start telemetry collection thread
    if (enabled_) {
        telemetry_thread_ = std::thread(&TelemetryCollector::collectLoop, this);
        initialized_ = true;
    }
}

void TelemetryCollector::stop() {
    if (!initialized_) return;
    
    should_stop_ = true;
    if (telemetry_thread_.joinable()) {
        telemetry_thread_.join();
    }
    flushData();
    initialized_ = false;
}

void TelemetryCollector::setEnabled(bool enabled) {
    enabled_ = enabled;
    if (enabled && !initialized_) {
        initialize();
    } else if (!enabled && initialized_) {
        stop();
    }
}

bool TelemetryCollector::isEnabled() const {
    return enabled_;
}

void TelemetryCollector::recordFeatureUsage(const std::string& feature_name) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    feature_usage_[feature_name]++;
    last_feature_usage_time_ = std::chrono::steady_clock::now();
}

void TelemetryCollector::recordPerformanceMetric(const std::string& metric_name, 
                                                double value, 
                                                const std::string& unit) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    performance_metrics_[metric_name].push_back({
        value,
        unit,
        std::chrono::steady_clock::now()
    });
    
    // Keep only the last 1000 measurements per metric to prevent memory bloat
    if (performance_metrics_[metric_name].size() > 1000) {
        performance_metrics_[metric_name].erase(
            performance_metrics_[metric_name].begin(),
            performance_metrics_[metric_name].begin() + 100
        );
    }
}

void TelemetryCollector::recordEvent(const std::string& event_type, 
                                    const std::map<std::string, std::string>& properties) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    events_.push_back({
        event_type,
        properties,
        std::chrono::steady_clock::now(),
        session_id_
    });
    
    // Keep only the last 1000 events to prevent memory bloat
    if (events_.size() > 1000) {
        events_.erase(events_.begin(), events_.begin() + 100);
    }
}

void TelemetryCollector::recordUserAction(const std::string& action, 
                                         const std::string& context) {
    if (!enabled_) return;
    
    std::map<std::string, std::string> properties;
    properties["context"] = context;
    properties["session_id"] = session_id_;
    
    recordEvent("user_action", properties);
}

void TelemetryCollector::collectLoop() {
    while (!should_stop_) {
        std::this_thread::sleep_for(std::chrono::minutes(5)); // Collect every 5 minutes
        
        if (enabled_) {
            flushData();
        }
    }
}

void TelemetryCollector::flushData() {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    // Prepare telemetry payload
    TelemetryPayload payload;
    payload.client_id = client_id_;
    payload.session_id = session_id_;
    payload.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Copy data to payload
    payload.feature_usage = feature_usage_;
    payload.performance_metrics = performance_metrics_;
    payload.events = std::move(events_);
    events_.clear(); // Clear events after moving
    
    // Send telemetry data (in a real implementation, this would send to a server)
    sendData(payload);
}

void TelemetryCollector::sendData(const TelemetryPayload& payload) {
    // In a real implementation, this would send data to a telemetry server
    // For now, we'll log the data or save to a local file for demonstration
    
    // Log telemetry data (anonymized)
    BTQ_LOG_INFO("Telemetry - Sending telemetry data for session: " +
                 payload.session_id.substr(0, 8) + "...");

    // Save to local file for demonstration purposes
    saveToFile(payload);
}

void TelemetryCollector::saveToFile(const TelemetryPayload& payload) {
    // Create a filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    
    std::string filename = "/tmp/btq_telemetry_" + ss.str() + ".json";
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        BTQ_LOG_WARNING("Telemetry - Could not open telemetry file for writing: " + filename);
        return;
    }
    
    // Write simplified JSON representation
    file << "{\n";
    file << "  \"client_id\": \"" << payload.client_id << "\",\n";
    file << "  \"session_id\": \"" << payload.session_id << "\",\n";
    file << "  \"timestamp\": " << payload.timestamp << ",\n";
    file << "  \"feature_usage_count\": " << payload.feature_usage.size() << ",\n";
    
    // Feature usage
    file << "  \"features\": {\n";
    bool first_feature = true;
    for (const auto& [feature, count] : payload.feature_usage) {
        if (!first_feature) file << ",\n";
        file << "    \"" << feature << "\": " << count;
        first_feature = false;
    }
    file << "\n  },\n";
    
    // Performance metrics count
    file << "  \"performance_metrics_count\": " << payload.performance_metrics.size() << "\n";
    file << "}\n";
    
    file.close();

    BTQ_LOG_DEBUG("Telemetry - Telemetry data saved to: " + filename);
}

std::string TelemetryCollector::generateSessionId() {
    uuid_t uuid;
    uuid_generate(uuid);
    
    char uuid_str[37]; // 36 chars + null terminator
    uuid_unparse(uuid, uuid_str);
    
    return std::string(uuid_str);
}

std::string TelemetryCollector::getClientId() {
    // In a real implementation, this would get a persistent client ID
    // For now, we'll generate one and store it in a temporary location
    std::string client_id_file = "/tmp/btq_client_id.txt";
    
    struct stat buffer;
    if (stat(client_id_file.c_str(), &buffer) == 0) {
        // File exists, read the client ID
        std::ifstream file(client_id_file);
        if (file.is_open()) {
            std::string id;
            std::getline(file, id);
            file.close();
            return id;
        }
    }
    
    // File doesn't exist, generate a new client ID
    uuid_t uuid;
    uuid_generate(uuid);
    
    char uuid_str[37];
    uuid_unparse(uuid, uuid_str);
    std::string new_client_id(uuid_str);
    
    // Save the new client ID to file
    std::ofstream file(client_id_file);
    if (file.is_open()) {
        file << new_client_id;
        file.close();
    }
    
    return new_client_id;
}

void TelemetryCollector::loadConfiguration() {
    // In a real implementation, this would load configuration from a file
    // For now, we'll just set defaults
    enabled_ = true; // Default to enabled
}

double TelemetryCollector::getAveragePerformanceMetric(const std::string& metric_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = performance_metrics_.find(metric_name);
    if (it == performance_metrics_.end() || it->second.empty()) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (const auto& measurement : it->second) {
        sum += measurement.value;
    }
    
    return sum / static_cast<double>(it->second.size());
}

int TelemetryCollector::getFeatureUsageCount(const std::string& feature_name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    auto it = feature_usage_.find(feature_name);
    if (it != feature_usage_.end()) {
        return it->second;
    }
    
    return 0;
}

void TelemetryCollector::resetFeatureUsage() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    feature_usage_.clear();
}

void TelemetryCollector::resetPerformanceMetrics() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    performance_metrics_.clear();
}

} // namespace btq