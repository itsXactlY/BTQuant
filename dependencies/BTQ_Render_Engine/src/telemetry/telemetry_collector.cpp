// Telemetry Collector — STUB
// Real implementation requires libuuid (uuid_generate, uuid_unparse) which is
// not available in this build environment. The class declaration in
// <telemetry_collector.h> stays for Phase 7.4 TSC telemetry interface
// compatibility, but this stub provides no-op implementations so the link
// succeeds.

#include "telemetry_collector.h"

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace btq {

std::unique_ptr<TelemetryCollector> TelemetryCollector::instance_;
std::mutex TelemetryCollector::mutex_;

TelemetryCollector& TelemetryCollector::getInstance() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!instance_) {
        instance_.reset(new TelemetryCollector());
    }
    return *instance_;
}

TelemetryCollector::TelemetryCollector()
    : enabled_(false), initialized_(false) {}

void TelemetryCollector::initialize() { initialized_ = true; enabled_ = true; }
void TelemetryCollector::stop()       { enabled_ = false; should_stop_.store(true); }
void TelemetryCollector::setEnabled(bool enabled) { enabled_ = enabled; }
bool   TelemetryCollector::isEnabled() const { return enabled_; }

void TelemetryCollector::recordFeatureUsage(const std::string&) {}
void TelemetryCollector::recordEvent(const std::string&, const std::map<std::string, std::string>&) {}
void TelemetryCollector::recordUserAction(const std::string&, const std::string&) {}

void TelemetryCollector::recordPerformanceMetric(const std::string& name,
                                                 double value,
                                                 const std::string& unit) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    performance_metrics_[name].push_back({value, unit, std::chrono::steady_clock::now()});
}

double TelemetryCollector::getAveragePerformanceMetric(const std::string& name) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    auto it = performance_metrics_.find(name);
    if (it == performance_metrics_.end() || it->second.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& m : it->second) sum += m.value;
    return sum / it->second.size();
}

int TelemetryCollector::getFeatureUsageCount(const std::string&) { return 0; }

void TelemetryCollector::resetFeatureUsage()      {}
void TelemetryCollector::resetPerformanceMetrics() { std::lock_guard<std::mutex> lock(data_mutex_); performance_metrics_.clear(); }

void TelemetryCollector::collectLoop() {}           // background thread body — no-op stub
void TelemetryCollector::flushData()    {}           // no-op stub
void TelemetryCollector::sendData(const TelemetryPayload&) {}
void TelemetryCollector::saveToFile(const TelemetryPayload&) {}
void TelemetryCollector::loadConfiguration() {}

std::string TelemetryCollector::generateSessionId() { return "stub-session"; }
std::string TelemetryCollector::getClientId()       { return "stub-client"; }

}  // namespace btq
