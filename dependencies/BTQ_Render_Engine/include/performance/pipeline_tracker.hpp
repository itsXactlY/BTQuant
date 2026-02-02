#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>

namespace btq {
namespace performance {

struct PipelineMetrics {
    uint64_t events_processed = 0;
    uint64_t events_per_second = 0;
    uint64_t bytes_processed = 0;
    uint64_t bytes_per_second = 0;
    double avg_processing_time_ms = 0.0;
    double min_processing_time_ms = 0.0;
    double max_processing_time_ms = 0.0;
    uint64_t total_processing_time_ns = 0;
    uint64_t dropped_events = 0;
    std::chrono::high_resolution_clock::time_point last_update;
};

struct PipelineStageMetrics {
    std::string stage_name;
    uint64_t events_processed = 0;
    uint64_t events_per_second = 0;
    uint64_t bytes_per_second = 0;
    double avg_stage_time_ms = 0.0;
    double min_stage_time_ms = 0.0;
    double max_stage_time_ms = 0.0;
    uint64_t total_stage_time_ns = 0;
    double bottleneck_score = 0.0; // 0.0 to 1.0, higher means more likely to be a bottleneck
    std::chrono::high_resolution_clock::time_point last_update;
};

struct BottleneckAnalysis {
    std::string bottleneck_stage;
    double bottleneck_severity_score = 0.0; // 0.0 to 1.0, higher means more severe
    double avg_queue_time_ms = 0.0;
    double avg_processing_time_ms = 0.0;
    uint64_t queue_size = 0;
    double cpu_utilization = 0.0; // 0.0 to 1.0
};

class PipelineTracker {
public:
    PipelineTracker();
    ~PipelineTracker();

    // Start tracking pipeline metrics
    void startTracking();
    
    // Stop tracking pipeline metrics
    void stopTracking();

    // Record an event being processed by the pipeline
    void recordEventProcessed(const std::string& stage_name, size_t data_size, 
                             std::chrono::nanoseconds processing_time);

    // Record data throughput
    void recordThroughput(size_t bytes_processed);

    // Get overall pipeline metrics
    PipelineMetrics getOverallMetrics() const;

    // Get metrics for a specific stage
    PipelineStageMetrics getStageMetrics(const std::string& stage_name) const;

    // Get all stage metrics
    std::map<std::string, PipelineStageMetrics> getAllStageMetrics() const;

    // Perform bottleneck analysis
    BottleneckAnalysis analyzeBottlenecks() const;

    // Reset all metrics
    void reset();

    // Export metrics report to a string
    std::string generateMetricsReport() const;

    // Export metrics report to a file
    void exportMetricsReport(const std::string& filename) const;

    // Get current throughput in events per second
    uint64_t getCurrentEventsPerSecond() const;

    // Get current throughput in bytes per second
    uint64_t getCurrentBytesPerSecond() const;

private:
    void updateThroughputCalculations();
    void updateStageThroughput(const std::string& stage_name);
    void calculateBottleneckScores();
    std::string getCurrentTimeString() const;

private:
    mutable std::mutex metrics_mutex_;
    
    // Overall pipeline metrics
    PipelineMetrics overall_metrics_;
    
    // Stage-specific metrics
    std::map<std::string, PipelineStageMetrics> stage_metrics_;
    
    // Throughput tracking
    std::map<std::string, uint64_t> stage_event_counts_last_;
    std::map<std::string, uint64_t> stage_byte_counts_last_;
    std::chrono::steady_clock::time_point last_throughput_calculation_;
    
    // Tracking state
    std::atomic<bool> is_tracking_;
    std::atomic<uint64_t> total_events_processed_;
    std::atomic<uint64_t> total_bytes_processed_;
    
    // Bottleneck analysis data
    std::map<std::string, std::vector<double>> stage_processing_times_;
    static constexpr size_t MAX_SAMPLE_COUNT = 1000; // Keep last 1000 samples for analysis
};

// Global pipeline tracker instance
PipelineTracker& getGlobalPipelineTracker();

} // namespace performance
} // namespace btq