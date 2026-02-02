#include "../../include/performance/pipeline_tracker.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace btq {
namespace performance {

PipelineTracker::PipelineTracker()
    : is_tracking_(false)
    , total_events_processed_(0)
    , total_bytes_processed_(0) {
    overall_metrics_.last_update = std::chrono::high_resolution_clock::now();
    last_throughput_calculation_ = std::chrono::steady_clock::now();
}

PipelineTracker::~PipelineTracker() {
    if (is_tracking_) {
        stopTracking();
    }
}

void PipelineTracker::startTracking() {
    is_tracking_ = true;
    overall_metrics_.last_update = std::chrono::high_resolution_clock::now();
    last_throughput_calculation_ = std::chrono::steady_clock::now();

    // Initialize throughput tracking
    total_events_processed_ = 0;
    total_bytes_processed_ = 0;
}

void PipelineTracker::stopTracking() {
    is_tracking_ = false;
}

void PipelineTracker::recordEventProcessed(const std::string& stage_name, size_t data_size,
                                         std::chrono::nanoseconds processing_time) {
    if (!is_tracking_) return;

    auto now = std::chrono::high_resolution_clock::now();
    auto processing_time_ms = std::chrono::duration<double, std::milli>(processing_time).count();

    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        // Update overall metrics
        overall_metrics_.events_processed++;
        overall_metrics_.bytes_processed += data_size;
        overall_metrics_.total_processing_time_ns += processing_time.count();

        // Update min/max processing times
        if (overall_metrics_.min_processing_time_ms == 0.0 ||
            processing_time_ms < overall_metrics_.min_processing_time_ms) {
            overall_metrics_.min_processing_time_ms = processing_time_ms;
        }
        if (processing_time_ms > overall_metrics_.max_processing_time_ms) {
            overall_metrics_.max_processing_time_ms = processing_time_ms;
        }

        overall_metrics_.last_update = now;

        // Update stage-specific metrics
        auto& stage_metric = stage_metrics_[stage_name];
        stage_metric.stage_name = stage_name;
        stage_metric.events_processed++;
        stage_metric.total_stage_time_ns += processing_time.count();

        // Update min/max stage times
        if (stage_metric.min_stage_time_ms == 0.0 ||
            processing_time_ms < stage_metric.min_stage_time_ms) {
            stage_metric.min_stage_time_ms = processing_time_ms;
        }
        if (processing_time_ms > stage_metric.max_stage_time_ms) {
            stage_metric.max_stage_time_ms = processing_time_ms;
        }

        stage_metric.last_update = now;

        // Store processing time for bottleneck analysis
        auto& times = stage_processing_times_[stage_name];
        times.push_back(processing_time_ms);
        if (times.size() > MAX_SAMPLE_COUNT) {
            times.erase(times.begin()); // Keep only the most recent samples
        }
    }

    total_events_processed_++;
    total_bytes_processed_ += data_size;

    // Update throughput calculations periodically
    if (total_events_processed_ % 100 == 0) { // Update every 100 events
        updateThroughputCalculations();
        updateStageThroughput(stage_name);
        calculateBottleneckScores();
    }
}

void PipelineTracker::recordThroughput(size_t bytes_processed) {
    if (!is_tracking_) return;

    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        overall_metrics_.bytes_processed += bytes_processed;
    }

    total_bytes_processed_ += bytes_processed;

    // Update throughput calculations periodically
    if (total_bytes_processed_ % 10000 == 0) { // Update every 10KB processed
        updateThroughputCalculations();
    }
}

PipelineMetrics PipelineTracker::getOverallMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    auto metrics = overall_metrics_;

    // Calculate average processing time
    if (metrics.events_processed > 0) {
        metrics.avg_processing_time_ms =
            static_cast<double>(metrics.total_processing_time_ns) /
            static_cast<double>(metrics.events_processed) / 1000000.0; // Convert ns to ms
    }

    // Update events per second and bytes per second
    metrics.events_per_second = getCurrentEventsPerSecond();
    metrics.bytes_per_second = getCurrentBytesPerSecond();

    return metrics;
}

PipelineStageMetrics PipelineTracker::getStageMetrics(const std::string& stage_name) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    auto it = stage_metrics_.find(stage_name);
    if (it != stage_metrics_.end()) {
        auto metrics = it->second;

        // Calculate average stage time
        if (metrics.events_processed > 0) {
            metrics.avg_stage_time_ms =
                static_cast<double>(metrics.total_stage_time_ns) /
                static_cast<double>(metrics.events_processed) / 1000000.0; // Convert ns to ms
        }

        // Update events per second for this stage
        // Note: This is a simplified approach - in a real system, you'd track stage-specific throughput
        metrics.events_per_second = getCurrentEventsPerSecond();

        // Calculate bytes per second for this stage (approximation)
        // In a real system, you'd track stage-specific byte counts
        metrics.bytes_per_second = getCurrentBytesPerSecond();

        return metrics;
    }

    return PipelineStageMetrics{};
}

std::map<std::string, PipelineStageMetrics> PipelineTracker::getAllStageMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    auto all_metrics = stage_metrics_;

    // Calculate averages for each stage
    for (auto& [stage_name, metrics] : all_metrics) {
        if (metrics.events_processed > 0) {
            metrics.avg_stage_time_ms =
                static_cast<double>(metrics.total_stage_time_ns) /
                static_cast<double>(metrics.events_processed) / 1000000.0; // Convert ns to ms
        }

        // Update events per second for this stage
        metrics.events_per_second = getCurrentEventsPerSecond();

        // Calculate bytes per second for this stage (approximation)
        metrics.bytes_per_second = getCurrentBytesPerSecond();
    }

    return all_metrics;
}

BottleneckAnalysis PipelineTracker::analyzeBottlenecks() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    BottleneckAnalysis analysis;
    analysis.bottleneck_severity_score = 0.0;

    if (stage_metrics_.empty()) {
        return analysis;
    }

    // Find the stage with the highest bottleneck score
    double max_bottleneck_score = 0.0;
    std::string bottleneck_stage = "";

    for (const auto& [stage_name, metrics] : stage_metrics_) {
        if (metrics.bottleneck_score > max_bottleneck_score) {
            max_bottleneck_score = metrics.bottleneck_score;
            bottleneck_stage = stage_name;
        }
    }

    // If no stage has a calculated bottleneck score, fall back to average processing time
    if (bottleneck_stage.empty()) {
        double max_avg_time = 0.0;
        for (const auto& [stage_name, metrics] : stage_metrics_) {
            if (metrics.avg_stage_time_ms > max_avg_time) {
                max_avg_time = metrics.avg_stage_time_ms;
                bottleneck_stage = stage_name;
            }
        }
    }

    if (!bottleneck_stage.empty()) {
        analysis.bottleneck_stage = bottleneck_stage;

        // Use the calculated bottleneck score if available
        auto it = stage_metrics_.find(bottleneck_stage);
        if (it != stage_metrics_.end()) {
            analysis.bottleneck_severity_score = it->second.bottleneck_score;
            analysis.avg_processing_time_ms = it->second.avg_stage_time_ms;
        }

        // Estimate queue time based on processing time variance
        const auto& processing_times = stage_processing_times_.at(bottleneck_stage);
        if (!processing_times.empty()) {
            double sum = 0.0;
            for (double time : processing_times) {
                sum += time;
            }
            double mean = sum / processing_times.size();

            double variance = 0.0;
            for (double time : processing_times) {
                variance += (time - mean) * (time - mean);
            }
            if (processing_times.size() > 1) {
                variance /= processing_times.size() - 1; // Sample variance
            }

            analysis.avg_queue_time_ms = std::sqrt(variance); // Rough estimate of queue time
        }
    }

    return analysis;
}

void PipelineTracker::reset() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    overall_metrics_ = PipelineMetrics{};
    stage_metrics_.clear();
    stage_processing_times_.clear();
    stage_event_counts_last_.clear();
    overall_metrics_.last_update = std::chrono::high_resolution_clock::now();

    total_events_processed_ = 0;
    total_bytes_processed_ = 0;

    last_throughput_calculation_ = std::chrono::steady_clock::now();
}

std::string PipelineTracker::generateMetricsReport() const {
    std::ostringstream report;
    report << "Pipeline Metrics Report\n";
    report << "=======================\n";
    report << "Generated: " << getCurrentTimeString() << "\n\n";
    
    auto overall = getOverallMetrics();
    report << "Overall Pipeline Metrics:\n";
    report << "-------------------------\n";
    report << "Events Processed: " << overall.events_processed << "\n";
    report << "Events Per Second: " << overall.events_per_second << "\n";
    report << "Bytes Processed: " << overall.bytes_processed << "\n";
    report << "Average Processing Time: " << std::fixed << std::setprecision(3) 
           << overall.avg_processing_time_ms << " ms\n";
    report << "Min Processing Time: " << overall.min_processing_time_ms << " ms\n";
    report << "Max Processing Time: " << overall.max_processing_time_ms << " ms\n";
    report << "Dropped Events: " << overall.dropped_events << "\n\n";
    
    auto all_stage_metrics = getAllStageMetrics();
    report << "Stage-Specific Metrics:\n";
    report << "-----------------------\n";
    for (const auto& [stage_name, metrics] : all_stage_metrics) {
        report << stage_name << ":\n";
        report << "  Events Processed: " << metrics.events_processed << "\n";
        report << "  Events Per Second: " << metrics.events_per_second << "\n";
        report << "  Bytes Per Second: " << metrics.bytes_per_second << "\n";
        report << "  Average Time: " << std::fixed << std::setprecision(3)
               << metrics.avg_stage_time_ms << " ms\n";
        report << "  Min Time: " << metrics.min_stage_time_ms << " ms\n";
        report << "  Max Time: " << metrics.max_stage_time_ms << " ms\n";
        report << "  Bottleneck Score: " << std::fixed << std::setprecision(3)
               << metrics.bottleneck_score << "\n";
    }
    
    auto bottleneck = analyzeBottlenecks();
    report << "\nBottleneck Analysis:\n";
    report << "--------------------\n";
    if (!bottleneck.bottleneck_stage.empty()) {
        report << "Bottleneck Stage: " << bottleneck.bottleneck_stage << "\n";
        report << "Severity Score: " << std::fixed << std::setprecision(3) 
               << bottleneck.bottleneck_severity_score << "\n";
        report << "Avg Processing Time: " << bottleneck.avg_processing_time_ms << " ms\n";
        report << "Estimated Queue Time: " << bottleneck.avg_queue_time_ms << " ms\n";
    } else {
        report << "No significant bottlenecks detected.\n";
    }
    
    return report.str();
}

void PipelineTracker::exportMetricsReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for metrics report: " << filename << std::endl;
        return;
    }
    
    file << generateMetricsReport();
    file.close();
}

uint64_t PipelineTracker::getCurrentEventsPerSecond() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_throughput_calculation_).count();
    
    if (elapsed > 0) {
        return static_cast<uint64_t>(total_events_processed_.load() / elapsed);
    }
    
    return 0;
}

uint64_t PipelineTracker::getCurrentBytesPerSecond() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_throughput_calculation_).count();
    
    if (elapsed > 0) {
        return static_cast<uint64_t>(total_bytes_processed_.load() / elapsed);
    }
    
    return 0;
}

std::string PipelineTracker::getCurrentTimeString() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();

    return ss.str();
}

void PipelineTracker::updateThroughputCalculations() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_throughput_calculation_).count();

    if (elapsed > 0) {
        // Calculate overall throughput
        uint64_t current_total_events = total_events_processed_.load();
        uint64_t current_total_bytes = total_bytes_processed_.load();

        overall_metrics_.events_per_second = static_cast<uint64_t>((current_total_events - overall_metrics_.events_processed) / elapsed);
        overall_metrics_.bytes_per_second = static_cast<uint64_t>((current_total_bytes - overall_metrics_.bytes_processed) / elapsed);

        // Update last values for next calculation
        overall_metrics_.events_processed = current_total_events;
        overall_metrics_.bytes_processed = current_total_bytes;

        last_throughput_calculation_ = now;
    }
}

void PipelineTracker::updateStageThroughput(const std::string& stage_name) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_throughput_calculation_).count();

    if (elapsed > 0) {
        auto it = stage_metrics_.find(stage_name);
        if (it != stage_metrics_.end()) {
            auto& stage_metric = it->second;

            // Calculate stage throughput
            auto it_last = stage_event_counts_last_.find(stage_name);
            uint64_t last_count = (it_last != stage_event_counts_last_.end()) ? it_last->second : 0;

            stage_metric.events_per_second = static_cast<uint64_t>((stage_metric.events_processed - last_count) / elapsed);

            // Update last count for next calculation
            stage_event_counts_last_[stage_name] = stage_metric.events_processed;
        }
    }
}

void PipelineTracker::calculateBottleneckScores() {
    // Calculate detailed bottleneck scores based on multiple factors
    for (auto& [stage_name, metrics] : stage_metrics_) {
        // Calculate utilization rate (events per second relative to capacity)
        // This is a simplified calculation - in a real system, you'd have capacity limits
        double utilization_factor = (metrics.events_per_second > 0) ?
                                   static_cast<double>(metrics.events_per_second) / 1000.0 : 0.0; // Assuming 1000 events/sec is max capacity

        // Calculate processing time factor (higher processing time indicates potential bottleneck)
        double processing_time_factor = (metrics.avg_stage_time_ms > 0) ?
                                       metrics.avg_stage_time_ms / 100.0 : 0.0; // Normalized against 100ms baseline

        // Calculate variance factor (high variance indicates instability)
        const auto& processing_times = stage_processing_times_[stage_name];
        if (!processing_times.empty()) {
            double sum = 0.0;
            for (double time : processing_times) {
                sum += time;
            }
            double mean = sum / processing_times.size();

            double variance = 0.0;
            for (double time : processing_times) {
                variance += (time - mean) * (time - mean);
            }
            if (processing_times.size() > 1) {
                variance /= processing_times.size() - 1; // Sample variance
            }

            double variance_factor = std::sqrt(variance) / mean; // Coefficient of variation

            // Combine factors to get bottleneck score
            metrics.bottleneck_score = std::min(1.0,
                                              (utilization_factor * 0.4) +
                                              (processing_time_factor * 0.4) +
                                              (variance_factor * 0.2));
        } else {
            metrics.bottleneck_score = utilization_factor * 0.6 + processing_time_factor * 0.4;
        }
    }
}

// Global pipeline tracker instance
static PipelineTracker g_pipeline_tracker;

PipelineTracker& getGlobalPipelineTracker() {
    return g_pipeline_tracker;
}

} // namespace performance
} // namespace btq