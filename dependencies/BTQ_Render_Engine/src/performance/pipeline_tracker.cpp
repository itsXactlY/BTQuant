#include "../../include/performance/pipeline_tracker.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>

namespace btq {
namespace performance {

PipelineTracker::PipelineTracker()
    : is_tracking_(false)
    , total_events_processed_(0)
    , total_bytes_processed_(0) {
    overall_metrics_.last_update = std::chrono::high_resolution_clock::now();
    overall_metrics_.events_processed_at_last_calc = 0;
    overall_metrics_.bytes_processed_at_last_calc = 0;
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

    // Ensure bottleneck scores are up to date before analysis
    // Create a temporary non-const copy to update scores
    // We'll recalculate scores for the analysis
    std::map<std::string, PipelineStageMetrics> temp_stage_metrics = stage_metrics_;
    std::map<std::string, std::vector<double>> temp_stage_processing_times = stage_processing_times_;

    // Calculate updated bottleneck scores for analysis
    for (auto& [stage_name, metrics] : temp_stage_metrics) {
        // Calculate utilization rate (events per second relative to capacity)
        double utilization_factor = (metrics.events_per_second > 0) ?
                                   std::min(0.3, static_cast<double>(metrics.events_per_second) / 10000.0) : 0.0; // Cap at 0.3

        // Calculate processing time factor (higher processing time indicates potential bottleneck)
        double processing_time_factor = (metrics.avg_stage_time_ms > 0) ?
                                       std::min(0.4, metrics.avg_stage_time_ms / 50.0) : 0.0; // Cap at 0.4

        // Calculate queue length factor (based on events processed vs throughput)
        double queue_length_factor = 0.0;
        if (metrics.events_per_second > 0 && metrics.avg_stage_time_ms > 0) {
            // Estimate queue length based on the relationship between arrival rate and service rate
            double arrival_rate = static_cast<double>(metrics.events_per_second);
            double service_rate = 5000.0 / metrics.avg_stage_time_ms; // Using 5000 as baseline max rate adjusted by processing time

            // If arrival rate exceeds service rate, there's a growing queue
            if (service_rate > 0) {
                if (arrival_rate > service_rate) {
                    queue_length_factor = 0.3; // Cap at 0.3 to prevent saturation
                } else {
                    queue_length_factor = std::min(0.3, arrival_rate / service_rate); // 0.0 to 0.3 based on utilization
                }
            } else {
                queue_length_factor = 0.3; // If service rate is 0, assume moderate bottleneck
            }
        }

        // Calculate variance factor (high variance indicates instability)
        const auto& processing_times = temp_stage_processing_times[stage_name];
        double variance_factor = 0.0;
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

            if (mean > 0) {
                variance_factor = std::sqrt(variance) / mean; // Coefficient of variation
            }
        }

        // Calculate resource contention factor based on processing time distribution
        double contention_factor = 0.0;
        if (!processing_times.empty()) {
            // Look at the top 10% of processing times to detect spikes
            auto sorted_times = processing_times;
            std::sort(sorted_times.begin(), sorted_times.end());

            size_t top_10_percent_idx = static_cast<size_t>(sorted_times.size() * 0.9);
            if (top_10_percent_idx < sorted_times.size() && metrics.avg_stage_time_ms > 0) {
                double p90_time = sorted_times[top_10_percent_idx];
                contention_factor = std::max(0.0, (p90_time - metrics.avg_stage_time_ms) / metrics.avg_stage_time_ms);
            }
        }

        // Combine factors to get bottleneck score
        double raw_score = (utilization_factor * 0.25) +
                          (processing_time_factor * 0.30) +
                          (queue_length_factor * 0.20) +
                          (variance_factor * 0.15) +
                          (contention_factor * 0.10);

        // Normalize to 0-1 range but allow differentiation
        metrics.bottleneck_score = std::min(1.0, raw_score);

        // If all scores are maxed out, we need to differentiate based on processing time
        if (metrics.bottleneck_score >= 0.99) {  // Near the maximum
            // Boost the score slightly based on processing time to differentiate
            double time_boost = metrics.avg_stage_time_ms / 100.0; // Normalize against 100ms
            metrics.bottleneck_score = std::min(1.0, raw_score + time_boost * 0.01);
        }
    }

    // Find the stage with the highest bottleneck score
    double max_bottleneck_score = 0.0;
    std::string bottleneck_stage = "";

    for (const auto& [stage_name, metrics] : temp_stage_metrics) {
        if (metrics.bottleneck_score > max_bottleneck_score) {
            max_bottleneck_score = metrics.bottleneck_score;
            bottleneck_stage = stage_name;
        }
    }

    // If all bottleneck scores are equal (e.g., all are 1.0 due to saturation),
    // fall back to the stage with the highest average processing time
    if (max_bottleneck_score > 0) {
        // Check if multiple stages have the same max score
        std::vector<std::string> tied_stages;
        for (const auto& [stage_name, metrics] : temp_stage_metrics) {
            if (std::abs(metrics.bottleneck_score - max_bottleneck_score) < 0.001) { // Account for floating point precision
                tied_stages.push_back(stage_name);
            }
        }

        // If multiple stages are tied, pick the one with the highest average processing time
        if (tied_stages.size() > 1) {
            std::string best_stage = tied_stages[0];
            double best_avg_time = 0.0;

            for (const auto& stage_name : tied_stages) {
                auto it = temp_stage_metrics.find(stage_name);
                if (it != temp_stage_metrics.end() && it->second.avg_stage_time_ms > best_avg_time) {
                    best_avg_time = it->second.avg_stage_time_ms;
                    best_stage = stage_name;
                }
            }
            bottleneck_stage = best_stage;
        }
    } else {
        // If no stage has a calculated bottleneck score, fall back to average processing time
        double max_avg_time = 0.0;
        for (const auto& [stage_name, metrics] : temp_stage_metrics) {
            if (metrics.avg_stage_time_ms > max_avg_time) {
                max_avg_time = metrics.avg_stage_time_ms;
                bottleneck_stage = stage_name;
            }
        }
    }

    if (!bottleneck_stage.empty()) {
        analysis.bottleneck_stage = bottleneck_stage;

        // Use the updated bottleneck score from our temporary calculation
        auto it = temp_stage_metrics.find(bottleneck_stage);
        if (it != temp_stage_metrics.end()) {
            analysis.bottleneck_severity_score = it->second.bottleneck_score;
            analysis.avg_processing_time_ms = it->second.avg_stage_time_ms;
        } else {
            // Fallback: find in original metrics if not in temp
            auto orig_it = stage_metrics_.find(bottleneck_stage);
            if (orig_it != stage_metrics_.end()) {
                analysis.avg_processing_time_ms = orig_it->second.avg_stage_time_ms;

                // If we didn't calculate a score in temp, use the stored one
                if (analysis.bottleneck_severity_score == 0.0) {
                    analysis.bottleneck_severity_score = orig_it->second.bottleneck_score;
                }
            }
        }

        // Ensure avg_processing_time_ms is set even if the above failed
        if (analysis.avg_processing_time_ms == 0.0) {
            auto orig_it = stage_metrics_.find(bottleneck_stage);
            if (orig_it != stage_metrics_.end()) {
                analysis.avg_processing_time_ms = orig_it->second.avg_stage_time_ms;
            }
        }

        // Estimate queue time based on processing time variance
        auto times_it = temp_stage_processing_times.find(bottleneck_stage);
        if (times_it != temp_stage_processing_times.end()) {
            const auto& processing_times = times_it->second;
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
        } else {
            // Fallback: look in original processing times
            auto orig_times_it = stage_processing_times_.find(bottleneck_stage);
            if (orig_times_it != stage_processing_times_.end()) {
                const auto& processing_times = orig_times_it->second;
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
    overall_metrics_.events_processed_at_last_calc = 0;
    overall_metrics_.bytes_processed_at_last_calc = 0;

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
        report << "Avg Processing Time: " << std::fixed << std::setprecision(3)
               << bottleneck.avg_processing_time_ms << " ms\n";
        report << "Estimated Queue Time: " << std::fixed << std::setprecision(3)
               << bottleneck.avg_queue_time_ms << " ms\n";
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

std::map<std::string, double> PipelineTracker::getStageThroughputRatios() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    std::map<std::string, double> ratios;

    // Calculate total events processed across all stages
    uint64_t total_events = 0;
    for (const auto& [stage_name, metrics] : stage_metrics_) {
        total_events += metrics.events_processed;
    }

    if (total_events > 0) {
        for (const auto& [stage_name, metrics] : stage_metrics_) {
            ratios[stage_name] = static_cast<double>(metrics.events_processed) / total_events;
        }
    }

    return ratios;
}

double PipelineTracker::getPipelineEfficiency() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    if (stage_metrics_.empty()) {
        return 0.0;
    }

    // Calculate efficiency as the ratio of the slowest stage to the fastest stage
    double min_avg_time = std::numeric_limits<double>::max();
    double max_avg_time = 0.0;

    for (const auto& [stage_name, metrics] : stage_metrics_) {
        if (metrics.avg_stage_time_ms > 0) {
            min_avg_time = std::min(min_avg_time, metrics.avg_stage_time_ms);
            max_avg_time = std::max(max_avg_time, metrics.avg_stage_time_ms);
        }
    }

    if (min_avg_time > 0 && max_avg_time > 0) {
        // Efficiency is the ratio of fastest to slowest (bounded between 0 and 1)
        return min_avg_time / max_avg_time;
    }

    return 1.0; // If all stages have 0 avg time, assume perfect efficiency
}

std::map<std::string, std::pair<double, double>> PipelineTracker::getStageProcessingTimePercentiles() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    std::map<std::string, std::pair<double, double>> percentiles;

    for (const auto& [stage_name, times] : stage_processing_times_) {
        if (times.empty()) {
            percentiles[stage_name] = std::make_pair(0.0, 0.0);
            continue;
        }

        auto sorted_times = times;
        std::sort(sorted_times.begin(), sorted_times.end());

        // Calculate P50 (median) and P95
        size_t p50_idx = static_cast<size_t>(sorted_times.size() * 0.5);
        size_t p95_idx = static_cast<size_t>(sorted_times.size() * 0.95);

        p50_idx = std::min(p50_idx, sorted_times.size() - 1);
        p95_idx = std::min(p95_idx, sorted_times.size() - 1);

        double p50 = sorted_times[p50_idx];
        double p95 = sorted_times[p95_idx];

        percentiles[stage_name] = std::make_pair(p50, p95);
    }

    return percentiles;
}

void PipelineTracker::updateThroughputCalculations() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_throughput_calculation_).count();

    if (elapsed > 0) {
        // Calculate overall throughput
        uint64_t current_total_events = total_events_processed_.load();
        uint64_t current_total_bytes = total_bytes_processed_.load();

        // Calculate throughput based on the difference since last calculation
        uint64_t events_processed_since_last = current_total_events - overall_metrics_.events_processed_at_last_calc;
        uint64_t bytes_processed_since_last = current_total_bytes - overall_metrics_.bytes_processed_at_last_calc;

        overall_metrics_.events_per_second = static_cast<uint64_t>(events_processed_since_last / elapsed);
        overall_metrics_.bytes_per_second = static_cast<uint64_t>(bytes_processed_since_last / elapsed);

        // Update last values for next calculation
        overall_metrics_.events_processed_at_last_calc = current_total_events;
        overall_metrics_.bytes_processed_at_last_calc = current_total_bytes;

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
        double utilization_factor = (metrics.events_per_second > 0) ?
                                   std::min(0.2, static_cast<double>(metrics.events_per_second) / 20000.0) : 0.0; // Cap at 0.2

        // Calculate processing time factor (higher processing time indicates potential bottleneck)
        double processing_time_factor = (metrics.avg_stage_time_ms > 0) ?
                                       std::min(0.5, metrics.avg_stage_time_ms / 30.0) : 0.0; // Cap at 0.5, more weight to processing time

        // Calculate queue length factor (based on events processed vs throughput)
        double queue_length_factor = 0.0;
        if (metrics.events_per_second > 0 && metrics.avg_stage_time_ms > 0) {
            // Estimate queue length based on the relationship between arrival rate and service rate
            double arrival_rate = static_cast<double>(metrics.events_per_second);
            double service_rate = 10000.0 / metrics.avg_stage_time_ms; // Using 10000 as baseline max rate adjusted by processing time

            // If arrival rate exceeds service rate, there's a growing queue
            if (service_rate > 0) {
                if (arrival_rate > service_rate) {
                    queue_length_factor = 0.2; // Cap at 0.2 to prevent saturation
                } else {
                    queue_length_factor = std::min(0.2, arrival_rate / service_rate); // 0.0 to 0.2 based on utilization
                }
            } else {
                queue_length_factor = 0.2; // If service rate is 0, assume moderate bottleneck
            }
        }

        // Calculate variance factor (high variance indicates instability)
        const auto& processing_times = stage_processing_times_[stage_name];
        double variance_factor = 0.0;
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

            if (mean > 0) {
                variance_factor = std::min(0.1, std::sqrt(variance) / mean); // Coefficient of variation, cap at 0.1
            }
        }

        // Calculate resource contention factor based on processing time distribution
        double contention_factor = 0.0;
        if (!processing_times.empty()) {
            // Look at the top 10% of processing times to detect spikes
            auto sorted_times = processing_times;
            std::sort(sorted_times.begin(), sorted_times.end());

            size_t top_10_percent_idx = static_cast<size_t>(sorted_times.size() * 0.9);
            if (top_10_percent_idx < sorted_times.size() && metrics.avg_stage_time_ms > 0) {
                double p90_time = sorted_times[top_10_percent_idx];
                contention_factor = std::min(0.05, std::max(0.0, (p90_time - metrics.avg_stage_time_ms) / metrics.avg_stage_time_ms)); // Cap at 0.05
            }
        }

        // Combine factors to get bottleneck score
        metrics.bottleneck_score = utilization_factor * 0.05 +
                                  processing_time_factor * 0.70 +  // Much higher weight to processing time
                                  queue_length_factor * 0.10 +
                                  variance_factor * 0.10 +
                                  contention_factor * 0.05;
    }
}

// Global pipeline tracker instance
static PipelineTracker g_pipeline_tracker;

PipelineTracker& getGlobalPipelineTracker() {
    return g_pipeline_tracker;
}

} // namespace performance
} // namespace btq