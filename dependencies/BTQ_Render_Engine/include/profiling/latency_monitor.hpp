#pragma once

/**
 * @file latency_monitor.hpp
 * @brief Tick-to-Photon Latency Monitoring
 * 
 * This implementation provides:
 * - High-resolution latency measurement
 * - Tick-to-photon pipeline tracking
 * - Statistical analysis (min, max, avg, percentiles)
 * - Latency histogram generation
 * - Real-time alerts for latency spikes
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

namespace btq {
namespace profiling {

/**
 * @brief High-resolution clock type
 */
using HighResClock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<HighResClock>;
using Nanoseconds = std::chrono::nanoseconds;
using Microseconds = std::chrono::microseconds;
using Milliseconds = std::chrono::milliseconds;

/**
 * @brief Latency measurement point in the pipeline
 */
enum class LatencyPoint : uint8_t {
    NETWORK_RECEIVE,        // Data received from network
    PARSE_START,            // JSON parsing started
    PARSE_END,              // JSON parsing completed
    QUEUE_ENQUEUE,          // Data enqueued to processing queue
    QUEUE_DEQUEUE,          // Data dequeued for processing
    ORDERBOOK_UPDATE,       // Orderbook updated
    OHLCV_UPDATE,           // OHLCV candle updated
    RENDER_START,           // Render frame started
    RENDER_END,             // Render frame completed
    GPU_SUBMIT,             // Command buffer submitted to GPU
    GPU_COMPLETE,           // GPU processing completed
    PRESENT,                // Frame presented to screen
    COUNT                   // Number of measurement points
};

/**
 * @brief Single latency measurement
 */
struct LatencyMeasurement {
    uint64_t id = 0;
    LatencyPoint point;
    TimePoint timestamp;
    uint64_t thread_id = 0;
    std::string metadata;
};

/**
 * @brief Complete tick-to-photon trace
 */
struct LatencyTrace {
    uint64_t id = 0;
    std::array<TimePoint, static_cast<size_t>(LatencyPoint::COUNT)> points;
    std::array<bool, static_cast<size_t>(LatencyPoint::COUNT)> valid{};
    std::string symbol;
    double price = 0.0;
    bool is_complete = false;
};

/**
 * @brief Latency statistics
 */
struct LatencyStats {
    double min_ns = 0.0;
    double max_ns = 0.0;
    double avg_ns = 0.0;
    double std_dev_ns = 0.0;
    double p50_ns = 0.0;    // Median
    double p90_ns = 0.0;    // 90th percentile
    double p95_ns = 0.0;    // 95th percentile
    double p99_ns = 0.0;    // 99th percentile
    size_t sample_count = 0;
    
    /**
     * @brief Get stats in microseconds
     */
    double min_us() const { return min_ns / 1000.0; }
    double max_us() const { return max_ns / 1000.0; }
    double avg_us() const { return avg_ns / 1000.0; }
    double p50_us() const { return p50_ns / 1000.0; }
    double p90_us() const { return p90_ns / 1000.0; }
    double p95_us() const { return p95_ns / 1000.0; }
    double p99_us() const { return p99_ns / 1000.0; }
    
    /**
     * @brief Get stats in milliseconds
     */
    double min_ms() const { return min_ns / 1000000.0; }
    double max_ms() const { return max_ns / 1000000.0; }
    double avg_ms() const { return avg_ns / 1000000.0; }
};

/**
 * @brief Latency histogram bin
 */
struct LatencyBin {
    double lower_bound_ns;
    double upper_bound_ns;
    size_t count;
    double percentage;
};

/**
 * @brief Latency histogram
 */
struct LatencyHistogram {
    std::vector<LatencyBin> bins;
    double bin_width_ns;
    size_t total_samples;
    double underflow_count;  // Samples below lowest bin
    double overflow_count;   // Samples above highest bin
};

/**
 * @brief Latency monitor configuration
 */
struct LatencyConfig {
    size_t max_samples = 10000;
    size_t histogram_bins = 50;
    double histogram_min_ns = 0.0;
    double histogram_max_ns = 100000000.0;  // 100ms
    bool enable_traces = true;
    size_t max_traces = 100;
    double alert_threshold_us = 1000.0;  // Alert if latency > 1ms
};

/**
 * @brief High-resolution latency monitor
 */
class LatencyMonitor {
public:
    LatencyMonitor() = default;
    
    /**
     * @brief Initialize the monitor
     */
    void initialize(const LatencyConfig& config = LatencyConfig{}) {
        config_ = config;
        samples_.reserve(config.max_samples);
    }
    
    /**
     * @brief Start a new trace
     */
    uint64_t beginTrace(const std::string& symbol = "", double price = 0.0) {
        uint64_t id = next_trace_id_++;
        
        if (!config_.enable_traces) {
            return id;
        }
        
        std::lock_guard<std::mutex> lock(traces_mutex_);
        
        LatencyTrace trace;
        trace.id = id;
        trace.symbol = symbol;
        trace.price = price;
        traces_[id] = std::move(trace);
        
        // Cleanup old traces
        while (traces_.size() > config_.max_traces) {
            traces_.erase(traces_.begin());
        }
        
        return id;
    }
    
    /**
     * @brief Record a measurement point
     */
    void recordPoint(uint64_t trace_id, LatencyPoint point, const std::string& metadata = "") {
        auto now = HighResClock::now();
        
        if (config_.enable_traces) {
            std::lock_guard<std::mutex> lock(traces_mutex_);
            
            auto it = traces_.find(trace_id);
            if (it != traces_.end()) {
                size_t idx = static_cast<size_t>(point);
                it->second.points[idx] = now;
                it->second.valid[idx] = true;
            }
        }
    }
    
    /**
     * @brief End a trace and calculate latency
     */
    void endTrace(uint64_t trace_id) {
        if (!config_.enable_traces) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(traces_mutex_);
        
        auto it = traces_.find(trace_id);
        if (it == traces_.end()) {
            return;
        }
        
        auto& trace = it->second;
        trace.is_complete = true;
        
        // Calculate tick-to-photon latency
        if (trace.valid[static_cast<size_t>(LatencyPoint::NETWORK_RECEIVE)] &&
            trace.valid[static_cast<size_t>(LatencyPoint::PRESENT)]) {
            
            auto start = trace.points[static_cast<size_t>(LatencyPoint::NETWORK_RECEIVE)];
            auto end = trace.points[static_cast<size_t>(LatencyPoint::PRESENT)];
            
            int64_t latency_ns = std::chrono::duration_cast<Nanoseconds>(end - start).count();
            
            addSample(static_cast<double>(latency_ns));
            
            // Check for alert
            if (latency_ns > config_.alert_threshold_us * 1000.0) {
                if (alert_callback_) {
                    alert_callback_(trace_id, latency_ns / 1000.0);
                }
            }
        }
    }
    
    /**
     * @brief Record a standalone latency sample
     */
    void recordLatency(double latency_ns) {
        addSample(latency_ns);
    }
    
    /**
     * @brief Record latency between two time points
     */
    void recordLatency(TimePoint start, TimePoint end) {
        int64_t latency_ns = std::chrono::duration_cast<Nanoseconds>(end - start).count();
        addSample(static_cast<double>(latency_ns));
    }
    
    /**
     * @brief Get current statistics
     */
    LatencyStats getStats() const {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        
        LatencyStats stats;
        stats.sample_count = samples_.size();
        
        if (samples_.empty()) {
            return stats;
        }
        
        // Calculate basic stats
        auto [min_it, max_it] = std::minmax_element(samples_.begin(), samples_.end());
        stats.min_ns = *min_it;
        stats.max_ns = *max_it;
        
        double sum = std::accumulate(samples_.begin(), samples_.end(), 0.0);
        stats.avg_ns = sum / samples_.size();
        
        // Calculate standard deviation
        double sq_sum = 0.0;
        for (double sample : samples_) {
            double diff = sample - stats.avg_ns;
            sq_sum += diff * diff;
        }
        stats.std_dev_ns = std::sqrt(sq_sum / samples_.size());
        
        // Calculate percentiles
        std::vector<double> sorted_samples = samples_;
        std::sort(sorted_samples.begin(), sorted_samples.end());
        
        auto percentile = [&sorted_samples](double p) {
            size_t idx = static_cast<size_t>(p / 100.0 * (sorted_samples.size() - 1));
            return sorted_samples[idx];
        };
        
        stats.p50_ns = percentile(50.0);
        stats.p90_ns = percentile(90.0);
        stats.p95_ns = percentile(95.0);
        stats.p99_ns = percentile(99.0);
        
        return stats;
    }
    
    /**
     * @brief Generate latency histogram
     */
    LatencyHistogram getHistogram() const {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        
        LatencyHistogram histogram;
        histogram.total_samples = samples_.size();
        histogram.bin_width_ns = (config_.histogram_max_ns - config_.histogram_min_ns) / config_.histogram_bins;
        histogram.underflow_count = 0;
        histogram.overflow_count = 0;
        
        if (samples_.empty()) {
            return histogram;
        }
        
        // Initialize bins
        histogram.bins.resize(config_.histogram_bins);
        for (size_t i = 0; i < config_.histogram_bins; ++i) {
            histogram.bins[i].lower_bound_ns = config_.histogram_min_ns + i * histogram.bin_width_ns;
            histogram.bins[i].upper_bound_ns = histogram.bins[i].lower_bound_ns + histogram.bin_width_ns;
            histogram.bins[i].count = 0;
        }
        
        // Fill bins
        for (double sample : samples_) {
            if (sample < config_.histogram_min_ns) {
                histogram.underflow_count++;
            } else if (sample >= config_.histogram_max_ns) {
                histogram.overflow_count++;
            } else {
                size_t bin_idx = static_cast<size_t>((sample - config_.histogram_min_ns) / histogram.bin_width_ns);
                bin_idx = std::min(bin_idx, config_.histogram_bins - 1);
                histogram.bins[bin_idx].count++;
            }
        }
        
        // Calculate percentages
        for (auto& bin : histogram.bins) {
            bin.percentage = static_cast<double>(bin.count) / histogram.total_samples * 100.0;
        }
        
        return histogram;
    }
    
    /**
     * @brief Set alert callback
     */
    void setAlertCallback(std::function<void(uint64_t trace_id, double latency_us)> callback) {
        alert_callback_ = std::move(callback);
    }
    
    /**
     * @brief Clear all samples
     */
    void clear() {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        samples_.clear();
    }
    
    /**
     * @brief Get sample count
     */
    size_t getSampleCount() const {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        return samples_.size();
    }

private:
    void addSample(double latency_ns) {
        std::lock_guard<std::mutex> lock(samples_mutex_);
        
        samples_.push_back(latency_ns);
        
        // Remove old samples if over limit
        while (samples_.size() > config_.max_samples) {
            samples_.pop_front();
        }
    }
    
    LatencyConfig config_;
    
    mutable std::mutex samples_mutex_;
    std::deque<double> samples_;
    
    mutable std::mutex traces_mutex_;
    std::unordered_map<uint64_t, LatencyTrace> traces_;
    
    std::atomic<uint64_t> next_trace_id_{1};
    
    std::function<void(uint64_t trace_id, double latency_us)> alert_callback_;
};

/**
 * @brief Scoped latency measurement
 */
class ScopedLatency {
public:
    ScopedLatency(LatencyMonitor& monitor, LatencyPoint start_point, LatencyPoint end_point)
        : monitor_(monitor)
        , start_point_(start_point)
        , end_point_(end_point)
        , start_time_(HighResClock::now())
    {}
    
    ~ScopedLatency() {
        auto end_time = HighResClock::now();
        monitor_.recordLatency(start_time_, end_time);
    }
    
private:
    LatencyMonitor& monitor_;
    LatencyPoint start_point_;
    LatencyPoint end_point_;
    TimePoint start_time_;
};

/**
 * @brief Pipeline latency tracker for detailed stage analysis
 */
class PipelineLatencyTracker {
public:
    /**
     * @brief Record latency for a pipeline stage
     */
    void recordStage(LatencyPoint stage, double latency_ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t idx = static_cast<size_t>(stage);
        stage_latencies_[idx].push_back(latency_ns);
        
        // Limit stored samples
        if (stage_latencies_[idx].size() > max_samples_per_stage_) {
            stage_latencies_[idx].pop_front();
        }
    }
    
    /**
     * @brief Get average latency for each stage
     */
    std::array<double, static_cast<size_t>(LatencyPoint::COUNT)> getStageAverages() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::array<double, static_cast<size_t>(LatencyPoint::COUNT)> averages{};
        
        for (size_t i = 0; i < static_cast<size_t>(LatencyPoint::COUNT); ++i) {
            if (!stage_latencies_[i].empty()) {
                double sum = std::accumulate(stage_latencies_[i].begin(), 
                                            stage_latencies_[i].end(), 0.0);
                averages[i] = sum / stage_latencies_[i].size();
            }
        }
        
        return averages;
    }
    
    /**
     * @brief Get stage name
     */
    static const char* getStageName(LatencyPoint stage) {
        switch (stage) {
            case LatencyPoint::NETWORK_RECEIVE: return "Network Receive";
            case LatencyPoint::PARSE_START: return "Parse Start";
            case LatencyPoint::PARSE_END: return "Parse End";
            case LatencyPoint::QUEUE_ENQUEUE: return "Queue Enqueue";
            case LatencyPoint::QUEUE_DEQUEUE: return "Queue Dequeue";
            case LatencyPoint::ORDERBOOK_UPDATE: return "Orderbook Update";
            case LatencyPoint::OHLCV_UPDATE: return "OHLCV Update";
            case LatencyPoint::RENDER_START: return "Render Start";
            case LatencyPoint::RENDER_END: return "Render End";
            case LatencyPoint::GPU_SUBMIT: return "GPU Submit";
            case LatencyPoint::GPU_COMPLETE: return "GPU Complete";
            case LatencyPoint::PRESENT: return "Present";
            default: return "Unknown";
        }
    }

private:
    mutable std::mutex mutex_;
    std::array<std::deque<double>, static_cast<size_t>(LatencyPoint::COUNT)> stage_latencies_;
    size_t max_samples_per_stage_ = 1000;
};

/**
 * @brief Real-time latency display helper
 */
class LatencyDisplay {
public:
    /**
     * @brief Format latency for display
     */
    static std::string formatLatency(double latency_ns) {
        if (latency_ns < 1000.0) {
            return std::to_string(static_cast<int64_t>(latency_ns)) + " ns";
        } else if (latency_ns < 1000000.0) {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.2f us", latency_ns / 1000.0);
            return std::string(buffer);
        } else if (latency_ns < 1000000000.0) {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.2f ms", latency_ns / 1000000.0);
            return std::string(buffer);
        } else {
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.2f s", latency_ns / 1000000000.0);
            return std::string(buffer);
        }
    }
    
    /**
     * @brief Format stats for display
     */
    static std::string formatStats(const LatencyStats& stats) {
        return "Min: " + formatLatency(stats.min_ns) + 
               ", Max: " + formatLatency(stats.max_ns) +
               ", Avg: " + formatLatency(stats.avg_ns) +
               ", P99: " + formatLatency(stats.p99_ns);
    }
};

} // namespace profiling
} // namespace btq
