#ifndef BTQ_PERFORMANCE_MEMORY_TRACKER_HPP
#define BTQ_PERFORMANCE_MEMORY_TRACKER_HPP

#include <mutex>
#include <thread>
#include <chrono>
#include <map>
#include <vector>
#include <string>

namespace btq {
namespace performance {

struct MemorySample {
    std::chrono::high_resolution_clock::time_point timestamp;
    size_t memory_usage_bytes;
};

struct AllocationInfo {
    size_t size;
    const void* ptr;
    std::chrono::high_resolution_clock::time_point timestamp;
    std::string tag;
};

struct LeakCandidate {
    const void* ptr;
    size_t size;
    double duration_seconds;
    std::string tag;
};

class MemoryTracker {
public:
    MemoryTracker();

    void startTracking();
    void stopTracking();

    void trackAllocation(size_t size, const void* ptr, const std::string& tag = "");
    void trackDeallocation(const void* ptr);

    size_t getCurrentMemoryUsage() const;
    size_t getPeakMemoryUsage() const;
    size_t getBaselineMemoryUsage() const;
    size_t getTotalAllocatedBytes() const;
    size_t getTotalDeallocatedBytes() const;
    size_t getCurrentAllocationCount() const;
    double getAverageMemoryGrowthRate() const;

    std::vector<MemorySample> getMemorySamples() const;
    std::vector<MemorySample> getRecentMemorySamples(size_t count) const;
    std::vector<AllocationInfo> getActiveAllocations() const;
    std::vector<LeakCandidate> identifyLeaks(double min_duration_seconds = 10.0) const;
    std::vector<LeakCandidate> getLargestUnreleasedAllocations(size_t max_count = 10) const;
    std::vector<LeakCandidate> getGrowingAllocations(double growth_threshold_percent = 10.0) const;

    void resetBaseline();
    void setSamplingInterval(int milliseconds);
    void setLeakThreshold(double seconds);
    void setTrendAnalysisWindow(double seconds);
    void exportMemoryReport(const std::string& filename) const;

    // Trend analysis methods
    double getMemoryGrowthRate(double window_seconds = 10.0) const;
    std::vector<MemorySample> getTrendData(double window_seconds = 30.0) const;
    bool isMemoryLeaking(double threshold_rate_bytes_per_second = 100000.0) const;

    // Visualization-ready data
    std::string getMemoryTrendAsJSON(double window_seconds = 30.0) const;
    std::vector<std::pair<double, size_t>> getMemoryTimelineForVisualization(double window_seconds = 30.0) const;

private:
    void trackingLoop();
    size_t getSystemMemoryUsage() const;
    std::string getCurrentTimeString() const;
    std::string formatBytes(size_t bytes) const;

private:
    bool is_tracking_;
    std::thread tracking_thread_;
    mutable std::mutex samples_mutex_;
    mutable std::mutex allocations_mutex_;

    std::vector<MemorySample> memory_samples_;
    std::map<const void*, AllocationInfo> active_allocations_;

    size_t peak_usage_bytes_;
    size_t baseline_usage_bytes_;
    size_t total_allocated_bytes_;
    size_t total_deallocated_bytes_;
    size_t current_allocation_count_;
    int sampling_interval_ms_;
    double leak_threshold_seconds_;
    double trend_analysis_window_seconds_;
};

// Global memory tracker instance
MemoryTracker& getGlobalMemoryTracker();

} // namespace performance
} // namespace btq

#endif // BTQ_PERFORMANCE_MEMORY_TRACKER_HPP