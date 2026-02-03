#include "performance/memory_tracker.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cmath>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
#elif __linux__
    #include <sys/resource.h>
    #include <unistd.h>
    #include <cstring>
    #include <fstream>
#endif

namespace btq {
namespace performance {

MemoryTracker::MemoryTracker() :
    is_tracking_(false),
    peak_usage_bytes_(0),
    baseline_usage_bytes_(0),
    total_allocated_bytes_(0),
    total_deallocated_bytes_(0),
    current_allocation_count_(0),
    sampling_interval_ms_(100),
    leak_threshold_seconds_(10.0),
    trend_analysis_window_seconds_(30.0) {

    // Initialize baseline memory usage
    baseline_usage_bytes_ = getSystemMemoryUsage();
}

void MemoryTracker::startTracking() {
    is_tracking_ = true;
    tracking_thread_ = std::thread(&MemoryTracker::trackingLoop, this);
}

void MemoryTracker::stopTracking() {
    is_tracking_ = false;
    if (tracking_thread_.joinable()) {
        tracking_thread_.join();
    }
}

void MemoryTracker::trackAllocation(size_t size, const void* ptr, const std::string& tag) {
    std::lock_guard<std::mutex> lock(allocations_mutex_);

    AllocationInfo info;
    info.size = size;
    info.ptr = ptr;
    info.timestamp = std::chrono::high_resolution_clock::now();
    info.tag = tag;

    active_allocations_[ptr] = info;
    total_allocated_bytes_ += size;
    current_allocation_count_++;

    // Update peak usage if needed
    size_t current_usage = getCurrentMemoryUsage();
    if (current_usage > peak_usage_bytes_) {
        peak_usage_bytes_ = current_usage;
    }
}

void MemoryTracker::trackDeallocation(const void* ptr) {
    std::lock_guard<std::mutex> lock(allocations_mutex_);

    auto it = active_allocations_.find(ptr);
    if (it != active_allocations_.end()) {
        total_deallocated_bytes_ += it->second.size;
        current_allocation_count_--;
        active_allocations_.erase(it);
    }
}

size_t MemoryTracker::getCurrentMemoryUsage() const {
    return getSystemMemoryUsage() - baseline_usage_bytes_;
}

size_t MemoryTracker::getSystemMemoryUsage() const {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<size_t>(pmc.WorkingSetSize);
    }
    return 0;
#elif __linux__
    // Try to get resident set size (RSS) from /proc/self/status
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key;
            size_t value;
            std::string unit;
            iss >> key >> value >> unit;
            // Convert from KB to bytes
            return value * 1024;
        }
    }
    // Fallback to getrusage if VmRSS not available
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<size_t>(usage.ru_maxrss * 1024); // Convert from KB to bytes
#else
    // Fallback - return 0 if platform not supported
    return 0;
#endif
}

size_t MemoryTracker::getPeakMemoryUsage() const {
    return peak_usage_bytes_;
}

size_t MemoryTracker::getBaselineMemoryUsage() const {
    return baseline_usage_bytes_;
}

size_t MemoryTracker::getTotalAllocatedBytes() const {
    return total_allocated_bytes_;
}

size_t MemoryTracker::getTotalDeallocatedBytes() const {
    return total_deallocated_bytes_;
}

size_t MemoryTracker::getCurrentAllocationCount() const {
    return current_allocation_count_;
}

double MemoryTracker::getAverageMemoryGrowthRate() const {
    std::lock_guard<std::mutex> lock(samples_mutex_);

    if (memory_samples_.size() < 2) {
        return 0.0; // Not enough samples to calculate growth rate
    }

    // Calculate average growth rate over the last N samples
    const size_t num_samples = std::min(static_cast<size_t>(50), memory_samples_.size());
    if (num_samples < 2) return 0.0;

    auto start_it = memory_samples_.end() - num_samples;
    auto end_it = memory_samples_.end();

    double total_change = 0.0;
    auto prev_sample = start_it;
    auto curr_sample = std::next(start_it);

    while (curr_sample != end_it) {
        double time_diff = std::chrono::duration<double>(
            curr_sample->timestamp - prev_sample->timestamp).count();
        double mem_diff = static_cast<double>(curr_sample->memory_usage_bytes) -
                         static_cast<double>(prev_sample->memory_usage_bytes);

        if (time_diff > 0) {
            total_change += mem_diff / time_diff; // bytes per second
        }

        ++prev_sample;
        ++curr_sample;
    }

    return total_change / (num_samples - 1);
}

std::vector<MemorySample> MemoryTracker::getMemorySamples() const {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    return memory_samples_;
}

std::vector<MemorySample> MemoryTracker::getRecentMemorySamples(size_t count) const {
    std::lock_guard<std::mutex> lock(samples_mutex_);
    if (memory_samples_.empty()) {
        return {};
    }

    size_t start_idx = memory_samples_.size() > count ?
                      memory_samples_.size() - count : 0;

    return std::vector<MemorySample>(
        memory_samples_.begin() + start_idx,
        memory_samples_.end()
    );
}

std::vector<AllocationInfo> MemoryTracker::getActiveAllocations() const {
    std::lock_guard<std::mutex> lock(allocations_mutex_);
    std::vector<AllocationInfo> result;
    for (const auto& pair : active_allocations_) {
        result.push_back(pair.second);
    }
    return result;
}

std::vector<LeakCandidate> MemoryTracker::identifyLeaks(double min_duration_seconds) const {
    std::vector<LeakCandidate> leaks;
    auto now = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(allocations_mutex_);
    for (const auto& pair : active_allocations_) {
        auto duration = std::chrono::duration<double>(now - pair.second.timestamp).count();
        if (duration >= min_duration_seconds) {
            LeakCandidate candidate;
            candidate.ptr = pair.first;
            candidate.size = pair.second.size;
            candidate.duration_seconds = duration;
            candidate.tag = pair.second.tag;
            leaks.push_back(candidate);
        }
    }

    // Sort by size descending (largest allocations first)
    std::sort(leaks.begin(), leaks.end(), [](const LeakCandidate& a, const LeakCandidate& b) {
        return a.size > b.size;
    });

    return leaks;
}

std::vector<LeakCandidate> MemoryTracker::getLargestUnreleasedAllocations(size_t max_count) const {
    auto active_allocs = getActiveAllocations();

    // Sort by size descending
    std::sort(active_allocs.begin(), active_allocs.end(),
              [](const AllocationInfo& a, const AllocationInfo& b) {
                  return a.size > b.size;
              });

    std::vector<LeakCandidate> result;
    size_t count = 0;
    for (const auto& alloc : active_allocs) {
        if (count >= max_count) break;

        LeakCandidate candidate;
        candidate.ptr = alloc.ptr;
        candidate.size = alloc.size;
        candidate.duration_seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - alloc.timestamp).count();
        candidate.tag = alloc.tag;
        result.push_back(candidate);

        count++;
    }

    return result;
}

void MemoryTracker::trackingLoop() {
    while (is_tracking_) {
        auto current_usage = getCurrentMemoryUsage();

        MemorySample sample;
        sample.timestamp = std::chrono::high_resolution_clock::now();
        sample.memory_usage_bytes = current_usage;

        {
            std::lock_guard<std::mutex> lock(samples_mutex_);
            memory_samples_.push_back(sample);

            // Keep only the last 10000 samples to prevent memory overflow
            if (memory_samples_.size() > 10000) {
                memory_samples_.erase(memory_samples_.begin(),
                                    memory_samples_.begin() + memory_samples_.size() - 10000);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(sampling_interval_ms_));
    }
}

void MemoryTracker::resetBaseline() {
    baseline_usage_bytes_ = getSystemMemoryUsage();
}

void MemoryTracker::setSamplingInterval(int milliseconds) {
    sampling_interval_ms_ = std::max(10, milliseconds); // Minimum 10ms interval
}

void MemoryTracker::setLeakThreshold(double seconds) {
    leak_threshold_seconds_ = std::max(1.0, seconds); // Minimum 1 second threshold
}

void MemoryTracker::setTrendAnalysisWindow(double seconds) {
    trend_analysis_window_seconds_ = std::max(1.0, seconds); // Minimum 1 second
}

double MemoryTracker::getMemoryGrowthRate(double window_seconds) const {
    std::lock_guard<std::mutex> lock(samples_mutex_);

    if (memory_samples_.empty()) {
        return 0.0;
    }

    auto cutoff_time = std::chrono::high_resolution_clock::now() -
                       std::chrono::duration<double>(window_seconds);

    // Find samples within the time window
    auto start_it = memory_samples_.begin();
    for (auto it = memory_samples_.rbegin(); it != memory_samples_.rend(); ++it) {
        if (it->timestamp < cutoff_time) {
            start_it = it.base();
            break;
        }
    }

    if (start_it == memory_samples_.end() ||
        std::distance(start_it, memory_samples_.end()) < 2) {
        return 0.0;
    }

    auto start_sample = start_it;
    auto end_sample = memory_samples_.rbegin();

    double time_diff = std::chrono::duration<double>(
        end_sample->timestamp - start_sample->timestamp).count();
    double mem_diff = static_cast<double>(end_sample->memory_usage_bytes) -
                     static_cast<double>(start_sample->memory_usage_bytes);

    if (time_diff > 0) {
        return mem_diff / time_diff; // bytes per second
    }

    return 0.0;
}

std::vector<MemorySample> MemoryTracker::getTrendData(double window_seconds) const {
    std::lock_guard<std::mutex> lock(samples_mutex_);

    if (memory_samples_.empty()) {
        return {};
    }

    auto cutoff_time = std::chrono::high_resolution_clock::now() -
                       std::chrono::duration<double>(window_seconds);

    std::vector<MemorySample> result;
    for (const auto& sample : memory_samples_) {
        if (sample.timestamp >= cutoff_time) {
            result.push_back(sample);
        }
    }

    return result;
}

bool MemoryTracker::isMemoryLeaking(double threshold_rate_bytes_per_second) const {
    double growth_rate = getMemoryGrowthRate(10.0); // Check growth over last 10 seconds
    return growth_rate > threshold_rate_bytes_per_second;
}

std::vector<LeakCandidate> MemoryTracker::getGrowingAllocations(double growth_threshold_percent) const {
    std::vector<LeakCandidate> growing_allocs;
    auto now = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(allocations_mutex_);

    // Group allocations by tag to identify growing patterns
    std::map<std::string, std::vector<const AllocationInfo*>> allocations_by_tag;
    for (const auto& pair : active_allocations_) {
        allocations_by_tag[pair.second.tag].push_back(&pair.second);
    }

    // Analyze each tag group for growth patterns
    for (const auto& tag_pair : allocations_by_tag) {
        const auto& allocs = tag_pair.second;
        if (allocs.empty()) continue;

        // Calculate average age and total size for this tag
        double total_age = 0.0;
        size_t total_size = 0;
        for (const auto* alloc : allocs) {
            total_age += std::chrono::duration<double>(now - alloc->timestamp).count();
            total_size += alloc->size;
        }

        double avg_age = total_age / allocs.size();

        // If there are many allocations with this tag and they're relatively old,
        // it might indicate a growing pattern
        if (allocs.size() > 5 && avg_age > 5.0) { // More than 5 allocations older than 5 seconds
            for (const auto* alloc : allocs) {
                LeakCandidate candidate;
                candidate.ptr = alloc->ptr;
                candidate.size = alloc->size;
                candidate.duration_seconds = std::chrono::duration<double>(now - alloc->timestamp).count();
                candidate.tag = alloc->tag;

                // Only include if allocation is significantly old
                if (candidate.duration_seconds > 10.0) {
                    growing_allocs.push_back(candidate);
                }
            }
        }
    }

    // Sort by duration (longest held allocations first)
    std::sort(growing_allocs.begin(), growing_allocs.end(),
              [](const LeakCandidate& a, const LeakCandidate& b) {
                  return a.duration_seconds > b.duration_seconds;
              });

    return growing_allocs;
}

void MemoryTracker::exportMemoryReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for memory report: " << filename << std::endl;
        return;
    }

    file << "Memory Usage Report\n";
    file << "==================\n";
    file << "Timestamp: " << getCurrentTimeString() << "\n\n";

    file << "Current Memory Usage: " << formatBytes(getCurrentMemoryUsage()) << "\n";
    file << "Peak Memory Usage: " << formatBytes(getPeakMemoryUsage()) << "\n";
    file << "Baseline Memory Usage: " << formatBytes(getBaselineMemoryUsage()) << "\n";
    file << "Total Allocated: " << formatBytes(getTotalAllocatedBytes()) << "\n";
    file << "Total Deallocated: " << formatBytes(getTotalDeallocatedBytes()) << "\n";
    file << "Current Allocation Count: " << getCurrentAllocationCount() << "\n";
    file << "Average Growth Rate: " << formatBytes(static_cast<size_t>(getAverageMemoryGrowthRate())) << "/sec\n\n";

    // Trend analysis
    file << "Trend Analysis (last 10s): " << formatBytes(static_cast<size_t>(getMemoryGrowthRate(10.0))) << "/sec\n";
    file << "Trend Analysis (last 30s): " << formatBytes(static_cast<size_t>(getMemoryGrowthRate(30.0))) << "/sec\n";
    file << "Is Memory Leaking: " << (isMemoryLeaking() ? "YES" : "NO") << "\n\n";

    // Active allocations
    auto active_allocs = getActiveAllocations();
    file << "Active Allocations (" << active_allocs.size() << "):\n";
    file << "------------------\n";
    for (const auto& alloc : active_allocs) {
        file << "Ptr: " << alloc.ptr
             << ", Size: " << formatBytes(alloc.size)
             << ", Tag: " << alloc.tag
             << ", Age: "
             << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - alloc.timestamp).count()
             << "s\n";
    }

    file << "\nTop 10 Largest Unreleased Allocations:\n";
    file << "------------------------------------\n";
    auto largest_allocs = getLargestUnreleasedAllocations(10);
    for (const auto& alloc : largest_allocs) {
        file << "Ptr: " << alloc.ptr
             << ", Size: " << formatBytes(alloc.size)
             << ", Duration: " << std::fixed << std::setprecision(2) << alloc.duration_seconds
             << "s, Tag: " << alloc.tag << "\n";
    }

    file << "\nPotential Leaks (>=" << leak_threshold_seconds_ << "s):\n";
    file << "---------------------\n";
    auto leaks = identifyLeaks(leak_threshold_seconds_);
    for (const auto& leak : leaks) {
        file << "Ptr: " << leak.ptr
             << ", Size: " << formatBytes(leak.size)
             << ", Duration: " << std::fixed << std::setprecision(2) << leak.duration_seconds
             << "s, Tag: " << leak.tag << "\n";
    }

    file << "\nGrowing Allocation Patterns:\n";
    file << "---------------------------\n";
    auto growing_allocs = getGrowingAllocations();
    for (const auto& alloc : growing_allocs) {
        file << "Ptr: " << alloc.ptr
             << ", Size: " << formatBytes(alloc.size)
             << ", Duration: " << std::fixed << std::setprecision(2) << alloc.duration_seconds
             << "s, Tag: " << alloc.tag << "\n";
    }

    file.close();
}

std::string MemoryTracker::getMemoryTrendAsJSON(double window_seconds) const {
    std::ostringstream json_stream;
    json_stream << "{\n";
    json_stream << "  \"timestamp\": \"" << getCurrentTimeString() << "\",\n";
    json_stream << "  \"current_usage_bytes\": " << getCurrentMemoryUsage() << ",\n";
    json_stream << "  \"peak_usage_bytes\": " << getPeakMemoryUsage() << ",\n";
    json_stream << "  \"baseline_usage_bytes\": " << getBaselineMemoryUsage() << ",\n";
    json_stream << "  \"growth_rate_bytes_per_sec\": " << getMemoryGrowthRate(window_seconds) << ",\n";
    json_stream << "  \"is_leaking\": " << (isMemoryLeaking() ? "true" : "false") << ",\n";
    json_stream << "  \"samples\": [\n";

    auto samples = getTrendData(window_seconds);
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        auto time_point = std::chrono::duration<double>(
            sample.timestamp.time_since_epoch()).count();

        json_stream << "    {\n";
        json_stream << "      \"timestamp\": " << time_point << ",\n";
        json_stream << "      \"memory_usage_bytes\": " << sample.memory_usage_bytes << "\n";
        json_stream << "    }";

        if (i < samples.size() - 1) {
            json_stream << ",";
        }
        json_stream << "\n";
    }

    json_stream << "  ]\n";
    json_stream << "}";

    return json_stream.str();
}

std::vector<std::pair<double, size_t>> MemoryTracker::getMemoryTimelineForVisualization(double window_seconds) const {
    auto samples = getTrendData(window_seconds);
    std::vector<std::pair<double, size_t>> timeline;

    if (samples.empty()) {
        return timeline;
    }

    // Use the first sample as reference time to keep numbers manageable
    auto reference_time = samples.front().timestamp;

    for (const auto& sample : samples) {
        auto time_offset = std::chrono::duration<double>(
            sample.timestamp - reference_time).count();
        timeline.emplace_back(time_offset, sample.memory_usage_bytes);
    }

    return timeline;
}

std::vector<MemoryTrendPoint> MemoryTracker::getDetailedTrendAnalysis(double window_seconds) const {
    std::lock_guard<std::mutex> lock(samples_mutex_);

    if (memory_samples_.empty()) {
        return {};
    }

    auto cutoff_time = std::chrono::high_resolution_clock::now() -
                       std::chrono::duration<double>(window_seconds);

    // Find samples within the time window
    std::vector<MemorySample> window_samples;
    for (const auto& sample : memory_samples_) {
        if (sample.timestamp >= cutoff_time) {
            window_samples.push_back(sample);
        }
    }

    if (window_samples.size() < 2) {
        return {};
    }

    std::vector<MemoryTrendPoint> trend_points;

    // Calculate statistics for the time window
    size_t min_usage = SIZE_MAX;
    size_t max_usage = 0;
    size_t total_usage = 0;

    for (const auto& sample : window_samples) {
        min_usage = std::min(min_usage, sample.memory_usage_bytes);
        max_usage = std::max(max_usage, sample.memory_usage_bytes);
        total_usage += sample.memory_usage_bytes;
    }

    size_t avg_usage = total_usage / window_samples.size();

    // Create trend points with additional metrics
    for (const auto& sample : window_samples) {
        MemoryTrendPoint point;
        point.timestamp = std::chrono::duration<double>(
            sample.timestamp.time_since_epoch()).count();
        point.memory_usage_bytes = sample.memory_usage_bytes;
        point.relative_to_min = static_cast<double>(sample.memory_usage_bytes - min_usage) /
                               std::max(1.0, static_cast<double>(min_usage));
        point.relative_to_avg = static_cast<double>(sample.memory_usage_bytes) /
                               std::max(1.0, static_cast<double>(avg_usage));
        point.trend_direction = 0; // Will calculate based on neighbors

        trend_points.push_back(point);
    }

    // Calculate trend direction for each point (compared to previous and next)
    for (size_t i = 1; i < trend_points.size() - 1; ++i) {
        double prev_diff = static_cast<double>(trend_points[i].memory_usage_bytes) -
                          static_cast<double>(trend_points[i-1].memory_usage_bytes);
        double next_diff = static_cast<double>(trend_points[i+1].memory_usage_bytes) -
                          static_cast<double>(trend_points[i].memory_usage_bytes);

        trend_points[i].trend_direction = (prev_diff + next_diff) / 2.0;
    }

    // Handle edge cases for first and last points
    if (trend_points.size() > 1) {
        trend_points[0].trend_direction = static_cast<double>(trend_points[1].memory_usage_bytes) -
                                        static_cast<double>(trend_points[0].memory_usage_bytes);
        trend_points.back().trend_direction = static_cast<double>(trend_points.back().memory_usage_bytes) -
                                            static_cast<double>(trend_points[trend_points.size()-2].memory_usage_bytes);
    }

    return trend_points;
}

std::string MemoryTracker::getFormattedTrendReport(double window_seconds) const {
    auto trend_points = getDetailedTrendAnalysis(window_seconds);

    if (trend_points.empty()) {
        return "No trend data available.";
    }

    std::ostringstream report;
    report << "Memory Trend Analysis Report\n";
    report << "============================\n";
    report << "Time Window: " << window_seconds << " seconds\n";
    report << "Sample Count: " << trend_points.size() << "\n\n";

    // Calculate summary statistics
    size_t min_usage = SIZE_MAX, max_usage = 0;
    size_t first_usage = trend_points.front().memory_usage_bytes;
    size_t last_usage = trend_points.back().memory_usage_bytes;

    for (const auto& point : trend_points) {
        min_usage = std::min(min_usage, point.memory_usage_bytes);
        max_usage = std::max(max_usage, point.memory_usage_bytes);
    }

    double growth_rate = ((static_cast<double>(last_usage) - static_cast<double>(first_usage)) /
                         static_cast<double>(first_usage)) * 100.0;

    report << "Summary Statistics:\n";
    report << "- Min Usage: " << formatBytes(min_usage) << "\n";
    report << "- Max Usage: " << formatBytes(max_usage) << "\n";
    report << "- Current Usage: " << formatBytes(last_usage) << "\n";
    report << "- Initial Usage: " << formatBytes(first_usage) << "\n";
    report << "- Growth Rate: " << std::fixed << std::setprecision(2) << growth_rate << "%\n\n";

    // Determine trend classification
    double avg_trend_dir = 0;
    for (const auto& point : trend_points) {
        avg_trend_dir += point.trend_direction;
    }
    avg_trend_dir /= trend_points.size();

    report << "Overall Trend: ";
    if (avg_trend_dir > 10000) {
        report << "Increasing Rapidly\n";
    } else if (avg_trend_dir > 1000) {
        report << "Increasing\n";
    } else if (avg_trend_dir > 100) {
        report << "Slightly Increasing\n";
    } else if (avg_trend_dir < -10000) {
        report << "Decreasing Rapidly\n";
    } else if (avg_trend_dir < -1000) {
        report << "Decreasing\n";
    } else if (avg_trend_dir < -100) {
        report << "Slightly Decreasing\n";
    } else {
        report << "Stable\n";
    }

    report << "\nRecent Activity:\n";
    report << "----------------\n";
    size_t recent_count = std::min(static_cast<size_t>(10), trend_points.size());
    for (size_t i = trend_points.size() - recent_count; i < trend_points.size(); ++i) {
        const auto& point = trend_points[i];
        report << "[" << getCurrentTimeStringFromTimestamp(point.timestamp) << "] "
               << formatBytes(point.memory_usage_bytes)
               << " (Δ" << (point.trend_direction >= 0 ? "+" : "")
               << formatBytes(static_cast<size_t>(std::abs(point.trend_direction))) << ")\n";
    }

    return report.str();
}

std::string MemoryTracker::getCurrentTimeStringFromTimestamp(double timestamp) const {
    using namespace std::chrono;
    auto time_point = system_clock::time_point() +
                     duration_cast<system_clock::duration>(duration<double>(timestamp));
    auto time_t = system_clock::to_time_t(time_point);
    auto fractional_seconds = timestamp - std::floor(timestamp);
    auto ms = static_cast<int>(fractional_seconds * 1000);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms;

    return ss.str();
}

std::string MemoryTracker::formatBytes(size_t bytes) const {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

std::string MemoryTracker::getCurrentTimeString() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();

    return ss.str();
}

// Global memory tracker instance
static MemoryTracker g_memory_tracker;

MemoryTracker& getGlobalMemoryTracker() {
    return g_memory_tracker;
}

} // namespace performance
} // namespace btq