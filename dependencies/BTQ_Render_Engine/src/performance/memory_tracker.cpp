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
    leak_threshold_seconds_(10.0) {

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

    file.close();
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