#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <stack>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <functional>

namespace BTQuant {

struct FunctionProfileData {
    uint64_t call_count = 0;
    uint64_t total_duration_ns = 0;
    uint64_t min_duration_ns = UINT64_MAX;
    uint64_t max_duration_ns = 0;
    std::chrono::high_resolution_clock::time_point start_time;

    // For calculating averages and statistics
    std::vector<uint64_t> duration_history;
    static constexpr size_t MAX_HISTORY_SIZE = 100;

    void add_duration(uint64_t duration_ns) {
        call_count++;
        total_duration_ns += duration_ns;

        if (duration_ns < min_duration_ns) {
            min_duration_ns = duration_ns;
        }
        if (duration_ns > max_duration_ns) {
            max_duration_ns = duration_ns;
        }

        duration_history.push_back(duration_ns);
        if (duration_history.size() > MAX_HISTORY_SIZE) {
            duration_history.erase(duration_history.begin());
        }
    }

    double get_average_duration_ms() const {
        if (call_count == 0) return 0.0;
        return static_cast<double>(total_duration_ns) / static_cast<double>(call_count) / 1000000.0;
    }

    double get_total_duration_ms() const {
        return static_cast<double>(total_duration_ns) / 1000000.0;
    }

    double get_min_duration_ms() const {
        return min_duration_ns == UINT64_MAX ? 0.0 : static_cast<double>(min_duration_ns) / 1000000.0;
    }

    double get_max_duration_ms() const {
        return static_cast<double>(max_duration_ns) / 1000000.0;
    }
};

struct CallTreeNode {
    std::string function_name;
    FunctionProfileData profile_data;
    std::vector<std::unique_ptr<CallTreeNode>> children;
    CallTreeNode* parent;

    CallTreeNode(const std::string& name, CallTreeNode* p = nullptr)
        : function_name(name), parent(p) {}
};

struct ThreadProfileData {
    std::unordered_map<std::string, FunctionProfileData> function_profiles;
    std::stack<std::chrono::high_resolution_clock::time_point> call_stack;
    std::stack<std::string> function_stack;

    // Hierarchical profiling data
    std::unique_ptr<CallTreeNode> call_tree_root;
    CallTreeNode* current_node;
};

class CPUProfiler {
public:
    CPUProfiler();
    ~CPUProfiler();

    // Start profiling a function
    void start_function(const std::string& function_name);

    // End profiling a function
    void end_function(const std::string& function_name);

    // Get profile data for a specific function
    FunctionProfileData get_function_profile(const std::string& function_name) const;

    // Get all profile data for the current thread
    std::unordered_map<std::string, FunctionProfileData> get_thread_profiles() const;

    // Get all profile data aggregated across all threads
    std::map<std::string, FunctionProfileData> get_aggregated_profiles() const;

    // Reset all profiling data
    void reset();

    // Enable/disable profiling
    void set_enabled(bool enabled);
    bool is_enabled() const;

    // Generate a human-readable report of CPU usage
    std::string generate_report() const;

    // Generate a hierarchical call tree report
    std::string generate_hierarchical_report() const;

    // Get top N functions by total time spent
    std::vector<std::pair<std::string, FunctionProfileData>> get_top_functions_by_total_time(int n = 10) const;

    // Get top N functions by average time spent
    std::vector<std::pair<std::string, FunctionProfileData>> get_top_functions_by_average_time(int n = 10) const;

    // Get call tree for visualization
    std::unique_ptr<CallTreeNode> get_call_tree() const;

    // Get flame graph data for visualization
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, double>>>> get_flame_graph_data() const;

    // Get function call statistics with percentiles
    std::map<std::string, FunctionProfileData> get_percentile_statistics(double percentile = 95.0) const;

    // Get exclusive and inclusive timing data
    struct TimingBreakdown {
        std::string function_name;
        double exclusive_time_ms;  // Time spent in function excluding children
        double inclusive_time_ms;  // Time spent in function including children
        uint64_t call_count;
    };

    std::vector<TimingBreakdown> get_timing_breakdown() const;

    // Enhanced detailed breakdown with additional metrics
    struct DetailedTimingBreakdown {
        std::string function_name;
        double exclusive_time_ms;      // Time spent in function excluding children
        double inclusive_time_ms;      // Time spent in function including children
        uint64_t call_count;          // Number of times function was called
        double min_time_ms;           // Minimum time taken by function
        double max_time_ms;           // Maximum time taken by function
        double avg_time_ms;           // Average time taken by function
        double variance_time_ms;      // Variance of execution time
        double std_deviation_ms;      // Standard deviation of execution time
        double percentage_of_total;   // Percentage of total CPU time
        std::vector<double> percentiles; // Various percentiles (50th, 90th, 95th, 99th)
    };

    std::vector<DetailedTimingBreakdown> get_detailed_timing_breakdown() const;

    // Enhanced function-level profiling with detailed breakdown
    struct FunctionLevelBreakdown {
        std::string function_name;
        uint64_t call_count;
        double total_time_ms;
        double exclusive_time_ms;
        double inclusive_time_ms;
        double min_time_ms;
        double max_time_ms;
        double avg_time_ms;
        double std_deviation_ms;
        double percentage_of_total;
        std::vector<double> percentiles; // 25th, 50th, 75th, 95th, 99th percentiles
        std::string thread_id;
    };

    std::vector<FunctionLevelBreakdown> get_function_level_breakdown() const;

    // Detailed CPU time breakdown by function with additional metrics
    struct DetailedCPUTimeBreakdown {
        std::string function_name;
        uint64_t call_count;
        double total_time_ms;
        double exclusive_time_ms;
        double inclusive_time_ms;
        double min_time_ms;
        double max_time_ms;
        double avg_time_ms;
        double std_deviation_ms;
        double percentage_of_total;
        std::vector<double> percentiles; // 25th, 50th, 75th, 90th, 95th, 99th percentiles
        std::string thread_id;
        double cpu_utilization;  // Estimated CPU utilization percentage
        uint64_t total_samples;  // Number of samples collected
        double variance_time_ms;      // Variance of execution time
        std::chrono::steady_clock::time_point first_call_time;
        std::chrono::steady_clock::time_point last_call_time;
    };

    std::vector<DetailedCPUTimeBreakdown> get_detailed_cpu_time_breakdown() const;

    // Get CPU time distribution by function
    std::map<std::string, double> get_cpu_time_distribution() const;

    // Get functions with similar execution patterns
    std::vector<std::vector<std::string>> get_function_similarity_clusters() const;

    // Generate detailed breakdown report showing where CPU time is spent
    std::string generate_detailed_breakdown_report() const;

    // Generate a report focusing on CPU time distribution
    std::string generate_cpu_time_distribution_report() const;

    // Generate a report showing function call overhead
    std::string generate_overhead_analysis_report() const;

    // Generate a detailed CPU time breakdown report
    std::string generate_detailed_cpu_time_breakdown_report() const;

    // Hot path detection - identifies most time-consuming call paths
    std::vector<std::vector<std::string>> get_hot_paths(int max_paths = 10) const;

    // Get profiling statistics summary
    struct ProfilingStats {
        uint64_t total_calls;
        double total_time_ms;
        double avg_time_per_call_ms;
        double min_time_per_call_ms;
        double max_time_per_call_ms;
        uint64_t unique_functions;
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point end_time;
        bool is_active;
    };

    ProfilingStats get_profiling_stats() const;

    // Get functions with highest variance in execution time
    std::vector<std::pair<std::string, double>> get_high_variance_functions(int n = 10) const;

    // Get functions with highest call frequency
    std::vector<std::pair<std::string, uint64_t>> get_most_frequent_functions(int n = 10) const;

    // Filter profiles by time range
    std::map<std::string, FunctionProfileData> get_filtered_profiles(
        double min_time_ms = 0.0,
        double max_time_ms = std::numeric_limits<double>::max()
    ) const;

    // RAII wrapper for automatic profiling
    class ProfileScope {
    public:
        ProfileScope(const std::string& function_name);
        ~ProfileScope();

    private:
        std::string function_name_;
    };

    // RAII wrapper with custom tag for more granular profiling
    class TaggedProfileScope {
    public:
        TaggedProfileScope(const std::string& function_name, const std::string& tag);
        ~TaggedProfileScope();

    private:
        std::string function_name_with_tag_;
    };

    // Sampling-based profiling methods
    void start_sampling_profiling(std::chrono::milliseconds interval = std::chrono::milliseconds(1));
    void stop_sampling_profiling();
    bool is_sampling() const;

    // Register callback for sampling results
    void register_sample_callback(std::function<void(const std::map<std::string, FunctionProfileData>&)> callback);

    // Export profiling data in various formats
    std::string export_to_json() const;
    std::string export_to_csv() const;

private:
    mutable std::mutex profiles_mutex_;
    std::unordered_map<std::thread::id, ThreadProfileData> thread_profiles_;
    std::atomic<bool> enabled_;

    // Sampling members
    std::atomic<bool> sampling_active_{false};
    std::thread sampling_thread_;
    std::chrono::milliseconds sampling_interval_{1};
    std::chrono::steady_clock::time_point sampling_start_time_;
    std::function<void(const std::map<std::string, FunctionProfileData>&)> sample_callback_;

    void sampling_loop();
    void print_tree_node(std::ostringstream& report, const CallTreeNode* node, int depth) const;
    std::unique_ptr<CallTreeNode> deep_copy_tree(const CallTreeNode* node) const;

    // Helper methods for enhanced profiling
    void calculate_exclusive_times(const CallTreeNode* node, std::map<std::string, double>& exclusive_times) const;
    std::vector<double> get_percentile_values(const std::vector<uint64_t>& values, double percentile) const;
    void flatten_tree_for_flame_graph(const CallTreeNode* node,
                                    std::vector<std::pair<std::string, double>>& result,
                                    const std::string& parent_path) const;
    std::string escape_json_string(const std::string& str) const;

    // Helper methods for hot path detection
    void collect_hot_paths_from_tree(const CallTreeNode* node,
                                   std::vector<std::string> current_path,
                                   std::vector<std::vector<std::string>>& hot_paths,
                                   int depth) const;
    double get_path_total_time(const std::vector<std::string>& path) const;
};

// Global CPU profiler instance
extern CPUProfiler g_cpu_profiler;

} // namespace BTQuant