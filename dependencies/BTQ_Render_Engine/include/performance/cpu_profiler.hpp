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

    // RAII wrapper for automatic profiling
    class ProfileScope {
    public:
        ProfileScope(const std::string& function_name);
        ~ProfileScope();

    private:
        std::string function_name_;
    };

    // Sampling-based profiling methods
    void start_sampling_profiling(std::chrono::milliseconds interval = std::chrono::milliseconds(1));
    void stop_sampling_profiling();
    bool is_sampling() const;

    // Register callback for sampling results
    void register_sample_callback(std::function<void(const std::map<std::string, FunctionProfileData>&)> callback);

private:
    mutable std::mutex profiles_mutex_;
    std::unordered_map<std::thread::id, ThreadProfileData> thread_profiles_;
    std::atomic<bool> enabled_;

    // Sampling members
    std::atomic<bool> sampling_active_{false};
    std::thread sampling_thread_;
    std::chrono::milliseconds sampling_interval_{1};
    std::function<void(const std::map<std::string, FunctionProfileData>&)> sample_callback_;

    void sampling_loop();
    void print_tree_node(std::ostringstream& report, const CallTreeNode* node, int depth) const;
    std::unique_ptr<CallTreeNode> deep_copy_tree(const CallTreeNode* node) const;
};

// Global CPU profiler instance
extern CPUProfiler g_cpu_profiler;

} // namespace BTQuant