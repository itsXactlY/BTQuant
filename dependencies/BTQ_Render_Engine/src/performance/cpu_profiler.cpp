#include "../include/performance/cpu_profiler.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <memory>

namespace BTQuant {

CPUProfiler::CPUProfiler() : enabled_(true) {}

CPUProfiler::~CPUProfiler() {
    stop_sampling_profiling();
    // Ensure all profiling data is properly cleaned up
    reset();
}

void CPUProfiler::start_function(const std::string& function_name) {
    if (!enabled_) return;

    auto thread_id = std::this_thread::get_id();
    auto start_time = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(profiles_mutex_);

    auto& thread_data = thread_profiles_[thread_id];
    thread_data.call_stack.push(start_time);
    thread_data.function_stack.push(function_name);

    // Update hierarchical call tree
    if (!thread_data.current_node) {
        // This is the root of the call tree
        thread_data.call_tree_root = std::make_unique<CallTreeNode>(function_name);
        thread_data.current_node = thread_data.call_tree_root.get();
    } else {
        // Find or create child node
        CallTreeNode* child_node = nullptr;
        for (auto& child : thread_data.current_node->children) {
            if (child->function_name == function_name) {
                child_node = child.get();
                break;
            }
        }

        if (!child_node) {
            // Create new child node
            auto new_child = std::make_unique<CallTreeNode>(function_name, thread_data.current_node);
            child_node = new_child.get();
            thread_data.current_node->children.push_back(std::move(new_child));
        }

        thread_data.current_node = child_node;
    }

    // Record start time for this node
    thread_data.current_node->profile_data.start_time = start_time;
}

void CPUProfiler::end_function(const std::string& function_name) {
    if (!enabled_) return;

    auto thread_id = std::this_thread::get_id();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(profiles_mutex_);

    auto thread_it = thread_profiles_.find(thread_id);
    if (thread_it == thread_profiles_.end()) {
        return; // No profiling data for this thread
    }

    auto& thread_data = thread_it->second;

    if (thread_data.function_stack.empty() || thread_data.call_stack.empty()) {
        return; // Stack is empty, nothing to end
    }

    // Verify that we're ending the correct function (LIFO)
    if (thread_data.function_stack.top() != function_name) {
        // Mismatch - this could happen if profiling wasn't started properly
        // For now, just skip this end call
        return;
    }

    auto start_time = thread_data.call_stack.top();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    // Record the profile data
    auto& func_profile = thread_data.function_profiles[function_name];
    func_profile.add_duration(static_cast<uint64_t>(duration));

    // Update the hierarchical node data
    if (thread_data.current_node) {
        thread_data.current_node->profile_data.add_duration(static_cast<uint64_t>(duration));
        // Move back up to parent node
        thread_data.current_node = thread_data.current_node->parent;
    }

    // Pop from stacks
    thread_data.call_stack.pop();
    thread_data.function_stack.pop();
}

FunctionProfileData CPUProfiler::get_function_profile(const std::string& function_name) const {
    std::lock_guard<std::mutex> lock(profiles_mutex_);

    auto thread_id = std::this_thread::get_id();
    auto thread_it = thread_profiles_.find(thread_id);

    if (thread_it != thread_profiles_.end()) {
        auto func_it = thread_it->second.function_profiles.find(function_name);
        if (func_it != thread_it->second.function_profiles.end()) {
            return func_it->second;
        }
    }

    return FunctionProfileData{}; // Return empty profile data
}

std::unordered_map<std::string, FunctionProfileData> CPUProfiler::get_thread_profiles() const {
    std::lock_guard<std::mutex> lock(profiles_mutex_);

    auto thread_id = std::this_thread::get_id();
    auto thread_it = thread_profiles_.find(thread_id);

    if (thread_it != thread_profiles_.end()) {
        return thread_it->second.function_profiles;
    }

    return std::unordered_map<std::string, FunctionProfileData>{};
}

std::string CPUProfiler::generate_hierarchical_report() const {
    std::ostringstream report;
    report << "Hierarchical CPU Profiling Report\n";
    report << "=================================\n";

    std::lock_guard<std::mutex> lock(profiles_mutex_);

    for (const auto& [thread_id, thread_data] : thread_profiles_) {
        report << "\nThread ID: " << thread_id << "\n";
        report << "------------------\n";

        if (thread_data.call_tree_root) {
            print_tree_node(report, thread_data.call_tree_root.get(), 0);
        } else {
            report << "No hierarchical profiling data for this thread.\n";
        }
    }

    return report.str();
}

void CPUProfiler::print_tree_node(std::ostringstream& report, const CallTreeNode* node, int depth) const {
    if (!node) return;

    std::string indent(depth * 2, ' ');

    double total_ms = node->profile_data.get_total_duration_ms();
    double avg_ms = node->profile_data.get_average_duration_ms();
    double min_ms = node->profile_data.get_min_duration_ms();
    double max_ms = node->profile_data.get_max_duration_ms();

    report << indent << node->function_name
           << " [Calls: " << node->profile_data.call_count
           << ", Total: " << std::fixed << std::setprecision(3) << total_ms << "ms"
           << ", Avg: " << avg_ms << "ms"
           << ", Min: " << min_ms << "ms"
           << ", Max: " << max_ms << "ms]\n";

    // Recursively print children
    for (const auto& child : node->children) {
        print_tree_node(report, child.get(), depth + 1);
    }
}

std::unique_ptr<CallTreeNode> CPUProfiler::get_call_tree() const {
    std::lock_guard<std::mutex> lock(profiles_mutex_);

    auto thread_id = std::this_thread::get_id();
    auto thread_it = thread_profiles_.find(thread_id);

    if (thread_it != thread_profiles_.end() && thread_it->second.call_tree_root) {
        // Deep copy the tree
        return deep_copy_tree(thread_it->second.call_tree_root.get());
    }

    return nullptr;
}

std::unique_ptr<CallTreeNode> CPUProfiler::deep_copy_tree(const CallTreeNode* node) const {
    if (!node) return nullptr;

    auto new_node = std::make_unique<CallTreeNode>(node->function_name, nullptr);
    new_node->profile_data = node->profile_data;

    for (const auto& child : node->children) {
        auto copied_child = deep_copy_tree(child.get());
        if (copied_child) {
            copied_child->parent = new_node.get();
            new_node->children.push_back(std::move(copied_child));
        }
    }

    return new_node;
}

std::map<std::string, FunctionProfileData> CPUProfiler::get_aggregated_profiles() const {
    std::lock_guard<std::mutex> lock(profiles_mutex_);

    std::map<std::string, FunctionProfileData> aggregated_profiles;

    for (const auto& [thread_id, thread_data] : thread_profiles_) {
        for (const auto& [func_name, func_data] : thread_data.function_profiles) {
            auto& agg_data = aggregated_profiles[func_name];

            agg_data.call_count += func_data.call_count;
            agg_data.total_duration_ns += func_data.total_duration_ns;

            if (func_data.min_duration_ns < agg_data.min_duration_ns) {
                agg_data.min_duration_ns = func_data.min_duration_ns;
            }

            if (func_data.max_duration_ns > agg_data.max_duration_ns) {
                agg_data.max_duration_ns = func_data.max_duration_ns;
            }

            // Add durations to history (up to max size)
            for (auto duration : func_data.duration_history) {
                agg_data.duration_history.push_back(duration);
                if (agg_data.duration_history.size() > FunctionProfileData::MAX_HISTORY_SIZE) {
                    agg_data.duration_history.erase(agg_data.duration_history.begin());
                }
            }
        }
    }

    return aggregated_profiles;
}

void CPUProfiler::reset() {
    std::lock_guard<std::mutex> lock(profiles_mutex_);

    // Clear all thread profiles
    for (auto& [thread_id, thread_data] : thread_profiles_) {
        thread_data.call_tree_root.reset();
        thread_data.current_node = nullptr;
    }

    thread_profiles_.clear();
}

void CPUProfiler::set_enabled(bool enabled) {
    enabled_ = enabled;
    if (!enabled) {
        reset();
    }
}

bool CPUProfiler::is_enabled() const {
    return enabled_;
}

bool CPUProfiler::is_sampling() const {
    return sampling_active_.load();
}

void CPUProfiler::register_sample_callback(std::function<void(const std::map<std::string, FunctionProfileData>&)> callback) {
    std::lock_guard<std::mutex> lock(profiles_mutex_);
    sample_callback_ = callback;
}

void CPUProfiler::start_sampling_profiling(std::chrono::milliseconds interval) {
    if (sampling_active_.load()) {
        stop_sampling_profiling();
    }

    sampling_interval_ = interval;
    sampling_active_.store(true);

    sampling_thread_ = std::thread(&CPUProfiler::sampling_loop, this);
}

void CPUProfiler::stop_sampling_profiling() {
    if (sampling_active_.load()) {
        sampling_active_.store(false);

        if (sampling_thread_.joinable()) {
            sampling_thread_.join();
        }
    }
}

void CPUProfiler::sampling_loop() {
    while (sampling_active_.load()) {
        std::this_thread::sleep_for(sampling_interval_);

        // Collect current profiling snapshot
        auto current_profiles = get_aggregated_profiles();

        // Call the registered callback if available
        std::lock_guard<std::mutex> lock(profiles_mutex_);
        if (sample_callback_) {
            sample_callback_(current_profiles);
        }
    }
}

std::string CPUProfiler::generate_report() const {
    std::ostringstream report;
    report << "CPU Profiling Report\n";
    report << "====================\n";

    auto all_profiles = get_aggregated_profiles();

    if (all_profiles.empty()) {
        report << "No CPU profiling data collected.\n";
        return report.str();
    }

    // Sort by total time spent (descending)
    std::vector<std::pair<std::string, FunctionProfileData>> sorted_profiles(
        all_profiles.begin(), all_profiles.end());

    std::sort(sorted_profiles.begin(), sorted_profiles.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_duration_ns > b.second.total_duration_ns;
              });

    // Calculate total time for percentage calculation
    uint64_t total_time_ns = 0;
    for (const auto& [func_name, data] : sorted_profiles) {
        total_time_ns += data.total_duration_ns;
    }

    report << std::fixed << std::setprecision(3);
    report << "Function Name                           Calls     Total(ms)  Avg(ms)    Min(ms)    Max(ms)    % of Total\n";
    report << "------------------------------------------------------------------------------------------------------\n";

    for (const auto& [func_name, data] : sorted_profiles) {
        double total_ms = data.get_total_duration_ms();
        double avg_ms = data.get_average_duration_ms();
        double min_ms = data.get_min_duration_ms();
        double max_ms = data.get_max_duration_ms();
        double percent_of_total = total_time_ns > 0 ?
            (static_cast<double>(data.total_duration_ns) / static_cast<double>(total_time_ns)) * 100.0 : 0.0;

        report << std::left << std::setw(38) << func_name.substr(0, 37);
        report << std::right << std::setw(10) << data.call_count;
        report << std::right << std::setw(9) << total_ms;
        report << std::right << std::setw(9) << avg_ms;
        report << std::right << std::setw(9) << min_ms;
        report << std::right << std::setw(9) << max_ms;
        report << std::right << std::setw(9) << percent_of_total << "%\n";
    }

    report << "\nTotal profiling time: " << static_cast<double>(total_time_ns) / 1000000.0 << " ms\n";
    report << "Number of unique functions profiled: " << all_profiles.size() << "\n";

    return report.str();
}

std::vector<std::pair<std::string, FunctionProfileData>>
CPUProfiler::get_top_functions_by_total_time(int n) const {
    auto all_profiles = get_aggregated_profiles();

    std::vector<std::pair<std::string, FunctionProfileData>> sorted_profiles(
        all_profiles.begin(), all_profiles.end());

    std::sort(sorted_profiles.begin(), sorted_profiles.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_duration_ns > b.second.total_duration_ns;
              });

    if (sorted_profiles.size() > static_cast<size_t>(n)) {
        sorted_profiles.resize(n);
    }

    return sorted_profiles;
}

std::vector<std::pair<std::string, FunctionProfileData>>
CPUProfiler::get_top_functions_by_average_time(int n) const {
    auto all_profiles = get_aggregated_profiles();

    std::vector<std::pair<std::string, FunctionProfileData>> sorted_profiles(
        all_profiles.begin(), all_profiles.end());

    std::sort(sorted_profiles.begin(), sorted_profiles.end(),
              [](const auto& a, const auto& b) {
                  return a.second.get_average_duration_ms() > b.second.get_average_duration_ms();
              });

    if (sorted_profiles.size() > static_cast<size_t>(n)) {
        sorted_profiles.resize(n);
    }

    return sorted_profiles;
}

// RAII wrapper implementation
CPUProfiler::ProfileScope::ProfileScope(const std::string& function_name)
    : function_name_(function_name) {
    BTQuant::g_cpu_profiler.start_function(function_name_);
}

CPUProfiler::ProfileScope::~ProfileScope() {
    BTQuant::g_cpu_profiler.end_function(function_name_);
}

// Global instance
CPUProfiler g_cpu_profiler;

} // namespace BTQuant