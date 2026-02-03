#include "../include/performance/cpu_profiler.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <memory>
#include <limits>
#include <cmath>
#include <fstream>
#include <cstdio>

namespace BTQuant {

CPUProfiler::CPUProfiler() : enabled_(true), sampling_start_time_(std::chrono::steady_clock::now()) {}

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
    sampling_start_time_ = std::chrono::steady_clock::now();

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

std::vector<double> CPUProfiler::get_percentile_values(const std::vector<uint64_t>& values, double percentile) const {
    if (values.empty()) {
        return {};
    }

    std::vector<uint64_t> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());

    std::vector<double> result;
    size_t index = static_cast<size_t>((percentile / 100.0) * sorted_values.size());
    if (index < sorted_values.size()) {
        result.push_back(static_cast<double>(sorted_values[index]));
    }

    return result;
}

std::string CPUProfiler::escape_json_string(const std::string& str) const {
    std::string result;
    result.reserve(str.length()); // Reserve space to minimize allocations

    for (char c : str) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b";  break;
            case '\f': result += "\\f";  break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    result += "\\u";
                    char buf[8];
                    snprintf(buf, sizeof(buf), "%04x", c);
                    result += buf;
                } else {
                    result += c;
                }
                break;
        }
    }

    return result;
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

std::vector<std::pair<std::string, std::vector<std::pair<std::string, double>>>>
CPUProfiler::get_flame_graph_data() const {
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, double>>>> flame_data;

    std::lock_guard<std::mutex> lock(profiles_mutex_);

    for (const auto& [thread_id, thread_data] : thread_profiles_) {
        std::vector<std::pair<std::string, double>> thread_flame_data;

        // Flatten the call tree to generate flame graph data
        if (thread_data.call_tree_root) {
            flatten_tree_for_flame_graph(thread_data.call_tree_root.get(), thread_flame_data, "");
        }

        flame_data.push_back({std::to_string(reinterpret_cast<uintptr_t>(&thread_id)), thread_flame_data});
    }

    return flame_data;
}

void CPUProfiler::flatten_tree_for_flame_graph(const CallTreeNode* node,
                                               std::vector<std::pair<std::string, double>>& result,
                                               const std::string& parent_path) const {
    if (!node) return;

    std::string current_path = parent_path.empty() ? node->function_name : parent_path + ";" + node->function_name;
    double time_ms = node->profile_data.get_total_duration_ms();

    result.push_back({current_path, time_ms});

    for (const auto& child : node->children) {
        flatten_tree_for_flame_graph(child.get(), result, current_path);
    }
}

std::map<std::string, FunctionProfileData>
CPUProfiler::get_percentile_statistics(double percentile) const {
    std::map<std::string, FunctionProfileData> percentile_stats;

    auto all_profiles = get_aggregated_profiles();

    for (const auto& [func_name, profile_data] : all_profiles) {
        FunctionProfileData stat = profile_data;

        if (!profile_data.duration_history.empty()) {
            std::vector<uint64_t> sorted_durations = profile_data.duration_history;
            std::sort(sorted_durations.begin(), sorted_durations.end());

            size_t index = static_cast<size_t>((percentile / 100.0) * sorted_durations.size());
            if (index >= sorted_durations.size()) {
                index = sorted_durations.size() - 1;
            }

            stat.min_duration_ns = sorted_durations[0];  // Actual min
            stat.max_duration_ns = sorted_durations[index];  // Percentile value
        }

        percentile_stats[func_name] = stat;
    }

    return percentile_stats;
}

std::vector<CPUProfiler::TimingBreakdown>
CPUProfiler::get_timing_breakdown() const {
    std::vector<TimingBreakdown> breakdowns;
    std::map<std::string, double> exclusive_times;

    // First, calculate exclusive times while holding the lock
    {
        std::lock_guard<std::mutex> lock(profiles_mutex_);

        // Calculate exclusive times for each thread
        for (const auto& [thread_id, thread_data] : thread_profiles_) {
            if (thread_data.call_tree_root) {
                calculate_exclusive_times(thread_data.call_tree_root.get(), exclusive_times);
            }
        }
    } // Release the lock here

    // Now aggregate inclusive times from all profiles (this will acquire the lock internally)
    auto all_profiles = get_aggregated_profiles();

    for (const auto& [func_name, profile_data] : all_profiles) {
        TimingBreakdown tb;
        tb.function_name = func_name;
        tb.inclusive_time_ms = profile_data.get_total_duration_ms();
        tb.exclusive_time_ms = exclusive_times.count(func_name) ?
                              exclusive_times[func_name] : 0.0;
        tb.call_count = profile_data.call_count;

        breakdowns.push_back(tb);
    }

    // Sort by inclusive time (most time-consuming first)
    std::sort(breakdowns.begin(), breakdowns.end(),
              [](const TimingBreakdown& a, const TimingBreakdown& b) {
                  return a.inclusive_time_ms > b.inclusive_time_ms;
              });

    return breakdowns;
}

void CPUProfiler::calculate_exclusive_times(const CallTreeNode* node,
                                           std::map<std::string, double>& exclusive_times) const {
    if (!node) return;

    // Calculate exclusive time (total time minus children's time)
    double total_time = node->profile_data.get_total_duration_ms();
    double children_time = 0.0;

    for (const auto& child : node->children) {
        children_time += child->profile_data.get_total_duration_ms();
        calculate_exclusive_times(child.get(), exclusive_times);  // Recursive call for deeper levels
    }

    double exclusive_time = total_time - children_time;

    // Accumulate exclusive time for this function name
    exclusive_times[node->function_name] += exclusive_time;
}

std::vector<CPUProfiler::DetailedTimingBreakdown>
CPUProfiler::get_detailed_timing_breakdown() const {
    std::vector<DetailedTimingBreakdown> detailed_breakdowns;
    std::map<std::string, double> exclusive_times;

    // Calculate exclusive times
    {
        std::lock_guard<std::mutex> lock(profiles_mutex_);

        for (const auto& [thread_id, thread_data] : thread_profiles_) {
            if (thread_data.call_tree_root) {
                calculate_exclusive_times(thread_data.call_tree_root.get(), exclusive_times);
            }
        }
    }

    auto all_profiles = get_aggregated_profiles();
    uint64_t total_time_ns = 0;

    // Calculate total time for percentage calculation
    for (const auto& [func_name, profile_data] : all_profiles) {
        total_time_ns += profile_data.total_duration_ns;
    }

    for (const auto& [func_name, profile_data] : all_profiles) {
        DetailedTimingBreakdown dtb;
        dtb.function_name = func_name;
        dtb.inclusive_time_ms = profile_data.get_total_duration_ms();
        dtb.exclusive_time_ms = exclusive_times.count(func_name) ?
                               exclusive_times[func_name] : 0.0;
        dtb.call_count = profile_data.call_count;
        dtb.min_time_ms = profile_data.get_min_duration_ms();
        dtb.max_time_ms = profile_data.get_max_duration_ms();
        dtb.avg_time_ms = profile_data.get_average_duration_ms();

        // Calculate variance and standard deviation
        if (profile_data.call_count > 1 && !profile_data.duration_history.empty()) {
            double sum_squares = 0.0;
            double mean = dtb.avg_time_ms;

            for (uint64_t duration_ns : profile_data.duration_history) {
                double duration_ms = static_cast<double>(duration_ns) / 1000000.0;
                double diff = duration_ms - mean;
                sum_squares += diff * diff;
            }

            dtb.variance_time_ms = sum_squares / profile_data.duration_history.size();
            dtb.std_deviation_ms = std::sqrt(dtb.variance_time_ms);
        } else {
            dtb.variance_time_ms = 0.0;
            dtb.std_deviation_ms = 0.0;
        }

        dtb.percentage_of_total = total_time_ns > 0 ?
                                 (static_cast<double>(profile_data.total_duration_ns) /
                                  static_cast<double>(total_time_ns)) * 100.0 : 0.0;

        // Calculate percentiles (50th, 90th, 95th, 99th)
        if (!profile_data.duration_history.empty()) {
            std::vector<uint64_t> sorted_durations = profile_data.duration_history;
            std::sort(sorted_durations.begin(), sorted_durations.end());

            auto get_percentile = [&](double percentile) -> double {
                if (sorted_durations.empty()) return 0.0;
                size_t index = static_cast<size_t>((percentile / 100.0) * sorted_durations.size());
                if (index >= sorted_durations.size()) index = sorted_durations.size() - 1;
                return static_cast<double>(sorted_durations[index]) / 1000000.0;
            };

            dtb.percentiles.push_back(get_percentile(50));  // Median
            dtb.percentiles.push_back(get_percentile(90));  // 90th percentile
            dtb.percentiles.push_back(get_percentile(95));  // 95th percentile
            dtb.percentiles.push_back(get_percentile(99));  // 99th percentile
        } else {
            dtb.percentiles = {0.0, 0.0, 0.0, 0.0};
        }

        detailed_breakdowns.push_back(dtb);
    }

    // Sort by inclusive time (most time-consuming first)
    std::sort(detailed_breakdowns.begin(), detailed_breakdowns.end(),
              [](const DetailedTimingBreakdown& a, const DetailedTimingBreakdown& b) {
                  return a.inclusive_time_ms > b.inclusive_time_ms;
              });

    return detailed_breakdowns;
}

std::vector<CPUProfiler::FunctionLevelBreakdown>
CPUProfiler::get_function_level_breakdown() const {
    std::vector<FunctionLevelBreakdown> function_breakdowns;
    std::map<std::string, double> exclusive_times;

    // Calculate exclusive times
    {
        std::lock_guard<std::mutex> lock(profiles_mutex_);

        for (const auto& [thread_id, thread_data] : thread_profiles_) {
            if (thread_data.call_tree_root) {
                calculate_exclusive_times(thread_data.call_tree_root.get(), exclusive_times);
            }
        }
    }

    auto all_profiles = get_aggregated_profiles();
    uint64_t total_time_ns = 0;

    // Calculate total time for percentage calculation
    for (const auto& [func_name, profile_data] : all_profiles) {
        total_time_ns += profile_data.total_duration_ns;
    }

    for (const auto& [func_name, profile_data] : all_profiles) {
        FunctionLevelBreakdown flb;
        flb.function_name = func_name;
        flb.call_count = profile_data.call_count;
        flb.total_time_ms = profile_data.get_total_duration_ms();
        flb.exclusive_time_ms = exclusive_times.count(func_name) ?
                               exclusive_times[func_name] : 0.0;
        flb.inclusive_time_ms = profile_data.get_total_duration_ms();
        flb.min_time_ms = profile_data.get_min_duration_ms();
        flb.max_time_ms = profile_data.get_max_duration_ms();
        flb.avg_time_ms = profile_data.get_average_duration_ms();

        // Calculate standard deviation
        if (profile_data.call_count > 1 && !profile_data.duration_history.empty()) {
            double sum_squares = 0.0;
            double mean = flb.avg_time_ms;

            for (uint64_t duration_ns : profile_data.duration_history) {
                double duration_ms = static_cast<double>(duration_ns) / 1000000.0;
                double diff = duration_ms - mean;
                sum_squares += diff * diff;
            }

            double variance = sum_squares / profile_data.duration_history.size();
            flb.std_deviation_ms = std::sqrt(variance);
        } else {
            flb.std_deviation_ms = 0.0;
        }

        flb.percentage_of_total = total_time_ns > 0 ?
                                 (static_cast<double>(profile_data.total_duration_ns) /
                                  static_cast<double>(total_time_ns)) * 100.0 : 0.0;

        // Calculate percentiles (25th, 50th, 75th, 95th, 99th)
        if (!profile_data.duration_history.empty()) {
            std::vector<uint64_t> sorted_durations = profile_data.duration_history;
            std::sort(sorted_durations.begin(), sorted_durations.end());

            auto get_percentile = [&](double percentile) -> double {
                if (sorted_durations.empty()) return 0.0;
                size_t index = static_cast<size_t>((percentile / 100.0) * sorted_durations.size());
                if (index >= sorted_durations.size()) index = sorted_durations.size() - 1;
                return static_cast<double>(sorted_durations[index]) / 1000000.0;
            };

            flb.percentiles.push_back(get_percentile(25));  // 25th percentile
            flb.percentiles.push_back(get_percentile(50));  // Median
            flb.percentiles.push_back(get_percentile(75));  // 75th percentile
            flb.percentiles.push_back(get_percentile(95));  // 95th percentile
            flb.percentiles.push_back(get_percentile(99));  // 99th percentile
        } else {
            flb.percentiles = {0.0, 0.0, 0.0, 0.0, 0.0};
        }

        // Add thread ID information
        flb.thread_id = "aggregated";

        function_breakdowns.push_back(flb);
    }

    // Sort by total time (most time-consuming first)
    std::sort(function_breakdowns.begin(), function_breakdowns.end(),
              [](const FunctionLevelBreakdown& a, const FunctionLevelBreakdown& b) {
                  return a.total_time_ms > b.total_time_ms;
              });

    return function_breakdowns;
}

std::vector<CPUProfiler::DetailedCPUTimeBreakdown>
CPUProfiler::get_detailed_cpu_time_breakdown() const {
    std::vector<DetailedCPUTimeBreakdown> detailed_breakdowns;
    std::map<std::string, double> exclusive_times;

    // Calculate exclusive times
    {
        std::lock_guard<std::mutex> lock(profiles_mutex_);

        for (const auto& [thread_id, thread_data] : thread_profiles_) {
            if (thread_data.call_tree_root) {
                calculate_exclusive_times(thread_data.call_tree_root.get(), exclusive_times);
            }
        }
    }

    auto all_profiles = get_aggregated_profiles();
    uint64_t total_time_ns = 0;

    // Calculate total time for percentage calculation
    for (const auto& [func_name, profile_data] : all_profiles) {
        total_time_ns += profile_data.total_duration_ns;
    }

    for (const auto& [func_name, profile_data] : all_profiles) {
        DetailedCPUTimeBreakdown dtb;
        dtb.function_name = func_name;
        dtb.call_count = profile_data.call_count;
        dtb.total_time_ms = profile_data.get_total_duration_ms();
        dtb.exclusive_time_ms = exclusive_times.count(func_name) ?
                               exclusive_times[func_name] : 0.0;
        dtb.inclusive_time_ms = profile_data.get_total_duration_ms(); // Same as total for root level
        dtb.min_time_ms = profile_data.get_min_duration_ms();
        dtb.max_time_ms = profile_data.get_max_duration_ms();
        dtb.avg_time_ms = profile_data.get_average_duration_ms();
        dtb.variance_time_ms = 0.0; // Will be calculated later if needed

        // Calculate variance and standard deviation
        if (profile_data.call_count > 1 && !profile_data.duration_history.empty()) {
            double sum_squares = 0.0;
            double mean = dtb.avg_time_ms;

            for (uint64_t duration_ns : profile_data.duration_history) {
                double duration_ms = static_cast<double>(duration_ns) / 1000000.0;
                double diff = duration_ms - mean;
                sum_squares += diff * diff;
            }

            double variance_time_ms = sum_squares / profile_data.duration_history.size();
            dtb.std_deviation_ms = std::sqrt(variance_time_ms);
        } else {
            dtb.std_deviation_ms = 0.0;
        }

        dtb.percentage_of_total = total_time_ns > 0 ?
                                 (static_cast<double>(profile_data.total_duration_ns) /
                                  static_cast<double>(total_time_ns)) * 100.0 : 0.0;

        // Calculate percentiles (25th, 50th, 75th, 90th, 95th, 99th)
        if (!profile_data.duration_history.empty()) {
            std::vector<uint64_t> sorted_durations = profile_data.duration_history;
            std::sort(sorted_durations.begin(), sorted_durations.end());

            auto get_percentile = [&](double percentile) -> double {
                if (sorted_durations.empty()) return 0.0;
                size_t index = static_cast<size_t>((percentile / 100.0) * sorted_durations.size());
                if (index >= sorted_durations.size()) index = sorted_durations.size() - 1;
                return static_cast<double>(sorted_durations[index]) / 1000000.0;
            };

            dtb.percentiles.push_back(get_percentile(25));  // 25th percentile
            dtb.percentiles.push_back(get_percentile(50));  // 50th percentile (median)
            dtb.percentiles.push_back(get_percentile(75));  // 75th percentile
            dtb.percentiles.push_back(get_percentile(90));  // 90th percentile
            dtb.percentiles.push_back(get_percentile(95));  // 95th percentile
            dtb.percentiles.push_back(get_percentile(99));  // 99th percentile
        } else {
            dtb.percentiles = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        }

        // Set thread ID (for aggregated data, we'll use a generic identifier)
        dtb.thread_id = "aggregated";

        // Estimate CPU utilization (simplified calculation)
        dtb.cpu_utilization = dtb.percentage_of_total; // For now, use percentage as proxy

        // Set sample count
        dtb.total_samples = profile_data.duration_history.size();

        // Set time bounds (these would need to be tracked separately in a full implementation)
        dtb.first_call_time = std::chrono::steady_clock::time_point{}; // Placeholder
        dtb.last_call_time = std::chrono::steady_clock::time_point{};  // Placeholder

        detailed_breakdowns.push_back(dtb);
    }

    // Sort by total time (most time-consuming first)
    std::sort(detailed_breakdowns.begin(), detailed_breakdowns.end(),
              [](const DetailedCPUTimeBreakdown& a, const DetailedCPUTimeBreakdown& b) {
                  return a.total_time_ms > b.total_time_ms;
              });

    return detailed_breakdowns;
}

std::map<std::string, double>
CPUProfiler::get_cpu_time_distribution() const {
    std::map<std::string, double> time_distribution;

    auto all_profiles = get_aggregated_profiles();
    uint64_t total_time_ns = 0;

    // Calculate total time for percentage calculation
    for (const auto& [func_name, profile_data] : all_profiles) {
        total_time_ns += profile_data.total_duration_ns;
    }

    if (total_time_ns == 0) {
        return time_distribution;
    }

    // Calculate percentage for each function
    for (const auto& [func_name, profile_data] : all_profiles) {
        double percentage = (static_cast<double>(profile_data.total_duration_ns) /
                            static_cast<double>(total_time_ns)) * 100.0;
        time_distribution[func_name] = percentage;
    }

    return time_distribution;
}

std::vector<std::vector<std::string>>
CPUProfiler::get_function_similarity_clusters() const {
    std::vector<std::vector<std::string>> clusters;

    // This is a simplified clustering algorithm based on execution time similarity
    auto all_profiles = get_aggregated_profiles();

    if (all_profiles.empty()) {
        return clusters;
    }

    // Group functions by similar average execution time (within 10% tolerance)
    std::vector<std::pair<std::string, double>> func_times;
    for (const auto& [func_name, profile_data] : all_profiles) {
        if (profile_data.call_count > 0) {
            func_times.push_back({func_name, profile_data.get_average_duration_ms()});
        }
    }

    // Sort by average execution time
    std::sort(func_times.begin(), func_times.end(),
              [](const auto& a, const auto& b) {
                  return a.second < b.second;
              });

    // Cluster functions with similar execution times
    if (!func_times.empty()) {
        std::vector<std::string> current_cluster;
        current_cluster.push_back(func_times[0].first);

        for (size_t i = 1; i < func_times.size(); ++i) {
            double prev_time = func_times[i-1].second;
            double curr_time = func_times[i].second;

            // If the difference is within 10% of the previous time, cluster them together
            if (curr_time <= prev_time * 1.1 && curr_time >= prev_time * 0.9) {
                current_cluster.push_back(func_times[i].first);
            } else {
                // Start a new cluster
                if (!current_cluster.empty()) {
                    clusters.push_back(current_cluster);
                }
                current_cluster.clear();
                current_cluster.push_back(func_times[i].first);
            }
        }

        // Add the last cluster
        if (!current_cluster.empty()) {
            clusters.push_back(current_cluster);
        }
    }

    return clusters;
}

std::string CPUProfiler::generate_detailed_breakdown_report() const {
    std::ostringstream report;
    report << "Detailed CPU Profiling Breakdown Report\n";
    report << "=======================================\n\n";

    auto function_breakdowns = get_function_level_breakdown();
    auto time_distribution = get_cpu_time_distribution();

    if (function_breakdowns.empty()) {
        report << "No profiling data collected.\n";
        return report.str();
    }

    report << std::fixed << std::setprecision(3);
    report << "Function-Level CPU Time Analysis\n";
    report << "--------------------------------\n";
    report << "Function Name                        Calls     Total(ms)  Exclusive(ms)  Inclusive(ms)  Avg(ms)    Min(ms)    Max(ms)    StdDev(ms)  %Total\n";
    report << "----------------------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& breakdown : function_breakdowns) {
        report << std::left << std::setw(35) << breakdown.function_name.substr(0, 34);
        report << std::right << std::setw(10) << breakdown.call_count;
        report << std::right << std::setw(11) << breakdown.total_time_ms;
        report << std::right << std::setw(13) << breakdown.exclusive_time_ms;
        report << std::right << std::setw(13) << breakdown.inclusive_time_ms;
        report << std::right << std::setw(9) << breakdown.avg_time_ms;
        report << std::right << std::setw(9) << breakdown.min_time_ms;
        report << std::right << std::setw(9) << breakdown.max_time_ms;
        report << std::right << std::setw(12) << breakdown.std_deviation_ms;
        report << std::right << std::setw(7) << time_distribution.at(breakdown.function_name) << "%\n";
    }

    report << "\nPercentile Analysis (Top 10 Functions by Total Time):\n";
    report << "----------------------------------------------------\n";
    report << "Function Name                        25th       50th       75th       95th       99th\n";
    report << "-------------------------------------------------------------------------------------\n";

    // Show percentiles for top 10 functions
    int count = 0;
    for (const auto& breakdown : function_breakdowns) {
        if (count++ >= 10) break;

        report << std::left << std::setw(35) << breakdown.function_name.substr(0, 34);
        if (breakdown.percentiles.size() >= 5) {
            report << std::right << std::setw(11) << breakdown.percentiles[0];  // 25th
            report << std::right << std::setw(11) << breakdown.percentiles[1];  // 50th (median)
            report << std::right << std::setw(11) << breakdown.percentiles[2];  // 75th
            report << std::right << std::setw(11) << breakdown.percentiles[3];  // 95th
            report << std::right << std::setw(11) << breakdown.percentiles[4];  // 99th
        }
        report << "\n";
    }

    // Show function similarity clusters
    auto clusters = get_function_similarity_clusters();
    report << "\nFunction Similarity Clusters (by execution time):\n";
    report << "-------------------------------------------------\n";
    for (size_t i = 0; i < clusters.size(); ++i) {
        report << "Cluster " << (i + 1) << " (" << clusters[i].size() << " functions): ";
        for (size_t j = 0; j < clusters[i].size(); ++j) {
            if (j > 0) report << ", ";
            report << clusters[i][j];
        }
        report << "\n";
    }

    // Summary statistics
    auto stats = get_profiling_stats();
    report << "\nSummary Statistics:\n";
    report << "-------------------\n";
    report << "Total functions profiled: " << stats.unique_functions << "\n";
    report << "Total calls: " << stats.total_calls << "\n";
    report << "Total CPU time: " << stats.total_time_ms << " ms\n";
    report << "Average time per call: " << stats.avg_time_per_call_ms << " ms\n";
    report << "Minimum time per call: " << stats.min_time_per_call_ms << " ms\n";
    report << "Maximum time per call: " << stats.max_time_per_call_ms << " ms\n";

    return report.str();
}

std::string CPUProfiler::generate_cpu_time_distribution_report() const {
    std::ostringstream report;
    report << "CPU Time Distribution Analysis Report\n";
    report << "=====================================\n\n";

    auto time_distribution = get_cpu_time_distribution();
    auto all_profiles = get_aggregated_profiles();

    if (time_distribution.empty()) {
        report << "No profiling data collected.\n";
        return report.str();
    }

    report << std::fixed << std::setprecision(3);
    report << "CPU Time Distribution by Function\n";
    report << "---------------------------------\n";
    report << "Function Name                        % of Total  Total Time(ms)  Calls\n";
    report << "--------------------------------------------------------------------\n";

    // Sort by percentage of total time
    std::vector<std::pair<std::string, double>> sorted_distribution(
        time_distribution.begin(), time_distribution.end());

    std::sort(sorted_distribution.begin(), sorted_distribution.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    for (const auto& [func_name, percentage] : sorted_distribution) {
        auto profile_it = all_profiles.find(func_name);
        if (profile_it != all_profiles.end()) {
            report << std::left << std::setw(35) << func_name.substr(0, 34);
            report << std::right << std::setw(10) << percentage << "%";
            report << std::right << std::setw(14) << profile_it->second.get_total_duration_ms();
            report << std::right << std::setw(8) << profile_it->second.call_count << "\n";
        }
    }

    // Create a simple text-based pie chart representation
    report << "\nCPU Time Distribution Visualization (Top 10 Functions):\n";
    report << "------------------------------------------------------\n";

    int displayed = 0;
    for (const auto& [func_name, percentage] : sorted_distribution) {
        if (displayed++ >= 10) break;

        report << std::left << std::setw(35) << func_name.substr(0, 34) << "|";

        // Draw a bar proportional to the percentage (scale to 40 characters)
        int bar_length = static_cast<int>(percentage * 0.4); // 100% = 40 chars
        for (int i = 0; i < bar_length && i < 40; ++i) {
            report << "=";
        }
        report << " " << percentage << "%\n";
    }

    // Calculate and show the cumulative percentage for top functions
    report << "\nCumulative CPU Time Analysis:\n";
    report << "----------------------------\n";
    double cumulative_percentage = 0.0;
    int top_functions_count = 0;
    for (const auto& [func_name, percentage] : sorted_distribution) {
        cumulative_percentage += percentage;
        top_functions_count++;
        report << "Top " << top_functions_count << " functions account for "
               << std::fixed << std::setprecision(2) << cumulative_percentage
               << "% of total CPU time\n";

        if (cumulative_percentage >= 80.0) {
            break; // Stop when we reach 80% of the total time
        }
    }

    return report.str();
}

std::string CPUProfiler::generate_overhead_analysis_report() const {
    std::ostringstream report;
    report << "Function Call Overhead Analysis Report\n";
    report << "======================================\n\n";

    auto all_profiles = get_aggregated_profiles();

    if (all_profiles.empty()) {
        report << "No profiling data collected.\n";
        return report.str();
    }

    report << std::fixed << std::setprecision(3);
    report << "Function Call Overhead Analysis\n";
    report << "-------------------------------\n";
    report << "Function Name                        Calls     Total(ms)  Avg(ms)    Min(ms)    Max(ms)    Variance\n";
    report << "------------------------------------------------------------------------------------------------\n";

    // Calculate overhead metrics for each function
    for (const auto& [func_name, profile_data] : all_profiles) {
        // Calculate variance
        double variance = 0.0;
        if (profile_data.call_count > 1 && !profile_data.duration_history.empty()) {
            double mean = profile_data.get_average_duration_ms();
            double sum_squares = 0.0;

            for (uint64_t duration_ns : profile_data.duration_history) {
                double duration_ms = static_cast<double>(duration_ns) / 1000000.0;
                double diff = duration_ms - mean;
                sum_squares += diff * diff;
            }

            variance = sum_squares / profile_data.duration_history.size();
        }

        report << std::left << std::setw(35) << func_name.substr(0, 34);
        report << std::right << std::setw(10) << profile_data.call_count;
        report << std::right << std::setw(11) << profile_data.get_total_duration_ms();
        report << std::right << std::setw(9) << profile_data.get_average_duration_ms();
        report << std::right << std::setw(9) << profile_data.get_min_duration_ms();
        report << std::right << std::setw(9) << profile_data.get_max_duration_ms();
        report << std::right << std::setw(9) << variance << "\n";
    }

    // Identify functions with high call frequency
    report << "\nHigh-Frequency Functions (>100 calls):\n";
    report << "-------------------------------------\n";
    for (const auto& [func_name, profile_data] : all_profiles) {
        if (profile_data.call_count > 100) {
            report << "- " << func_name << ": " << profile_data.call_count << " calls, "
                   << profile_data.get_average_duration_ms() << " ms avg\n";
        }
    }

    // Identify functions with high variance (potential bottlenecks)
    report << "\nHigh-Variance Functions (Potential Bottlenecks):\n";
    report << "------------------------------------------------\n";
    std::vector<std::pair<std::string, double>> high_variance_funcs;

    for (const auto& [func_name, profile_data] : all_profiles) {
        if (profile_data.call_count > 1 && !profile_data.duration_history.empty()) {
            double mean = profile_data.get_average_duration_ms();
            double sum_squares = 0.0;

            for (uint64_t duration_ns : profile_data.duration_history) {
                double duration_ms = static_cast<double>(duration_ns) / 1000000.0;
                double diff = duration_ms - mean;
                sum_squares += diff * diff;
            }

            double variance = sum_squares / profile_data.duration_history.size();
            if (variance > 0.1) { // Threshold for high variance
                high_variance_funcs.push_back({func_name, variance});
            }
        }
    }

    std::sort(high_variance_funcs.begin(), high_variance_funcs.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    for (const auto& [func_name, variance] : high_variance_funcs) {
        auto profile_it = all_profiles.find(func_name);
        if (profile_it != all_profiles.end()) {
            report << "- " << func_name << ": variance=" << variance << " ms², "
                   << profile_it->second.call_count << " calls, "
                   << profile_it->second.get_average_duration_ms() << " ms avg\n";
        }
    }

    // Calculate efficiency metrics
    report << "\nEfficiency Metrics:\n";
    report << "-------------------\n";
    uint64_t total_calls = 0;
    double total_time = 0.0;
    for (const auto& [func_name, profile_data] : all_profiles) {
        total_calls += profile_data.call_count;
        total_time += profile_data.get_total_duration_ms();
    }

    if (total_calls > 0) {
        report << "Total function calls: " << total_calls << "\n";
        report << "Total profiling time: " << total_time << " ms\n";
        report << "Average time per call: " << (total_time / total_calls) << " ms\n";
    }

    return report.str();
}

std::string CPUProfiler::generate_detailed_cpu_time_breakdown_report() const {
    std::ostringstream report;
    report << "Detailed CPU Time Breakdown Report\n";
    report << "==================================\n\n";

    auto detailed_breakdowns = get_detailed_cpu_time_breakdown();

    if (detailed_breakdowns.empty()) {
        report << "No profiling data collected.\n";
        return report.str();
    }

    report << std::fixed << std::setprecision(3);
    report << "Function-Level CPU Time Analysis\n";
    report << "--------------------------------\n";
    report << "Function Name                        Calls     Total(ms)  Exclusive(ms)  Inclusive(ms)  Avg(ms)    Min(ms)    Max(ms)    StdDev(ms)  %Total   Samples\n";
    report << "--------------------------------------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& breakdown : detailed_breakdowns) {
        report << std::left << std::setw(35) << breakdown.function_name.substr(0, 34);
        report << std::right << std::setw(10) << breakdown.call_count;
        report << std::right << std::setw(11) << breakdown.total_time_ms;
        report << std::right << std::setw(13) << breakdown.exclusive_time_ms;
        report << std::right << std::setw(13) << breakdown.inclusive_time_ms;
        report << std::right << std::setw(9) << breakdown.avg_time_ms;
        report << std::right << std::setw(9) << breakdown.min_time_ms;
        report << std::right << std::setw(9) << breakdown.max_time_ms;
        report << std::right << std::setw(12) << breakdown.std_deviation_ms;
        report << std::right << std::setw(7) << breakdown.percentage_of_total << "%";
        report << std::right << std::setw(8) << breakdown.total_samples << "\n";
    }

    report << "\nPercentile Analysis (Top 10 Functions by Total Time):\n";
    report << "----------------------------------------------------\n";
    report << "Function Name                        25th       50th       75th       90th       95th       99th\n";
    report << "-------------------------------------------------------------------------------------------------------\n";

    // Show percentiles for top 10 functions
    int count = 0;
    for (const auto& breakdown : detailed_breakdowns) {
        if (count++ >= 10) break;

        report << std::left << std::setw(35) << breakdown.function_name.substr(0, 34);
        if (breakdown.percentiles.size() >= 6) {
            report << std::right << std::setw(11) << breakdown.percentiles[0];  // 25th
            report << std::right << std::setw(11) << breakdown.percentiles[1];  // 50th (median)
            report << std::right << std::setw(11) << breakdown.percentiles[2];  // 75th
            report << std::right << std::setw(11) << breakdown.percentiles[3];  // 90th
            report << std::right << std::setw(11) << breakdown.percentiles[4];  // 95th
            report << std::right << std::setw(11) << breakdown.percentiles[5];  // 99th
        }
        report << "\n";
    }

    // Highlight the most expensive functions
    report << "\nMost CPU-Intensive Functions (>1% of total CPU time):\n";
    report << "----------------------------------------------------\n";
    for (const auto& breakdown : detailed_breakdowns) {
        if (breakdown.percentage_of_total > 1.0) {
            report << "- " << breakdown.function_name << ": " << breakdown.percentage_of_total
                   << "% of total CPU time (" << breakdown.total_time_ms << " ms total, "
                   << breakdown.call_count << " calls)\n";
        }
    }

    // Summary statistics
    auto stats = get_profiling_stats();
    report << "\nSummary Statistics:\n";
    report << "-------------------\n";
    report << "Total functions profiled: " << stats.unique_functions << "\n";
    report << "Total calls: " << stats.total_calls << "\n";
    report << "Total CPU time: " << stats.total_time_ms << " ms\n";
    report << "Average time per call: " << stats.avg_time_per_call_ms << " ms\n";
    report << "Active profiling: " << (stats.is_active ? "Yes" : "No") << "\n";

    return report.str();
}

std::vector<std::vector<std::string>>
CPUProfiler::get_hot_paths(int max_paths) const {
    std::vector<std::vector<std::string>> hot_paths;

    std::lock_guard<std::mutex> lock(profiles_mutex_);

    for (const auto& [thread_id, thread_data] : thread_profiles_) {
        if (thread_data.call_tree_root) {
            std::vector<std::vector<std::string>> thread_hot_paths;
            collect_hot_paths_from_tree(thread_data.call_tree_root.get(), {}, thread_hot_paths, 0);

            // Sort paths by total time and take top ones
            std::sort(thread_hot_paths.begin(), thread_hot_paths.end(),
                      [this](const std::vector<std::string>& a, const std::vector<std::string>& b) {
                          double time_a = get_path_total_time(a);
                          double time_b = get_path_total_time(b);
                          return time_a > time_b;
                      });

            // Add to global list
            for (const auto& path : thread_hot_paths) {
                if (hot_paths.size() >= static_cast<size_t>(max_paths)) break;
                hot_paths.push_back(path);
            }
        }
    }

    // Sort all paths by total time and return top ones
    std::sort(hot_paths.begin(), hot_paths.end(),
              [this](const std::vector<std::string>& a, const std::vector<std::string>& b) {
                  double time_a = get_path_total_time(a);
                  double time_b = get_path_total_time(b);
                  return time_a > time_b;
              });

    if (hot_paths.size() > static_cast<size_t>(max_paths)) {
        hot_paths.resize(max_paths);
    }

    return hot_paths;
}

void CPUProfiler::collect_hot_paths_from_tree(const CallTreeNode* node,
                                              std::vector<std::string> current_path,
                                              std::vector<std::vector<std::string>>& hot_paths,
                                              int depth) const {
    if (!node) return;

    // Prevent infinite recursion by limiting depth
    if (depth > 50) {
        return;
    }

    current_path.push_back(node->function_name);

    // Only add paths that have meaningful duration (more than 0.1ms total)
    if (node->profile_data.get_total_duration_ms() > 0.1) {
        hot_paths.push_back(current_path);
    }

    // Recursively explore children
    for (const auto& child : node->children) {
        collect_hot_paths_from_tree(child.get(), current_path, hot_paths, depth + 1);
    }
}

double CPUProfiler::get_path_total_time(const std::vector<std::string>& path) const {
    double total_time = 0.0;

    for (const auto& func_name : path) {
        auto all_profiles = get_aggregated_profiles();
        auto it = all_profiles.find(func_name);
        if (it != all_profiles.end()) {
            total_time += it->second.get_total_duration_ms();
        }
    }

    return total_time;
}

CPUProfiler::ProfilingStats
CPUProfiler::get_profiling_stats() const {
    ProfilingStats stats{};

    auto all_profiles = get_aggregated_profiles();

    stats.unique_functions = all_profiles.size();
    stats.is_active = enabled_;

    for (const auto& [func_name, profile_data] : all_profiles) {
        stats.total_calls += profile_data.call_count;
        stats.total_time_ms += profile_data.get_total_duration_ms();

        if (profile_data.call_count > 0) {
            double avg_time = profile_data.get_average_duration_ms();
            if (stats.avg_time_per_call_ms == 0.0 || avg_time < stats.min_time_per_call_ms) {
                stats.min_time_per_call_ms = avg_time;
            }
            if (avg_time > stats.max_time_per_call_ms) {
                stats.max_time_per_call_ms = avg_time;
            }
        }
    }

    if (stats.total_calls > 0) {
        stats.avg_time_per_call_ms = stats.total_time_ms / static_cast<double>(stats.total_calls);
    }

    // Set start and end times based on the sampling state
    std::lock_guard<std::mutex> lock(profiles_mutex_);
    if (sampling_active_.load()) {
        stats.start_time = sampling_start_time_;
        stats.end_time = std::chrono::steady_clock::now();
    }

    return stats;
}

std::vector<std::pair<std::string, double>>
CPUProfiler::get_high_variance_functions(int n) const {
    std::vector<std::pair<std::string, double>> high_variance_funcs;

    auto all_profiles = get_aggregated_profiles();

    for (const auto& [func_name, profile_data] : all_profiles) {
        if (profile_data.call_count > 1 && !profile_data.duration_history.empty()) {
            double mean = profile_data.get_average_duration_ms();
            double sum_squares = 0.0;

            for (uint64_t duration_ns : profile_data.duration_history) {
                double duration_ms = static_cast<double>(duration_ns) / 1000000.0;
                double diff = duration_ms - mean;
                sum_squares += diff * diff;
            }

            double variance = sum_squares / profile_data.duration_history.size();
            high_variance_funcs.push_back({func_name, variance});
        }
    }

    // Sort by variance (highest first)
    std::sort(high_variance_funcs.begin(), high_variance_funcs.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    if (high_variance_funcs.size() > static_cast<size_t>(n)) {
        high_variance_funcs.resize(n);
    }

    return high_variance_funcs;
}

std::vector<std::pair<std::string, uint64_t>>
CPUProfiler::get_most_frequent_functions(int n) const {
    std::vector<std::pair<std::string, uint64_t>> frequent_funcs;

    auto all_profiles = get_aggregated_profiles();

    for (const auto& [func_name, profile_data] : all_profiles) {
        frequent_funcs.push_back({func_name, profile_data.call_count});
    }

    // Sort by call count (highest first)
    std::sort(frequent_funcs.begin(), frequent_funcs.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    if (frequent_funcs.size() > static_cast<size_t>(n)) {
        frequent_funcs.resize(n);
    }

    return frequent_funcs;
}

std::map<std::string, FunctionProfileData>
CPUProfiler::get_filtered_profiles(double min_time_ms, double max_time_ms) const {
    std::map<std::string, FunctionProfileData> filtered_profiles;

    auto all_profiles = get_aggregated_profiles();

    for (const auto& [func_name, profile_data] : all_profiles) {
        double total_time = profile_data.get_total_duration_ms();
        if (total_time >= min_time_ms && total_time <= max_time_ms) {
            filtered_profiles[func_name] = profile_data;
        }
    }

    return filtered_profiles;
}

std::string CPUProfiler::export_to_json() const {
    std::ostringstream json_stream;
    json_stream << "{\n  \"profiling_data\": [\n";

    bool first_entry = true;
    auto all_profiles = get_aggregated_profiles();

    for (const auto& [func_name, profile_data] : all_profiles) {
        if (!first_entry) {
            json_stream << ",\n";
        }

        json_stream << "    {\n";
        json_stream << "      \"function_name\": \"" << escape_json_string(func_name) << "\",\n";
        json_stream << "      \"call_count\": " << profile_data.call_count << ",\n";
        json_stream << "      \"total_time_ms\": " << profile_data.get_total_duration_ms() << ",\n";
        json_stream << "      \"average_time_ms\": " << profile_data.get_average_duration_ms() << ",\n";
        json_stream << "      \"min_time_ms\": " << profile_data.get_min_duration_ms() << ",\n";
        json_stream << "      \"max_time_ms\": " << profile_data.get_max_duration_ms() << "\n";
        json_stream << "    }";

        first_entry = false;
    }

    json_stream << "\n  ],\n";

    // Add timing breakdown
    auto timing_breakdown = get_timing_breakdown();
    json_stream << "  \"timing_breakdown\": [\n";

    for (size_t i = 0; i < timing_breakdown.size(); ++i) {
        if (i > 0) json_stream << ",\n";

        json_stream << "    {\n";
        json_stream << "      \"function_name\": \"" << escape_json_string(timing_breakdown[i].function_name) << "\",\n";
        json_stream << "      \"exclusive_time_ms\": " << timing_breakdown[i].exclusive_time_ms << ",\n";
        json_stream << "      \"inclusive_time_ms\": " << timing_breakdown[i].inclusive_time_ms << ",\n";
        json_stream << "      \"call_count\": " << timing_breakdown[i].call_count << "\n";
        json_stream << "    }";
    }

    json_stream << "\n  ]\n}";

    return json_stream.str();
}

std::string CPUProfiler::export_to_csv() const {
    std::ostringstream csv_stream;

    // Header
    csv_stream << "Function Name,Call Count,Total Time (ms),Average Time (ms),Min Time (ms),Max Time (ms),Exclusive Time (ms),Inclusive Time (ms)\n";

    auto timing_breakdown = get_timing_breakdown();

    for (const auto& breakdown : timing_breakdown) {
        // Find the corresponding profile data
        auto all_profiles = get_aggregated_profiles();
        auto profile_it = all_profiles.find(breakdown.function_name);

        if (profile_it != all_profiles.end()) {
            const auto& profile_data = profile_it->second;
            csv_stream << "\"" << profile_it->first << "\",";
            csv_stream << profile_data.call_count << ",";
            csv_stream << profile_data.get_total_duration_ms() << ",";
            csv_stream << profile_data.get_average_duration_ms() << ",";
            csv_stream << profile_data.get_min_duration_ms() << ",";
            csv_stream << profile_data.get_max_duration_ms() << ",";
            csv_stream << breakdown.exclusive_time_ms << ",";
            csv_stream << breakdown.inclusive_time_ms << "\n";
        }
    }

    return csv_stream.str();
}

// TaggedProfileScope implementation
CPUProfiler::TaggedProfileScope::TaggedProfileScope(const std::string& function_name, const std::string& tag) {
    function_name_with_tag_ = function_name + "[" + tag + "]";
    BTQuant::g_cpu_profiler.start_function(function_name_with_tag_);
}

CPUProfiler::TaggedProfileScope::~TaggedProfileScope() {
    BTQuant::g_cpu_profiler.end_function(function_name_with_tag_);
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