#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <functional>

namespace BTQuant {

struct PanelRenderStats {
    std::chrono::high_resolution_clock::time_point start_time;
    uint64_t last_render_time_us = 0;
    uint64_t total_render_time_us = 0;
    uint64_t min_render_time_us = 0;
    uint64_t max_render_time_us = 0;
    uint64_t render_count = 0;
    std::string panel_title;
    std::vector<uint64_t> render_time_history;

    // Additional fields for bottleneck detection
    uint64_t slow_render_threshold_us = 10000; // 10ms threshold for slow renders
    uint64_t slow_render_count = 0; // Count of renders exceeding threshold
};

class PanelProfiler {
public:
    PanelProfiler();

    // Start timing a panel render
    void start_panel_render(uint32_t panel_id, const std::string& panel_title);

    // End timing a panel render
    void end_panel_render(uint32_t panel_id);

    // Get stats for a specific panel
    PanelRenderStats get_panel_stats(uint32_t panel_id) const;

    // Get stats for all panels
    std::vector<std::pair<uint32_t, PanelRenderStats>> get_all_panel_stats() const;

    // Get average render time for a specific panel (in milliseconds)
    double get_average_render_time_ms(uint32_t panel_id) const;

    // Get last render time for a specific panel (in milliseconds)
    double get_last_render_time_ms(uint32_t panel_id) const;

    // Get slow render count for a specific panel
    uint64_t get_slow_render_count(uint32_t panel_id) const;

    // Get percentage of slow renders for a specific panel
    double get_slow_render_percentage(uint32_t panel_id) const;

    // Set slow render threshold in microseconds (default is 10000us = 10ms)
    void set_slow_render_threshold(uint64_t threshold_us);

    // Get slowest panels (top N panels by average render time)
    std::vector<std::pair<uint32_t, PanelRenderStats>> get_slowest_panels(size_t top_n = 5) const;

    // Get panels with most slow renders (exceeding threshold)
    std::vector<std::pair<uint32_t, PanelRenderStats>> get_bottleneck_panels(size_t top_n = 5) const;

    // Reset all profiling stats
    void reset_stats();

    // Enable/disable profiling
    void set_enabled(bool enabled);
    bool is_enabled() const;

    // Generate a human-readable report of panel render times
    std::string generate_report() const;

    // Generate a bottleneck-focused report
    std::string generate_bottleneck_report() const;

    // Register a callback for when a slow render is detected
    void register_slow_render_callback(std::function<void(uint32_t, const std::string&, uint64_t)> callback);

    // Get panels ranked by render time variance (higher variance = more inconsistent performance)
    std::vector<std::pair<uint32_t, double>> get_panel_variance_ranking() const;

    // Get panels ranked by outlier ratio (panels with inconsistent render times)
    std::vector<std::pair<uint32_t, double>> get_panel_outlier_ratio_ranking() const;

    // Get panels ranked by resource intensity (combination of average time and slow render frequency)
    std::vector<std::pair<uint32_t, double>> get_panel_resource_intensity_ranking() const;

    // Generate a detailed bottleneck analysis report with multiple perspectives
    std::string generate_detailed_bottleneck_report() const;

private:
    std::unordered_map<uint32_t, PanelRenderStats> profiling_data_;
    mutable std::mutex profiling_data_mutex_;
    std::atomic<bool> enabled_;
    std::atomic<uint64_t> slow_render_threshold_us_{10000}; // Default 10ms threshold
    std::function<void(uint32_t, const std::string&, uint64_t)> slow_render_callback_{nullptr};

    static constexpr size_t MAX_HISTORY_SIZE = 100;
};

// Global panel profiler instance
extern PanelProfiler g_panel_profiler;

} // namespace BTQuant