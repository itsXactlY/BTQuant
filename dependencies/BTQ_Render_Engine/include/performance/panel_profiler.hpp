#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>

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
    
    // Reset all profiling stats
    void reset_stats();
    
    // Enable/disable profiling
    void set_enabled(bool enabled);
    bool is_enabled() const;
    
    // Generate a human-readable report of panel render times
    std::string generate_report() const;

private:
    std::unordered_map<uint32_t, PanelRenderStats> profiling_data_;
    mutable std::mutex profiling_data_mutex_;
    std::atomic<bool> enabled_;
    
    static constexpr size_t MAX_HISTORY_SIZE = 100;
};

// Global panel profiler instance
extern PanelProfiler g_panel_profiler;

} // namespace BTQuant