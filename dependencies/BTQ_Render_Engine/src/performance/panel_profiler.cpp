#include "../../include/performance/panel_profiler.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace BTQuant {

PanelProfiler::PanelProfiler() : enabled_(true) {}

void PanelProfiler::start_panel_render(uint32_t panel_id, const std::string& panel_title) {
    if (!enabled_) return;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Store the start time for this panel
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    profiling_data_[panel_id].start_time = start_time;
    profiling_data_[panel_id].panel_title = panel_title;
}

void PanelProfiler::end_panel_render(uint32_t panel_id) {
    if (!enabled_) return;

    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    auto it = profiling_data_.find(panel_id);
    if (it != profiling_data_.end()) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - it->second.start_time);
        
        // Update the panel's render time
        it->second.last_render_time_us = duration.count();
        it->second.total_render_time_us += duration.count();
        it->second.render_count++;
        
        // Update min/max render times
        if (it->second.min_render_time_us == 0 || 
            it->second.min_render_time_us > duration.count()) {
            it->second.min_render_time_us = duration.count();
        }
        if (it->second.max_render_time_us < duration.count()) {
            it->second.max_render_time_us = duration.count();
        }
        
        // Add to history for averaging
        it->second.render_time_history.push_back(duration.count());
        if (it->second.render_time_history.size() > MAX_HISTORY_SIZE) {
            it->second.render_time_history.erase(
                it->second.render_time_history.begin());
        }
    }
}

PanelRenderStats PanelProfiler::get_panel_stats(uint32_t panel_id) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    auto it = profiling_data_.find(panel_id);
    if (it != profiling_data_.end()) {
        return it->second;
    }
    return PanelRenderStats{};
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_all_panel_stats() const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, PanelRenderStats>> result;
    
    for (const auto& [panel_id, stats] : profiling_data_) {
        result.push_back({panel_id, stats});
    }
    
    return result;
}

double PanelProfiler::get_average_render_time_ms(uint32_t panel_id) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    auto it = profiling_data_.find(panel_id);
    if (it != profiling_data_.end() && it->second.render_count > 0) {
        double avg_microseconds = 
            static_cast<double>(it->second.total_render_time_us) / 
            static_cast<double>(it->second.render_count);
        return avg_microseconds / 1000.0; // Convert to milliseconds
    }
    return 0.0;
}

double PanelProfiler::get_last_render_time_ms(uint32_t panel_id) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    auto it = profiling_data_.find(panel_id);
    if (it != profiling_data_.end()) {
        return static_cast<double>(it->second.last_render_time_us) / 1000.0; // Convert to milliseconds
    }
    return 0.0;
}

void PanelProfiler::reset_stats() {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    profiling_data_.clear();
}

void PanelProfiler::set_enabled(bool enabled) {
    enabled_ = enabled;
    if (!enabled) {
        reset_stats();
    }
}

bool PanelProfiler::is_enabled() const {
    return enabled_;
}

std::string PanelProfiler::generate_report() const {
    std::ostringstream report;
    report << "Panel Render Time Report\n";
    report << "========================\n";
    
    auto all_stats = get_all_panel_stats();
    
    if (all_stats.empty()) {
        report << "No panel render data collected.\n";
        return report.str();
    }
    
    // Sort by average render time (descending)
    std::sort(all_stats.begin(), all_stats.end(), 
              [](const auto& a, const auto& b) {
                  return a.second.total_render_time_us / a.second.render_count > 
                         b.second.total_render_time_us / b.second.render_count;
              });
    
    for (const auto& [panel_id, stats] : all_stats) {
        double avg_ms = get_average_render_time_ms(panel_id);
        double min_ms = static_cast<double>(stats.min_render_time_us) / 1000.0;
        double max_ms = static_cast<double>(stats.max_render_time_us) / 1000.0;
        double last_ms = get_last_render_time_ms(panel_id);
        
        report << "Panel ID: " << panel_id 
               << ", Title: " << stats.panel_title << "\n";
        report << "  Average: " << std::fixed << std::setprecision(3) << avg_ms << " ms\n";
        report << "  Min: " << std::fixed << std::setprecision(3) << min_ms << " ms\n";
        report << "  Max: " << std::fixed << std::setprecision(3) << max_ms << " ms\n";
        report << "  Last: " << std::fixed << std::setprecision(3) << last_ms << " ms\n";
        report << "  Count: " << stats.render_count << " renders\n";
        report << "\n";
    }
    
    return report.str();
}

// Global instance
PanelProfiler g_panel_profiler;

} // namespace BTQuant