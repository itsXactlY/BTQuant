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

        // Check if this render exceeded the slow render threshold
        uint64_t threshold = slow_render_threshold_us_.load();
        if (duration.count() > threshold) {
            it->second.slow_render_count++;

            // Call the slow render callback if registered
            if (slow_render_callback_) {
                slow_render_callback_(panel_id, it->second.panel_title, duration.count());
            }
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

uint64_t PanelProfiler::get_slow_render_count(uint32_t panel_id) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    auto it = profiling_data_.find(panel_id);
    if (it != profiling_data_.end()) {
        return it->second.slow_render_count;
    }
    return 0;
}

double PanelProfiler::get_slow_render_percentage(uint32_t panel_id) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    auto it = profiling_data_.find(panel_id);
    if (it != profiling_data_.end() && it->second.render_count > 0) {
        return (static_cast<double>(it->second.slow_render_count) /
                static_cast<double>(it->second.render_count)) * 100.0;
    }
    return 0.0;
}

void PanelProfiler::set_slow_render_threshold(uint64_t threshold_us) {
    slow_render_threshold_us_.store(threshold_us);
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_slowest_panels(size_t top_n) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, PanelRenderStats>> result;

    // Copy all panel stats
    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 0) { // Only include panels that have been rendered
            result.push_back({panel_id, stats});
        }
    }

    // Sort by average render time (descending)
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  double avg_a = static_cast<double>(a.second.total_render_time_us) /
                                static_cast<double>(a.second.render_count);
                  double avg_b = static_cast<double>(b.second.total_render_time_us) /
                                static_cast<double>(b.second.render_count);
                  return avg_a > avg_b;
              });

    // Limit to top N
    if (result.size() > top_n) {
        result.resize(top_n);
    }

    return result;
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_bottleneck_panels(size_t top_n) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, PanelRenderStats>> result;

    // Copy all panel stats
    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 0) { // Only include panels that have been rendered
            result.push_back({panel_id, stats});
        }
    }

    // Sort by slow render count (descending)
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  return a.second.slow_render_count > b.second.slow_render_count;
              });

    // Limit to top N
    if (result.size() > top_n) {
        result.resize(top_n);
    }

    return result;
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

std::string PanelProfiler::generate_bottleneck_report() const {
    std::ostringstream report;
    report << "Panel Bottleneck Analysis Report\n";
    report << "================================\n\n";

    auto slowest_panels = get_slowest_panels(10);
    auto bottleneck_panels = get_bottleneck_panels(10);

    if (slowest_panels.empty() && bottleneck_panels.empty()) {
        report << "No panel render data collected.\n";
        return report.str();
    }

    // Report slowest panels by average render time
    if (!slowest_panels.empty()) {
        report << "Top Slowest Panels (by average render time):\n";
        report << "------------------------------------------\n";

        for (size_t i = 0; i < slowest_panels.size(); ++i) {
            const auto& [panel_id, stats] = slowest_panels[i];
            double avg_ms = get_average_render_time_ms(panel_id);
            double min_ms = static_cast<double>(stats.min_render_time_us) / 1000.0;
            double max_ms = static_cast<double>(stats.max_render_time_us) / 1000.0;
            double last_ms = get_last_render_time_ms(panel_id);
            double slow_pct = get_slow_render_percentage(panel_id);

            report << (i + 1) << ". Panel ID: " << panel_id
                   << ", Title: " << stats.panel_title << "\n";
            report << "   Average: " << std::fixed << std::setprecision(3) << avg_ms << " ms\n";
            report << "   Min: " << std::fixed << std::setprecision(3) << min_ms << " ms\n";
            report << "   Max: " << std::fixed << std::setprecision(3) << max_ms << " ms\n";
            report << "   Last: " << std::fixed << std::setprecision(3) << last_ms << " ms\n";
            report << "   Slow Renders: " << stats.slow_render_count << "/" << stats.render_count
                   << " (" << std::fixed << std::setprecision(1) << slow_pct << "%)\n";
            report << "\n";
        }
    }

    // Report bottleneck panels by slow render count
    if (!bottleneck_panels.empty()) {
        report << "Top Bottleneck Panels (by slow render count):\n";
        report << "--------------------------------------------\n";

        for (size_t i = 0; i < bottleneck_panels.size(); ++i) {
            const auto& [panel_id, stats] = bottleneck_panels[i];
            double avg_ms = get_average_render_time_ms(panel_id);
            double slow_pct = get_slow_render_percentage(panel_id);

            report << (i + 1) << ". Panel ID: " << panel_id
                   << ", Title: " << stats.panel_title << "\n";
            report << "   Average: " << std::fixed << std::setprecision(3) << avg_ms << " ms\n";
            report << "   Slow Renders: " << stats.slow_render_count << "/" << stats.render_count
                   << " (" << std::fixed << std::setprecision(1) << slow_pct << "%)\n";
            report << "\n";
        }
    }

    // Summary statistics
    report << "Summary:\n";
    report << "--------\n";
    auto all_stats = get_all_panel_stats();
    uint64_t total_slow_renders = 0;
    uint64_t total_renders = 0;

    for (const auto& [panel_id, stats] : all_stats) {
        total_slow_renders += stats.slow_render_count;
        total_renders += stats.render_count;
    }

    double overall_slow_pct = total_renders > 0 ?
        (static_cast<double>(total_slow_renders) / static_cast<double>(total_renders)) * 100.0 : 0.0;

    report << "Total Slow Renders: " << total_slow_renders << "/" << total_renders
           << " (" << std::fixed << std::setprecision(1) << overall_slow_pct << "%)\n";
    report << "Slow Render Threshold: " << slow_render_threshold_us_.load() / 1000.0 << " ms\n";

    return report.str();
}

void PanelProfiler::register_slow_render_callback(std::function<void(uint32_t, const std::string&, uint64_t)> callback) {
    slow_render_callback_ = callback;
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
        double slow_pct = get_slow_render_percentage(panel_id);

        report << "Panel ID: " << panel_id
               << ", Title: " << stats.panel_title << "\n";
        report << "  Average: " << std::fixed << std::setprecision(3) << avg_ms << " ms\n";
        report << "  Min: " << std::fixed << std::setprecision(3) << min_ms << " ms\n";
        report << "  Max: " << std::fixed << std::setprecision(3) << max_ms << " ms\n";
        report << "  Last: " << std::fixed << std::setprecision(3) << last_ms << " ms\n";
        report << "  Count: " << stats.render_count << " renders\n";
        report << "  Slow Renders: " << stats.slow_render_count << " (" << std::fixed << std::setprecision(1) << slow_pct << "%)\n";
        report << "\n";
    }

    return report.str();
}

// Global instance
PanelProfiler g_panel_profiler;

} // namespace BTQuant