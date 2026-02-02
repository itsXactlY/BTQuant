#include "../../include/performance/panel_profiler.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <numeric>

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

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_panel_variance_ranking() const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> variance_ranking;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 1 && !stats.render_time_history.empty()) {
            // Calculate variance of render times
            double mean = static_cast<double>(stats.total_render_time_us) / static_cast<double>(stats.render_count);
            double variance = 0.0;

            for (uint64_t render_time : stats.render_time_history) {
                double diff = static_cast<double>(render_time) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<double>(stats.render_time_history.size());

            variance_ranking.push_back({panel_id, variance});
        }
    }

    // Sort by variance (descending) - higher variance indicates more inconsistent performance
    std::sort(variance_ranking.begin(), variance_ranking.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return variance_ranking;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_panel_outlier_ratio_ranking() const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> outlier_ratio_ranking;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 2 && !stats.render_time_history.empty()) {
            // Calculate outliers using interquartile range method
            auto history = stats.render_time_history; // copy for sorting
            std::sort(history.begin(), history.end());

            size_t n = history.size();
            double q1_val = static_cast<double>(history[n / 4]);
            double q3_val = static_cast<double>(history[(3 * n) / 4]);
            double iqr = q3_val - q1_val;
            double lower_bound = q1_val - 1.5 * iqr;
            double upper_bound = q3_val + 1.5 * iqr;

            size_t outliers = 0;
            for (uint64_t time : stats.render_time_history) {
                if (static_cast<double>(time) < lower_bound || static_cast<double>(time) > upper_bound) {
                    outliers++;
                }
            }

            double outlier_ratio = static_cast<double>(outliers) / static_cast<double>(stats.render_time_history.size());
            outlier_ratio_ranking.push_back({panel_id, outlier_ratio});
        }
    }

    // Sort by outlier ratio (descending) - higher ratio indicates more outlier renders
    std::sort(outlier_ratio_ranking.begin(), outlier_ratio_ranking.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return outlier_ratio_ranking;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_panel_resource_intensity_ranking() const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> intensity_ranking;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 0) {
            // Calculate resource intensity as a combination of average render time and frequency of slow renders
            double avg_time = static_cast<double>(stats.total_render_time_us) / static_cast<double>(stats.render_count);
            double slow_render_ratio = static_cast<double>(stats.slow_render_count) / static_cast<double>(stats.render_count);

            // Weighted score combining both factors
            double intensity_score = avg_time * (1.0 + slow_render_ratio * 10.0);
            intensity_ranking.push_back({panel_id, intensity_score});
        }
    }

    // Sort by intensity score (descending)
    std::sort(intensity_ranking.begin(), intensity_ranking.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return intensity_ranking;
}

std::string PanelProfiler::generate_detailed_bottleneck_report() const {
    std::ostringstream report;
    report << "Comprehensive Panel Bottleneck Analysis Report\n";
    report << "==============================================\n\n";

    auto all_stats = get_all_panel_stats();

    if (all_stats.empty()) {
        report << "No panel render data collected.\n";
        return report.str();
    }

    // Get rankings by different criteria
    auto variance_ranking = get_panel_variance_ranking();
    auto outlier_ranking = get_panel_outlier_ratio_ranking();
    auto intensity_ranking = get_panel_resource_intensity_ranking();
    auto slowest_panels = get_slowest_panels(10);
    auto bottleneck_panels = get_bottleneck_panels(10);

    // Summary statistics
    report << "Summary:\n";
    report << "--------\n";
    uint64_t total_slow_renders = 0;
    uint64_t total_renders = 0;
    double total_avg_render_time = 0.0;
    size_t valid_panels = 0;

    for (const auto& [panel_id, stats] : all_stats) {
        total_slow_renders += stats.slow_render_count;
        total_renders += stats.render_count;
        if (stats.render_count > 0) {
            total_avg_render_time += static_cast<double>(stats.total_render_time_us) / static_cast<double>(stats.render_count);
            valid_panels++;
        }
    }

    double overall_slow_pct = total_renders > 0 ?
        (static_cast<double>(total_slow_renders) / static_cast<double>(total_renders)) * 100.0 : 0.0;
    double overall_avg_render_time = valid_panels > 0 ? total_avg_render_time / static_cast<double>(valid_panels) / 1000.0 : 0.0;

    report << "Total Panels: " << all_stats.size() << "\n";
    report << "Total Renders: " << total_renders << "\n";
    report << "Total Slow Renders: " << total_slow_renders << " (" << std::fixed << std::setprecision(1) << overall_slow_pct << "%)\n";
    report << "Overall Avg Render Time: " << std::fixed << std::setprecision(3) << overall_avg_render_time << " ms\n";
    report << "Slow Render Threshold: " << slow_render_threshold_us_.load() / 1000.0 << " ms\n\n";

    // High Resource Intensity Panels
    if (!intensity_ranking.empty()) {
        report << "High Resource Intensity Panels:\n";
        report << "-------------------------------\n";
        for (size_t i = 0; i < std::min(size_t(5), intensity_ranking.size()); ++i) {
            uint32_t panel_id = intensity_ranking[i].first;
            double intensity_score = intensity_ranking[i].second;
            auto stats_it = std::find_if(all_stats.begin(), all_stats.end(),
                                         [panel_id](const auto& pair) { return pair.first == panel_id; });

            if (stats_it != all_stats.end()) {
                const auto& [id, stats] = *stats_it;
                double avg_ms = get_average_render_time_ms(panel_id);
                double slow_pct = get_slow_render_percentage(panel_id);

                report << (i + 1) << ". Panel ID: " << panel_id
                       << ", Title: " << stats.panel_title << "\n";
                report << "   Avg: " << std::fixed << std::setprecision(3) << avg_ms << " ms, "
                       << "Slow: " << std::fixed << std::setprecision(1) << slow_pct << "%, "
                       << "Intensity Score: " << std::fixed << std::setprecision(2) << intensity_score << "\n";
                report << "\n";
            }
        }
    }

    // High Variance Panels (Unstable Performance)
    if (!variance_ranking.empty()) {
        report << "High Variance Panels (Unstable Performance):\n";
        report << "-------------------------------------------\n";
        for (size_t i = 0; i < std::min(size_t(5), variance_ranking.size()); ++i) {
            uint32_t panel_id = variance_ranking[i].first;
            double variance = variance_ranking[i].second;
            auto stats_it = std::find_if(all_stats.begin(), all_stats.end(),
                                         [panel_id](const auto& pair) { return pair.first == panel_id; });

            if (stats_it != all_stats.end()) {
                const auto& [id, stats] = *stats_it;
                double avg_ms = get_average_render_time_ms(panel_id);
                double min_ms = static_cast<double>(stats.min_render_time_us) / 1000.0;
                double max_ms = static_cast<double>(stats.max_render_time_us) / 1000.0;

                report << (i + 1) << ". Panel ID: " << panel_id
                       << ", Title: " << stats.panel_title << "\n";
                report << "   Avg: " << std::fixed << std::setprecision(3) << avg_ms << " ms, "
                       << "Min: " << std::fixed << std::setprecision(3) << min_ms << " ms, "
                       << "Max: " << std::fixed << std::setprecision(3) << max_ms << " ms, "
                       << "Variance: " << variance << "\n";
                report << "\n";
            }
        }
    }

    // Outlier Panels (Irregular Performance)
    if (!outlier_ranking.empty()) {
        report << "Outlier-Prone Panels (Irregular Performance):\n";
        report << "--------------------------------------------\n";
        for (size_t i = 0; i < std::min(size_t(5), outlier_ranking.size()); ++i) {
            uint32_t panel_id = outlier_ranking[i].first;
            double outlier_ratio = outlier_ranking[i].second;
            auto stats_it = std::find_if(all_stats.begin(), all_stats.end(),
                                         [panel_id](const auto& pair) { return pair.first == panel_id; });

            if (stats_it != all_stats.end()) {
                const auto& [id, stats] = *stats_it;
                double avg_ms = get_average_render_time_ms(panel_id);
                double slow_pct = get_slow_render_percentage(panel_id);

                report << (i + 1) << ". Panel ID: " << panel_id
                       << ", Title: " << stats.panel_title << "\n";
                report << "   Avg: " << std::fixed << std::setprecision(3) << avg_ms << " ms, "
                       << "Slow: " << std::fixed << std::setprecision(1) << slow_pct << "%, "
                       << "Outlier Ratio: " << std::fixed << std::setprecision(2) << (outlier_ratio * 100.0) << "%\n";
                report << "\n";
            }
        }
    }

    // Traditional slowest panels
    if (!slowest_panels.empty()) {
        report << "Traditional Slowest Panels:\n";
        report << "---------------------------\n";
        for (size_t i = 0; i < std::min(size_t(5), slowest_panels.size()); ++i) {
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

    // Recommendations section
    report << "Recommendations:\n";
    report << "----------------\n";
    if (!intensity_ranking.empty()) {
        uint32_t worst_panel_id = intensity_ranking[0].first;
        auto worst_panel_it = std::find_if(all_stats.begin(), all_stats.end(),
                                           [worst_panel_id](const auto& pair) { return pair.first == worst_panel_id; });

        if (worst_panel_it != all_stats.end()) {
            const auto& [id, stats] = *worst_panel_it;
            report << "- Priority optimization target: Panel '" << stats.panel_title
                   << "' (ID: " << id << ") due to high resource intensity\n";
        }
    }

    if (!variance_ranking.empty()) {
        uint32_t unstable_panel_id = variance_ranking[0].first;
        auto unstable_panel_it = std::find_if(all_stats.begin(), all_stats.end(),
                                              [unstable_panel_id](const auto& pair) { return pair.first == unstable_panel_id; });

        if (unstable_panel_it != all_stats.end()) {
            const auto& [id, stats] = *unstable_panel_it;
            report << "- Investigate inconsistent performance in: Panel '" << stats.panel_title
                   << "' (ID: " << id << ") due to high variance\n";
        }
    }

    if (!outlier_ranking.empty()) {
        uint32_t outlier_panel_id = outlier_ranking[0].first;
        auto outlier_panel_it = std::find_if(all_stats.begin(), all_stats.end(),
                                             [outlier_panel_id](const auto& pair) { return pair.first == outlier_panel_id; });

        if (outlier_panel_it != all_stats.end()) {
            const auto& [id, stats] = *outlier_panel_it;
            report << "- Examine irregular behavior in: Panel '" << stats.panel_title
                   << "' (ID: " << id << ") due to high outlier ratio\n";
        }
    }

    return report.str();
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