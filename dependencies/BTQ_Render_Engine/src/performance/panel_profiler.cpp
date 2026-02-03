#include "../../include/performance/panel_profiler.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <cmath>

namespace BTQuant {

PanelProfiler::PanelProfiler() : enabled_(true) {}

void PanelProfiler::start_panel_render(uint32_t panel_id, const std::string& panel_title) {
    if (!enabled_) return;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Store the start time for this panel
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    profiling_data_[panel_id].start_time = start_time;
    profiling_data_[panel_id].panel_title = panel_title;

    // Also track in active renders
    {
        std::lock_guard<std::mutex> active_lock(active_renders_mutex_);
        active_renders_[panel_id] = start_time;
    }
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

    // Remove from active renders
    {
        std::lock_guard<std::mutex> active_lock(active_renders_mutex_);
        active_renders_.erase(panel_id);
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

    // Additional analysis sections
    auto sudden_degradation = get_sudden_degradation_panels(5);
    auto volatility_ranking = get_performance_volatility_ranking();
    auto potential_warnings = get_potential_bottleneck_warnings(5);

    // Sudden Degradation Section
    if (!sudden_degradation.empty()) {
        report << "Panels with Sudden Performance Degradation:\n";
        report << "------------------------------------------\n";
        for (size_t i = 0; i < std::min(size_t(5), sudden_degradation.size()); ++i) {
            uint32_t panel_id = sudden_degradation[i].first;
            double degradation_factor = sudden_degradation[i].second;
            auto stats_it = std::find_if(all_stats.begin(), all_stats.end(),
                                         [panel_id](const auto& pair) { return pair.first == panel_id; });

            if (stats_it != all_stats.end()) {
                const auto& [id, stats] = *stats_it;
                double avg_ms = get_average_render_time_ms(panel_id);

                report << (i + 1) << ". Panel ID: " << panel_id
                       << ", Title: " << stats.panel_title << "\n";
                report << "   Avg: " << std::fixed << std::setprecision(3) << avg_ms << " ms, "
                       << "Degradation Factor: " << std::fixed << std::setprecision(2) << degradation_factor * 100.0 << "%\n";
                report << "\n";
            }
        }
    }

    // Performance Volatility Section
    if (!volatility_ranking.empty()) {
        report << "Panels with Highest Performance Volatility:\n";
        report << "------------------------------------------\n";
        for (size_t i = 0; i < std::min(size_t(5), volatility_ranking.size()); ++i) {
            uint32_t panel_id = volatility_ranking[i].first;
            double volatility = volatility_ranking[i].second;
            auto stats_it = std::find_if(all_stats.begin(), all_stats.end(),
                                         [panel_id](const auto& pair) { return pair.first == panel_id; });

            if (stats_it != all_stats.end()) {
                const auto& [id, stats] = *stats_it;
                double avg_ms = get_average_render_time_ms(panel_id);

                report << (i + 1) << ". Panel ID: " << panel_id
                       << ", Title: " << stats.panel_title << "\n";
                report << "   Avg: " << std::fixed << std::setprecision(3) << avg_ms << " ms, "
                       << "Volatility: " << std::fixed << std::setprecision(3) << volatility << "\n";
                report << "\n";
            }
        }
    }

    // Potential Bottleneck Warnings Section
    if (!potential_warnings.empty()) {
        report << "Potential Future Bottleneck Warnings:\n";
        report << "-------------------------------------\n";
        for (size_t i = 0; i < std::min(size_t(5), potential_warnings.size()); ++i) {
            uint32_t panel_id = potential_warnings[i].first;
            double proximity = potential_warnings[i].second;
            auto stats_it = std::find_if(all_stats.begin(), all_stats.end(),
                                         [panel_id](const auto& pair) { return pair.first == panel_id; });

            if (stats_it != all_stats.end()) {
                const auto& [id, stats] = *stats_it;
                double avg_ms = get_average_render_time_ms(panel_id);

                report << (i + 1) << ". Panel ID: " << panel_id
                       << ", Title: " << stats.panel_title << "\n";
                report << "   Avg: " << std::fixed << std::setprecision(3) << avg_ms << " ms, "
                       << "Proximity to Threshold: " << std::fixed << std::setprecision(1) << proximity * 100.0 << "%\n";
                report << "\n";
            }
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

    if (!sudden_degradation.empty()) {
        uint32_t degraded_panel_id = sudden_degradation[0].first;
        auto degraded_panel_it = std::find_if(all_stats.begin(), all_stats.end(),
                                              [degraded_panel_id](const auto& pair) { return pair.first == degraded_panel_id; });

        if (degraded_panel_it != all_stats.end()) {
            const auto& [id, stats] = *degraded_panel_it;
            report << "- Urgent attention needed for: Panel '" << stats.panel_title
                   << "' (ID: " << id << ") due to sudden performance degradation\n";
        }
    }

    if (!potential_warnings.empty()) {
        uint32_t warning_panel_id = potential_warnings[0].first;
        auto warning_panel_it = std::find_if(all_stats.begin(), all_stats.end(),
                                             [warning_panel_id](const auto& pair) { return pair.first == warning_panel_id; });

        if (warning_panel_it != all_stats.end()) {
            const auto& [id, stats] = *warning_panel_it;
            report << "- Monitor closely: Panel '" << stats.panel_title
                   << "' (ID: " << id << ") - approaching bottleneck status\n";
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

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_high_percentile_panels(double percentile) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, PanelRenderStats>> result;

    if (profiling_data_.empty()) {
        return result;
    }

    // Copy all panel stats with valid render counts
    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 0) {
            result.push_back({panel_id, stats});
        }
    }

    // Sort by average render time (ascending for percentile calculation)
    std::sort(result.begin(), result.end(),
              [this](const auto& a, const auto& b) {
                  return get_average_render_time_ms(a.first) < get_average_render_time_ms(b.first);
              });

    // Calculate the index corresponding to the requested percentile
    size_t percentile_index = static_cast<size_t>((percentile / 100.0) * result.size());

    // Return panels that are above the specified percentile
    if (percentile_index < result.size()) {
        std::vector<std::pair<uint32_t, PanelRenderStats>> high_percentile_result(
            result.begin() + percentile_index, result.end());
        return high_percentile_result;
    }

    return {};
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_panels_above_threshold(double threshold_ms) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, PanelRenderStats>> result;

    uint64_t threshold_us = static_cast<uint64_t>(threshold_ms * 1000); // Convert ms to us

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 0) {
            double avg_time_us = static_cast<double>(stats.total_render_time_us) / static_cast<double>(stats.render_count);
            if (avg_time_us >= threshold_us) {
                result.push_back({panel_id, stats});
            }
        }
    }

    // Sort by average render time (descending)
    std::sort(result.begin(), result.end(),
              [this](const auto& a, const auto& b) {
                  return get_average_render_time_ms(a.first) > get_average_render_time_ms(b.first);
              });

    return result;
}

std::pair<double, double> PanelProfiler::get_render_trend_ms(uint32_t panel_id) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    auto it = profiling_data_.find(panel_id);
    if (it == profiling_data_.end() || it->second.render_time_history.empty()) {
        return {0.0, 0.0};
    }

    const auto& history = it->second.render_time_history;
    size_t total_size = history.size();

    // Calculate recent average (last N renders)
    size_t recent_count = std::min(static_cast<size_t>(RECENT_RENDER_COUNT), total_size);
    size_t recent_start_idx = total_size - recent_count;

    double recent_sum = 0.0;
    for (size_t i = recent_start_idx; i < total_size; ++i) {
        recent_sum += static_cast<double>(history[i]) / 1000.0; // Convert to ms
    }
    double recent_avg = recent_count > 0 ? recent_sum / recent_count : 0.0;

    // Calculate historical average (excluding recent renders)
    size_t historical_count = total_size - recent_count;
    double historical_sum = 0.0;
    for (size_t i = 0; i < recent_start_idx; ++i) {
        historical_sum += static_cast<double>(history[i]) / 1000.0; // Convert to ms
    }
    double historical_avg = historical_count > 0 ? historical_sum / historical_count : 0.0;

    return {recent_avg, historical_avg};
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_degrading_panels(size_t top_n) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> degrading_panels;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_time_history.size() >= RECENT_RENDER_COUNT * 2) { // Need enough data for comparison
            auto [recent_avg, historical_avg] = get_render_trend_ms(panel_id);

            if (historical_avg > 0.0 && recent_avg > historical_avg) {
                double degradation_factor = recent_avg / historical_avg;
                degrading_panels.push_back({panel_id, degradation_factor});
            }
        }
    }

    // Sort by degradation factor (descending)
    std::sort(degrading_panels.begin(), degrading_panels.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Limit to top N
    if (degrading_panels.size() > top_n) {
        degrading_panels.resize(top_n);
    }

    return degrading_panels;
}

std::vector<std::pair<uint32_t, uint64_t>> PanelProfiler::get_active_render_times() const {
    std::vector<std::pair<uint32_t, uint64_t>> active_times;

    {
        std::lock_guard<std::mutex> lock(active_renders_mutex_);
        auto now = std::chrono::high_resolution_clock::now();

        for (const auto& [panel_id, start_time] : active_renders_) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                now - start_time);
            active_times.push_back({panel_id, duration.count()});
        }
    }

    // Sort by render time (descending)
    std::sort(active_times.begin(), active_times.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return active_times;
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_peak_render_time_panels(size_t top_n) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, PanelRenderStats>> result;

    // Copy all panel stats
    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 0) { // Only include panels that have been rendered
            result.push_back({panel_id, stats});
        }
    }

    // Sort by max render time (descending)
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  return a.second.max_render_time_us > b.second.max_render_time_us;
              });

    // Limit to top N
    if (result.size() > top_n) {
        result.resize(top_n);
    }

    return result;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_increasing_trend_panels(size_t top_n) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> increasing_trend_panels;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_time_history.size() >= RECENT_RENDER_COUNT * 2) { // Need enough data for comparison
            auto [recent_avg, historical_avg] = get_render_trend_ms(panel_id);

            if (historical_avg > 0.0 && recent_avg > historical_avg) {
                double increase_ratio = (recent_avg - historical_avg) / historical_avg; // Percentage increase
                increasing_trend_panels.push_back({panel_id, increase_ratio});
            } else if (historical_avg > 0.0 && recent_avg <= historical_avg) {
                // Even if decreasing, we might want to track the stability
                double change_ratio = (recent_avg - historical_avg) / historical_avg;
                increasing_trend_panels.push_back({panel_id, change_ratio});
            }
        }
    }

    // Sort by increase ratio (descending) - most increasing panels first
    std::sort(increasing_trend_panels.begin(), increasing_trend_panels.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Limit to top N
    if (increasing_trend_panels.size() > top_n) {
        increasing_trend_panels.resize(top_n);
    }

    return increasing_trend_panels;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_resource_utilization_ranking() const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> utilization_ranking;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 0) {
            // Calculate resource utilization as a combination of factors:
            // 1. Average render time
            // 2. Frequency of slow renders
            // 3. Peak render time impact
            double avg_time = static_cast<double>(stats.total_render_time_us) / static_cast<double>(stats.render_count);
            double slow_render_ratio = static_cast<double>(stats.slow_render_count) / static_cast<double>(stats.render_count);
            double peak_impact = static_cast<double>(stats.max_render_time_us) / static_cast<double>(avg_time > 0 ? avg_time : 1.0);

            // Weighted score combining all factors
            double utilization_score = avg_time * (1.0 + slow_render_ratio * 5.0 + peak_impact * 0.5);
            utilization_ranking.push_back({panel_id, utilization_score});
        }
    }

    // Sort by utilization score (descending)
    std::sort(utilization_ranking.begin(), utilization_ranking.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return utilization_ranking;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_comprehensive_bottleneck_ranking() const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> bottleneck_ranking;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 0) {
            // Comprehensive scoring algorithm considering multiple factors:
            // 1. Average render time (primary factor)
            double avg_time = static_cast<double>(stats.total_render_time_us) / static_cast<double>(stats.render_count);

            // 2. Frequency of slow renders (secondary factor)
            double slow_render_ratio = static_cast<double>(stats.slow_render_count) / static_cast<double>(stats.render_count);

            // 3. Peak render time impact (tertiary factor)
            double peak_impact = static_cast<double>(stats.max_render_time_us) / static_cast<double>(avg_time > 0 ? avg_time : 1.0);

            // 4. Variance in render times (indicates instability)
            double mean = avg_time;
            double variance = 0.0;
            for (uint64_t render_time : stats.render_time_history) {
                double diff = static_cast<double>(render_time) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<double>(stats.render_time_history.size());
            double stability_factor = 1.0 + (variance / (mean > 0 ? mean : 1.0));

            // 5. Recent trend (is performance getting worse?) - calculate inline to avoid mutex issues
            double trend_factor = 1.0;
            if (stats.render_time_history.size() >= RECENT_RENDER_COUNT * 2) { // Need enough data for comparison
                // Calculate recent average (last N renders)
                size_t recent_count = std::min(static_cast<size_t>(RECENT_RENDER_COUNT), stats.render_time_history.size());
                size_t recent_start_idx = stats.render_time_history.size() - recent_count;

                double recent_sum = 0.0;
                for (size_t i = recent_start_idx; i < stats.render_time_history.size(); ++i) {
                    recent_sum += static_cast<double>(stats.render_time_history[i]) / 1000.0; // Convert to ms
                }
                double recent_avg = recent_count > 0 ? recent_sum / recent_count : 0.0;

                // Calculate historical average (excluding recent renders)
                size_t historical_count = stats.render_time_history.size() - recent_count;
                double historical_sum = 0.0;
                for (size_t i = 0; i < recent_start_idx; ++i) {
                    historical_sum += static_cast<double>(stats.render_time_history[i]) / 1000.0; // Convert to ms
                }
                double historical_avg = historical_count > 0 ? historical_sum / historical_count : 0.0;

                if (historical_avg > 0.0 && recent_avg > historical_avg) {
                    trend_factor = 1.0 + ((recent_avg - historical_avg) / historical_avg);
                }
            }

            // Weighted comprehensive score
            double bottleneck_score = avg_time *
                                    (1.0 + slow_render_ratio * 8.0 +  // Heavy weight on slow renders
                                     peak_impact * 0.5 +              // Moderate weight on peaks
                                     stability_factor * 0.3 +         // Light weight on stability
                                     trend_factor * 1.0);             // Moderate weight on trends

            bottleneck_ranking.push_back({panel_id, bottleneck_score});
        }
    }

    // Sort by bottleneck score (descending) - highest scores are biggest bottlenecks
    std::sort(bottleneck_ranking.begin(), bottleneck_ranking.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return bottleneck_ranking;
}

std::vector<std::tuple<uint32_t, std::string, double, double, uint64_t, double>> PanelProfiler::get_top_bottleneck_details(size_t top_n) const {
    std::vector<std::tuple<uint32_t, std::string, double, double, uint64_t, double>> bottleneck_details;

    // Get comprehensive bottleneck ranking (this will acquire its own lock)
    auto bottleneck_ranking = get_comprehensive_bottleneck_ranking();

    // Take top N bottlenecks and get detailed information
    size_t count = std::min(top_n, bottleneck_ranking.size());
    for (size_t i = 0; i < count; ++i) {
        uint32_t panel_id = bottleneck_ranking[i].first;
        double bottleneck_score = bottleneck_ranking[i].second;

        // Get stats for this panel (this will acquire its own lock)
        auto stats = get_panel_stats(panel_id);
        if (!stats.panel_title.empty() || stats.render_count > 0) { // Check if valid stats
            double avg_render_time_ms = get_average_render_time_ms(panel_id);
            double slow_render_percentage = get_slow_render_percentage(panel_id);
            uint64_t total_renders = stats.render_count;

            bottleneck_details.emplace_back(
                panel_id,
                stats.panel_title,
                avg_render_time_ms,
                bottleneck_score,
                total_renders,
                slow_render_percentage
            );
        }
    }

    return bottleneck_details;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_sudden_degradation_panels(size_t top_n) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> degradation_panels;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_time_history.size() >= RECENT_RENDER_COUNT * 3) { // Need sufficient data
            // Split history into early and late periods
            size_t total_size = stats.render_time_history.size();
            size_t early_end = total_size / 3;  // First third
            size_t middle_start = early_end;
            size_t middle_end = 2 * total_size / 3;  // Second third
            size_t late_start = middle_end;  // Last third

            // Calculate averages for each period
            double early_avg = 0.0;
            for (size_t i = 0; i < early_end; ++i) {
                early_avg += static_cast<double>(stats.render_time_history[i]);
            }
            early_avg /= static_cast<double>(early_end);

            double late_avg = 0.0;
            for (size_t i = late_start; i < total_size; ++i) {
                late_avg += static_cast<double>(stats.render_time_history[i]);
            }
            late_avg /= static_cast<double>(total_size - late_start);

            // Calculate degradation factor (how much worse the late period is compared to early)
            if (early_avg > 0.0 && late_avg > early_avg) {
                double degradation_factor = (late_avg - early_avg) / early_avg;

                // Only consider significant degradations (more than 50% increase)
                if (degradation_factor > 0.5) {
                    degradation_panels.push_back({panel_id, degradation_factor});
                }
            }
        }
    }

    // Sort by degradation factor (descending) - most degraded panels first
    std::sort(degradation_panels.begin(), degradation_panels.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Limit to top N
    if (degradation_panels.size() > top_n) {
        degradation_panels.resize(top_n);
    }

    return degradation_panels;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_performance_volatility_ranking() const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> volatility_ranking;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_time_history.size() >= 5) { // Need minimum data points
            // Calculate coefficient of variation (standard deviation / mean)
            double mean = static_cast<double>(stats.total_render_time_us) / static_cast<double>(stats.render_count);

            if (mean > 0.0) {
                double variance = 0.0;
                for (uint64_t render_time : stats.render_time_history) {
                    double diff = static_cast<double>(render_time) - mean;
                    variance += diff * diff;
                }
                variance /= static_cast<double>(stats.render_time_history.size());

                double std_dev = std::sqrt(variance);
                double coefficient_of_variation = std_dev / mean;

                volatility_ranking.push_back({panel_id, coefficient_of_variation});
            }
        }
    }

    // Sort by coefficient of variation (descending) - higher volatility first
    std::sort(volatility_ranking.begin(), volatility_ranking.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return volatility_ranking;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_potential_bottleneck_warnings(size_t top_n) const {
    std::lock_guard<std::mutex> lock(profiling_data_mutex_);
    std::vector<std::pair<uint32_t, double>> potential_bottlenecks;

    for (const auto& [panel_id, stats] : profiling_data_) {
        if (stats.render_count > 0) {
            double avg_time = static_cast<double>(stats.total_render_time_us) / static_cast<double>(stats.render_count);
            double current_threshold = slow_render_threshold_us_.load() / 1000.0; // Convert to ms

            // Calculate how close the average render time is to the slow render threshold
            // Values closer to 1.0 indicate approaching bottleneck status
            double proximity_to_threshold = avg_time / current_threshold;

            // Also consider recent trend - if performance is getting worse
            if (stats.render_time_history.size() >= RECENT_RENDER_COUNT * 2) {
                // Calculate recent average (last N renders)
                size_t recent_count = std::min(static_cast<size_t>(RECENT_RENDER_COUNT), stats.render_time_history.size());
                size_t recent_start_idx = stats.render_time_history.size() - recent_count;

                double recent_sum = 0.0;
                for (size_t i = recent_start_idx; i < stats.render_time_history.size(); ++i) {
                    recent_sum += static_cast<double>(stats.render_time_history[i]) / 1000.0; // Convert to ms
                }
                double recent_avg = recent_count > 0 ? recent_sum / recent_count : 0.0;

                // Calculate historical average (excluding recent renders)
                size_t historical_count = stats.render_time_history.size() - recent_count;
                double historical_sum = 0.0;
                for (size_t i = 0; i < recent_start_idx; ++i) {
                    historical_sum += static_cast<double>(stats.render_time_history[i]) / 1000.0; // Convert to ms
                }
                double historical_avg = historical_count > 0 ? historical_sum / historical_count : 0.0;

                // Adjust proximity based on trend
                if (historical_avg > 0.0 && recent_avg > historical_avg) {
                    double trend_factor = recent_avg / historical_avg;
                    proximity_to_threshold *= trend_factor; // Amplify if trending toward bottleneck
                }
            }

            // Only include panels that are approaching bottleneck status (within 20% of threshold)
            // or showing concerning trends
            if (proximity_to_threshold >= 0.8 || (proximity_to_threshold >= 0.5 && proximity_to_threshold > 0.0)) {
                potential_bottlenecks.push_back({panel_id, proximity_to_threshold});
            }
        }
    }

    // Sort by proximity to threshold (descending) - closest to becoming bottlenecks first
    std::sort(potential_bottlenecks.begin(), potential_bottlenecks.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Limit to top N
    if (potential_bottlenecks.size() > top_n) {
        potential_bottlenecks.resize(top_n);
    }

    return potential_bottlenecks;
}

double PanelProfiler::calculate_standard_deviation(const std::vector<uint64_t>& values) const {
    if (values.empty()) {
        return 0.0;
    }

    // Calculate mean
    double sum = 0.0;
    for (uint64_t val : values) {
        sum += static_cast<double>(val);
    }
    double mean = sum / static_cast<double>(values.size());

    // Calculate variance
    double variance = 0.0;
    for (uint64_t val : values) {
        double diff = static_cast<double>(val) - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(values.size());

    // Standard deviation is square root of variance
    return std::sqrt(variance);
}

// Global instance
PanelProfiler g_panel_profiler;

} // namespace BTQuant