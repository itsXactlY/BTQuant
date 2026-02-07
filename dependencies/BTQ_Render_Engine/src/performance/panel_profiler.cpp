#include "performance/panel_profiler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace BTQuant {

// Global panel profiler instance
PanelProfiler g_panel_profiler;

PanelProfiler::PanelProfiler() : enabled_(true) {}

void PanelProfiler::start_panel_render(uint32_t panel_id, const std::string& panel_title) {
  if (!enabled_) return;

  std::lock_guard<std::mutex> lock(active_renders_mutex_);
  active_renders_[panel_id] = std::chrono::high_resolution_clock::now();

  std::lock_guard<std::mutex> data_lock(profiling_data_mutex_);
  if (profiling_data_.find(panel_id) == profiling_data_.end()) {
    profiling_data_[panel_id] = PanelRenderStats{};
    profiling_data_[panel_id].panel_title = panel_title;
  }
  profiling_data_[panel_id].start_time = std::chrono::high_resolution_clock::now();
}

void PanelProfiler::end_panel_render(uint32_t panel_id) {
  if (!enabled_) return;

  auto end_time = std::chrono::high_resolution_clock::now();

  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  auto it = profiling_data_.find(panel_id);
  if (it != profiling_data_.end()) {
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - it->second.start_time)
            .count();

    it->second.last_render_time_us = duration;
    it->second.total_render_time_us += duration;
    it->second.render_count++;

    if (static_cast<int64_t>(duration) < static_cast<int64_t>(it->second.min_render_time_us) || it->second.min_render_time_us == 0) {
      it->second.min_render_time_us = duration;
    }
    if (static_cast<int64_t>(duration) > static_cast<int64_t>(it->second.max_render_time_us)) {
      it->second.max_render_time_us = duration;
    }

    if (static_cast<int64_t>(duration) > static_cast<int64_t>(it->second.slow_render_threshold_us)) {
      it->second.slow_render_count++;
      if (slow_render_callback_) {
        slow_render_callback_(panel_id, it->second.panel_title, duration);
      }
    }

    it->second.render_time_history.push_back(duration);
    if (it->second.render_time_history.size() > MAX_HISTORY_SIZE) {
      it->second.render_time_history.erase(it->second.render_time_history.begin());
    }

    it->second.cumulative_squared_time_us += duration * duration;

    if (!it->second.first_render_recorded) {
      it->second.first_render_time = end_time;
      it->second.first_render_recorded = true;
    }
  }

  std::lock_guard<std::mutex> active_lock(active_renders_mutex_);
  active_renders_.erase(panel_id);
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
  for (const auto& [id, stats] : profiling_data_) {
    result.emplace_back(id, stats);
  }
  return result;
}

double PanelProfiler::get_average_render_time_ms(uint32_t panel_id) const {
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  auto it = profiling_data_.find(panel_id);
  if (it != profiling_data_.end() && it->second.render_count > 0) {
    return static_cast<double>(it->second.total_render_time_us) /
           static_cast<double>(it->second.render_count) / 1000.0;
  }
  return 0.0;
}

double PanelProfiler::get_last_render_time_ms(uint32_t panel_id) const {
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  auto it = profiling_data_.find(panel_id);
  if (it != profiling_data_.end()) {
    return static_cast<double>(it->second.last_render_time_us) / 1000.0;
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
    return static_cast<double>(it->second.slow_render_count) /
           static_cast<double>(it->second.render_count) * 100.0;
  }
  return 0.0;
}

void PanelProfiler::set_slow_render_threshold(uint64_t threshold_us) {
  slow_render_threshold_us_ = threshold_us;
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_slowest_panels(
    size_t top_n) const {
  auto all_stats = get_all_panel_stats();
  std::sort(all_stats.begin(), all_stats.end(), [](const auto& a, const auto& b) {
    double avg_a = a.second.render_count > 0
                       ? static_cast<double>(a.second.total_render_time_us) / a.second.render_count
                       : 0;
    double avg_b = b.second.render_count > 0
                       ? static_cast<double>(b.second.total_render_time_us) / b.second.render_count
                       : 0;
    return avg_a > avg_b;
  });

  if (all_stats.size() > top_n) {
    all_stats.resize(top_n);
  }
  return all_stats;
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_bottleneck_panels(
    size_t top_n) const {
  auto all_stats = get_all_panel_stats();
  std::sort(all_stats.begin(), all_stats.end(), [](const auto& a, const auto& b) {
    return a.second.slow_render_count > b.second.slow_render_count;
  });

  if (all_stats.size() > top_n) {
    all_stats.resize(top_n);
  }
  return all_stats;
}

void PanelProfiler::reset_stats() {
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  profiling_data_.clear();
}

void PanelProfiler::set_enabled(bool enabled) { enabled_ = enabled; }

bool PanelProfiler::is_enabled() const { return enabled_; }

std::string PanelProfiler::generate_report() const {
  std::ostringstream oss;
  oss << "=== Panel Render Performance Report ===\n";
  auto all_stats = get_all_panel_stats();
  for (const auto& [id, stats] : all_stats) {
    double avg_ms = stats.render_count > 0 ? static_cast<double>(stats.total_render_time_us) /
                                                 stats.render_count / 1000.0
                                           : 0;
    oss << "Panel " << id << " (" << stats.panel_title << "): "
        << "Avg: " << avg_ms << "ms, "
        << "Last: " << stats.last_render_time_us / 1000.0 << "ms, "
        << "Count: " << stats.render_count << "\n";
  }
  return oss.str();
}

std::string PanelProfiler::generate_bottleneck_report() const {
  std::ostringstream oss;
  oss << "=== Bottleneck Analysis Report ===\n";
  auto bottlenecks = get_bottleneck_panels(10);
  for (const auto& [id, stats] : bottlenecks) {
    if (stats.slow_render_count > 0) {
      oss << "Panel " << id << " (" << stats.panel_title << "): " << stats.slow_render_count
          << " slow renders (" << get_slow_render_percentage(id) << "%)\n";
    }
  }
  return oss.str();
}

void PanelProfiler::register_slow_render_callback(
    std::function<void(uint32_t, const std::string&, uint64_t)> callback) {
  slow_render_callback_ = std::move(callback);
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_panel_variance_ranking() const {
  std::vector<std::pair<uint32_t, double>> result;
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  for (const auto& [id, stats] : profiling_data_) {
    if (!stats.render_time_history.empty()) {
      double variance = calculate_standard_deviation(stats.render_time_history);
      result.emplace_back(id, variance);
    }
  }
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  return result;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_panel_outlier_ratio_ranking() const {
  return get_panel_variance_ranking();
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_panel_resource_intensity_ranking()
    const {
  std::vector<std::pair<uint32_t, double>> result;
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  for (const auto& [id, stats] : profiling_data_) {
    if (stats.render_count > 0) {
      double avg = static_cast<double>(stats.total_render_time_us) / stats.render_count;
      double slow_ratio = static_cast<double>(stats.slow_render_count) / stats.render_count;
      double intensity = avg * (1.0 + slow_ratio);
      result.emplace_back(id, intensity);
    }
  }
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  return result;
}

std::string PanelProfiler::generate_detailed_bottleneck_report() const {
  return generate_bottleneck_report();
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_high_percentile_panels(
    double /*percentile*/) const {
  return get_slowest_panels(5);
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_panels_above_threshold(
    double threshold_ms) const {
  std::vector<std::pair<uint32_t, PanelRenderStats>> result;
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  for (const auto& [id, stats] : profiling_data_) {
    if (stats.render_count > 0) {
      double avg_ms = static_cast<double>(stats.total_render_time_us) / stats.render_count / 1000.0;
      if (avg_ms > threshold_ms) {
        result.emplace_back(id, stats);
      }
    }
  }
  return result;
}

std::pair<double, double> PanelProfiler::get_render_trend_ms(uint32_t panel_id) const {
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  auto it = profiling_data_.find(panel_id);
  if (it != profiling_data_.end() && !it->second.render_time_history.empty()) {
    const auto& history = it->second.render_time_history;
    size_t recent_count = std::min(history.size(), RECENT_RENDER_COUNT);

    double recent_sum = 0;
    for (size_t i = history.size() - recent_count; i < history.size(); ++i) {
      recent_sum += history[i];
    }
    double recent_avg = recent_sum / recent_count / 1000.0;

    double total_sum = std::accumulate(history.begin(), history.end(), 0ULL);
    double historical_avg = total_sum / history.size() / 1000.0;

    return {recent_avg, historical_avg};
  }
  return {0.0, 0.0};
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_degrading_panels(size_t top_n) const {
  std::vector<std::pair<uint32_t, double>> result;
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  for (const auto& [id, stats] : profiling_data_) {
    auto [recent, historical] = get_render_trend_ms(id);
    if (historical > 0) {
      double degradation = recent / historical;
      if (degradation > 1.0) {
        result.emplace_back(id, degradation);
      }
    }
  }
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  if (result.size() > top_n) {
    result.resize(top_n);
  }
  return result;
}

std::vector<std::pair<uint32_t, uint64_t>> PanelProfiler::get_active_render_times() const {
  std::vector<std::pair<uint32_t, uint64_t>> result;
  auto now = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(active_renders_mutex_);
  for (const auto& [id, start_time] : active_renders_) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time).count();
    result.emplace_back(id, duration);
  }
  return result;
}

std::vector<std::pair<uint32_t, PanelRenderStats>> PanelProfiler::get_peak_render_time_panels(
    size_t top_n) const {
  auto all_stats = get_all_panel_stats();
  std::sort(all_stats.begin(), all_stats.end(), [](const auto& a, const auto& b) {
    return a.second.max_render_time_us > b.second.max_render_time_us;
  });
  if (all_stats.size() > top_n) {
    all_stats.resize(top_n);
  }
  return all_stats;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_increasing_trend_panels(
    size_t top_n) const {
  return get_degrading_panels(top_n);
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_resource_utilization_ranking() const {
  return get_panel_resource_intensity_ranking();
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_comprehensive_bottleneck_ranking()
    const {
  return get_panel_resource_intensity_ranking();
}

std::vector<std::tuple<uint32_t, std::string, double, double, uint64_t, double>>
PanelProfiler::get_top_bottleneck_details(size_t top_n) const {
  std::vector<std::tuple<uint32_t, std::string, double, double, uint64_t, double>> result;
  auto bottlenecks = get_bottleneck_panels(top_n);
  for (const auto& [id, stats] : bottlenecks) {
    double avg_ms = stats.render_count > 0 ? static_cast<double>(stats.total_render_time_us) /
                                                 stats.render_count / 1000.0
                                           : 0;
    double slow_pct = stats.render_count > 0 ? static_cast<double>(stats.slow_render_count) /
                                                   stats.render_count * 100.0
                                             : 0;
    result.emplace_back(id, stats.panel_title, avg_ms,
                        static_cast<double>(stats.max_render_time_us) / 1000.0,
                        stats.slow_render_count, slow_pct);
  }
  return result;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_sudden_degradation_panels(
    size_t top_n) const {
  return get_degrading_panels(top_n);
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_performance_volatility_ranking() const {
  return get_panel_variance_ranking();
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_potential_bottleneck_warnings(
    size_t top_n) const {
  return get_degrading_panels(top_n);
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_highest_variance_panels(
    size_t top_n) const {
  auto variance_ranking = get_panel_variance_ranking();
  if (variance_ranking.size() > top_n) {
    variance_ranking.resize(top_n);
  }
  return variance_ranking;
}

std::vector<std::pair<uint32_t, std::pair<double, double>>>
PanelProfiler::get_high_percentile_render_times(size_t top_n) const {
  std::vector<std::pair<uint32_t, std::pair<double, double>>> result;
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  for (const auto& [id, stats] : profiling_data_) {
    if (!stats.render_time_history.empty()) {
      double p95 = calculate_percentile(stats.render_time_history, 95.0) / 1000.0;
      double p99 = calculate_percentile(stats.render_time_history, 99.0) / 1000.0;
      result.emplace_back(id, std::make_pair(p95, p99));
    }
  }
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
  if (result.size() > top_n) {
    result.resize(top_n);
  }
  return result;
}

std::vector<std::pair<uint32_t, double>> PanelProfiler::get_longest_running_panels(
    size_t top_n) const {
  std::vector<std::pair<uint32_t, double>> result;
  auto now = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  for (const auto& [id, stats] : profiling_data_) {
    if (stats.first_render_recorded) {
      auto duration =
          std::chrono::duration_cast<std::chrono::seconds>(now - stats.first_render_time).count();
      result.emplace_back(id, static_cast<double>(duration));
    }
  }
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  if (result.size() > top_n) {
    result.resize(top_n);
  }
  return result;
}

std::tuple<double, double, double, double, double> PanelProfiler::get_detailed_panel_metrics(
    uint32_t panel_id) const {
  std::lock_guard<std::mutex> lock(profiling_data_mutex_);
  auto it = profiling_data_.find(panel_id);
  if (it != profiling_data_.end() && it->second.render_count > 0) {
    const auto& stats = it->second;
    double avg_ms = static_cast<double>(stats.total_render_time_us) / stats.render_count / 1000.0;
    double std_dev_ms = calculate_standard_deviation(stats.render_time_history) / 1000.0;
    double p95_ms = calculate_percentile(stats.render_time_history, 95.0) / 1000.0;
    double p99_ms = calculate_percentile(stats.render_time_history, 99.0) / 1000.0;
    double variance = std_dev_ms * std_dev_ms;
    return {avg_ms, std_dev_ms, p95_ms, p99_ms, variance};
  }
  return {0.0, 0.0, 0.0, 0.0, 0.0};
}

double PanelProfiler::calculate_standard_deviation(const std::vector<uint64_t>& values) const {
  if (values.empty()) return 0.0;
  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  double mean = sum / values.size();
  double sq_sum = 0.0;
  for (auto v : values) {
    sq_sum += (v - mean) * (v - mean);
  }
  return std::sqrt(sq_sum / values.size());
}

double PanelProfiler::calculate_percentile(const std::vector<uint64_t>& values,
                                           double percentile) const {
  if (values.empty()) return 0.0;
  std::vector<uint64_t> sorted_values = values;
  std::sort(sorted_values.begin(), sorted_values.end());
  size_t idx = static_cast<size_t>(percentile / 100.0 * (sorted_values.size() - 1));
  return static_cast<double>(sorted_values[idx]);
}

}  // namespace BTQuant
