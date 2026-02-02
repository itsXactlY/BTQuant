#include "../../include/components/performance_monitor_panel.hpp"

#include <iomanip>
#include <sstream>

#include "imgui.h"
#include "../../include/performance/panel_profiler.hpp"

namespace BTQuant {

PerformanceMonitorPanel::PerformanceMonitorPanel(const PanelConfig& config) : PanelBase(config) {}

void PerformanceMonitorPanel::update(float dt) {
  // Update performance metrics periodically
  update_timer_ += dt;
  if (update_timer_ >= update_interval_) {
    // Get current metrics from global performance monitor
    auto metrics = g_performance_monitor.get_metrics();

    // Update our local copy
    current_metrics_ = metrics;

    update_timer_ = 0.0f;
  }
}

void PerformanceMonitorPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();

  // Controls
  if (ImGui::Button("Reset")) {
    g_performance_monitor.reset();
  }
  ImGui::SameLine();
  ImGui::Text(" | Update Interval: %.1fs", update_interval_);
  ImGui::SameLine();
  ImGui::SliderFloat("##Interval", &update_interval_, 0.1f, 5.0f, "%.1f");

  ImGui::Separator();

  // Display current metrics in a table
  if (ImGui::BeginTable("PerformanceMetrics", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 150.0f);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableSetupColumn("Unit", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();

    // FPS
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("FPS");
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%.1f", g_performance_monitor.get_fps());
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("Frames/sec");

    // Frame Time
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Frame Time");
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%.2f", g_performance_monitor.get_frame_time_ms());
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("ms");

    // Memory Usage
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Memory Usage");
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%.1f", g_performance_monitor.get_memory_usage_percent());
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("%% of total");

    // Data Processed
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Data Processed");
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%zu", g_performance_monitor.get_data_processed_count());
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("items");

    // Indicators Calculated
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Indicators Calc.");
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%zu", g_performance_monitor.get_indicators_calculated_count());
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("calculations");

    // Data Processing Time
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Data Proc. Time");
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%.2f", g_performance_monitor.get_data_processing_time_ms());
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("ms");

    ImGui::EndTable();
  }

  // Panel render times section
  if (ImGui::CollapsingHeader("Panel Render Times")) {
    auto panel_stats = g_panel_profiler.get_all_panel_stats();

    if (!panel_stats.empty()) {
      if (ImGui::BeginTable("PanelRenderTimes", 5,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
        ImGui::TableSetupColumn("Panel ID", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("Title", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("Avg Time (ms)", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Last Time (ms)", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Slow Renders (%)", ImGuiTableColumnFlags_WidthFixed, 120.0f);

        ImGui::TableHeadersRow();

        for (const auto& [panel_id, stats] : panel_stats) {
          if (stats.render_count > 0) { // Only show panels that have been rendered
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%u", panel_id);

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%s", stats.panel_title.c_str());

            ImGui::TableSetColumnIndex(2);
            double avg_ms = g_panel_profiler.get_average_render_time_ms(panel_id);
            ImGui::Text("%.3f", avg_ms);

            ImGui::TableSetColumnIndex(3);
            double last_ms = g_panel_profiler.get_last_render_time_ms(panel_id);
            ImGui::Text("%.3f", last_ms);

            ImGui::TableSetColumnIndex(4);
            double slow_pct = g_panel_profiler.get_slow_render_percentage(panel_id);
            ImGui::Text("%.1f%% (%lu/%lu)", slow_pct, stats.slow_render_count, stats.render_count);
          }
        }

        ImGui::EndTable();
      }
    } else {
      ImGui::Text("No panel render data collected yet.");
    }
  }

  // Bottleneck analysis section
  if (ImGui::CollapsingHeader("Bottleneck Analysis")) {
    auto slowest_panels = g_panel_profiler.get_slowest_panels(5);

    if (!slowest_panels.empty()) {
      ImGui::Text("Top 5 Slowest Panels (by average render time):");

      if (ImGui::BeginTable("SlowestPanels", 4,
                            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Panel Title", ImGuiTableColumnFlags_WidthFixed, 150.0f);
        ImGui::TableSetupColumn("Avg Time (ms)", ImGuiTableColumnFlags_WidthFixed, 100.0f);
        ImGui::TableSetupColumn("Slow Renders (%)", ImGuiTableColumnFlags_WidthFixed, 120.0f);

        ImGui::TableHeadersRow();

        for (size_t i = 0; i < slowest_panels.size(); ++i) {
          const auto& [panel_id, stats] = slowest_panels[i];

          ImGui::TableNextRow();

          ImGui::TableSetColumnIndex(0);
          ImGui::Text("%zu", i + 1);

          ImGui::TableSetColumnIndex(1);
          ImGui::Text("%s", stats.panel_title.c_str());

          ImGui::TableSetColumnIndex(2);
          double avg_ms = g_panel_profiler.get_average_render_time_ms(panel_id);
          ImGui::Text("%.3f", avg_ms);

          ImGui::TableSetColumnIndex(3);
          double slow_pct = g_panel_profiler.get_slow_render_percentage(panel_id);
          ImGui::Text("%.1f%%", slow_pct);
        }

        ImGui::EndTable();
      }
    } else {
      ImGui::Text("No bottleneck data available yet.");
    }
  }

  // Detailed metrics section
  if (ImGui::CollapsingHeader("Detailed Metrics")) {
    auto all_metrics = g_performance_monitor.get_metrics();

    if (ImGui::BeginTable(
            "DetailedMetrics", 4,
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
      ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 150.0f);
      ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 100.0f);
      ImGui::TableSetupColumn("Min", ImGuiTableColumnFlags_WidthFixed, 80.0f);
      ImGui::TableSetupColumn("Max", ImGuiTableColumnFlags_WidthFixed, 80.0f);

      ImGui::TableHeadersRow();

      for (const auto& metric : all_metrics) {
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", metric.name.c_str());

        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.2f", metric.value);

        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.2f", metric.min_value);

        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.2f", metric.max_value);
      }

      ImGui::EndTable();
    }
  }

  end_panel_window();
}

void PerformanceMonitorPanel::set_update_interval(float interval) { update_interval_ = interval; }

float PerformanceMonitorPanel::get_update_interval() const { return update_interval_; }

}  // namespace BTQuant