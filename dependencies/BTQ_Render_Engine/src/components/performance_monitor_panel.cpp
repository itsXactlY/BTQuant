#include "../../include/components/performance_monitor_panel.hpp"

#include <iomanip>
#include <sstream>

#include "imgui.h"

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