#include "../../include/components/status_bar_panel.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>

#include "imgui.h"

namespace BTQuant {

StatusBarPanel::StatusBarPanel(const PanelConfig& config,
                               std::shared_ptr<HotSpineDataBridge> bridge,
                               std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
  config_.resizable = false;
  config_.movable = false;
}

void StatusBarPanel::update(float dt) {
  update_connection_status();
  update_performance_metrics();

  // Calculate FPS
  static float frame_time_accumulator = 0.0f;
  static int frame_count = 0;
  frame_time_accumulator += dt;
  frame_count++;

  if (frame_time_accumulator >= 1.0f) {
    fps_ = frame_count / frame_time_accumulator;
    frame_time_ms_ = (frame_time_accumulator / frame_count) * 1000.0f;
    frame_time_accumulator = 0.0f;
    frame_count = 0;
  }
}

void StatusBarPanel::render() {
  // Status bar should be a fixed bar at the top, not a window
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, 30));

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 4));
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));

  // Check if we're in a valid ImGui frame scope to prevent assertion errors
  ImGuiContext& g = *GImGui;
  if (!g.WithinFrameScope) {
      // If we're not within a frame scope, skip rendering this frame to avoid the assertion
      // The status bar will be rendered in the next frame when the scope is valid
      return;
  }

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                           ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                           ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse;

  ImGui::Begin("StatusBar", nullptr, flags);

  // Left section - Connection status
  render_connection_status();
  ImGui::SameLine();

  ImGui::Separator();
  ImGui::SameLine();

  // Center section - Performance metrics
  render_performance_metrics();
  ImGui::SameLine();

  ImGui::Separator();
  ImGui::SameLine();

  // Right section - Time
  render_time_display();

  ImGui::End();

  ImGui::PopStyleColor(1);
  ImGui::PopStyleVar(3);
}

void StatusBarPanel::update_connection_status() {
  // Check bridge exists
  if (!bridge_) {
    connection_status_ = false;
    connection_text_ = "No Bridge";
    return;
  }

  // Check if we have active symbols (SHM is readable)
  auto active_symbols = bridge_->getActiveSymbols();
  if (active_symbols.empty()) {
    connection_status_ = false;
    connection_text_ = "No Symbols";
    return;
  }

  // Check data freshness via processor metrics
  if (processor_) {
    auto metrics = processor_->getPerformanceMetrics();
    bool data_flowing = (metrics.trades_per_second > 0 || metrics.orderbooks_per_second > 0);
    connection_status_ = data_flowing;
    connection_text_ = data_flowing ? "Live" : "Stale";
  } else {
    connection_status_ = true;
    connection_text_ = "Connected";
  }
}

void StatusBarPanel::update_performance_metrics() {
  if (processor_) {
    performance_metrics_ = processor_->getPerformanceMetrics();
  }
}

void StatusBarPanel::render_connection_status() {
  ImVec4 color =
      connection_status_ ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
  ImGui::TextColored(color, "●");
  ImGui::SameLine();
  ImGui::Text("%s", connection_text_.c_str());
}

void StatusBarPanel::render_performance_metrics() {
  ImGui::Text("Trades: %.0f/s", performance_metrics_.trades_per_second);
  ImGui::SameLine();
  ImGui::Text("Books: %.0f/s", performance_metrics_.orderbooks_per_second);
  ImGui::SameLine();
  ImGui::Text("Latency: %.2fms", performance_metrics_.avg_latency_ms);
  ImGui::SameLine();
  ImGui::Text("FPS: %.1f (%.2fms)", fps_, frame_time_ms_);
}

void StatusBarPanel::render_time_display() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto tm = std::localtime(&time_t);

  std::stringstream ss;
  ss << std::put_time(tm, "%H:%M:%S");

  ImGui::Text("%s", ss.str().c_str());
}

}  // namespace BTQuant