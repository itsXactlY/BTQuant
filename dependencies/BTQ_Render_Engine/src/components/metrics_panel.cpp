#include "../../include/components/metrics_panel.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "../../include/symbol_registry.hpp"
#include "imgui.h"

namespace BTQuant {

MetricsPanel::MetricsPanel(const PanelConfig& config,
                           std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config),
      processor_(processor) {
  update_metrics();
}

void MetricsPanel::update(float dt) {
  update_timer_ += dt;
  if (update_timer_ >= UPDATE_INTERVAL) {
    update_metrics();
    update_timer_ = 0.0f;
  }
}

void MetricsPanel::render_content() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_debug_info();

  end_panel_window();
}

void MetricsPanel::update_metrics() {
  // Debug info is rendered directly from the processor metrics
}

void MetricsPanel::render_debug_info() {
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 1.0f, 0.8f, 1.0f));
  ImGui::Text("=== MARKET DEBUG ===");
  ImGui::PopStyleColor();
  ImGui::Separator();

  // Pipeline Health
  ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Pipeline Health:");
  if (processor_) {
    auto metrics = processor_->getPerformanceMetrics();
    ImGui::Text("  Trades/sec: %.0f", metrics.trades_per_second);
    ImGui::Text("  Books/sec: %.0f", metrics.orderbooks_per_second);
    ImGui::Text("  Avg Latency: %.2f ms", metrics.avg_latency_ms);
  } else {
    ImGui::Text("  Processor not connected");
  }

  ImGui::Separator();

  // Symbol Registry Info
  ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Symbol Registry:");
  auto& registry = SymbolRegistry::instance();
  auto exchanges = registry.get_exchanges();
  ImGui::Text("  Exchanges loaded: %zu", exchanges.size());
  for (const auto& exchange : exchanges) {
    auto symbols = registry.get_exchange_symbols(exchange);
    ImGui::Text("    %s: %zu symbols", exchange.c_str(), symbols.size());
  }

  ImGui::Separator();

  // Frame timing
  ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Frame Timing:");
  float fps = ImGui::GetIO().Framerate;
  float frame_time_ms = 1000.0f / fps;
  ImGui::Text("  FPS: %.1f", fps);
  ImGui::Text("  Frame Time: %.2f ms", frame_time_ms);

  ImGui::Separator();

  // Active Timeframes
  ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Active Timeframes:");
  ImGui::Text("  1ms, 10ms, 100ms, 500ms");
  ImGui::Text("  1s, 3s, 5s, 15s (Strict)");

  ImGui::Separator();

  // System
  ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "System:");
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  ImGui::Text("  Time: %s", std::ctime(&time_t));
}

void MetricsPanel::render_metric_grid() {}
void MetricsPanel::render_metric_card(const Metric& metric, float width) {
  (void)metric;
  (void)width;
}
ImVec4 MetricsPanel::get_metric_color(float value, bool is_positive_good) {
  (void)value;
  (void)is_positive_good;
  return {1, 1, 1, 1};
}

}  // namespace BTQuant