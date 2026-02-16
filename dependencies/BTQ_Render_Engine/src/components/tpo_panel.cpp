#include "../../include/components/tpo_panel.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../../include/analytics/tpoengine.h"
#include "../../include/symbol_registry.hpp"
#include "components/quant_workspace_component.hpp"
#include "components/theme_manager.hpp"
#include "imgui.h"
#include "implot.h"

// Define dummy structures for compilation
namespace Data {
struct Cluster {
  double centerX = 0.0;
  double centerY = 0.0;
  double width = 0.0;
  double height = 0.0;
  double askVolume = 0.0;
  double bidVolume = 0.0;
};

struct Stats {
  uint64_t lastUpdateTimeNs = 0;
};
}  // namespace Data

// DEPRECATED - Legacy hotspine
namespace BTQuant {

TpoPanel::TpoPanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                   std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {}

void TpoPanel::update(float /*dt*/) {
  // Update logic if needed
}

void TpoPanel::render() {
  begin_panel_window();

  // Enhanced toolbar with more options
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }
  ImGui::SameLine();

  // Time window configuration
  static float time_window = 3600.0f * 4.0f;  // Default 4 hours
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100);
  ImGui::SliderFloat("Time Window (s)", &time_window, 60.0f, 86400.0f, "%.0f s");

  // Determine active symbol
  uint32_t active_symbol_id = symbol_id_;
  if (active_symbol_id == 0 && bridge_) {
    // Fallback or use global symbol if needed
  }

  if (active_symbol_id == 0 || !processor_) {
    ImGui::Text("No symbol selected or processor unavailable");
    end_panel_window();
    return;
  }

  // Get analytics data from processor
  auto analytics = processor_->getSymbolAnalytics(active_symbol_id);

  // Create TPO Engine and Process Data
  // In a real implementation, we would maintain the TPOEngine state incrementally
  // For now, we rebuild it for the visible range or recent history

  // TPO Configuration
  double tpo_tick_size = 0.5;  // Default fallback
  auto sym_info = SymbolRegistry::instance().get_symbol_info(active_symbol_id);
  if (sym_info && sym_info->tick_size > 0.0) {
    tpo_tick_size = sym_info->tick_size;
  }

  TPOEngine tpo_engine(tpo_tick_size);

  // Convert recent candles to OHLCVCandle and process
  // Note: analytics.candles is std::deque<OHLCVCandle>
  // We process all available candles for now (or filter by time window)

  auto current_time = std::chrono::high_resolution_clock::now();
  auto current_time_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(current_time.time_since_epoch()).count();
  uint64_t window_start_ns =
      current_time_ns - static_cast<uint64_t>(time_window * 1000000000.0);  // Simple window

  // Check if we have candles
  // We prefer 1-minute candles for TPO accuracy
  using TimeFrame = BTQuant::RenderEngine::TimeFrame;
  const std::vector<BTQuant::RenderEngine::OHLCVCandle>* candles_ptr = nullptr;

  if (analytics.candles.count(TimeFrame::TF_1MIN)) {
    candles_ptr = &analytics.candles.at(TimeFrame::TF_1MIN);
  } else if (analytics.candles.count(TimeFrame::TF_5MIN)) {
    candles_ptr = &analytics.candles.at(TimeFrame::TF_5MIN);
  } else if (analytics.candles.count(TimeFrame::TF_30MIN)) {
    candles_ptr = &analytics.candles.at(TimeFrame::TF_30MIN);
  }

  if (candles_ptr && !candles_ptr->empty()) {
    for (const auto& candle : *candles_ptr) {
      if (candle.timestamp >= window_start_ns) {
        tpo_engine.process_candle(candle);
      }
    }
  }

  const auto& profile = tpo_engine.get_tpo_profile();
  double poc = profile.get_poc();
  auto va = profile.get_value_area(70.0);

  // Statistics for auto-fit
  int max_tpo_width = 0;
  for (const auto& [price, letters] : profile.price_to_letters) {
    max_tpo_width = std::max(max_tpo_width, (int)letters.length());
  }

  // Rendering
  if (ImPlot::BeginPlot("##TPOProfile", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_Crosshairs)) {
    ImPlot::SetupAxes("TPO Count", "Price");

    // Auto-fit axes
    ImPlot::SetupAxesLimits(0, max_tpo_width + 2, 0, 100, ImPlotCond_Always);  // X is count
    if (analytics.last_trade_price > 0) {
      double p = analytics.last_trade_price;
      ImPlot::SetupAxisLimits(ImAxis_Y1, p * 0.995, p * 1.005, ImPlotCond_Once);
    }

    ImDrawList* draw_list = ImPlot::GetPlotDrawList();

    // Render TPO Profile
    // Iterate through profile.price_to_letters (Map is sorted by Price)
    for (auto it = profile.price_to_letters.rbegin(); it != profile.price_to_letters.rend(); ++it) {
      double price = it->first;
      const std::string& letters = it->second;

      // MMT Single Print Detection
      bool is_single = profile.is_single_print(price);

      for (size_t i = 0; i < letters.length(); ++i) {
        char letter = letters[i];

        // Calculate screen position
        // X = i (0-based index)
        // Y = price

        // We want to draw a box/character centered at (i + 0.5, price)
        ImVec2 pos = ImPlot::PlotToPixels((double)i + 0.5, price);

        // Color Logic
        ImU32 color = IM_COL32(200, 200, 200, 255);  // Default Gray
        if (is_single) {
          color = IM_COL32(135, 206, 250, 200);  // MMT Light Blue
        } else if (std::abs(price - poc) < tpo_tick_size / 2.0) {
          color = IM_COL32(255, 215, 0, 255);  // Gold for POC
        } else if (price >= va.first && price <= va.second) {
          color = IM_COL32(50, 205, 50, 200);  // Greenish for Value Area
        }

        // Draw Block
        // Determine block size in pixels
        ImVec2 p1 = ImPlot::PlotToPixels((double)i, price - tpo_tick_size / 2.0);
        ImVec2 p2 = ImPlot::PlotToPixels((double)i + 1.0, price + tpo_tick_size / 2.0);

        // draw_list->AddRectFilled(p1, p2, color);

        // Draw Letter
        // Use simple text for now
        char text[2] = {letter, '\0'};
        // Center text ?
        draw_list->AddText(ImVec2(pos.x - 4, pos.y - 6), color, text);
      }
    }

    // Draw POC Line
    if (poc > 0) {
      double x[2] = {0, (double)max_tpo_width};
      double y[2] = {poc, poc};
      ImPlot::PlotLine("POC", x, y, 2);
    }

    ImPlot::EndPlot();
  }

  ImGui::SetCursorPos(ImVec2(10, 45));
  ImGui::TextColored(ImVec4(1, 1, 0, 0.5f), "TPO Profile | POC: %.2f | VA: %.2f - %.2f", poc,
                     va.first, va.second);

  end_panel_window();
}

}  // namespace BTQuant
