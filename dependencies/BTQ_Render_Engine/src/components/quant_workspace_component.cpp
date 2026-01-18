#include "../../include/components/quant_workspace_component.hpp"
#include <iostream>

namespace BTQuant {

QuantWorkspaceComponent::QuantWorkspaceComponent(
    std::shared_ptr<HotSpineDataBridge> bridge)
    : UIComponent({0, 0}, {0, 0}), bridge_(bridge) {}

void QuantWorkspaceComponent::update(float dt) {
  // Data polling is handled by the bridge in the main loop
}

void QuantWorkspaceComponent::render_gui() {
  auto instruments = bridge_->GetInstruments();

  ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_FirstUseEver);

  if (ImGui::Begin("BTQuant Workspace", nullptr,
                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_MenuBar)) {

    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("Order Flow", nullptr, false);
        ImGui::MenuItem("Market Depth", nullptr, false);
        ImGui::EndMenu();
      }
      ImGui::EndMenuBar();
    }

    // Design: Use an ImGui Table (e.g., 3 columns for 3 exchanges) to organize
    // charts automatically.
    if (ImGui::BeginTable("WorkspaceGrid", 3,
                          ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
                              ImGuiTableFlags_SizingStretchSame)) {
      for (auto const &[id, instrument] : instruments) {
        ImGui::TableNextColumn();
        render_instrument_chart(id, *instrument);
      }
      ImGui::EndTable();
    }
    ImGui::End();
  }
}

// Custom Candlestick implementation with Volume Bars
static void PlotCandlesticks(const char *, const double *xs,
                             const double *opens, const double *closes,
                             const double *lows, const double *highs,
                             const double *volumes, int count) {
  // Use ImPlot's draw list for precise, high-performance rendering
  for (int i = 0; i < count; ++i) {
    ImVec4 color = (closes[i] >= opens[i]) ? ImVec4(0, 0.8f, 0.2f, 1)
                                           : ImVec4(0.9f, 0.1f, 0.1f, 1);

    // 1. Wick (Vertical Line)
    ImPlot::PushStyleColor(ImPlotCol_Line, color);
    double wick_x[2] = {xs[i], xs[i]};
    double wick_y[2] = {lows[i], highs[i]};
    ImPlot::PlotLine("##Wick", wick_x, wick_y, 2);
    ImPlot::PopStyleColor();

    // 2. Body (Filled Rectangle)
    ImVec2 open_pixels = ImPlot::PlotToPixels(xs[i] - 0.25, opens[i]);
    ImVec2 close_pixels = ImPlot::PlotToPixels(xs[i] + 0.25, closes[i]);

    ImPlot::GetPlotDrawList()->AddRectFilled(
        open_pixels, close_pixels, ImGui::ColorConvertFloat4ToU32(color));

    // 3. Volume Bar (Semi-transparent at bottom)
    if (volumes) {
      float vol_scale = 0.1f; // Scale volume to fit at the bottom
      ImVec2 vol_min = ImPlot::PlotToPixels(
          xs[i] - 0.2, 0); // Assuming Y axis starts at 0 or auto-fits
      ImVec2 vol_max =
          ImPlot::PlotToPixels(xs[i] + 0.2, volumes[i] * vol_scale);

      ImPlot::GetPlotDrawList()->AddRectFilled(
          vol_min, vol_max,
          ImGui::ColorConvertFloat4ToU32(
              ImVec4(color.x, color.y, color.z, 0.3f)));
    }
  }
}

void QuantWorkspaceComponent::render_instrument_chart(
    const std::string &id, const MarketInstrument &instrument) {
  std::lock_guard<std::mutex> lock(instrument.mutex);

  ImGui::PushID(id.c_str());
  ImGui::Text("%s - %s", instrument.symbol.c_str(),
              instrument.exchange.c_str());

  if (ImPlot::BeginPlot("##Candles", ImVec2(-1, 300))) {
    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxis(ImAxis_Y1, nullptr,
                      ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_Opposite);

    if (instrument.count > 0) {
      PlotCandlesticks("Price", instrument.timestamps.data(),
                       instrument.opens.data(), instrument.closes.data(),
                       instrument.lows.data(), instrument.highs.data(),
                       instrument.volumes.data(), instrument.count);

      // Technical Indicator Placeholder: Moving Average
      if (instrument.count >= 20) {
        // Plot simulated EMA
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1, 1, 0, 1));
        ImPlot::PlotLine("EMA 20", instrument.timestamps.data(),
                         instrument.closes.data(), instrument.count);
        ImPlot::PopStyleColor();
      }
    }

    ImPlot::EndPlot();
  }

  // Add Statistical Insight and Order Book Mini-view
  ImGui::Columns(2, "Details", false);
  render_stats_panel(instrument);
  ImGui::NextColumn();
  render_order_book_mini(instrument);
  ImGui::Columns(1);

  ImGui::PopID();
}

void QuantWorkspaceComponent::render_order_book_mini(
    const MarketInstrument &instrument) {
  // Mini Order Book View
  ImGui::TextDisabled("LOB Snapshot");
  if (ImGui::BeginTable("MiniLOB", 2, ImGuiTableFlags_SizingFixedFit)) {
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 60.0f);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 60.0f);

    // Asks (Red)
    for (int i = 2; i >= 0; --i) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "%.2f",
                         instrument.ask_prices[i]);
      ImGui::TableNextColumn();
      ImGui::Text("%.3f", instrument.ask_sizes[i]);
    }

    // Mid
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    float mid = (instrument.ask_prices[0] + instrument.bid_prices[0]) * 0.5f;
    ImGui::TextColored(ImVec4(1, 1, 1, 0.5f), "--- %.2f ---", mid);
    ImGui::TableNextColumn();

    // Bids (Green)
    for (int i = 0; i < 3; ++i) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "%.2f",
                         instrument.bid_prices[i]);
      ImGui::TableNextColumn();
      ImGui::Text("%.3f", instrument.bid_sizes[i]);
    }
    ImGui::EndTable();
  }
}

void QuantWorkspaceComponent::render_stats_panel(
    const MarketInstrument &instrument) {
  if (instrument.count > 0) {
    double last = instrument.closes.back();
    double prev =
        (instrument.count > 1) ? instrument.closes[instrument.count - 2] : last;
    double change = last - prev;
    double pct = (prev != 0) ? (change / prev) * 100.0 : 0.0;

    ImGui::Text("Last: %.2f", last);
    ImGui::TextColored(change >= 0 ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
                       "Chg: %+.2f (%+.2f%%)", change, pct);
    ImGui::Text("Vol: %.1f", instrument.volumes.back());
  }
}

} // namespace BTQuant
