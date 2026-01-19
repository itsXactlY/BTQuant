#include "../../include/components/quant_workspace_component.hpp"
#include "implot_internal.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace BTQuant {

QuantWorkspaceComponent::QuantWorkspaceComponent(
    std::shared_ptr<HotSpineDataBridge> bridge)
    : UIComponent({0, 0}, {0, 0}), bridge_(bridge) {

  // Ensure ImPlot context is created (must be called once)
  static bool implot_init = false;
  if (!implot_init) {
    ImPlot::CreateContext();
    implot_init = true;
  }
}

void QuantWorkspaceComponent::update(float dt) {}

void QuantWorkspaceComponent::render_gui() {
  auto &instruments = bridge_->GetAllInstruments();
  std::lock_guard<std::mutex> lock(bridge_->GetMapMutex());

  // Set the "Neon" theme globally for this component context if needed or via
  // local style push
  ImGuiIO &io = ImGui::GetIO();

  static std::vector<std::string> closed_symbols;

  for (auto it = instruments.begin(); it != instruments.end();) {
    const std::string &symbol = it->first;
    auto &inst = it->second;

    // Check if symbol was closed
    auto closed_it =
        std::find(closed_symbols.begin(), closed_symbols.end(), symbol);
    if (closed_it != closed_symbols.end()) {
      ++it;
      continue;
    }

    bool open = true;
    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);

    // Cyber-Cyan Title / Neon-Red Accents would be handled by style pushes
    if (ImGui::Begin(symbol.c_str(), &open)) {
      render_instrument_chart(symbol, *inst);
    }
    ImGui::End();

    if (!open) {
      closed_symbols.push_back(symbol);
    }
    ++it;
  }
}

void QuantWorkspaceComponent::render_instrument_chart(
    const std::string &symbol, const InstrumentStore &inst) {
  std::lock_guard<std::mutex> inst_lock(inst.data_mutex);

  if (inst.timestamps.empty()) {
    ImGui::Text("Initializing Stream for %s...", symbol.c_str());
    return;
  }

  // Cyber-Cyan: #00F0FF (0xFFFFF000), Neon-Red: #FF0033 (0xFF3300FF)
  ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::GetColorU32(ImVec4(
                                             0.0f, 0.94f, 1.0f, 1.0f))); // Cyan

  if (ImPlot::BeginPlot(symbol.c_str(), ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxis(ImAxis_X1, "Time", ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxis(ImAxis_Y2, "Volume",
                      ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_NoGridLines |
                          ImPlotAxisFlags_NoTickLabels);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y2, 0,
                                       1000000); // For Volume alignment

    const double *dates = inst.timestamps.data();
    const double *opens = inst.opens.data();
    const double *closes = inst.closes.data();
    const double *lows = inst.lows.data();
    const double *highs = inst.highs.data();
    int count = (int)inst.timestamps.size();

    // Plot 1: Candlesticks (Manual high-perf implementation)
    if (ImPlot::BeginItem("OHLC")) {
      ImDrawList *draw_list = ImPlot::GetPlotDrawList();
      double width = 0.25;
      if (count > 1) {
        width = (dates[1] - dates[0]) * 0.25;
      }

      for (int i = 0; i < count; ++i) {
        ImVec2 open_pos = ImPlot::PlotToPixels(dates[i] - width, opens[i]);
        ImVec2 close_pos = ImPlot::PlotToPixels(dates[i] + width, closes[i]);
        ImVec2 low_pos = ImPlot::PlotToPixels(dates[i], lows[i]);
        ImVec2 high_pos = ImPlot::PlotToPixels(dates[i], highs[i]);

        // Neon Red for down, Cyber Cyan for up
        ImU32 color = (opens[i] > closes[i])
                          ? ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.2f, 1.0f))
                          : // Neon Red
                          ImGui::GetColorU32(
                              ImVec4(0.0f, 0.94f, 1.0f, 1.0f)); // Cyber Cyan

        draw_list->AddLine(low_pos, high_pos, color);
        draw_list->AddRectFilled(open_pos, close_pos, color);

        ImPlot::FitPoint(ImPlotPoint(dates[i], lows[i]));
        ImPlot::FitPoint(ImPlotPoint(dates[i], highs[i]));
      }
      ImPlot::EndItem();
    }

    // Plot 2: Volume Profile (PlotBarsH on Y-axis)
    if (!inst.m_vol_profile.empty()) {
      std::vector<double> vp_prices;
      std::vector<double> vp_volumes;
      for (auto const &[price, vol] : inst.m_vol_profile) {
        vp_prices.push_back(price);
        vp_volumes.push_back(vol);
      }

      ImPlot::SetAxis(ImAxis_Y1); // Align to Price Axis
      ImPlot::SetNextFillStyle(
          ImVec4(0.0f, 0.94f, 1.0f, 0.3f)); // Transparent Cyan
      // Volume scale is arbitrary on Y axis context, but we use PlotBarsH which
      // uses Y as positions Horizontal Bars: Y = positions (prices), X = values
      // (volumes) We want them on the right, so we might need a custom plotter
      // or just use the Y axis
      ImPlot::PlotBars("VolProfile", vp_prices.data(), vp_volumes.data(),
                       (int)vp_prices.size(), 0.5, ImPlotBarsFlags_Horizontal);
    }

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor();
}

} // namespace BTQuant
