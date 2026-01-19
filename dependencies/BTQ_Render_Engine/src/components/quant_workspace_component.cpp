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
  // Use BeginTable as a high-performance grid layout fallback since DockSpace
  // requires a specific ImGui branch not present in the current environment.

  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->WorkPos);
  ImGui::SetNextWindowSize(ImGui::GetMainViewport()->WorkSize);

  if (ImGui::Begin("Quant Workspace", nullptr,
                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                       ImGuiWindowFlags_NoMove |
                       ImGuiWindowFlags_NoBringToFrontOnFocus)) {

    auto &instruments = bridge_->GetAllInstruments();
    std::lock_guard<std::mutex> lock(bridge_->GetMapMutex());

    int columns = instruments.size() > 1 ? 2 : 1;
    if (instruments.size() > 4)
      columns = 3;

    if (ImGui::BeginTable("ChartsGrid", columns,
                          ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_BordersInner)) {
      for (auto &[symbol, inst] : instruments) {
        ImGui::TableNextColumn();
        // Label the cell
        ImGui::TextColored(ImVec4(0, 0.95f, 1, 1), "[ %s ]", symbol.c_str());
        render_instrument_chart(symbol, *inst);
      }
      ImGui::EndTable();
    }
    ImGui::End();
  }
}

void QuantWorkspaceComponent::render_instrument_chart(
    const std::string &symbol, const InstrumentData &inst) {
  std::lock_guard<std::mutex> inst_lock(inst.data_mutex);

  if (inst.timestamps.empty()) {
    ImGui::Text("No data for %s", symbol.c_str());
    return;
  }

  if (ImPlot::BeginPlot(symbol.c_str(), ImVec2(-1, -1))) {
    // Setup Axis
    ImPlot::SetupAxis(ImAxis_X1, "Time", ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_AutoFit);

    // Setup values
    const double *dates = inst.timestamps.data();
    const double *opens = inst.opens.data();
    const double *closes = inst.closes.data();
    const double *lows = inst.lows.data();
    const double *highs = inst.highs.data();
    int count = (int)inst.timestamps.size();

    // Custom Candlestick Plotting (Manual)
    if (ImPlot::BeginItem("OHLC")) {
      ImDrawList *draw_list = ImPlot::GetPlotDrawList();

      // Width calculation
      double width = 0.25;
      if (count > 1) {
        width = (dates[1] - dates[0]) * 0.25;
      }

      for (int i = 0; i < count; ++i) {
        ImVec2 open_pos = ImPlot::PlotToPixels(dates[i] - width, opens[i]);
        ImVec2 close_pos = ImPlot::PlotToPixels(dates[i] + width, closes[i]);
        ImVec2 low_pos = ImPlot::PlotToPixels(dates[i], lows[i]);
        ImVec2 high_pos = ImPlot::PlotToPixels(dates[i], highs[i]);

        ImU32 color =
            ImGui::GetColorU32(opens[i] > closes[i] ? ImVec4(1, 0.2f, 0.2f, 1)
                                                    : ImVec4(0.2f, 1, 0.4f, 1));

        draw_list->AddLine(low_pos, high_pos, color);
        draw_list->AddRectFilled(open_pos, close_pos, color);

        // Optional: Fit data
        ImPlot::FitPoint(ImPlotPoint(dates[i], lows[i]));
        ImPlot::FitPoint(ImPlotPoint(dates[i], highs[i]));
      }
      ImPlot::EndItem();
    }

    // Heatmap Layer (Orderflow)
    if (!inst.heatmap_prices.empty()) {
      // Optional: PlotHeatmap or PlotScatter for orderflow
      // ImPlot::PlotScatter("Heatmap", inst.heatmap_times.data(),
      // inst.heatmap_prices.data(), (int)inst.heatmap_times.size());
    }

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant
