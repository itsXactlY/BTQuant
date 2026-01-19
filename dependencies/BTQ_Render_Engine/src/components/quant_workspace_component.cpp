#include "../../include/components/quant_workspace_component.hpp"
#include "implot_internal.h"
#include <algorithm>
#include <iostream>
#include <vector>

namespace BTQuant {

QuantWorkspaceComponent::QuantWorkspaceComponent(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : UIComponent({0, 0}, {0, 0}), bridge_(bridge), processor_(processor) {

  // Ensure ImPlot context is created (must be called once)
  static bool implot_init = false;
  if (!implot_init) {
    ImPlot::CreateContext();
    implot_init = true;
  }

  // Create chart manager and indicator renderer
  chart_manager_ = std::make_unique<ChartManager>(bridge, processor);
  indicator_renderer_ = std::make_unique<IndicatorRenderer>(nullptr, processor);

  std::cout << "[QuantWorkspaceComponent] Enhanced version initialized"
            << std::endl;
}

void QuantWorkspaceComponent::initialize_vulkan_resources(VulkanCore *core) {
  indicator_renderer_->initialize_vulkan_resources();
}

void QuantWorkspaceComponent::update(float dt) { chart_manager_->update(); }

void QuantWorkspaceComponent::render_gui() {
  auto &instruments = bridge_->GetAllInstruments();
  std::lock_guard<std::mutex> lock(bridge_->GetMapMutex());

  // Render chart controls
  if (show_chart_controls_) {
    render_chart_controls();
  }

  // Render indicator selector
  if (show_indicator_selector_) {
    render_indicator_selector();
  }

  // Render all visible charts
  for (const auto &chart : chart_manager_->get_visible_charts()) {
    auto it = instruments.find(chart.symbol);
    if (it != instruments.end()) {
      bool open = true;
      ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);

      if (ImGui::Begin((chart.symbol + " - " +
                        std::to_string(static_cast<int>(chart.timeframe)))
                           .c_str(),
                       &open)) {
        render_instrument_chart(chart.chart_id, chart.symbol, *it->second,
                                chart.timeframe);
      }
      ImGui::End();

      if (!open) {
        chart_manager_->destroy_chart(chart.chart_id);
      }
    }
  }
}

void QuantWorkspaceComponent::render_chart_controls() {
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Chart Controls", &show_chart_controls_)) {
    // Timeframe selector
    const char *timeframes[] = {"1 Minute", "5 Minutes", "15 Minutes",
                                "1 Hour",   "4 Hours",   "1 Day"};
    int selected = static_cast<int>(selected_timeframe_);
    if (ImGui::Combo("Timeframe", &selected, timeframes,
                     IM_ARRAYSIZE(timeframes))) {
      selected_timeframe_ = static_cast<RenderEngine::TimeFrame>(selected);
    }

    ImGui::Separator();

    // Create chart button
    if (ImGui::Button("Create New Chart")) {
      // For now, create chart for first available symbol
      auto &instruments = bridge_->GetAllInstruments();
      if (!instruments.empty()) {
        chart_manager_->create_chart(instruments.begin()->first,
                                     selected_timeframe_);
      }
    }

    ImGui::Separator();

    // Chart list
    ImGui::Text("Active Charts: %zu", chart_manager_->get_charts().size());
    for (const auto &[id, chart] : chart_manager_->get_charts()) {
      std::string chart_label =
          chart.symbol + " (" +
          std::to_string(static_cast<int>(chart.timeframe)) + ")";
      if (ImGui::Checkbox(chart_label.c_str(),
                          &const_cast<ChartInstance &>(chart).visible)) {
        chart_manager_->toggle_chart_visibility(id);
      }
    }
  }
  ImGui::End();
}

void QuantWorkspaceComponent::render_indicator_selector() {
  ImGui::SetNextWindowPos(ImVec2(270, 10), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Indicator Selector", &show_indicator_selector_)) {
    // Default indicator configuration for all charts
    static IndicatorConfig global_config;

    ImGui::Text("Global Indicators");
    ImGui::Separator();

    ImGui::Checkbox("Show SMA 10", &global_config.show_sma_10);
    ImGui::Checkbox("Show SMA 20", &global_config.show_sma_20);
    ImGui::Checkbox("Show SMA 50", &global_config.show_sma_50);
    ImGui::Checkbox("Show EMA 10", &global_config.show_ema_10);
    ImGui::Checkbox("Show EMA 20", &global_config.show_ema_20);
    ImGui::Checkbox("Show EMA 50", &global_config.show_ema_50);
    ImGui::Checkbox("Show RSI", &global_config.show_rsi);
    ImGui::Checkbox("Show MACD", &global_config.show_macd);
    ImGui::Checkbox("Show Bollinger Bands", &global_config.show_bollinger);
    ImGui::Checkbox("Show Stochastic", &global_config.show_stochastic);

    // Apply to all charts
    if (ImGui::Button("Apply to All")) {
      for (const auto &chart : chart_manager_->get_charts()) {
        indicator_configs_[chart.first] = global_config;
      }
    }
  }
  ImGui::End();
}

void QuantWorkspaceComponent::render_instrument_chart(
    uint32_t chart_id, const std::string &symbol, const InstrumentStore &inst,
    RenderEngine::TimeFrame timeframe) {
  std::lock_guard<std::mutex> inst_lock(inst.data_mutex);

  if (inst.timestamps.empty()) {
    ImGui::Text("Initializing Stream for %s...", symbol.c_str());
    return;
  }

  // Get indicator configuration for this chart
  auto &indicator_config = indicator_configs_[chart_id];

  // Prepare indicator parameters
  std::vector<IndicatorParams> indicators;

  if (indicator_config.show_sma_10) {
    indicators.push_back({IndicatorType::SMA_10,
                          10,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.94f, 1.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_sma_20) {
    indicators.push_back({IndicatorType::SMA_20,
                          20,
                          0,
                          0,
                          2.0,
                          {1.0f, 0.0f, 0.3f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_sma_50) {
    indicators.push_back({IndicatorType::SMA_50,
                          50,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.5f, 0.5f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_ema_10) {
    indicators.push_back({IndicatorType::EMA_10,
                          10,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.5f, 0.5f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_ema_20) {
    indicators.push_back({IndicatorType::EMA_20,
                          20,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.0f, 0.5f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_ema_50) {
    indicators.push_back({IndicatorType::EMA_50,
                          50,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.5f, 0.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_rsi) {
    indicators.push_back({IndicatorType::RSI_14,
                          14,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.8f, 0.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_macd) {
    indicators.push_back({IndicatorType::MACD,
                          12,
                          26,
                          9,
                          2.0,
                          {0.8f, 0.4f, 0.0f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_bollinger) {
    indicators.push_back({IndicatorType::BOLLINGER_MID,
                          20,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.6f, 0.6f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_stochastic) {
    indicators.push_back({IndicatorType::STOCHASTIC_K,
                          14,
                          3,
                          0,
                          2.0,
                          {0.6f, 0.0f, 0.6f, 1.0f},
                          1.0f,
                          true});
  }

  if (indicator_config.show_waddah_explosion) {
    indicators.push_back({IndicatorType::WADDAH_ATTAR_EXPLOSION,
                          20,
                          40,
                          150,
                          2.0,
                          {1.0f, 1.0f, 0.0f, 1.0f},
                          1.5f,
                          true});
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
      ImPlot::PlotBars("VolProfile", vp_prices.data(), vp_volumes.data(),
                       (int)vp_prices.size(), 0.5, ImPlotBarsFlags_Horizontal);
    }

    // Plot indicators
    indicator_renderer_->render_indicators(inst.symbol_id, timeframe,
                                           indicators);

    ImPlot::EndPlot();
  }
  ImPlot::PopStyleColor();
}

void QuantWorkspaceComponent::clear_data() {
  chart_manager_.reset();
  indicator_renderer_.reset();
  indicator_configs_.clear();
}

} // namespace BTQuant
