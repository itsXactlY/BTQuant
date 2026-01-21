#include "../../include/components/chart_panel.hpp"
#include "implot.h"
#include <cstring>

namespace BTQuant {

ChartPanel::ChartPanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    ChartManager *chart_manager)
    : PanelBase(config), bridge_(bridge), processor_(processor),
      chart_manager_(chart_manager) {
  indicator_renderer_ = new IndicatorRenderer(nullptr, processor_);
}

void ChartPanel::initialize() {
  // Create the chart
  auto id_opt = chart_manager_->getSymbolId(symbol_);
  uint32_t symbol_id = id_opt ? *id_opt : 10007; // Default BTC-USDT
  chart_id_ =
      chart_manager_->create_chart(symbol_, exchange_, symbol_id, timeframe_);
}

void ChartPanel::update(float dt) {
  // Chart manager handles updates
  (void)dt;
}

void ChartPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  // Get the chart instance
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it == charts.end()) {
    ImGui::Text("Chart not found");
    end_panel_window();
    return;
  }

  const ChartInstance &chart = it->second;

  // Render chart controls in a collapsible header
  if (ImGui::CollapsingHeader("Chart Controls",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    render_chart_controls();
  }

  if (ImGui::CollapsingHeader("Indicators", ImGuiTreeNodeFlags_DefaultOpen)) {
    render_indicator_selector();
  }

  // Render the chart
  render_instrument_chart(chart);

  end_panel_window();
}

void ChartPanel::set_symbol(const std::string &symbol,
                            const std::string &exchange) {
  symbol_ = symbol;
  exchange_ = exchange;
  // Recreate chart with new symbol
  if (chart_id_ != 0) {
    chart_manager_->destroy_chart(chart_id_);
  }
  initialize();
}

void ChartPanel::set_timeframe(RenderEngine::TimeFrame timeframe) {
  timeframe_ = timeframe;
  // Recreate chart with new timeframe
  if (chart_id_ != 0) {
    chart_manager_->destroy_chart(chart_id_);
  }
  initialize();
}

void ChartPanel::render_chart_controls() {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));

  // Symbol input
  static char symbol_input[32] = "BTC-USDT";
  std::strncpy(symbol_input, symbol_.c_str(), sizeof(symbol_input) - 1);
  if (ImGui::InputText("Symbol", symbol_input, sizeof(symbol_input))) {
    set_symbol(symbol_input, exchange_);
  }
  ImGui::SameLine();

  // Timeframe selector
  const char *timeframes[] = {"1 Min",  "5 Min",  "15 Min",
                              "1 Hour", "4 Hour", "1 Day"};
  int selected = static_cast<int>(timeframe_);
  if (ImGui::Combo("Timeframe", &selected, timeframes,
                   IM_ARRAYSIZE(timeframes))) {
    set_timeframe(static_cast<RenderEngine::TimeFrame>(selected));
  }

  ImGui::PopStyleVar();
}

void ChartPanel::render_indicator_selector() {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));

  ImGui::Checkbox("SMA 10", &indicator_config_.show_sma_10);
  ImGui::SameLine();
  ImGui::Checkbox("SMA 20", &indicator_config_.show_sma_20);
  ImGui::SameLine();
  ImGui::Checkbox("SMA 50", &indicator_config_.show_sma_50);

  ImGui::Checkbox("EMA 10", &indicator_config_.show_ema_10);
  ImGui::SameLine();
  ImGui::Checkbox("EMA 20", &indicator_config_.show_ema_20);
  ImGui::SameLine();
  ImGui::Checkbox("EMA 50", &indicator_config_.show_ema_50);

  ImGui::Checkbox("RSI", &indicator_config_.show_rsi);
  ImGui::SameLine();
  ImGui::Checkbox("MACD", &indicator_config_.show_macd);
  ImGui::SameLine();
  ImGui::Checkbox("Bollinger", &indicator_config_.show_bollinger);

  ImGui::PopStyleVar();
}

void ChartPanel::render_instrument_chart(const ChartInstance &chart) {
  if (chart.dates.empty()) {
    ImGui::Text("Loading chart data for %s...", symbol_.c_str());
    return;
  }

  // Diagnostic info
  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                     "Candles: %zu | Last: %.2f", chart.dates.size(),
                     chart.closes.empty() ? 0.0 : chart.closes.back());

  // Prepare indicators
  std::vector<IndicatorParams> indicators;
  if (indicator_config_.show_sma_10) {
    IndicatorParams params;
    params.type = IndicatorType::SMA_10;
    params.period1 = 10;
    params.line_width = 2.0f;
    params.color = {0.0f, 0.94f, 1.0f, 1.0f};
    params.visible = true;
    indicators.push_back(params);
  }
  if (indicator_config_.show_sma_20) {
    IndicatorParams params;
    params.type = IndicatorType::SMA_20;
    params.period1 = 20;
    params.line_width = 2.0f;
    params.color = {1.0f, 0.84f, 0.0f, 1.0f};
    params.visible = true;
    indicators.push_back(params);
  }
  if (indicator_config_.show_sma_50) {
    IndicatorParams params;
    params.type = IndicatorType::SMA_50;
    params.period1 = 50;
    params.line_width = 2.0f;
    params.color = {1.0f, 0.0f, 1.0f, 1.0f};
    params.visible = true;
    indicators.push_back(params);
  }
  if (indicator_config_.show_ema_10) {
    IndicatorParams params;
    params.type = IndicatorType::EMA_10;
    params.period1 = 10;
    params.line_width = 2.0f;
    params.color = {0.0f, 1.0f, 0.5f, 1.0f};
    params.visible = true;
    indicators.push_back(params);
  }
  if (indicator_config_.show_ema_20) {
    IndicatorParams params;
    params.type = IndicatorType::EMA_20;
    params.period1 = 20;
    params.line_width = 2.0f;
    params.color = {0.5f, 0.5f, 1.0f, 1.0f};
    params.visible = true;
    indicators.push_back(params);
  }

  // Neon Chart Styling
  ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0.05f, 0.05f, 0.05f, 1.0f));
  ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0.02f, 0.02f, 0.02f, 1.0f));
  ImPlot::PushStyleColor(ImPlotCol_PlotBorder, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
  ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(10, 10));

  if (ImPlot::BeginPlot("##Chart", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoTitle)) {
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_RangeFit);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxisLinks(ImAxis_Y1, nullptr, nullptr);

    double candle_width = 60.0; // Default 1 min
    if (timeframe_ == RenderEngine::TimeFrame::TF_5MIN)
      candle_width = 300.0;
    else if (timeframe_ == RenderEngine::TimeFrame::TF_15MIN)
      candle_width = 900.0;
    else if (timeframe_ == RenderEngine::TimeFrame::TF_1HOUR)
      candle_width = 3600.0;
    else if (timeframe_ == RenderEngine::TimeFrame::TF_4HOUR)
      candle_width = 14400.0;
    else if (timeframe_ == RenderEngine::TimeFrame::TF_1DAY)
      candle_width = 86400.0;

    // Candlestick rendering
    std::vector<double> up_wick_x, up_wick_y;
    std::vector<double> down_wick_x, down_wick_y;
    std::vector<double> up_b_x, up_b_y1, up_b_y2;
    std::vector<double> down_b_x, down_b_y1, down_b_y2;

    for (size_t i = 0; i < chart.dates.size(); ++i) {
      double x = chart.dates[i];
      if (x == 0)
        continue;

      if (chart.closes[i] >= chart.opens[i]) {
        // Bullish Wick
        up_wick_x.push_back(x);
        up_wick_x.push_back(x);
        up_wick_y.push_back(chart.highs[i]);
        up_wick_y.push_back(chart.lows[i]);

        // Bullish Body
        up_b_x.push_back(x);
        up_b_y1.push_back(chart.opens[i]);
        up_b_y2.push_back(chart.closes[i]);
      } else {
        // Bearish Wick
        down_wick_x.push_back(x);
        down_wick_x.push_back(x);
        down_wick_y.push_back(chart.highs[i]);
        down_wick_y.push_back(chart.lows[i]);

        // Bearish Body
        down_b_x.push_back(x);
        down_b_y1.push_back(chart.opens[i]);
        down_b_y2.push_back(chart.closes[i]);
      }
    }

    // Draw Wicks (Neon Style)
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.0f);
    if (!up_wick_x.empty()) {
      ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 0.4f, 1.0f));
      ImPlot::PlotLine("##UpWicks", up_wick_x.data(), up_wick_y.data(),
                       (int)up_wick_x.size(), ImPlotLineFlags_Segments);
    }
    if (!down_wick_x.empty()) {
      ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.2f, 1.0f));
      ImPlot::PlotLine("##DownWicks", down_wick_x.data(), down_wick_y.data(),
                       (int)down_wick_x.size(), ImPlotLineFlags_Segments);
    }
    ImPlot::PopStyleVar();

    // Draw Bodies (Using PlotBars for robustness and performance)
    if (!up_b_x.empty()) {
      ImPlot::SetNextFillStyle(ImVec4(0.0f, 1.0f, 0.4f, 0.6f));
      std::vector<double> up_body_heights;
      up_body_heights.reserve(up_b_x.size());
      for (size_t i = 0; i < up_b_x.size(); ++i) {
        up_body_heights.push_back(up_b_y2[i] - up_b_y1[i]);
      }
      ImPlot::PlotBars("##UpBodies", up_b_x.data(), up_body_heights.data(),
                       (int)up_b_x.size(), candle_width * 0.82, up_b_y1[0]);
    }
    if (!down_b_x.empty()) {
      ImPlot::SetNextFillStyle(ImVec4(1.0f, 0.0f, 0.2f, 0.6f));
      std::vector<double> down_body_heights;
      down_body_heights.reserve(down_b_x.size());
      for (size_t i = 0; i < down_b_x.size(); ++i) {
        down_body_heights.push_back(down_b_y2[i] - down_b_y1[i]);
      }
      ImPlot::PlotBars("##DownBodies", down_b_x.data(),
                       down_body_heights.data(), (int)down_b_x.size(),
                       candle_width * 0.82, down_b_y1[0]);
    }

    // Render Indicators
    if (!indicators.empty()) {
      indicator_renderer_->render_indicators(symbol_, timeframe_, indicators);
    }

    ImPlot::EndPlot();
  }

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor(3);
}

} // namespace BTQuant