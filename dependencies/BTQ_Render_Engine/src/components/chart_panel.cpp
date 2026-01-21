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
  (void)dt; // Suppress unused parameter warning
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

  // Timeframe selector with all supported timeframes
  const char *timeframes[] = {"1 Min",  "5 Min",  "15 Min", "1 Hour",
                              "4 Hour", "1 Day",  "1 Sec",  "5 Sec",
                              "15 Sec", "30 Sec", "500ms",  "100ms"};
  static int selected = 0;
  if (ImGui::Combo("Timeframe", &selected, timeframes,
                   IM_ARRAYSIZE(timeframes))) {
    // Map combo index to TimeFrame enum
    RenderEngine::TimeFrame tf;
    switch (selected) {
    case 0:
      tf = RenderEngine::TimeFrame::TF_1MIN;
      break;
    case 1:
      tf = RenderEngine::TimeFrame::TF_5MIN;
      break;
    case 2:
      tf = RenderEngine::TimeFrame::TF_15MIN;
      break;
    case 3:
      tf = RenderEngine::TimeFrame::TF_1HOUR;
      break;
    case 4:
      tf = RenderEngine::TimeFrame::TF_4HOUR;
      break;
    case 5:
      tf = RenderEngine::TimeFrame::TF_1DAY;
      break;
    case 6:
      tf = RenderEngine::TimeFrame::TF_1SEC;
      break;
    case 7:
      tf = RenderEngine::TimeFrame::TF_5SEC;
      break;
    case 8:
      tf = RenderEngine::TimeFrame::TF_15SEC;
      break;
    case 9:
      tf = RenderEngine::TimeFrame::TF_30SEC;
      break;
    case 10:
      tf = RenderEngine::TimeFrame::TF_500MS;
      break;
    case 11:
      tf = RenderEngine::TimeFrame::TF_100MS;
      break;
    default:
      tf = RenderEngine::TimeFrame::TF_1MIN;
      break;
    }
    set_timeframe(tf);
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

  // Prepare indicators list
  std::vector<IndicatorParams> indicators;

  if (indicator_config_.show_sma_10) {
    IndicatorParams p;
    p.type = IndicatorType::SMA_10;
    p.period1 = 10;
    p.color = ImVec4(0.0f, 0.94f, 1.0f, 1.0f); // Cyan
    p.visible = true;
    indicators.push_back(p);
  }
  if (indicator_config_.show_sma_20) {
    IndicatorParams p;
    p.type = IndicatorType::SMA_20;
    p.period1 = 20;
    p.color = ImVec4(1.0f, 0.84f, 0.0f, 1.0f); // Gold
    p.visible = true;
    indicators.push_back(p);
  }
  if (indicator_config_.show_sma_50) {
    IndicatorParams p;
    p.type = IndicatorType::SMA_50;
    p.period1 = 50;
    p.color = ImVec4(1.0f, 0.0f, 1.0f, 1.0f); // Magenta
    p.visible = true;
    indicators.push_back(p);
  }
  if (indicator_config_.show_ema_10) {
    IndicatorParams p;
    p.type = IndicatorType::EMA_10;
    p.period1 = 10;
    p.color = ImVec4(0.0f, 1.0f, 0.5f, 1.0f);
    p.visible = true;
    indicators.push_back(p);
  }
  if (indicator_config_.show_ema_20) {
    IndicatorParams p;
    p.type = IndicatorType::EMA_20;
    p.period1 = 20;
    p.color = ImVec4(0.5f, 0.5f, 1.0f, 1.0f);
    p.visible = true;
    indicators.push_back(p);
  }
  if (indicator_config_.show_ema_50) {
    IndicatorParams p;
    p.type = IndicatorType::EMA_50;
    p.period1 = 50;
    p.color = ImVec4(1.0f, 0.5f, 0.0f, 1.0f);
    p.visible = true;
    indicators.push_back(p);
  }
  if (indicator_config_.show_rsi) {
    IndicatorParams p;
    p.type = IndicatorType::RSI_14;
    p.period1 = 14;
    p.color = ImVec4(1.0f, 0.8f, 0.0f, 1.0f);
    p.visible = true;
    indicators.push_back(p);
  }
  if (indicator_config_.show_macd) {
    IndicatorParams p;
    p.type = IndicatorType::MACD;
    p.period1 = 12;
    p.period2 = 26;
    p.period3 = 9;
    p.color = ImVec4(0.8f, 0.2f, 0.8f, 1.0f);
    p.visible = true;
    indicators.push_back(p);
  }
  if (indicator_config_.show_bollinger) {
    IndicatorParams p;
    p.type = IndicatorType::BOLLINGER_MID;
    p.period1 = 20;
    p.std_dev = 2.0;
    p.color = ImVec4(0.5f, 0.8f, 0.5f, 1.0f);
    p.visible = true;
    indicators.push_back(p);
  }

  // Render the chart
  ImVec2 plot_size = ImVec2(ImGui::GetContentRegionAvail().x,
                            ImGui::GetContentRegionAvail().y - 20);

  if (ImPlot::BeginPlot(symbol_.c_str(), plot_size)) {
    // Setup axes with time formatting
    ImPlot::SetupAxis(ImAxis_X1, "Time", ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);
    ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisFormat(ImAxis_Y1, "%.2f");

    int count = static_cast<int>(chart.dates.size());
    if (count > 0) {
      // Calculate candle width based on timeframe
      double candle_width = 30.0;
      switch (timeframe_) {
      case RenderEngine::TimeFrame::TF_100MS:
        candle_width = 0.08;
        break;
      case RenderEngine::TimeFrame::TF_500MS:
        candle_width = 0.4;
        break;
      case RenderEngine::TimeFrame::TF_1SEC:
        candle_width = 0.8;
        break;
      case RenderEngine::TimeFrame::TF_5SEC:
        candle_width = 4.0;
        break;
      case RenderEngine::TimeFrame::TF_15SEC:
        candle_width = 12.0;
        break;
      case RenderEngine::TimeFrame::TF_30SEC:
        candle_width = 24.0;
        break;
      case RenderEngine::TimeFrame::TF_1MIN:
        candle_width = 45.0;
        break;
      case RenderEngine::TimeFrame::TF_5MIN:
        candle_width = 225.0;
        break;
      case RenderEngine::TimeFrame::TF_15MIN:
        candle_width = 675.0;
        break;
      case RenderEngine::TimeFrame::TF_1HOUR:
        candle_width = 2700.0;
        break;
      case RenderEngine::TimeFrame::TF_4HOUR:
        candle_width = 10800.0;
        break;
      case RenderEngine::TimeFrame::TF_1DAY:
        candle_width = 64800.0;
        break;
      }

      // Separate bullish and bearish candles
      std::vector<double> bull_x, bull_open, bull_close, bull_high, bull_low;
      std::vector<double> bear_x, bear_open, bear_close, bear_high, bear_low;

      for (int i = 0; i < count; ++i) {
        if (chart.closes[i] >= chart.opens[i]) {
          bull_x.push_back(chart.dates[i]);
          bull_open.push_back(chart.opens[i]);
          bull_close.push_back(chart.closes[i]);
          bull_high.push_back(chart.highs[i]);
          bull_low.push_back(chart.lows[i]);
        } else {
          bear_x.push_back(chart.dates[i]);
          bear_open.push_back(chart.opens[i]);
          bear_close.push_back(chart.closes[i]);
          bear_high.push_back(chart.highs[i]);
          bear_low.push_back(chart.lows[i]);
        }
      }

      // Draw bullish wicks and bodies (green)
      if (!bull_x.empty()) {
        int n = static_cast<int>(bull_x.size());
        // Wicks
        std::vector<double> wick_x, wick_y;
        for (int i = 0; i < n; ++i) {
          wick_x.push_back(bull_x[i]);
          wick_x.push_back(bull_x[i]);
          wick_y.push_back(bull_low[i]);
          wick_y.push_back(bull_high[i]);
        }
        ImPlot::SetNextLineStyle(ImVec4(0.0f, 1.0f, 0.4f, 1.0f), 1.0f);
        ImPlot::PlotLine("##BullWick", wick_x.data(), wick_y.data(),
                         static_cast<int>(wick_x.size()),
                         ImPlotLineFlags_Segments);

        // Bodies - using filled rects
        std::vector<double> body_h;
        for (int i = 0; i < n; ++i) {
          body_h.push_back(bull_close[i] - bull_open[i]);
        }
        ImPlot::SetNextFillStyle(ImVec4(0.0f, 1.0f, 0.4f, 0.7f));
        ImPlot::PlotBars("##BullBody", bull_x.data(), body_h.data(), n,
                         candle_width * 0.8, ImPlotBarsFlags_None, 0,
                         sizeof(double));
      }

      // Draw bearish wicks and bodies (red)
      if (!bear_x.empty()) {
        int n = static_cast<int>(bear_x.size());
        // Wicks
        std::vector<double> wick_x, wick_y;
        for (int i = 0; i < n; ++i) {
          wick_x.push_back(bear_x[i]);
          wick_x.push_back(bear_x[i]);
          wick_y.push_back(bear_low[i]);
          wick_y.push_back(bear_high[i]);
        }
        ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.2f, 1.0f), 1.0f);
        ImPlot::PlotLine("##BearWick", wick_x.data(), wick_y.data(),
                         static_cast<int>(wick_x.size()),
                         ImPlotLineFlags_Segments);

        // Bodies
        std::vector<double> body_h;
        for (int i = 0; i < n; ++i) {
          body_h.push_back(bear_close[i] - bear_open[i]); // negative
        }
        ImPlot::SetNextFillStyle(ImVec4(1.0f, 0.0f, 0.2f, 0.7f));
        ImPlot::PlotBars("##BearBody", bear_x.data(), body_h.data(), n,
                         candle_width * 0.8, ImPlotBarsFlags_None, 0,
                         sizeof(double));
      }
    }

    // Render indicators using the IndicatorRenderer
    indicator_renderer_->render_indicators(symbol_, timeframe_, indicators);

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant