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
                     "Candles: %zu | Last: %.2f",
                     chart.dates.size(),
                     chart.closes.empty() ? 0.0 : chart.closes.back());

  // Prepare indicators
  std::vector<IndicatorParams> indicators;
  if (indicator_config_.show_sma_10) {
    indicators.push_back({IndicatorType::SMA_10, 10, 0, 0, 2.0f,
                         {0.0f, 0.94f, 1.0f, 1.0f}, 1.0f, true});
  }
  if (indicator_config_.show_sma_20) {
    indicators.push_back({IndicatorType::SMA_20, 20, 0, 0, 2.0f,
                         {1.0f, 0.84f, 0.0f, 1.0f}, 1.0f, true});
  }
  if (indicator_config_.show_sma_50) {
    indicators.push_back({IndicatorType::SMA_50, 50, 0, 0, 2.0f,
                         {1.0f, 0.0f, 1.0f, 1.0f}, 1.0f, true});
  }
  if (indicator_config_.show_ema_10) {
    indicators.push_back({IndicatorType::EMA_10, 10, 0, 0, 2.0f,
                         {0.0f, 1.0f, 0.5f, 1.0f}, 1.0f, true});
  }
  if (indicator_config_.show_ema_20) {
    indicators.push_back({IndicatorType::EMA_20, 20, 0, 0, 2.0f,
                         {0.5f, 0.5f, 1.0f, 1.0f}, 1.0f, true});
  }
  if (indicator_config_.show_ema_50) {
    indicators.push_back({IndicatorType::EMA_50, 50, 0, 0, 2.0f,
                         {1.0f, 0.5f, 0.0f, 1.0f}, 1.0f, true});
  }
  if (indicator_config_.show_rsi) {
    indicators.push_back({IndicatorType::RSI, 14, 0, 0, 1.5f,
                         {1.0f, 0.8f, 0.0f, 1.0f}, 1.0f, false});
  }
  if (indicator_config_.show_macd) {
    indicators.push_back({IndicatorType::MACD, 12, 26, 9, 1.5f,
                         {0.8f, 0.2f, 0.8f, 1.0f}, 1.0f, false});
  }
  if (indicator_config_.show_bollinger) {
    indicators.push_back({IndicatorType::BOLLINGER_BANDS, 20, 0, 0, 1.0f,
                         {0.5f, 0.8f, 0.5f, 1.0f}, 1.0f, true});
  }

  // Render the chart using indicator renderer
  ImVec2 plot_size = ImVec2(ImGui::GetContentRegionAvail().x,
                           ImGui::GetContentRegionAvail().y - 20);

  if (ImPlot::BeginPlot(symbol_.c_str(), plot_size)) {
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_Time, 0);
    ImPlot::SetupAxisFormat(ImAxis_Y1, "%.2f");

    // Plot candlestick
    indicator_renderer_->render_candlestick(chart);

    // Plot indicators
    for (const auto& indicator : indicators) {
      indicator_renderer_->render_indicator(chart, indicator);
    }

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant