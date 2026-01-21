#include "../../include/components/chart_panel.hpp"
#include "implot.h"
#include <algorithm>
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

  // Render the chart (no indicators)
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

  // Timeframe selector (1ms-15sec only)
  const char *timeframes[] = {"1ms", "10ms", "100ms", "500ms",
                              "1s",  "3s",   "5s",    "15s"};
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

  // Diagnostic info & Controls
  ImGui::Checkbox("Auto-follow", &follow_latest_);
  ImGui::SameLine();
  ImGui::TextColored(
      ImVec4(0.0f, 1.0f, 0.8f, 1.0f),
      " | Candles: %zu | Last: %.2f | TF: %.4gs", chart.dates.size(),
      chart.closes.empty() ? 0.0f : chart.closes.back(),
      RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) /
          1000000.0);

  // Neon Chart Styling
  ImPlot::PushStyleColor(ImPlotCol_FrameBg, ImVec4(0.05f, 0.05f, 0.05f, 1.0f));
  ImPlot::PushStyleColor(ImPlotCol_PlotBg, ImVec4(0.02f, 0.02f, 0.02f, 1.0f));
  ImPlot::PushStyleColor(ImPlotCol_PlotBorder, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
  ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(10, 10));

  if (ImPlot::BeginPlot("##Chart", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoTitle |
                            ImPlotFlags_Crosshairs)) {

    // Setup Axes
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);

    // Auto-follow logic
    if (follow_latest_) {
      double time_max = chart.dates.back();
      // Show last 300 units (seconds)
      double duration_sec =
          RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) /
          1000000.0;
      double window_size = std::max(duration_sec * 300.0, 10.0); // At least 10s
      ImPlot::SetupAxisLimits(ImAxis_X1, time_max - window_size,
                              time_max + window_size * 0.05, ImPlotCond_Always);
      ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 0, ImPlotCond_Always); // Auto-fit Y
    } else {
      // Manual mode: use Cond_Once for initial view
      double time_min = chart.dates.front();
      double time_max = chart.dates.back();
      ImPlot::SetupAxisLimits(ImAxis_X1, time_min, time_max, ImPlotCond_Once);
      ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 0, ImPlotCond_Once);
    }

    // Get current visible axis limits for viewport culling
    ImPlotRect limits = ImPlot::GetPlotLimits();
    double view_x_min = limits.X.Min;
    double view_x_max = limits.X.Max;

    // Binary search for visible range
    size_t start_idx = 0;
    size_t end_idx = chart.dates.size();

    auto lower =
        std::lower_bound(chart.dates.begin(), chart.dates.end(), view_x_min);
    if (lower != chart.dates.begin())
      --lower;
    start_idx = std::distance(chart.dates.begin(), lower);

    auto upper =
        std::upper_bound(chart.dates.begin(), chart.dates.end(), view_x_max);
    if (upper != chart.dates.end())
      ++upper;
    end_idx = std::distance(chart.dates.begin(), upper);

    end_idx = std::min(end_idx, chart.dates.size());

    // Calculate candle width based on timeframe
    double duration_sec =
        RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) /
        1000000.0;
    double candle_half_width = duration_sec * 0.4;

    // Safeguard for very small TFs
    if (candle_half_width < 0.000001)
      candle_half_width = 0.000001;

    ImDrawList *draw_list = ImPlot::GetPlotDrawList();

    // Draw ONLY visible candles
    for (size_t i = start_idx; i < end_idx; ++i) {
      double x = chart.dates[i];
      if (x == 0)
        continue;

      float open = chart.opens[i];
      float high = chart.highs[i];
      float low = chart.lows[i];
      float close = chart.closes[i];

      bool bullish = close >= open;
      ImU32 color =
          bullish ? IM_COL32(0, 255, 100, 200) : IM_COL32(255, 50, 80, 200);
      ImU32 wick_color =
          bullish ? IM_COL32(0, 255, 100, 255) : IM_COL32(255, 50, 80, 255);

      // Transform to screen coordinates
      ImVec2 wick_top = ImPlot::PlotToPixels(x, high);
      ImVec2 wick_bot = ImPlot::PlotToPixels(x, low);
      ImVec2 body_tl =
          ImPlot::PlotToPixels(x - candle_half_width, bullish ? close : open);
      ImVec2 body_br =
          ImPlot::PlotToPixels(x + candle_half_width, bullish ? open : close);

      // Draw wick (vertical line)
      draw_list->AddLine(wick_top, wick_bot, wick_color, 1.0f);

      // Draw body (filled rectangle)
      draw_list->AddRectFilled(body_tl, body_br, color);
    }

    ImPlot::EndPlot();
  }

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor(3);
}

} // namespace BTQuant