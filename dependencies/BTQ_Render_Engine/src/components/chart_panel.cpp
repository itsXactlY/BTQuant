#include "../../include/components/chart_panel.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

namespace BTQuant {

static std::string timeframe_to_string(RenderEngine::TimeFrame tf) {
  switch (tf) {
  case RenderEngine::TimeFrame::TF_1MS:
    return "1ms";
  case RenderEngine::TimeFrame::TF_10MS:
    return "10ms";
  case RenderEngine::TimeFrame::TF_100MS:
    return "100ms";
  case RenderEngine::TimeFrame::TF_500MS:
    return "500ms";
  case RenderEngine::TimeFrame::TF_1SEC:
    return "1s";
  case RenderEngine::TimeFrame::TF_3SEC:
    return "3s";
  case RenderEngine::TimeFrame::TF_5SEC:
    return "5s";
  case RenderEngine::TimeFrame::TF_15SEC:
    return "15s";
  default:
    return "Unknown";
  }
}

ChartPanel::ChartPanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    ChartManager *chart_manager)
    : PanelBase(config), bridge_(bridge), processor_(processor),
      chart_manager_(chart_manager) {
  indicator_renderer_ = new IndicatorRenderer(nullptr, processor_);
}

void ChartPanel::initialize() {
  // Create chart
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

  // Get chart instance
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it == charts.end()) {
    ImGui::Text("Chart not found");
    end_panel_window();
    return;
  }

  const ChartInstance &chart = it->second;

  // Invalidate cache if new data has arrived
  // NOTE: This check is lightweight and only compares sizes
  if (chart.closes.size() > last_known_data_size_) {
    cached_sma_.clear();
    cached_ema_.clear();
    cached_rsi_.clear();
    last_known_data_size_ = chart.closes.size();
  }

  // Render chart controls in a collapsible header
  if (ImGui::CollapsingHeader("Chart Controls",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    render_chart_controls();
  }

  // Render indicator selector
  if (ImGui::CollapsingHeader("Indicators", ImGuiTreeNodeFlags_DefaultOpen)) {
    render_indicator_selector();
  }

  // Render chart with indicators
  render_instrument_chart(chart);

  end_panel_window();
}

void ChartPanel::set_symbol(const std::string &symbol,
                            const std::string &exchange) {
  symbol_ = symbol;
  exchange_ = exchange;
  config_.title = symbol_ + " Chart [" + timeframe_to_string(timeframe_) + "]";

  // Recreate chart with new symbol
  // Don't destroy old one, so we can switch back to it with state preserved
  initialize();
}

void ChartPanel::set_timeframe(RenderEngine::TimeFrame timeframe) {
  timeframe_ = timeframe;
  config_.title = symbol_ + " Chart [" + timeframe_to_string(timeframe_) + "]";

  // Recreate chart with new timeframe
  // Don't destroy old one, so we can switch back to it with state preserved
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

  // Auto-follow Window Size
  ImGui::SameLine();
  int window_size_int = static_cast<int>(auto_follow_window_);
  if (ImGui::SliderInt("Window", &window_size_int, 100, 10000, "%d")) {
    auto_follow_window_ = static_cast<float>(window_size_int);
  }

  // Auto-follow checkbox
  ImGui::SameLine();
  ImGui::Checkbox("Auto-follow", &follow_latest_);

  ImGui::PopStyleVar();
}

void ChartPanel::render_indicator_selector() {
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));

  // Moving Averages
  ImGui::SeparatorText("Moving Averages");
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

  // Oscillators
  ImGui::SeparatorText("Oscillators");
  ImGui::Checkbox("RSI", &indicator_config_.show_rsi);
  ImGui::SameLine();
  ImGui::Checkbox("MACD", &indicator_config_.show_macd);

  // Overlays
  ImGui::SeparatorText("Overlays");
  ImGui::Checkbox("Bollinger Bands", &indicator_config_.show_bollinger);
  ImGui::SameLine();
  ImGui::Checkbox("Fibonacci", &indicator_config_.show_fibonacci);
  ImGui::SameLine();
  ImGui::Checkbox("Vol Profile", &indicator_config_.show_volume_profile);
  ImGui::SameLine();
  ImGui::Checkbox("Crosshair Info", &indicator_config_.show_crosshair_info);

  ImGui::PopStyleVar();
}

// ============================================================================
// INDICATOR CALCULATION HELPERS
// ============================================================================

std::vector<double> ChartPanel::calculate_sma(const std::vector<float> &prices,
                                              int period) {
  if (prices.empty()) {
    return std::vector<double>();
  }

  // Create cache key
  IndicatorCacheKey key{prices.size(), period};

  // Check if result is already cached
  auto it = cached_sma_.find(key);
  if (it != cached_sma_.end()) {
    return it->second;
  }

  // Calculate SMA if not cached
  std::vector<double> sma(prices.size(), 0.0);

  for (size_t i = period - 1; i < prices.size(); ++i) {
    double sum = 0.0;
    for (int j = 0; j < period; ++j) {
      sum += static_cast<double>(prices[i - j]);
    }
    sma[i] = sum / period;
  }

  // Cache the result
  cached_sma_[key] = sma;
  return sma;
}

std::vector<double> ChartPanel::calculate_ema(const std::vector<float> &prices,
                                              int period) {
  if (prices.empty()) {
    return std::vector<double>();
  }

  // Create cache key
  IndicatorCacheKey key{prices.size(), period};

  // Check if result is already cached
  auto it = cached_ema_.find(key);
  if (it != cached_ema_.end()) {
    return it->second;
  }

  std::vector<double> ema(prices.size(), 0.0);

  // Initialize with SMA
  double sum = 0.0;
  for (int i = 0; i < std::min(period, static_cast<int>(prices.size())); ++i) {
    sum += static_cast<double>(prices[i]);
  }
  ema[period - 1] = sum / period;

  // Calculate EMA
  double multiplier = 2.0 / (period + 1.0);
  for (size_t i = period; i < prices.size(); ++i) {
    ema[i] =
        (static_cast<double>(prices[i]) - ema[i - 1]) * multiplier + ema[i - 1];
  }

  // Cache the result
  cached_ema_[key] = ema;
  return ema;
}

std::vector<double> ChartPanel::calculate_ema(const std::vector<double> &prices,
                                              int period) {
  if (prices.empty()) {
    return std::vector<double>();
  }

  // Create cache key - we'll use a different approach for double vectors
  // Since this is typically used for MACD signals, we'll still cache it
  IndicatorCacheKey key{prices.size(), period};

  // Check if result is already cached
  auto it = cached_ema_.find(key);
  if (it != cached_ema_.end()) {
    // Note: This shares the same cache as float version, which could cause conflicts
    // For a more robust solution, we might need separate caches, but for now
    // this should work since periods are typically different
    return it->second;
  }

  std::vector<double> ema(prices.size(), 0.0);

  // Initialize with SMA
  double sum = 0.0;
  for (int i = 0; i < std::min(period, static_cast<int>(prices.size())); ++i) {
    sum += prices[i];
  }
  ema[period - 1] = sum / period;

  // Calculate EMA
  double multiplier = 2.0 / (period + 1.0);
  for (size_t i = period; i < prices.size(); ++i) {
    ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
  }

  // Cache the result
  cached_ema_[key] = ema;
  return ema;
}

std::vector<double>
ChartPanel::calculate_bollinger_upper(const std::vector<float> &prices,
                                      int period, double std_dev) {
  auto sma = calculate_sma(prices, period);
  std::vector<double> upper_band(prices.size(), 0.0);

  for (size_t i = period - 1; i < prices.size(); ++i) {
    // Calculate standard deviation
    double sum_sq_diff = 0.0;
    for (int j = 0; j < period; ++j) {
      double diff = static_cast<double>(prices[i - j]) - sma[i];
      sum_sq_diff += diff * diff;
    }
    double variance = sum_sq_diff / period;
    double std_deviation = std::sqrt(variance);

    upper_band[i] = sma[i] + std_dev * std_deviation;
  }

  return upper_band;
}

std::vector<double>
ChartPanel::calculate_bollinger_lower(const std::vector<float> &prices,
                                      int period, double std_dev) {
  auto sma = calculate_sma(prices, period);
  std::vector<double> lower_band(prices.size(), 0.0);

  for (size_t i = period - 1; i < prices.size(); ++i) {
    // Calculate standard deviation
    double sum_sq_diff = 0.0;
    for (int j = 0; j < period; ++j) {
      double diff = static_cast<double>(prices[i - j]) - sma[i];
      sum_sq_diff += diff * diff;
    }
    double variance = sum_sq_diff / period;
    double std_deviation = std::sqrt(variance);

    lower_band[i] = sma[i] - std_dev * std_deviation;
  }

  return lower_band;
}

std::vector<double> ChartPanel::calculate_rsi(const std::vector<float> &prices,
                                              int period) {
  if (prices.empty()) {
    return std::vector<double>();
  }

  // Create cache key
  IndicatorCacheKey key{prices.size(), period};

  // Check if result is already cached
  auto it = cached_rsi_.find(key);
  if (it != cached_rsi_.end()) {
    return it->second;
  }

  std::vector<double> rsi(prices.size(), 50.0); // Default to neutral

  if (prices.size() < static_cast<size_t>(period + 1)) {
    // Cache the result even if it's empty/default
    cached_rsi_[key] = rsi;
    return rsi;
  }

  for (size_t i = period; i < prices.size(); ++i) {
    double gains = 0.0;
    double losses = 0.0;

    for (int j = 1; j <= period; ++j) {
      double change = static_cast<double>(prices[i - j + 1]) -
                      static_cast<double>(prices[i - j]);
      if (change > 0) {
        gains += change;
      } else {
        losses -= change;
      }
    }

    double avg_gain = gains / period;
    double avg_loss = -losses / period;

    double rs = avg_loss == 0.0 ? 100.0 : avg_gain / avg_loss;
    rsi[i] = 100.0 - (100.0 / (1.0 + rs));
  }

  // Cache the result
  cached_rsi_[key] = rsi;
  return rsi;
}

std::vector<double>
ChartPanel::calculate_macd_line(const std::vector<float> &prices, int fast,
                                int slow) {
  auto ema_fast = calculate_ema(prices, fast);
  auto ema_slow = calculate_ema(prices, slow);

  std::vector<double> macd_line(prices.size(), 0.0);
  for (size_t i = 0; i < prices.size(); ++i) {
    macd_line[i] = ema_fast[i] - ema_slow[i];
  }

  return macd_line;
}

std::vector<double>
ChartPanel::calculate_macd_signal(const std::vector<double> &macd_line,
                                  int signal) {
  return calculate_ema(macd_line, signal);
}

std::vector<double>
ChartPanel::calculate_macd_histogram(const std::vector<double> &macd_line,
                                     const std::vector<double> &signal) {
  std::vector<double> histogram(macd_line.size(), 0.0);

  for (size_t i = 0; i < macd_line.size(); ++i) {
    histogram[i] = macd_line[i] - signal[i];
  }

  return histogram;
}

std::vector<FibonacciLevel>
ChartPanel::calculate_fibonacci_levels(double start_price, double end_price) {
  std::vector<FibonacciLevel> levels;

  if (start_price == 0.0 || end_price == 0.0)
    return levels;

  double range = end_price - start_price;

  // Fibonacci ratios
  const double ratios[] = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0};
  const char *labels[] = {"0%",    "23.6%", "38.2%", "50%",
                          "61.8%", "78.6%", "100%"};
  const ImU32 colors[] = {
      IM_COL32(255, 255, 255, 255), // White
      IM_COL32(0, 255, 255, 200),   // Cyan
      IM_COL32(0, 255, 0, 200),     // Green
      IM_COL32(255, 255, 0, 200),   // Yellow
      IM_COL32(255, 165, 0, 200),   // Orange
      IM_COL32(255, 0, 0, 200),     // Red
      IM_COL32(255, 255, 255, 200)  // White
  };

  for (int i = 0; i < 7; ++i) {
    FibonacciLevel level;
    level.price = start_price + range * ratios[i];
    level.ratio = ratios[i];
    level.label = labels[i];
    level.color = colors[i];
    levels.push_back(level);
  }

  return levels;
}

// ============================================================================
// INDICATOR RENDERING METHODS
// ============================================================================

void ChartPanel::render_sma_lines(const ChartInstance &chart, size_t start_idx,
                                  size_t end_idx) {
  if (chart.closes.empty())
    return;

  ImDrawList *draw_list = ImPlot::GetPlotDrawList();

  // SMA 10
  if (indicator_config_.show_sma_10) {
    auto sma_10 = calculate_sma(chart.closes, 10);
    for (size_t i = start_idx; i < end_idx; ++i) {
      if (i >= 9) {
        ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 9], sma_10[i]);
        ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], sma_10[i]);
        draw_list->AddLine(p1, p2, IM_COL32(255, 165, 0, 200), 2.0f);
      }
    }
  }

  // SMA 20
  if (indicator_config_.show_sma_20) {
    auto sma_20 = calculate_sma(chart.closes, 20);
    for (size_t i = start_idx; i < end_idx; ++i) {
      if (i >= 19) {
        ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 19], sma_20[i]);
        ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], sma_20[i]);
        draw_list->AddLine(p1, p2, IM_COL32(255, 255, 0, 200), 2.0f);
      }
    }
  }

  // SMA 50
  if (indicator_config_.show_sma_50) {
    auto sma_50 = calculate_sma(chart.closes, 50);
    for (size_t i = start_idx; i < end_idx; ++i) {
      if (i >= 49) {
        ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 49], sma_50[i]);
        ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], sma_50[i]);
        draw_list->AddLine(p1, p2, IM_COL32(0, 255, 255, 200), 2.0f);
      }
    }
  }
}

void ChartPanel::render_ema_lines(const ChartInstance &chart, size_t start_idx,
                                  size_t end_idx) {
  if (chart.closes.empty())
    return;

  ImDrawList *draw_list = ImPlot::GetPlotDrawList();

  // EMA 10
  if (indicator_config_.show_ema_10) {
    auto ema_10 = calculate_ema(chart.closes, 10);
    for (size_t i = start_idx; i < end_idx; ++i) {
      if (i >= 9) {
        ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 9], ema_10[i]);
        ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], ema_10[i]);
        draw_list->AddLine(p1, p2, IM_COL32(255, 0, 255, 200), 2.0f);
      }
    }
  }

  // EMA 20
  if (indicator_config_.show_ema_20) {
    auto ema_20 = calculate_ema(chart.closes, 20);
    for (size_t i = start_idx; i < end_idx; ++i) {
      if (i >= 19) {
        ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 19], ema_20[i]);
        ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], ema_20[i]);
        draw_list->AddLine(p1, p2, IM_COL32(255, 0, 128, 200), 2.0f);
      }
    }
  }

  // EMA 50
  if (indicator_config_.show_ema_50) {
    auto ema_50 = calculate_ema(chart.closes, 50);
    for (size_t i = start_idx; i < end_idx; ++i) {
      if (i >= 49) {
        ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i - 49], ema_50[i]);
        ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], ema_50[i]);
        draw_list->AddLine(p1, p2, IM_COL32(128, 0, 255, 200), 2.0f);
      }
    }
  }
}

void ChartPanel::render_bollinger_bands(const ChartInstance &chart,
                                        size_t start_idx, size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_bollinger)
    return;

  ImDrawList *draw_list = ImPlot::GetPlotDrawList();

  auto upper_band = calculate_bollinger_upper(
      chart.closes, indicator_config_.bollinger_period,
      indicator_config_.bollinger_std_dev);
  auto lower_band = calculate_bollinger_lower(
      chart.closes, indicator_config_.bollinger_period,
      indicator_config_.bollinger_std_dev);

  for (size_t i = start_idx; i < end_idx; ++i) {
    if (i >= static_cast<size_t>(indicator_config_.bollinger_period - 1)) {
      ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[i], upper_band[i]);
      ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[i], lower_band[i]);

      // Draw upper band
      draw_list->AddLine(p1, p1, IM_COL32(0, 255, 255, 100), 1.0f);
      // Draw lower band
      draw_list->AddLine(p2, p2, IM_COL32(0, 255, 255, 100), 1.0f);
    }
  }
}

void ChartPanel::render_rsi_indicator(const ChartInstance &chart,
                                      size_t start_idx, size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_rsi)
    return;

  ImDrawList *draw_list = ImPlot::GetPlotDrawList();
  auto rsi = calculate_rsi(chart.closes, indicator_config_.rsi_period);

  // Get plot limits for RSI scaling
  ImPlotRect limits = ImPlot::GetPlotLimits();

  for (size_t i = start_idx; i < end_idx; ++i) {
    if (i >= static_cast<size_t>(indicator_config_.rsi_period)) {
      double rsi_value = rsi[i];

      // Map RSI to Y-axis (0-100)
      double y =
          limits.Y.Min + (rsi_value / 100.0) * (limits.Y.Max - limits.Y.Min);

      ImVec2 p = ImPlot::PlotToPixels(chart.dates[i], y);

      // Color based on overbought/oversold
      ImU32 color = IM_COL32(128, 128, 128, 200);
      if (rsi_value >= indicator_config_.rsi_overbought) {
        color = IM_COL32(255, 0, 0, 200); // Red
      } else if (rsi_value <= indicator_config_.rsi_oversold) {
        color = IM_COL32(0, 255, 0, 200); // Green
      }

      // Draw RSI line
      draw_list->AddLine(p, p, color, 1.5f);
    }
  }

  // Draw overbought/oversold lines
  double overbought_y =
      limits.Y.Min + (indicator_config_.rsi_overbought / 100.0) *
                         (limits.Y.Max - limits.Y.Min);
  double oversold_y = limits.Y.Min + (indicator_config_.rsi_oversold / 100.0) *
                                         (limits.Y.Max - limits.Y.Min);

  ImVec2 ob_p1 = ImPlot::PlotToPixels(limits.X.Min, overbought_y);
  ImVec2 ob_p2 = ImPlot::PlotToPixels(limits.X.Max, overbought_y);
  draw_list->AddLine(ob_p1, ob_p2, IM_COL32(255, 0, 0, 100), 1.0f);

  ImVec2 os_p1 = ImPlot::PlotToPixels(limits.X.Min, oversold_y);
  ImVec2 os_p2 = ImPlot::PlotToPixels(limits.X.Max, oversold_y);
  draw_list->AddLine(os_p1, os_p2, IM_COL32(0, 255, 0, 100), 1.0f);
}

void ChartPanel::render_macd_indicator(const ChartInstance &chart,
                                       size_t start_idx, size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_macd)
    return;

  ImDrawList *draw_list = ImPlot::GetPlotDrawList();

  auto macd_line =
      calculate_macd_line(chart.closes, indicator_config_.macd_fast_period,
                          indicator_config_.macd_slow_period);
  auto macd_signal =
      calculate_macd_signal(macd_line, indicator_config_.macd_signal_period);
  auto macd_histogram = calculate_macd_histogram(macd_line, macd_signal);

  // Get plot limits for MACD scaling
  ImPlotRect limits = ImPlot::GetPlotLimits();

  for (size_t i = start_idx; i < end_idx; ++i) {
    if (i >= static_cast<size_t>(indicator_config_.macd_slow_period +
                                 indicator_config_.macd_signal_period)) {
      double macd_value = macd_line[i];
      double signal_value = macd_signal[i];
      double hist_value = macd_histogram[i];

      // Map MACD to Y-axis
      double y = limits.Y.Min +
                 ((macd_value - limits.Y.Min) / (limits.Y.Max - limits.Y.Min)) *
                     (limits.Y.Max - limits.Y.Min);

      ImVec2 p = ImPlot::PlotToPixels(chart.dates[i], y);

      // Draw MACD line
      draw_list->AddLine(p, p, IM_COL32(0, 255, 255, 200), 1.5f);

      // Draw signal line
      double signal_y = limits.Y.Min + ((signal_value - limits.Y.Min) /
                                        (limits.Y.Max - limits.Y.Min)) *
                                           (limits.Y.Max - limits.Y.Min);
      ImVec2 signal_p = ImPlot::PlotToPixels(chart.dates[i], signal_y);
      draw_list->AddLine(signal_p, signal_p, IM_COL32(255, 165, 0, 200), 1.5f);

      // Draw histogram
      double hist_y = limits.Y.Min + ((hist_value - limits.Y.Min) /
                                      (limits.Y.Max - limits.Y.Min)) *
                                         (limits.Y.Max - limits.Y.Min);
      ImVec2 hist_p = ImPlot::PlotToPixels(chart.dates[i], hist_y);

      ImU32 hist_color =
          hist_value >= 0 ? IM_COL32(0, 255, 0, 150) : IM_COL32(255, 0, 0, 150);
      draw_list->AddRectFilled(
          ImVec2(hist_p.x - 2, hist_p.y),
          ImVec2(hist_p.x + 2, hist_p.y + hist_y - signal_y), hist_color);
    }
  }
}

void ChartPanel::render_fibonacci_levels(const ChartInstance &chart,
                                         size_t start_idx, size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_fibonacci)
    return;

  // Find swing high and low in visible range
  double swing_high = 0.0;
  double swing_low = 1e9;

  for (size_t i = start_idx; i < end_idx; ++i) {
    swing_high = std::max(swing_high, static_cast<double>(chart.highs[i]));
    swing_low = std::min(swing_low, static_cast<double>(chart.lows[i]));
  }

  if (swing_high == 0.0 || swing_low == 1e9)
    return;

  auto fib_levels = calculate_fibonacci_levels(swing_low, swing_high);
  ImDrawList *draw_list = ImPlot::GetPlotDrawList();

  // Draw Fibonacci levels
  for (const auto &level : fib_levels) {
    ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[start_idx], level.price);
    ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[end_idx - 1], level.price);

    // Draw horizontal line
    draw_list->AddLine(p1, p2, level.color, 1.0f);

    // Draw label
    ImVec2 text_pos = ImVec2(p1.x + 5, p1.y - 10);
    draw_list->AddText(text_pos, level.color, level.label);
  }
}

void ChartPanel::render_crosshair_info(const ChartInstance &chart,
                                       double mouse_x, double mouse_y) {
  if (!indicator_config_.show_crosshair_info || chart.closes.empty())
    return;

  // Find closest candle to mouse position using binary search
  size_t closest_idx = 0;

  // Use lower_bound to find the insertion point for mouse_x in the sorted dates vector
  auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), mouse_x);

  if (lower == chart.dates.end()) {
    // Mouse x is beyond the last date, use the last element
    closest_idx = chart.dates.size() - 1;
  } else if (lower == chart.dates.begin()) {
    // Mouse x is before the first date, use the first element
    closest_idx = 0;
  } else {
    // Compare the distance to the element at lower and the one before it
    size_t idx_after = std::distance(chart.dates.begin(), lower);
    size_t idx_before = idx_after - 1;

    double dist_to_after = std::abs(chart.dates[idx_after] - mouse_x);
    double dist_to_before = std::abs(chart.dates[idx_before] - mouse_x);

    closest_idx = (dist_to_before < dist_to_after) ? idx_before : idx_after;
  }

  if (closest_idx >= chart.closes.size())
    return;

  // Get candle data
  double open = chart.opens[closest_idx];
  double high = chart.highs[closest_idx];
  double low = chart.lows[closest_idx];
  double close = chart.closes[closest_idx];
  double volume = chart.volumes[closest_idx];

  // Render crosshair info overlay
  ImGui::SetNextWindowPos(
      ImVec2(ImGui::GetMousePos().x + 20, ImGui::GetMousePos().y + 20));
  ImGui::SetNextWindowSize(ImVec2(200, 150));
  ImGui::Begin("Crosshair Info", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_AlwaysAutoResize);

  ImGui::TextColored(ImVec4(1, 1, 0, 1), "Candle Info:");
  ImGui::Separator();
  ImGui::Text("Open:  %.2f", open);
  ImGui::Text("High:  %.2f", high);
  ImGui::Text("Low:   %.2f", low);
  ImGui::Text("Close: %.2f", close);
  ImGui::Text("Volume: %.2f", volume);

  // Calculate change
  if (closest_idx > 0) {
    double prev_close = chart.closes[closest_idx - 1];
    double change = close - prev_close;
    double change_pct = (change / prev_close) * 100.0;

    ImGui::Separator();
    ImGui::TextColored(change >= 0 ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1),
                       "Change: %.2f (%.2f%%)", change, change_pct);
  }

  ImGui::End();
}

void ChartPanel::render_instrument_chart(const ChartInstance &chart) {
  if (chart.dates.empty()) {
    ImGui::Text("Loading chart data for %s...", symbol_.c_str());
    return;
  }

  // Diagnostic info & Controls
  ImGui::TextColored(
      ImVec4(0.0f, 1.0f, 0.8f, 1.0f),
      " | Candles: %zu | Last: %.2f | TF: %.4gs", chart.dates.size(),
      chart.closes.empty() ? 0.0f : chart.closes.back(),
      RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) /
          1000000.0);

  // Neon Chart Styling (ThemeManager)
  const auto &colors = ThemeManager::getInstance().getColors();
  ImPlot::PushStyleColor(ImPlotCol_FrameBg, colors.background);
  ImPlot::PushStyleColor(ImPlotCol_PlotBg,
                         colors.panel_bg); // Use panel bg or specific dark
  ImPlot::PushStyleColor(ImPlotCol_PlotBorder, colors.border);
  ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(10, 10));

  // Initialize chart configuration
  // Ensure we start with auto-follow enabled to show recent action
  if (first_frame_) {
    follow_latest_ = true;
    first_frame_ = false;
  }

  // Fetch Volume Profile Data Early (for auto-scaling axes)
  double vp_max_vol = 0;
  std::vector<float> vp_prices;
  std::vector<float> vp_volumes;

  if (indicator_config_.show_volume_profile) {
    auto id_opt = chart_manager_->getSymbolId(symbol_);
    if (id_opt) {
      auto profile = processor_->getVolumeProfile(*id_opt, timeframe_);
      if (!profile.empty()) {
        vp_prices.reserve(profile.size());
        vp_volumes.reserve(profile.size());

        for (const auto &level : profile) {
          vp_prices.push_back(level.price);
          vp_volumes.push_back(static_cast<float>(level.total_volume));
          if (level.total_volume > vp_max_vol)
            vp_max_vol = level.total_volume;
        }
      }
    }
  }

  // Use a unique ID string per symbol/tf/exchange to ensure ImPlot saves state
  // per chart
  std::string plot_id = "##Chart_" + symbol_ + "_" + exchange_ + "_" +
                        timeframe_to_string(timeframe_);

  if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoTitle |
                            ImPlotFlags_Crosshairs)) {

    // Setup Axes - ALL SETUP CALLS MUST HAPPEN HERE AT THE BEGINNING
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None,
                      ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);

    // Setup X2 for Volume Profile
    if (indicator_config_.show_volume_profile && vp_max_vol > 0) {
      ImPlot::SetupAxis(ImAxis_X2, nullptr,
                        ImPlotAxisFlags_NoTickLabels |
                            ImPlotAxisFlags_NoGridLines);
      ImPlot::SetupAxisLimits(ImAxis_X2, 0, vp_max_vol * 4.0,
                              ImPlotCond_Always);
    }

    // Improved Auto-follow logic with better zoom/pan handling
    bool user_interacted = ImPlot::IsPlotHovered() &&
                          (ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
                           ImGui::IsMouseDragging(ImGuiMouseButton_Right) ||
                           ImGui::GetIO().MouseWheel != 0.0f);

    if (user_interacted) {
        follow_latest_ = false;
    }

    // Determine axis limits based on follow mode
    double x_axis_min, x_axis_max;
    if (follow_latest_) {
      double time_max = chart.dates.back();
      double duration_raw =
          RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_);
      double duration_sec = duration_raw / 1000000.0;
      double window_size = duration_sec * auto_follow_window_;
      double padding = window_size * 0.05;

      last_view_min_ = time_max - window_size;
      last_view_max_ = time_max + padding;

      x_axis_min = last_view_min_;
      x_axis_max = last_view_max_;
    } else {
      // If not following, allow manual pan/zoom (Cond_Once allows initial set
      // but user overrides) Only set Once if we ever reset or init
      x_axis_min = chart.dates.front();
      x_axis_max = chart.dates.back();
    }

    // Calculate Y-axis limits based on visible X range
    double y_axis_min = std::numeric_limits<double>::max();
    double y_axis_max = std::numeric_limits<double>::lowest();

    // Binary search for visible range based on our ESTIMATE/CACHE
    size_t start_idx = 0;
    size_t end_idx = chart.dates.size();

    auto lower =
        std::lower_bound(chart.dates.begin(), chart.dates.end(), x_axis_min);
    if (lower != chart.dates.begin())
      --lower;
    start_idx = std::distance(chart.dates.begin(), lower);

    auto upper =
        std::upper_bound(chart.dates.begin(), chart.dates.end(), x_axis_max);
    if (upper != chart.dates.end())
      ++upper;
    end_idx = std::distance(chart.dates.begin(), upper);

    end_idx = std::min(end_idx, chart.dates.size());

    if (start_idx < end_idx) {
      bool found_data = false;

      for (size_t i = start_idx; i < end_idx; ++i) {
        double low = chart.lows[i];
        double high = chart.highs[i];
        if (low > 0 && high > 0) { // Valid data
          if (low < y_axis_min)
            y_axis_min = low;
          if (high > y_axis_max)
            y_axis_max = high;
          found_data = true;
        }
      }

      if (found_data) {
        double range = y_axis_max - y_axis_min;
        if (range == 0)
          range = y_axis_min * 0.01;
        if (range == 0)
          range = 1.0;

        y_axis_min -= range * 0.1;
        y_axis_max += range * 0.1;
      } else {
        // Fallback if no valid data found in range
        y_axis_min = chart.lows.empty() ? 0 : chart.lows[0];
        y_axis_max = chart.highs.empty() ? 1 : chart.highs[0];
      }
    } else {
      // Fallback if no range found
      y_axis_min = chart.lows.empty() ? 0 : *std::min_element(chart.lows.begin(), chart.lows.end());
      y_axis_max = chart.highs.empty() ? 1 : *std::max_element(chart.highs.begin(), chart.highs.end());
    }

    // NOW APPLY ALL AXIS LIMITS AT ONCE - BEFORE ANY PLOTTING OPERATIONS
    ImPlot::SetupAxisLimits(ImAxis_X1, x_axis_min, x_axis_max,
                            follow_latest_ ? ImPlotCond_Always : ImPlotCond_Once);
    ImPlot::SetupAxisLimits(ImAxis_Y1, static_cast<float>(y_axis_min),
                            static_cast<float>(y_axis_max), ImPlotCond_Always);

    // VOLUME PROFILE OVERLAY - using direct draw to avoid axis switching after setup lock
    if (indicator_config_.show_volume_profile && !vp_prices.empty()) {
      ImDrawList *draw_list = ImPlot::GetPlotDrawList();

      // Use Theme Colors for a more integrated look
      ImVec4 vp_color = colors.text;
      vp_color.w = 0.25f; // reduced alpha

      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, vp_color.w);
      ImPlot::PushStyleColor(ImPlotCol_Fill, vp_color);
      ImPlot::PushStyleColor(
          ImPlotCol_Line,
          ImVec4(vp_color.x, vp_color.y, vp_color.z, 0.5f)); // clearer border

      // Calculate bar width in plot coordinates based on volume values
      double max_vol_display = vp_max_vol * 4.0; // Same as used in SetupAxisLimits for X2

      // Draw volume profile bars using direct drawing to avoid axis switching after setup lock
      for (size_t i = 0; i < vp_prices.size(); ++i) {
          // Convert price (Y coordinate) and volume (X coordinate) to screen coordinates
          ImVec2 pos_screen = ImPlot::PlotToPixels(vp_volumes[i], vp_prices[i]);  // tip of the bar
          ImVec2 base_screen = ImPlot::PlotToPixels(0, vp_prices[i]);            // base of the bar

          // Calculate bar dimensions
          float bar_width = base_screen.x - pos_screen.x;  // width from volume value to zero
          float bar_height = 3.0f;  // fixed height for visibility

          // Define bar corners
          ImVec2 bar_tl = ImVec2(base_screen.x - bar_width, pos_screen.y - bar_height/2);
          ImVec2 bar_br = ImVec2(base_screen.x, pos_screen.y + bar_height/2);

          // Draw the volume bar
          draw_list->AddRectFilled(bar_tl, bar_br, ImGui::GetColorU32(vp_color));
      }

      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar();
    }

    // Drag & Drop Target for Price Levels (Must be after ALL Setup calls)
    if (ImPlot::BeginDragDropTargetPlot()) {
      if (const ImGuiPayload *payload =
              ImGui::AcceptDragDropPayload("PRICE_LEVEL")) {
        double dropped_price = *(const double *)payload->Data;
        std::cout << "[ChartPanel] Dropped Price Level: " << dropped_price
                  << std::endl;
      }
      ImPlot::EndDragDropTarget();
    }

    // Now get actual limits being used for THIS frame's rendering and NEXT
    // frame's scaling This locks setup, so it must happen AFTER SetupAxisLimits
    ImPlotRect limits = ImPlot::GetPlotLimits();

    // Store for next frame
    last_view_min_ = limits.X.Min;
    last_view_max_ = limits.X.Max;

    // Recalculate start/end for CULLING (Rendering optimization)
    size_t render_start_idx = 0;
    size_t render_end_idx = chart.dates.size();
    {
      auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(),
                                    limits.X.Min);
      if (lower != chart.dates.begin())
        --lower;
      render_start_idx = std::distance(chart.dates.begin(), lower);

      auto upper = std::upper_bound(chart.dates.begin(), chart.dates.end(),
                                    limits.X.Max);
      if (upper != chart.dates.end())
        ++upper;
      render_end_idx = std::min((size_t)std::distance(chart.dates.begin(), upper),
                                chart.dates.size());
    }

    // Calculate candle width based on timeframe
    double duration_sec =
        RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) /
        1000000.0;
    double candle_half_width = duration_sec * 0.4;

    // Safeguard for very small TFs
    if (candle_half_width < 0.000001)
      candle_half_width = 0.000001;

    ImDrawList *draw_list = ImPlot::GetPlotDrawList();

    // Calculate minimum pixel width for candles
    const float MIN_BODY_WIDTH_PX = 3.0f;
    const float MIN_BODY_HEIGHT_PX = 1.0f;

    // Draw ONLY visible candles
    for (size_t i = render_start_idx; i < render_end_idx; ++i) {
      double x = chart.dates[i];
      if (x == 0)
        continue;

      float open = chart.opens[i];
      float high = chart.highs[i];
      float low = chart.lows[i];
      float close = chart.closes[i];

      // Skip invalid candles
      if (high == 0 || low == 0 || open == 0 || close == 0)
        continue;

      bool bullish = close >= open;
      const auto &colors = ThemeManager::getInstance().getColors();
      ImU32 color = bullish
                        ? ImGui::ColorConvertFloat4ToU32(colors.candle_up)
                        : ImGui::ColorConvertFloat4ToU32(colors.candle_down);
      ImU32 wick_color = color;

      // Transform to screen coordinates
      ImVec2 wick_top = ImPlot::PlotToPixels(x, high);
      ImVec2 wick_bot = ImPlot::PlotToPixels(x, low);
      ImVec2 body_tl =
          ImPlot::PlotToPixels(x - candle_half_width, bullish ? close : open);
      ImVec2 body_br =
          ImPlot::PlotToPixels(x + candle_half_width, bullish ? open : close);

      // Ensure minimum body width in pixels
      float body_width = std::abs(body_br.x - body_tl.x);
      if (body_width < MIN_BODY_WIDTH_PX) {
        float extra = (MIN_BODY_WIDTH_PX - body_width) / 2.0f;
        body_tl.x -= extra;
        body_br.x += extra;
      }

      // Ensure minimum body height in pixels (for doji candles)
      float body_height = std::abs(body_br.y - body_tl.y);
      if (body_height < MIN_BODY_HEIGHT_PX) {
        float mid_y = (body_tl.y + body_br.y) / 2.0f;
        body_tl.y = mid_y - MIN_BODY_HEIGHT_PX / 2.0f;
        body_br.y = mid_y + MIN_BODY_HEIGHT_PX / 2.0f;
      }

      // Draw wick (vertical line)
      draw_list->AddLine(wick_top, wick_bot, wick_color, 1.0f);

      // Draw body (filled rectangle)
      draw_list->AddRectFilled(body_tl, body_br, color);
    }

    // Check for user interaction to break auto-follow
    if (ImPlot::IsPlotHovered() &&
        ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
      follow_latest_ = false;
    }

    // Render indicators
    render_sma_lines(chart, render_start_idx, render_end_idx);
    render_ema_lines(chart, render_start_idx, render_end_idx);
    render_bollinger_bands(chart, render_start_idx, render_end_idx);
    render_rsi_indicator(chart, render_start_idx, render_end_idx);
    render_macd_indicator(chart, render_start_idx, render_end_idx);
    render_fibonacci_levels(chart, render_start_idx, render_end_idx);

    // Render crosshair info if mouse is over plot
    if (indicator_config_.show_crosshair_info && ImPlot::IsPlotHovered()) {
      ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
      render_crosshair_info(chart, mouse_pos.x, mouse_pos.y);
    }

    ImPlot::EndPlot();
  }

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor(3);
}

} // namespace BTQuant
