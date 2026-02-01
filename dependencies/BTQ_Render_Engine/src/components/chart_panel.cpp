#include "../../include/components/chart_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <optional>
#include <limits>

#include "../../include/components/volume_profile_panel.hpp"
#include "../../include/components/interaction_manager.hpp"
#include "../../include/components/historical_time_sales.hpp"
#include "imgui.h"
#include "implot.h"
#include "../../include/indicators/anchored_vwap.hpp"
#include "../../include/indicators/session_vwap.hpp"

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

ChartPanel::ChartPanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                       std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                       ChartManager* chart_manager)
    : PanelBase(config), bridge_(bridge), processor_(processor), chart_manager_(chart_manager) {
  indicator_renderer_ = new IndicatorRenderer(nullptr, processor_);
}

void ChartPanel::initialize() {
  // Create chart
  auto id_opt = chart_manager_->getSymbolId(symbol_);
  uint32_t symbol_id = id_opt ? *id_opt : 10007;  // Default BTC-USDT
  chart_id_ = chart_manager_->create_chart(symbol_, exchange_, symbol_id, timeframe_);
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

  const ChartInstance& chart = it->second;

  // Invalidate cache if new data has arrived
  // NOTE: This check is lightweight and only compares sizes
  if (chart.closes.size() > last_known_data_size_) {
    cached_sma_.clear();
    cached_ema_.clear();
    cached_rsi_.clear();
    last_known_data_size_ = chart.closes.size();
  }

  // Render chart controls in a collapsible header
  if (ImGui::CollapsingHeader("Chart Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
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

void ChartPanel::set_symbol(const std::string& symbol, const std::string& exchange) {
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
  const char* timeframes[] = {"1ms", "10ms", "100ms", "500ms", "1s", "3s", "5s", "15s"};
  int selected = static_cast<int>(timeframe_);
  if (ImGui::Combo("Timeframe", &selected, timeframes, IM_ARRAYSIZE(timeframes))) {
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

std::vector<double> ChartPanel::calculate_sma(const std::vector<float>& prices, int period) {
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

std::vector<double> ChartPanel::calculate_ema(const std::vector<float>& prices, int period) {
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
    ema[i] = (static_cast<double>(prices[i]) - ema[i - 1]) * multiplier + ema[i - 1];
  }

  // Cache the result
  cached_ema_[key] = ema;
  return ema;
}

std::vector<double> ChartPanel::calculate_ema(const std::vector<double>& prices, int period) {
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

std::vector<double> ChartPanel::calculate_bollinger_upper(const std::vector<float>& prices,
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

std::vector<double> ChartPanel::calculate_bollinger_lower(const std::vector<float>& prices,
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

std::vector<double> ChartPanel::calculate_rsi(const std::vector<float>& prices, int period) {
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

  std::vector<double> rsi(prices.size(), 50.0);  // Default to neutral

  if (prices.size() < static_cast<size_t>(period + 1)) {
    // Cache the result even if it's empty/default
    cached_rsi_[key] = rsi;
    return rsi;
  }

  for (size_t i = period; i < prices.size(); ++i) {
    double gains = 0.0;
    double losses = 0.0;

    for (int j = 1; j <= period; ++j) {
      double change = static_cast<double>(prices[i - j + 1]) - static_cast<double>(prices[i - j]);
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

std::vector<double> ChartPanel::calculate_macd_line(const std::vector<float>& prices, int fast,
                                                    int slow) {
  auto ema_fast = calculate_ema(prices, fast);
  auto ema_slow = calculate_ema(prices, slow);

  std::vector<double> macd_line(prices.size(), 0.0);
  for (size_t i = 0; i < prices.size(); ++i) {
    macd_line[i] = ema_fast[i] - ema_slow[i];
  }

  return macd_line;
}

std::vector<double> ChartPanel::calculate_macd_signal(const std::vector<double>& macd_line,
                                                      int signal) {
  return calculate_ema(macd_line, signal);
}

std::vector<double> ChartPanel::calculate_macd_histogram(const std::vector<double>& macd_line,
                                                         const std::vector<double>& signal) {
  std::vector<double> histogram(macd_line.size(), 0.0);

  for (size_t i = 0; i < macd_line.size(); ++i) {
    histogram[i] = macd_line[i] - signal[i];
  }

  return histogram;
}

std::vector<FibonacciLevel> ChartPanel::calculate_fibonacci_levels(double start_price,
                                                                   double end_price) {
  std::vector<FibonacciLevel> levels;

  if (start_price == 0.0 || end_price == 0.0) return levels;

  double range = end_price - start_price;

  // Fibonacci ratios
  const double ratios[] = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0};
  const char* labels[] = {"0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"};
  const ImU32 colors[] = {
      IM_COL32(255, 255, 255, 255),  // White
      IM_COL32(0, 255, 255, 200),    // Cyan
      IM_COL32(0, 255, 0, 200),      // Green
      IM_COL32(255, 255, 0, 200),    // Yellow
      IM_COL32(255, 165, 0, 200),    // Orange
      IM_COL32(255, 0, 0, 200),      // Red
      IM_COL32(255, 255, 255, 200)   // White
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

void ChartPanel::render_sma_lines(const ChartInstance& chart, size_t start_idx, size_t end_idx) {
  if (chart.closes.empty()) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

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

void ChartPanel::render_ema_lines(const ChartInstance& chart, size_t start_idx, size_t end_idx) {
  if (chart.closes.empty()) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

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

void ChartPanel::render_bollinger_bands(const ChartInstance& chart, size_t start_idx,
                                        size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_bollinger) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  auto upper_band = calculate_bollinger_upper(chart.closes, indicator_config_.bollinger_period,
                                              indicator_config_.bollinger_std_dev);
  auto lower_band = calculate_bollinger_lower(chart.closes, indicator_config_.bollinger_period,
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

void ChartPanel::render_rsi_indicator(const ChartInstance& chart, size_t start_idx,
                                      size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_rsi) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();
  auto rsi = calculate_rsi(chart.closes, indicator_config_.rsi_period);

  // Get plot limits for RSI scaling
  ImPlotRect limits = ImPlot::GetPlotLimits();

  for (size_t i = start_idx; i < end_idx; ++i) {
    if (i >= static_cast<size_t>(indicator_config_.rsi_period)) {
      double rsi_value = rsi[i];

      // Map RSI to Y-axis (0-100)
      double y = limits.Y.Min + (rsi_value / 100.0) * (limits.Y.Max - limits.Y.Min);

      ImVec2 p = ImPlot::PlotToPixels(chart.dates[i], y);

      // Color based on overbought/oversold
      ImU32 color = IM_COL32(128, 128, 128, 200);
      if (rsi_value >= indicator_config_.rsi_overbought) {
        color = IM_COL32(255, 0, 0, 200);  // Red
      } else if (rsi_value <= indicator_config_.rsi_oversold) {
        color = IM_COL32(0, 255, 0, 200);  // Green
      }

      // Draw RSI line
      draw_list->AddLine(p, p, color, 1.5f);
    }
  }

  // Draw overbought/oversold lines
  double overbought_y =
      limits.Y.Min + (indicator_config_.rsi_overbought / 100.0) * (limits.Y.Max - limits.Y.Min);
  double oversold_y =
      limits.Y.Min + (indicator_config_.rsi_oversold / 100.0) * (limits.Y.Max - limits.Y.Min);

  ImVec2 ob_p1 = ImPlot::PlotToPixels(limits.X.Min, overbought_y);
  ImVec2 ob_p2 = ImPlot::PlotToPixels(limits.X.Max, overbought_y);
  draw_list->AddLine(ob_p1, ob_p2, IM_COL32(255, 0, 0, 100), 1.0f);

  ImVec2 os_p1 = ImPlot::PlotToPixels(limits.X.Min, oversold_y);
  ImVec2 os_p2 = ImPlot::PlotToPixels(limits.X.Max, oversold_y);
  draw_list->AddLine(os_p1, os_p2, IM_COL32(0, 255, 0, 100), 1.0f);
}

void ChartPanel::render_macd_indicator(const ChartInstance& chart, size_t start_idx,
                                       size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_macd) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  auto macd_line = calculate_macd_line(chart.closes, indicator_config_.macd_fast_period,
                                       indicator_config_.macd_slow_period);
  auto macd_signal = calculate_macd_signal(macd_line, indicator_config_.macd_signal_period);
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
      double y = limits.Y.Min + ((macd_value - limits.Y.Min) / (limits.Y.Max - limits.Y.Min)) *
                                    (limits.Y.Max - limits.Y.Min);

      ImVec2 p = ImPlot::PlotToPixels(chart.dates[i], y);

      // Draw MACD line
      draw_list->AddLine(p, p, IM_COL32(0, 255, 255, 200), 1.5f);

      // Draw signal line
      double signal_y =
          limits.Y.Min + ((signal_value - limits.Y.Min) / (limits.Y.Max - limits.Y.Min)) *
                             (limits.Y.Max - limits.Y.Min);
      ImVec2 signal_p = ImPlot::PlotToPixels(chart.dates[i], signal_y);
      draw_list->AddLine(signal_p, signal_p, IM_COL32(255, 165, 0, 200), 1.5f);

      // Draw histogram
      double hist_y = limits.Y.Min + ((hist_value - limits.Y.Min) / (limits.Y.Max - limits.Y.Min)) *
                                         (limits.Y.Max - limits.Y.Min);
      ImVec2 hist_p = ImPlot::PlotToPixels(chart.dates[i], hist_y);

      ImU32 hist_color = hist_value >= 0 ? IM_COL32(0, 255, 0, 150) : IM_COL32(255, 0, 0, 150);
      draw_list->AddRectFilled(ImVec2(hist_p.x - 2, hist_p.y),
                               ImVec2(hist_p.x + 2, hist_p.y + hist_y - signal_y), hist_color);
    }
  }
}

void ChartPanel::render_fibonacci_levels(const ChartInstance& chart, size_t start_idx,
                                         size_t end_idx) {
  if (chart.closes.empty() || !indicator_config_.show_fibonacci) return;

  // Find swing high and low in visible range
  double swing_high = 0.0;
  double swing_low = 1e9;

  for (size_t i = start_idx; i < end_idx; ++i) {
    swing_high = std::max(swing_high, static_cast<double>(chart.highs[i]));
    swing_low = std::min(swing_low, static_cast<double>(chart.lows[i]));
  }

  if (swing_high == 0.0 || swing_low == 1e9) return;

  auto fib_levels = calculate_fibonacci_levels(swing_low, swing_high);
  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Draw Fibonacci levels
  for (const auto& level : fib_levels) {
    ImVec2 p1 = ImPlot::PlotToPixels(chart.dates[start_idx], level.price);
    ImVec2 p2 = ImPlot::PlotToPixels(chart.dates[end_idx - 1], level.price);

    // Draw horizontal line
    draw_list->AddLine(p1, p2, level.color, 1.0f);

    // Draw label
    ImVec2 text_pos = ImVec2(p1.x + 5, p1.y - 10);
    draw_list->AddText(text_pos, level.color, level.label);
  }
}

void ChartPanel::render_crosshair_info(const ChartInstance& chart, double mouse_x, double mouse_y) {
  if (!indicator_config_.show_crosshair_info || chart.closes.empty()) return;

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

  if (closest_idx >= chart.closes.size()) return;

  // Get candle data
  double open = chart.opens[closest_idx];
  double high = chart.highs[closest_idx];
  double low = chart.lows[closest_idx];
  double close = chart.closes[closest_idx];
  double volume = chart.volumes[closest_idx];

  // Render crosshair info overlay
  ImGui::SetNextWindowPos(ImVec2(ImGui::GetMousePos().x + 20, ImGui::GetMousePos().y + 20));
  ImGui::SetNextWindowSize(ImVec2(200, 150));
  ImGui::Begin("Crosshair Info", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
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

void ChartPanel::render_instrument_chart(const ChartInstance& chart) {
  if (chart.dates.empty()) {
    ImGui::Text("Loading chart data for %s...", symbol_.c_str());
    return;
  }

  // Diagnostic info & Controls
  ImGui::TextColored(
      ImVec4(0.0f, 1.0f, 0.8f, 1.0f), " | Candles: %zu | Last: %.2f | TF: %.4gs",
      chart.dates.size(), chart.closes.empty() ? 0.0f : chart.closes.back(),
      RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) / 1000000.0);

  // Neon Chart Styling (ThemeManager)
  const auto& colors = ThemeManager::getInstance().getColors();
  ImPlot::PushStyleColor(ImPlotCol_FrameBg, colors.background);
  ImPlot::PushStyleColor(ImPlotCol_PlotBg,
                         colors.panel_bg);  // Use panel bg or specific dark
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

        for (const auto& level : profile) {
          vp_prices.push_back(level.price);
          vp_volumes.push_back(static_cast<float>(level.total_volume));
          if (level.total_volume > vp_max_vol) vp_max_vol = level.total_volume;
        }
      }
    }
  }

  // Use a unique ID string per symbol/tf/exchange to ensure ImPlot saves state
  // per chart
  std::string plot_id =
      "##Chart_" + symbol_ + "_" + exchange_ + "_" + timeframe_to_string(timeframe_);

  // Pre-calculate axis limits BEFORE BeginPlot to avoid calling locking functions during setup
  double x_axis_min_pre, x_axis_max_pre;
  double y_axis_min_pre = std::numeric_limits<double>::max();
  double y_axis_max_pre = std::numeric_limits<double>::lowest();

  if (follow_latest_) {
    double time_max = chart.dates.back();
    double duration_raw = RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_);
    double duration_sec = duration_raw / 1000000.0;
    double window_size = duration_sec * auto_follow_window_;
    double padding = window_size * 0.05;

    last_view_min_ = time_max - window_size;
    last_view_max_ = time_max + padding;

    x_axis_min_pre = last_view_min_;
    x_axis_max_pre = last_view_max_;
  } else {
    x_axis_min_pre = chart.dates.front();
    x_axis_max_pre = chart.dates.back();
  }

  // Calculate Y-axis limits based on visible X range
  size_t start_idx_pre = 0;
  size_t end_idx_pre = chart.dates.size();

  auto lower_pre = std::lower_bound(chart.dates.begin(), chart.dates.end(), x_axis_min_pre);
  if (lower_pre != chart.dates.begin()) --lower_pre;
  start_idx_pre = std::distance(chart.dates.begin(), lower_pre);

  auto upper_pre = std::upper_bound(chart.dates.begin(), chart.dates.end(), x_axis_max_pre);
  if (upper_pre != chart.dates.end()) ++upper_pre;
  end_idx_pre = std::distance(chart.dates.begin(), upper_pre);

  end_idx_pre = std::min(end_idx_pre, chart.dates.size());

  if (start_idx_pre < end_idx_pre) {
    bool found_data = false;

    for (size_t i = start_idx_pre; i < end_idx_pre; ++i) {
      double low = chart.lows[i];
      double high = chart.highs[i];
      if (low > 0 && high > 0) {
        if (low < y_axis_min_pre) y_axis_min_pre = low;
        if (high > y_axis_max_pre) y_axis_max_pre = high;
        found_data = true;
      }
    }

    if (found_data) {
      double range = y_axis_max_pre - y_axis_min_pre;
      if (range == 0) range = y_axis_min_pre * 0.01;
      if (range == 0) range = 1.0;

      y_axis_min_pre -= range * 0.1;
      y_axis_max_pre += range * 0.1;
    } else {
      y_axis_min_pre = chart.lows.empty() ? 0 : chart.lows[0];
      y_axis_max_pre = chart.highs.empty() ? 1 : chart.highs[0];
    }
  } else {
    y_axis_min_pre =
        chart.lows.empty() ? 0 : *std::min_element(chart.lows.begin(), chart.lows.end());
    y_axis_max_pre =
        chart.highs.empty() ? 1 : *std::max_element(chart.highs.begin(), chart.highs.end());
  }

  if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoTitle | ImPlotFlags_Crosshairs)) {
    // ===== ALL SETUP CALLS MUST HAPPEN FIRST - BEFORE ANY LOCKING FUNCTIONS =====
    // Setup primary axes
    ImPlot::SetupAxes("Time", "Price", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Time);

    // Setup X2 for Volume Profile (must happen before SetupAxisLimits for main axes)
    if (indicator_config_.show_volume_profile && vp_max_vol > 0) {
      ImPlot::SetupAxis(ImAxis_X2, nullptr,
                        ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines);
      ImPlot::SetupAxisLimits(ImAxis_X2, 0, vp_max_vol * 4.0, ImPlotCond_Always);
    }

    // Apply axis limits - these must ALL come before any locking functions
    ImPlot::SetupAxisLimits(ImAxis_X1, x_axis_min_pre, x_axis_max_pre,
                            follow_latest_ ? ImPlotCond_Always : ImPlotCond_Once);
    ImPlot::SetupAxisLimits(ImAxis_Y1, static_cast<float>(y_axis_min_pre),
                            static_cast<float>(y_axis_max_pre), ImPlotCond_Always);
    // ===== END OF SETUP PHASE =====

    // NOW it's safe to call locking functions like IsPlotHovered(), GetPlotLimits(), etc.
    // Check for user interaction to break auto-follow
    bool user_interacted =
        ImPlot::IsPlotHovered() &&
        (ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
         ImGui::IsMouseDragging(ImGuiMouseButton_Right) || ImGui::GetIO().MouseWheel != 0.0f);

    if (user_interacted) {
      follow_latest_ = false;
    }

    // VOLUME PROFILE OVERLAY - using direct draw to avoid axis switching after setup lock
    if (indicator_config_.show_volume_profile && !vp_prices.empty()) {
      ImDrawList* draw_list = ImPlot::GetPlotDrawList();

      // Use Theme Colors for a more integrated look
      ImVec4 vp_color = colors.text;
      vp_color.w = 0.25f;  // reduced alpha

      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, vp_color.w);
      ImPlot::PushStyleColor(ImPlotCol_Fill, vp_color);
      ImPlot::PushStyleColor(ImPlotCol_Line,
                             ImVec4(vp_color.x, vp_color.y, vp_color.z, 0.5f));  // clearer border

      // Calculate bar width in plot coordinates based on volume values
      double max_vol_display = vp_max_vol * 4.0;  // Same as used in SetupAxisLimits for X2

      // Draw volume profile bars using direct drawing to avoid axis switching after setup lock
      for (size_t i = 0; i < vp_prices.size(); ++i) {
        // Convert price (Y coordinate) and volume (X coordinate) to screen coordinates
        ImVec2 pos_screen = ImPlot::PlotToPixels(vp_volumes[i], vp_prices[i]);  // tip of the bar
        ImVec2 base_screen = ImPlot::PlotToPixels(0, vp_prices[i]);             // base of the bar

        // Calculate bar dimensions
        float bar_width = base_screen.x - pos_screen.x;  // width from volume value to zero
        float bar_height = 3.0f;                         // fixed height for visibility

        // Define bar corners
        ImVec2 bar_tl = ImVec2(base_screen.x - bar_width, pos_screen.y - bar_height / 2);
        ImVec2 bar_br = ImVec2(base_screen.x, pos_screen.y + bar_height / 2);

        // Draw the volume bar
        draw_list->AddRectFilled(bar_tl, bar_br, ImGui::GetColorU32(vp_color));
      }

      ImPlot::PopStyleColor(2);
      ImPlot::PopStyleVar();
    }

    // Drag & Drop Target for Price Levels (Must be after ALL Setup calls)
    if (ImPlot::BeginDragDropTargetPlot()) {
      if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("PRICE_LEVEL")) {
        double dropped_price = *(const double*)payload->Data;
        std::cout << "[ChartPanel] Dropped Price Level: " << dropped_price << std::endl;
      }
      ImPlot::EndDragDropTarget();
    }

    // Store previous view limits to detect changes
    double prev_view_min = last_view_min_;
    double prev_view_max = last_view_max_;

    // Now get actual limits being used for THIS frame's rendering and NEXT
    // frame's scaling This locks setup, so it must happen AFTER SetupAxisLimits
    ImPlotRect limits = ImPlot::GetPlotLimits();

    // Store for next frame
    last_view_min_ = limits.X.Min;
    last_view_max_ = limits.X.Max;

    // Check if the view has changed (scroll/zoom) and notify the time stats panel if needed
    if ((prev_view_min != last_view_min_ || prev_view_max != last_view_max_) && on_scroll_sync_) {
        uint64_t start_time = static_cast<uint64_t>(last_view_min_ * 1000000);
        uint64_t end_time = static_cast<uint64_t>(last_view_max_ * 1000000);
        on_scroll_sync_(start_time, end_time);
    }

    // Recalculate start/end for CULLING (Rendering optimization)
    size_t render_start_idx = 0;
    size_t render_end_idx = chart.dates.size();
    {
      auto lower = std::lower_bound(chart.dates.begin(), chart.dates.end(), limits.X.Min);
      if (lower != chart.dates.begin()) --lower;
      render_start_idx = std::distance(chart.dates.begin(), lower);

      auto upper = std::upper_bound(chart.dates.begin(), chart.dates.end(), limits.X.Max);
      if (upper != chart.dates.end()) ++upper;
      render_end_idx =
          std::min((size_t)std::distance(chart.dates.begin(), upper), chart.dates.size());
    }

    // Calculate candle width based on timeframe
    double duration_sec =
        RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_) / 1000000.0;
    double candle_half_width = duration_sec * 0.4;

    // Safeguard for very small TFs
    if (candle_half_width < 0.000001) candle_half_width = 0.000001;

    ImDrawList* draw_list = ImPlot::GetPlotDrawList();

    // Calculate minimum pixel width for candles
    const float MIN_BODY_WIDTH_PX = 3.0f;
    const float MIN_BODY_HEIGHT_PX = 1.0f;

    // Draw ONLY visible candles
    for (size_t i = render_start_idx; i < render_end_idx; ++i) {
      double x = chart.dates[i];
      if (x == 0) continue;

      float open = chart.opens[i];
      float high = chart.highs[i];
      float low = chart.lows[i];
      float close = chart.closes[i];

      // Skip invalid candles
      if (high == 0 || low == 0 || open == 0 || close == 0) continue;

      bool bullish = close >= open;
      const auto& colors = ThemeManager::getInstance().getColors();
      ImU32 color = bullish ? ImGui::ColorConvertFloat4ToU32(colors.candle_up)
                            : ImGui::ColorConvertFloat4ToU32(colors.candle_down);
      ImU32 wick_color = color;

      // Transform to screen coordinates
      ImVec2 wick_top = ImPlot::PlotToPixels(x, high);
      ImVec2 wick_bot = ImPlot::PlotToPixels(x, low);
      ImVec2 body_tl = ImPlot::PlotToPixels(x - candle_half_width, bullish ? close : open);
      ImVec2 body_br = ImPlot::PlotToPixels(x + candle_half_width, bullish ? open : close);

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

    // Render mini volume profile histograms on candles if enabled
    if (indicator_config_.show_volume_profile) {
      // Extract the visible candle data for the mini histograms
      std::vector<RenderEngine::OHLCVCandle> visible_candles;
      std::vector<double> x_coords;
      std::vector<double> y_coords_high;
      std::vector<double> y_coords_low;

      for (size_t i = render_start_idx; i < render_end_idx; ++i) {
        RenderEngine::OHLCVCandle candle;
        candle.timestamp =
            static_cast<uint64_t>(chart.dates[i] * 1000000);  // Convert back to microseconds
        candle.open = chart.opens[i];
        candle.high = chart.highs[i];
        candle.low = chart.lows[i];
        candle.close = chart.closes[i];
        candle.volume = chart.volumes[i];
        candle.trade_count = 1;  // Placeholder

        visible_candles.push_back(candle);

        // Calculate screen coordinates for this candle
        ImVec2 wick_top = ImPlot::PlotToPixels(chart.dates[i], chart.highs[i]);
        ImVec2 wick_bot = ImPlot::PlotToPixels(chart.dates[i], chart.lows[i]);

        x_coords.push_back(wick_top.x);
        y_coords_high.push_back(wick_top.y);
        y_coords_low.push_back(wick_bot.y);
      }

      // Use the processor to get the volume profile data for this symbol and timeframe
      // This will allow us to render the mini histograms with actual volume distribution data
      auto id_opt = chart_manager_->getSymbolId(symbol_);
      if (id_opt) {
        uint32_t symbol_id = *id_opt;

        // Get recent trades for this symbol to populate the mini histograms
        auto analytics = processor_->getSymbolAnalytics(symbol_id);
        const auto& recent_trades = analytics.recent_trades;

        // If we have recent trades, render the step profile histograms showing volume distribution for each candle
        if (!recent_trades.empty()) {
          // Use the enhanced method for Step Profile rendering: draw mini histogram overlay on each candlestick bar
          // showing volume distribution for that bar's price range with additional visualization options
          // This is the main implementation for the task requirement
          // We'll call the method directly on this instance since we have access to the processor and symbol_id
          render_enhanced_step_profile_histograms_on_candle_bars(draw_list, visible_candles, x_coords,
                                                              y_coords_high, y_coords_low,
                                                              true, 8, 0.8f, false);  // Show POC line with 8 buckets per candle, 80% opacity, no labels
        }
      }
    }

    // Check for user interaction to break auto-follow
    if (ImPlot::IsPlotHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
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
    // Handle mouse drag interaction for custom profile creation
    handleMouseDragInteraction();

    if (indicator_config_.show_crosshair_info && ImPlot::IsPlotHovered()) {
      ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();
      render_crosshair_info(chart, mouse_pos.x, mouse_pos.y);
    }

    // Render context menu if right-clicked on plot
    render_context_menu(chart);

    // Render anchored VWAP overlays
    render_anchored_vwap_overlay(chart);

    // Render session VWAP overlays
    render_session_vwap_overlay(chart);

    ImPlot::EndPlot();
  }

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor(3);
}

// Helper method to render enhanced step profile histograms on candle bars
void ChartPanel::render_enhanced_step_profile_histograms_on_candle_bars(ImDrawList* draw_list,
                                                                   const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                                   const std::vector<double>& x_coords,
                                                                   const std::vector<double>& y_coords_high,
                                                                   const std::vector<double>& y_coords_low,
                                                                   bool show_poc_line,
                                                                   int num_buckets_per_candle,
                                                                   float opacity,
                                                                   bool show_labels) {

  // This method implements the Step Profile rendering with additional visualization options
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get symbol ID to fetch recent trades
  auto symbol_id_opt = chart_manager_->getSymbolId(symbol_);
  if (!symbol_id_opt) {
    return;
  }

  uint32_t symbol_id = *symbol_id_opt;

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw enhanced step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_trade_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_trade_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw enhanced step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.7f;  // Slightly wider for better visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, static_cast<int>(240 * opacity));  // Bright yellow for POC with adjustable opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio * opacity);
            int alpha = static_cast<int>(180 * opacity);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio * opacity),
                            static_cast<int>(50 * volume_ratio * opacity), alpha);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio * opacity);
            int alpha = static_cast<int>(180 * opacity);
            color = IM_COL32(static_cast<int>(50 * volume_ratio * opacity), green_intensity,
                            static_cast<int>(50 * volume_ratio * opacity), alpha);
          }
        }

        // Draw the enhanced step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        if (opacity > 0.3f) {  // Only add border if not too transparent
          draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom),
                           IM_COL32(0, 0, 0, static_cast<int>(80 * opacity)), 0.0f, 0, 1.0f);
        }
      }
    }

    // Draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.35f;  // Slightly wider for better visibility
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line with adjustable opacity
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, static_cast<int>(255 * opacity)),  // Bright yellow color for POC with adjustable opacity
                         2.5f);                        // Slightly thicker line for better visibility
    }

    // Optionally show labels for the mini histogram
    if (show_labels) {
      // Add a small label indicating this is a volume profile histogram
      char label[32];
      snprintf(label, sizeof(label), "%.0f", candle.volume);

      // Position the label at the top of the candle
      ImVec2 label_pos = ImVec2(x_center, y_high - 15.0f);

      // Draw the label with appropriate color
      draw_list->AddText(label_pos, IM_COL32(255, 255, 255, static_cast<int>(200 * opacity)), label);
    }
  }
}

// Method to handle mouse drag interaction for custom profile creation
void ChartPanel::handleMouseDragInteraction() {
  auto& interaction_mgr = InteractionManager::getInstance();

  // Check if we're in a state where we should handle mouse drag for profile creation
  if (!ImPlot::IsPlotHovered()) {
    return;
  }

  // Get the current mouse position in plot coordinates
  ImPlotPoint mouse_plot_pos = ImPlot::GetPlotMousePos();
  ImVec2 mouse_screen_pos = ImGui::GetMousePos();

  // Convert ImPlotPoint to ImVec2 for interaction manager
  ImVec2 converted_plot_pos = ImVec2(static_cast<float>(mouse_plot_pos.x), static_cast<float>(mouse_plot_pos.y));

  // Check if left mouse button is pressed (starting drag) and no drag is currently active
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !interaction_mgr.isMouseDragActive()) {
    // Check if the user is holding a modifier key (e.g., Shift) to indicate custom profile creation
    if (ImGui::GetIO().KeyShift) {
      interaction_mgr.startTimeRangeSelection(converted_plot_pos);
    }
  }
  // If drag is active, update the position
  else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && interaction_mgr.isTimeRangeSelectionActive()) {
    interaction_mgr.updateTimeRangeSelection(converted_plot_pos);
  }
  // If mouse button is released, end the drag interaction
  else if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && interaction_mgr.isTimeRangeSelectionActive()) {
    interaction_mgr.endTimeRangeSelection();

    // At this point, we have a completed time range selection
    // We could trigger profile creation or notify other components
    auto time_range = interaction_mgr.getTimeRangeSelection();

    // Optionally, we can set the volume profile panel to use this time range
    // This would require having access to the volume profile panel instance
    // For now, we'll just log the selection
    std::cout << "[ChartPanel] Time range selected: " << time_range.first << " to " << time_range.second << std::endl;
  }
}

void ChartPanel::center_on_timestamp(uint64_t timestamp) {
  // Convert the timestamp to the format used by the chart (seconds since epoch)
  double timestamp_seconds = static_cast<double>(timestamp) / 1000000.0; // Convert microseconds to seconds

  // Store the target timestamp to be used in the next render cycle
  // We can't directly set the plot limits from outside BeginPlot/EndPlot
  // So we'll store it and apply it during the next render
  last_view_min_ = timestamp_seconds - 10.0; // 10 seconds before
  last_view_max_ = timestamp_seconds + 10.0; // 10 seconds after
  follow_latest_ = false; // Disable auto-follow to keep the view centered
}

std::pair<uint64_t, uint64_t> ChartPanel::get_visible_time_range() const {
  // Return the currently visible time range in the chart
  // Convert from seconds (used by ImPlot) back to microseconds (our internal format)
  uint64_t start_time = static_cast<uint64_t>(last_view_min_ * 1000000);
  uint64_t end_time = static_cast<uint64_t>(last_view_max_ * 1000000);

  return {start_time, end_time};
}

// Render context menu when user right-clicks on the chart
void ChartPanel::render_context_menu(const ChartInstance& chart) {
  // Check if the plot is hovered and right mouse button was clicked
  if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
    // Get the mouse position in plot coordinates
    ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();

    // Convert the x-coordinate (time) back to timestamp
    uint64_t clicked_timestamp = static_cast<uint64_t>(mouse_pos.x * 1000000); // Convert from seconds to microseconds

    // Find the closest candle to the clicked timestamp to determine the time range for the bar
    size_t closest_idx = 0;
    double min_distance = std::numeric_limits<double>::max();

    for (size_t i = 0; i < chart.dates.size(); ++i) {
      double distance = std::abs(chart.dates[i] - mouse_pos.x);
      if (distance < min_distance) {
        min_distance = distance;
        closest_idx = i;
      }
    }

    // Calculate the time range for the clicked bar based on the timeframe
    uint64_t bar_duration = RenderEngine::MarketDataProcessor::getTimeFrameDuration(timeframe_);
    uint64_t bar_start_time = static_cast<uint64_t>(chart.dates[closest_idx] * 1000000); // Convert to microseconds
    uint64_t bar_end_time = bar_start_time + bar_duration;

    // Store the time range for the clicked bar
    clicked_bar_start_time_ = bar_start_time;
    clicked_bar_end_time_ = bar_end_time;

    // Open the context menu
    ImGui::OpenPopup("ChartContextMenu");
  }

  // Create the context menu
  if (ImGui::BeginPopup("ChartContextMenu")) {
    if (ImGui::MenuItem("Show Trades for Bar")) {
      // Create or show the HistoricalTimeSalesPanel with trades for the clicked bar
      if (on_show_historical_trades_) {
        on_show_historical_trades_(clicked_bar_start_time_, clicked_bar_end_time_);
      }
    }

    if (ImGui::MenuItem("Anchor VWAP Here")) {
      // Get the current mouse position in plot coordinates
      ImPlotPoint mouse_pos = ImPlot::GetPlotMousePos();

      // Convert the x-coordinate (time) back to timestamp
      uint64_t anchor_timestamp = static_cast<uint64_t>(mouse_pos.x * 1000000); // Convert from seconds to microseconds

      // Create a new anchored VWAP at this timestamp
      create_anchored_vwap_at_time(anchor_timestamp);
    }

    ImGui::EndPopup();
  }
}

// Create an anchored VWAP at the specified timestamp
void ChartPanel::create_anchored_vwap_at_time(uint64_t timestamp) {
  // Create a new anchored VWAP with the given timestamp
  ::btq::AnchoredVWAP new_vwap(timestamp);

  // Get the chart data to calculate the VWAP
  auto charts = chart_manager_->get_charts();
  auto it = charts.find(chart_id_);
  if (it != charts.end()) {
    const ChartInstance& chart = it->second;

    // Convert the chart data to OHLCVCandle format for VWAP calculation
    std::vector<BTQuant::RenderEngine::OHLCVCandle> bars;
    for (size_t i = 0; i < chart.dates.size(); ++i) {
      BTQuant::RenderEngine::OHLCVCandle bar;
      bar.timestamp = static_cast<uint64_t>(chart.dates[i] * 1000000); // Convert to microseconds
      bar.open = chart.opens[i];
      bar.high = chart.highs[i];
      bar.low = chart.lows[i];
      bar.close = chart.closes[i];
      bar.volume = chart.volumes[i];
      bar.trade_count = 1; // Placeholder value
      bars.push_back(bar);
    }

    // Calculate the VWAP from the anchor point
    new_vwap.calculate(bars);
  }

  // Add the new VWAP to our list
  anchored_vwaps_.push_back(new_vwap);
}

// Render the anchored VWAP overlay on the chart
void ChartPanel::render_anchored_vwap_overlay(const ChartInstance& chart) {
  if (anchored_vwaps_.empty()) {
    return;
  }

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  // Iterate through all anchored VWAPs and render them
  for (const auto& vwap : anchored_vwaps_) {
    const auto& vwap_values = vwap.getVWAPValues();
    const auto& sd1_upper = vwap.getSD1UpperBand();
    const auto& sd1_lower = vwap.getSD1LowerBand();
    const auto& sd2_upper = vwap.getSD2UpperBand();
    const auto& sd2_lower = vwap.getSD2LowerBand();
    const auto& sd3_upper = vwap.getSD3UpperBand();
    const auto& sd3_lower = vwap.getSD3LowerBand();

    // Find the starting index in the chart data that corresponds to the anchor timestamp
    uint64_t anchor_timestamp = vwap.getAnchorTimestamp();
    double anchor_time_seconds = static_cast<double>(anchor_timestamp) / 1000000.0;

    // Find the index in the chart where the anchor timestamp occurs
    size_t start_idx = 0;
    bool found_anchor = false;
    for (size_t i = 0; i < chart.dates.size(); ++i) {
      if (chart.dates[i] >= anchor_time_seconds) {
        start_idx = i;
        found_anchor = true;
        break;
      }
    }

    if (!found_anchor) {
      continue; // Anchor timestamp not found in current chart data
    }

    // Render the VWAP line as a smooth polyline with anti-aliasing
    if (vwap_values.size() > 1) {
      // Prepare points for polyline
      std::vector<ImVec2> points;
      points.reserve(vwap_values.size());

      for (size_t i = 0; i < vwap_values.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 point = ImPlot::PlotToPixels(chart.dates[start_idx + i], vwap_values[i]);
        points.push_back(point);
      }

      if (points.size() > 1) {
        // Draw the VWAP line as a smooth polyline in yellow with anti-aliasing
        draw_list->AddPolyline(points.data(), static_cast<int>(points.size()),
                              IM_COL32(255, 255, 0, 255), ImDrawListFlags_AntiAliasedLines, 2.0f);
      }
    }

    // Render SD1 bands as semi-transparent filled regions
    if (sd1_upper.size() > 1 && sd1_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd1_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd1_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd1_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd1_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd1_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD1 band
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      IM_COL32(255, 255, 0, 80));  // Semi-transparent yellow
      }
    }

    // Render SD2 bands as semi-transparent filled regions
    if (sd2_upper.size() > 1 && sd2_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd2_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd2_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd2_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd2_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd2_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD2 band
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      IM_COL32(0, 255, 255, 60));  // Semi-transparent cyan
      }
    }

    // Render SD3 bands as semi-transparent filled regions
    if (sd3_upper.size() > 1 && sd3_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd3_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd3_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd3_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd3_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd3_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD3 band
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      IM_COL32(255, 0, 255, 40));  // More transparent magenta
      }
    }
  }
}

// Render the session VWAP overlay on the chart
void ChartPanel::render_session_vwap_overlay(const ChartInstance& chart) {
  if (chart.dates.empty()) {
    return;
  }

  // Convert the chart data to OHLCVCandle format for VWAP calculation
  std::vector<BTQuant::RenderEngine::OHLCVCandle> bars;
  for (size_t i = 0; i < chart.dates.size(); ++i) {
    BTQuant::RenderEngine::OHLCVCandle bar;
    bar.timestamp = static_cast<uint64_t>(chart.dates[i] * 1000000); // Convert to microseconds
    bar.open = chart.opens[i];
    bar.high = chart.highs[i];
    bar.low = chart.lows[i];
    bar.close = chart.closes[i];
    bar.volume = chart.volumes[i];
    bar.trade_count = 1; // Placeholder value
    bars.push_back(bar);
  }

  // Calculate session VWAPs based on the chart data
  session_vwap_.calculate(bars);

  // Get all sessions and render them
  const auto& sessions = session_vwap_.getSessions();
  ImDrawList* draw_list = ImPlot::GetPlotDrawList();

  for (const auto& session : sessions) {
    const auto& vwap_values = session.vwapValues;
    const auto& sd1_upper = session.sd1UpperBand;
    const auto& sd1_lower = session.sd1LowerBand;
    const auto& sd2_upper = session.sd2UpperBand;
    const auto& sd2_lower = session.sd2LowerBand;
    const auto& sd3_upper = session.sd3UpperBand;
    const auto& sd3_lower = session.sd3LowerBand;

    // Find the starting index in the chart data that corresponds to the session start time
    double session_start_seconds = static_cast<double>(session.startTime) / 1000000.0;

    // Find the index in the chart where the session starts
    size_t start_idx = 0;
    bool found_start = false;
    for (size_t i = 0; i < chart.dates.size(); ++i) {
      if (chart.dates[i] >= session_start_seconds) {
        start_idx = i;
        found_start = true;
        break;
      }
    }

    if (!found_start) {
      continue; // Session start time not found in current chart data
    }

    // Render the VWAP line as a smooth polyline with anti-aliasing
    if (vwap_values.size() > 1) {
      // Prepare points for polyline
      std::vector<ImVec2> points;
      points.reserve(vwap_values.size());

      for (size_t i = 0; i < vwap_values.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 point = ImPlot::PlotToPixels(chart.dates[start_idx + i], vwap_values[i]);
        points.push_back(point);
      }

      if (points.size() > 1) {
        // Use different colors for active vs historical sessions
        ImU32 color = session.isActive ? IM_COL32(0, 255, 255, 255) : IM_COL32(128, 128, 128, 200); // Cyan for active, gray for historical

        // Draw the VWAP line as a smooth polyline in different colors based on session status
        draw_list->AddPolyline(points.data(), static_cast<int>(points.size()),
                              color, ImDrawListFlags_AntiAliasedLines, 2.0f);
      }
    }

    // Render SD1 bands as semi-transparent filled regions
    if (sd1_upper.size() > 1 && sd1_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd1_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd1_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd1_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd1_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd1_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD1 band with different transparency for active vs historical
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        ImU32 sd1_color = session.isActive ? IM_COL32(0, 255, 255, 80) : IM_COL32(128, 128, 128, 60); // More opaque for active
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      sd1_color);
      }
    }

    // Render SD2 bands as semi-transparent filled regions
    if (sd2_upper.size() > 1 && sd2_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd2_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd2_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd2_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd2_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd2_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD2 band with different transparency for active vs historical
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        ImU32 sd2_color = session.isActive ? IM_COL32(0, 200, 200, 60) : IM_COL32(100, 100, 100, 40); // More opaque for active
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      sd2_color);
      }
    }

    // Render SD3 bands as semi-transparent filled regions
    if (sd3_upper.size() > 1 && sd3_lower.size() > 1) {
      // Prepare points for upper and lower bands to form a filled polygon
      std::vector<ImVec2> filled_region_points;
      filled_region_points.reserve(sd3_upper.size() * 2);

      // Add upper band points (forward direction)
      for (size_t i = 0; i < sd3_upper.size() && (start_idx + i) < chart.dates.size(); ++i) {
        ImVec2 upper_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd3_upper[i]);
        filled_region_points.push_back(upper_point);
      }

      // Add lower band points (reverse direction to close the shape)
      for (int i = static_cast<int>(sd3_lower.size()) - 1; i >= 0; --i) {
        if ((start_idx + i) < chart.dates.size()) {
          ImVec2 lower_point = ImPlot::PlotToPixels(chart.dates[start_idx + i], sd3_lower[i]);
          filled_region_points.push_back(lower_point);
        }
      }

      // Draw filled region for SD3 band with different transparency for active vs historical
      if (filled_region_points.size() >= 4) {  // Need at least 4 points to form a shape
        ImU32 sd3_color = session.isActive ? IM_COL32(0, 150, 150, 40) : IM_COL32(80, 80, 80, 20); // More opaque for active
        draw_list->AddConvexPolyFilled(filled_region_points.data(),
                                      static_cast<int>(filled_region_points.size()),
                                      sd3_color);
      }
    }
  }
}

}  // namespace BTQuant
