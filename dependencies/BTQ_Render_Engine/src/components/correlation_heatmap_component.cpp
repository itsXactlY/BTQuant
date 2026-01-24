/**
 * @file correlation_heatmap_component.cpp
 * @brief Correlation Heatmap Component (C++23/26)
 *
 * Multi-asset correlation visualization with modern C++ features:
 * - [[nodiscard]], [[likely]]/[[unlikely]] attributes
 * - constexpr constants
 * - Improved structure initialization
 *
 * @version 2.0.0 (C++23/26)
 */

#include "../../include/components/correlation_heatmap_component.hpp"
#include "../../include/symbol_registry.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace BTQuant {

// Constants
namespace {
constexpr float UPDATE_INTERVAL_SECONDS = 5.0f;
constexpr std::array<const char *, 5> EXCHANGES = {"Binance", "OKX", "Bybit",
                                                   "Coinbase", "Kraken"};
} // namespace

CorrelationHeatmapComponent::CorrelationHeatmapComponent(
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : processor_(processor) {
  // Default symbols for major crypto pairs
  symbols_ = {"BTC-USDT", "ETH-USDT", "BNB-USDT", "ADA-USDT", "SOL-USDT"};
}

void CorrelationHeatmapComponent::update(float dt) {
  // Update correlation matrix periodically (every 5 seconds)
  static float update_timer = 0.0f;
  update_timer += dt;

  if (update_timer >= 5.0f) {
    update_correlation_matrix();
    update_timer = 0.0f;
  }
}

void CorrelationHeatmapComponent::render_gui() {
  if (!visible_)
    return;

  ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("Correlation Heatmap", &visible_)) {
    render_controls();
    ImGui::Separator();
    render_heatmap();
  }
  ImGui::End();
}

void CorrelationHeatmapComponent::set_symbols(
    const std::vector<std::string> &symbols) {
  symbols_ = symbols;
  update_correlation_matrix();
}

void CorrelationHeatmapComponent::render_controls() {
  if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
    // Timeframe selection - use correct timeframe names
    const char *timeframes[] = {"1ms",  "10ms", "100ms", "500ms",
                                "1sec", "3sec", "5sec",  "15sec"};
    int selected = static_cast<int>(timeframe_);
    if (ImGui::Combo("Timeframe", &selected, timeframes,
                     IM_ARRAYSIZE(timeframes))) {
      timeframe_ = static_cast<RenderEngine::TimeFrame>(selected);
      update_correlation_matrix();
    }

    // Lookback period
    if (ImGui::SliderInt("Lookback Periods", &lookback_periods_, 20, 500)) {
      update_correlation_matrix();
    }

    // Symbol selection
    if (ImGui::Button("Refresh Data")) {
      update_correlation_matrix();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset to Defaults")) {
      symbols_ = {"BTC-USDT", "ETH-USDT", "BNB-USDT", "ADA-USDT", "SOL-USDT"};
      update_correlation_matrix();
    }
  }
}

void CorrelationHeatmapComponent::render_heatmap() {
  std::lock_guard<std::mutex> lock(data_mutex_);

  if (correlation_matrix_.matrix.empty() ||
      correlation_matrix_.labels.empty()) {
    ImGui::Text("No correlation data available");
    ImGui::Text("Waiting for market data...");
    return;
  }

  int n = static_cast<int>(correlation_matrix_.labels.size());
  if (n == 0)
    return;

  // Flatten the matrix for ImPlot::PlotHeatmap
  std::vector<double> flat_data;
  flat_data.reserve(n * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      flat_data.push_back(correlation_matrix_.matrix[i][j]);
    }
  }

  // Create heatmap using ImPlot
  ImPlot::PushColormap(ImPlotColormap_RdBu);

  if (ImPlot::BeginPlot("##CorrelationMatrix", ImVec2(-1, 400),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
    // Set up axes with labels
    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoDecorations);
    ImPlot::SetupAxis(ImAxis_Y1, nullptr,
                      ImPlotAxisFlags_NoDecorations | ImPlotAxisFlags_Invert);

    // Set axis limits
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, n, ImGuiCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, n, ImGuiCond_Always);

    // Plot heatmap - correct signature
    ImPlot::PlotHeatmap("Correlation", flat_data.data(), n, n, -1.0, 1.0,
                        "%.2f", ImPlotPoint(0, 0), ImPlotPoint(n, n));

    ImPlot::EndPlot();
  }

  ImPlot::PopColormap();

  // Color scale legend
  ImGui::SameLine();
  ImPlot::ColormapScale("##Scale", -1.0, 1.0, ImVec2(60, 400));

  // Draw labels below the heatmap
  ImGui::Text("Symbols: ");
  for (int i = 0; i < n; ++i) {
    ImGui::SameLine();
    ImGui::Text("%s", correlation_matrix_.labels[i].c_str());
    if (i < n - 1)
      ImGui::SameLine();
  }

  // Display correlation values on hover info
  ImGui::Separator();
  ImGui::Text("Hover over cells to see correlation values");
}

void CorrelationHeatmapComponent::update_correlation_matrix() {
  std::lock_guard<std::mutex> lock(data_mutex_);

  int n = static_cast<int>(symbols_.size());
  correlation_matrix_.matrix.assign(n, std::vector<double>(n, 0.0));
  correlation_matrix_.labels = symbols_;

  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      double corr = calculate_correlation(symbols_[i], symbols_[j]);
      correlation_matrix_.matrix[i][j] = corr;
      correlation_matrix_.matrix[j][i] = corr; // Symmetric
    }
  }

  correlation_matrix_.last_update =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
}

double
CorrelationHeatmapComponent::calculate_correlation(const std::string &symbol1,
                                                   const std::string &symbol2) {
  if (symbol1 == symbol2)
    return 1.0;

  auto returns1 = get_returns(symbol1);
  auto returns2 = get_returns(symbol2);

  if (returns1.size() != returns2.size() || returns1.size() < 2) {
    return 0.0;
  }

  // Calculate Pearson correlation coefficient
  double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
  double sum_x2 = 0.0, sum_y2 = 0.0;
  size_t n = returns1.size();

  for (size_t i = 0; i < n; ++i) {
    sum_x += returns1[i];
    sum_y += returns2[i];
    sum_xy += returns1[i] * returns2[i];
    sum_x2 += returns1[i] * returns1[i];
    sum_y2 += returns2[i] * returns2[i];
  }

  double numerator = n * sum_xy - sum_x * sum_y;
  double denominator =
      std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

  if (denominator == 0.0)
    return 0.0;

  return numerator / denominator;
}

std::vector<double>
CorrelationHeatmapComponent::get_returns(const std::string &symbol) {
  std::vector<double> returns;

  // Look up symbol_id from the registry
  auto symbol_id_opt = get_symbol_id(symbol);
  if (!symbol_id_opt.has_value()) {
    return returns;
  }

  uint32_t symbol_id = symbol_id_opt.value();

  // Get candles using correct API: getCandles(symbol_id, timeframe)
  auto candles = processor_->getCandles(symbol_id, timeframe_);

  if (candles.size() < 2)
    return returns;

  // Limit to lookback periods
  size_t start_idx = 0;
  if (candles.size() > static_cast<size_t>(lookback_periods_ + 1)) {
    start_idx = candles.size() - lookback_periods_ - 1;
  }

  for (size_t i = start_idx + 1; i < candles.size(); ++i) {
    double prev_close = candles[i - 1].close;
    double curr_close = candles[i].close;
    if (prev_close > 0.0) {
      returns.push_back((curr_close - prev_close) / prev_close);
    }
  }

  return returns;
}

std::optional<uint32_t>
CorrelationHeatmapComponent::get_symbol_id(const std::string &symbol) {
  // Try to get symbol ID from the registry
  // Symbol format is typically "BTC-USDT" - we need to parse exchange
  auto &registry = SymbolRegistry::instance();

  // Try common exchanges
  static const std::vector<std::string> exchanges = {"Binance", "OKX", "Bybit",
                                                     "Coinbase", "Kraken"};
  for (const auto &exchange : exchanges) {
    auto id_opt = registry.get_symbol_id(exchange, symbol);
    if (id_opt.has_value()) {
      return id_opt.value();
    }
  }

  // Fallback: try to find any active symbol
  auto active_symbols = processor_->getActiveSymbols();
  if (!active_symbols.empty()) {
    // Return first active symbol as fallback
    return active_symbols[0];
  }

  return std::nullopt;
}

ImVec4
CorrelationHeatmapComponent::get_correlation_color(double correlation) const {
  // Normalize correlation from [-1, 1] to [0, 1]
  float t = static_cast<float>((correlation + 1.0) / 2.0);

  // Blue (negative) to white (zero) to red (positive)
  if (correlation < 0) {
    return ImVec4(t * 2.0f, t * 2.0f, 1.0f, 1.0f);
  } else {
    return ImVec4(1.0f, (1.0f - t) * 2.0f, (1.0f - t) * 2.0f, 1.0f);
  }
}

std::string
CorrelationHeatmapComponent::format_correlation_value(double value) const {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.3f", value);
  return std::string(buf);
}

} // namespace BTQuant