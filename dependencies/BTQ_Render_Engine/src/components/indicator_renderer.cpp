#include "../../include/components/indicator_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "../../include/analytics/technical_analysis.hpp"  // Added include
#include "../../include/market_data_processor.hpp"
#include "../../include/symbol_registry.hpp"
#include "implot.h"

namespace BTQuant {

IndicatorRenderer::IndicatorRenderer(VulkanCore* vulkan_core,
                                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : vulkan_core_(vulkan_core), processor_(processor) {
  // Default indicators configuration
  default_indicators_ = {
      {IndicatorType::SMA_10, 10, 0, 0, 2.0, {0.0f, 0.94f, 1.0f, 1.0f}, 1.0f, true},
      {IndicatorType::SMA_20, 20, 0, 0, 2.0, {1.0f, 0.0f, 0.3f, 1.0f}, 1.0f, true},
      {IndicatorType::RSI_14, 14, 0, 0, 2.0, {0.5f, 0.5f, 0.5f, 1.0f}, 1.0f, false},
      {IndicatorType::MACD, 12, 26, 9, 2.0, {0.0f, 0.8f, 0.0f, 1.0f}, 1.0f, false},
      {IndicatorType::BOLLINGER_MID, 20, 0, 0, 2.0, {0.0f, 0.5f, 0.5f, 1.0f}, 1.0f, false}};
}

void IndicatorRenderer::initialize_vulkan_resources() {
  // Placeholder for Vulkan resource initialization
}

void IndicatorRenderer::cleanup_vulkan_resources() {
  // Placeholder for cleanup
}

void IndicatorRenderer::render_indicators(const std::string& symbol,
                                          RenderEngine::TimeFrame timeframe,
                                          const std::vector<IndicatorParams>& indicators) {
  for (const auto& params : indicators) {
    if (!params.visible) continue;

    switch (params.type) {
      case IndicatorType::SMA_10:
      case IndicatorType::SMA_20:
      case IndicatorType::SMA_50:
        render_sma(symbol, timeframe, params);
        break;
      case IndicatorType::EMA_10:
      case IndicatorType::EMA_20:
      case IndicatorType::EMA_50:
        render_ema(symbol, timeframe, params);
        break;
      case IndicatorType::RSI_14:
        render_rsi(symbol, timeframe, params);
        break;
      case IndicatorType::MACD:
        render_macd(symbol, timeframe, params);
        break;
      case IndicatorType::BOLLINGER_MID:
        render_bollinger(symbol, timeframe, params);
        break;
      case IndicatorType::STOCHASTIC_K:
        render_stochastic(symbol, timeframe, params);
        break;
      default:
        break;
    }
  }
}

uint32_t IndicatorRenderer::getSymbolId(const std::string& symbol) const {
  static std::unordered_map<std::string, uint32_t> symbol_cache;
  auto it = symbol_cache.find(symbol);
  if (it != symbol_cache.end()) return it->second;

  auto all_symbols = SymbolRegistry::instance().get_all_symbols();
  for (const auto& info : all_symbols) {
    if (info.symbol == symbol) {
      symbol_cache[symbol] = info.id;
      return info.id;
    }
  }
  return 0;
}

// Helper to convert candles
std::vector<TechnicalIndicators::OHLCV> convert_candles(
    const std::vector<RenderEngine::OHLCVCandle>& input) {
  std::vector<TechnicalIndicators::OHLCV> output;
  output.reserve(input.size());
  for (const auto& c : input) {
    output.push_back({c.open, c.high, c.low, c.close, c.volume, c.timestamp});
  }
  return output;
}

// Helper to cast uint64 timestamps to double for ImPlot
std::vector<double> cast_timestamps(const std::vector<uint64_t>& timestamps) {
  std::vector<double> output;
  output.reserve(timestamps.size());
  for (auto ts : timestamps) output.push_back(static_cast<double>(ts));
  return output;
}

void IndicatorRenderer::render_sma(const std::string& symbol, RenderEngine::TimeFrame timeframe,
                                   const IndicatorParams& params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.empty()) return;

  auto ti_candles = convert_candles(candles);
  auto result = TechnicalIndicators::simple_moving_average(ti_candles, params.period1);

  if (result.values.empty()) return;

  auto time_values = cast_timestamps(result.timestamps);

  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);
  ImPlot::PlotLine(("SMA" + std::to_string(params.period1)).c_str(), time_values.data(),
                   result.values.data(), static_cast<int>(result.values.size()));
  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_ema(const std::string& symbol, RenderEngine::TimeFrame timeframe,
                                   const IndicatorParams& params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.empty()) return;

  auto ti_candles = convert_candles(candles);
  auto result = TechnicalIndicators::exponential_moving_average(ti_candles, params.period1);

  if (result.values.empty()) return;

  auto time_values = cast_timestamps(result.timestamps);

  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);
  ImPlot::PlotLine(("EMA" + std::to_string(params.period1)).c_str(), time_values.data(),
                   result.values.data(), static_cast<int>(result.values.size()));
  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_rsi(const std::string& symbol, RenderEngine::TimeFrame timeframe,
                                   const IndicatorParams& params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.empty()) return;

  auto ti_candles = convert_candles(candles);
  auto result = TechnicalIndicators::rsi(ti_candles, params.period1);

  if (result.values.empty()) return;

  auto time_values = cast_timestamps(result.timestamps);

  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);
  ImPlot::PlotLine(("RSI" + std::to_string(params.period1)).c_str(), time_values.data(),
                   result.values.data(), static_cast<int>(result.values.size()));

  // Draw RSI levels
  ImPlotRect limits = ImPlot::GetPlotLimits();
  double x_min = limits.X.Min;
  double x_max = limits.X.Max;

  ImVec2 p1 = ImPlot::PlotToPixels(x_min, 30);
  ImVec2 p2 = ImPlot::PlotToPixels(x_max, 30);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

  p1 = ImPlot::PlotToPixels(x_min, 70);
  p2 = ImPlot::PlotToPixels(x_max, 70);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_macd(const std::string& symbol, RenderEngine::TimeFrame timeframe,
                                    const IndicatorParams& params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.empty()) return;

  auto ti_candles = convert_candles(candles);
  auto results =
      TechnicalIndicators::macd(ti_candles, params.period1, params.period2, params.period3);
  // results[0]: MACD Line, [1]: Signal, [2]: Histogram

  if (results.size() < 3) return;

  auto& macd = results[0];
  auto& signal = results[1];
  auto& hist = results[2];

  if (macd.values.empty()) return;

  auto macd_time = cast_timestamps(macd.timestamps);
  auto signal_time = cast_timestamps(signal.timestamps);

  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  ImPlot::PlotLine("MACD", macd_time.data(), macd.values.data(),
                   static_cast<int>(macd.values.size()));
  ImPlot::PlotLine("Signal", signal_time.data(), signal.values.data(),
                   static_cast<int>(signal.values.size()));

  // Histogram
  if (!hist.values.empty()) {
    auto hist_time = cast_timestamps(hist.timestamps);
    ImPlot::PlotBars("MACD Hist", hist_time.data(), hist.values.data(),
                     static_cast<int>(hist.values.size()), 0.67);
  }

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_bollinger(const std::string& symbol,
                                         RenderEngine::TimeFrame timeframe,
                                         const IndicatorParams& params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.empty()) return;

  auto ti_candles = convert_candles(candles);
  auto results = TechnicalIndicators::bollinger_bands(ti_candles, params.period1, params.std_dev);
  // results[0]: Middle, [1]: Upper, [2]: Lower

  if (results.size() < 3) return;
  auto& mid = results[0];
  auto& upper = results[1];
  auto& lower = results[2];

  if (mid.values.empty()) return;
  auto time_values = cast_timestamps(mid.timestamps);

  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  ImPlot::PlotLine("Bollinger Mid", time_values.data(), mid.values.data(),
                   static_cast<int>(mid.values.size()));
  ImPlot::PlotLine("Bollinger Upper", time_values.data(), upper.values.data(),
                   static_cast<int>(upper.values.size()));
  ImPlot::PlotLine("Bollinger Lower", time_values.data(), lower.values.data(),
                   static_cast<int>(lower.values.size()));

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_stochastic(const std::string& symbol,
                                          RenderEngine::TimeFrame timeframe,
                                          const IndicatorParams& params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.empty()) return;

  auto ti_candles = convert_candles(candles);
  auto results = TechnicalIndicators::stochastic(ti_candles, params.period1, params.period2);
  // results[0]: K, [1]: D

  if (results.size() < 2) return;
  auto& k = results[0];
  auto& d = results[1];

  if (k.values.empty()) return;

  auto k_time = cast_timestamps(k.timestamps);
  auto d_time = cast_timestamps(d.timestamps);

  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  ImPlot::PlotLine("Stoch K", k_time.data(), k.values.data(), static_cast<int>(k.values.size()));
  ImPlot::PlotLine("Stoch D", d_time.data(), d.values.data(), static_cast<int>(d.values.size()));

  ImPlotRect limits = ImPlot::GetPlotLimits();
  ImVec2 p1 = ImPlot::PlotToPixels(limits.X.Min, 20);
  ImVec2 p2 = ImPlot::PlotToPixels(limits.X.Max, 20);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

  p1 = ImPlot::PlotToPixels(limits.X.Min, 80);
  p2 = ImPlot::PlotToPixels(limits.X.Max, 80);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

}  // namespace BTQuant
