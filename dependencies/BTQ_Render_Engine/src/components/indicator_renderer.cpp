#include "../../include/components/indicator_renderer.hpp"
#include "../../include/market_data_processor.hpp"
#include "../../include/symbol_registry.hpp"
#include "implot.h"
#include <cmath>
#include <iostream>

namespace BTQuant {

IndicatorRenderer::IndicatorRenderer(
    VulkanCore *vulkan_core,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : vulkan_core_(vulkan_core), processor_(processor) {
  default_indicators_ = {{IndicatorType::SMA_10,
                          10,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.94f, 1.0f, 1.0f},
                          1.0f,
                          true},
                         {IndicatorType::SMA_20,
                          20,
                          0,
                          0,
                          2.0,
                          {1.0f, 0.0f, 0.3f, 1.0f},
                          1.0f,
                          true},
                         {IndicatorType::RSI_14,
                          14,
                          0,
                          0,
                          2.0,
                          {0.5f, 0.5f, 0.5f, 1.0f},
                          1.0f,
                          false},
                         {IndicatorType::MACD,
                          12,
                          26,
                          9,
                          2.0,
                          {0.0f, 0.8f, 0.0f, 1.0f},
                          1.0f,
                          false},
                         {IndicatorType::BOLLINGER_MID,
                          20,
                          0,
                          0,
                          2.0,
                          {0.0f, 0.5f, 0.5f, 1.0f},
                          1.0f,
                          false}};
}

void IndicatorRenderer::initialize_vulkan_resources() {
  // In a real implementation, this would initialize Vulkan resources
  // for indicator rendering
}

void IndicatorRenderer::cleanup_vulkan_resources() {
  // In a real implementation, this would cleanup Vulkan resources
}

void IndicatorRenderer::render_indicators(
    const std::string &symbol, RenderEngine::TimeFrame timeframe,
    const std::vector<IndicatorParams> &indicators) {
  // For now, use ImPlot to render indicators on CPU
  // In a real implementation, this would use GPU rendering

  for (const auto &params : indicators) {
    if (!params.visible) {
      continue;
    }

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

uint32_t IndicatorRenderer::getSymbolId(const std::string &symbol) const {
  // Check cache first
  static std::unordered_map<std::string, uint32_t> symbol_cache;
  auto it = symbol_cache.find(symbol);
  if (it != symbol_cache.end()) {
    return it->second;
  }

  // Query SymbolRegistry for the actual ID used by HotSpine
  auto all_symbols = SymbolRegistry::instance().get_all_symbols();
  for (const auto &info : all_symbols) {
    if (info.symbol == symbol) {
      symbol_cache[symbol] = info.id;
      return info.id;
    }
  }

  return 0; // Unknown symbol
}

void IndicatorRenderer::render_sma(const std::string &symbol,
                                   RenderEngine::TimeFrame timeframe,
                                   const IndicatorParams &params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.size() < static_cast<size_t>(params.period1)) {
    return;
  }

  std::vector<double> sma_values;
  sma_values.reserve(candles.size());

  for (size_t i = 0; i < candles.size(); ++i) {
    if (i < static_cast<size_t>(params.period1) - 1) {
      sma_values.push_back(0);
      continue;
    }

    double sum = 0;
    for (size_t j = i - params.period1 + 1; j <= i; ++j) {
      sum += candles[j].close;
    }

    sma_values.push_back(sum / params.period1);
  }

  // Prepare time values (dates) for plotting
  std::vector<double> time_values;
  time_values.reserve(candles.size());
  for (const auto& candle : candles) {
    time_values.push_back(candle.timestamp);
  }

  // Plot SMA
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  std::string label = "SMA" + std::to_string(params.period1);
  ImPlot::PlotLine(label.c_str(), time_values.data(), sma_values.data(),
                   static_cast<int>(sma_values.size()));

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_ema(const std::string &symbol,
                                   RenderEngine::TimeFrame timeframe,
                                   const IndicatorParams &params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.size() < static_cast<size_t>(params.period1)) {
    return;
  }

  std::vector<double> ema_values;
  ema_values.reserve(candles.size());

  double multiplier = 2.0 / (params.period1 + 1);
  double ema = candles[0].close;
  ema_values.push_back(ema);

  for (size_t i = 1; i < candles.size(); ++i) {
    ema = (candles[i].close - ema) * multiplier + ema;
    ema_values.push_back(ema);
  }

  // Prepare time values (dates) for plotting
  std::vector<double> time_values;
  time_values.reserve(candles.size());
  for (const auto& candle : candles) {
    time_values.push_back(candle.timestamp);
  }

  // Plot EMA
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  std::string label = "EMA" + std::to_string(params.period1);
  ImPlot::PlotLine(label.c_str(), time_values.data(), ema_values.data(),
                   static_cast<int>(ema_values.size()));

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_rsi(const std::string &symbol,
                                   RenderEngine::TimeFrame timeframe,
                                   const IndicatorParams &params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.size() < static_cast<size_t>(params.period1 + 1)) {
    return;
  }

  std::vector<double> rsi_values;
  rsi_values.reserve(candles.size());

  std::vector<double> gains;
  std::vector<double> losses;

  for (size_t i = 1; i < candles.size(); ++i) {
    double change = candles[i].close - candles[i - 1].close;
    gains.push_back(std::max(change, 0.0));
    losses.push_back(std::max(-change, 0.0));
  }

  for (size_t i = 0; i < gains.size(); ++i) {
    if (i < static_cast<size_t>(params.period1 - 1)) {
      rsi_values.push_back(0);
      continue;
    }

    double avg_gain = 0;
    double avg_loss = 0;

    for (size_t j = i - params.period1 + 1; j <= i; ++j) {
      avg_gain += gains[j];
      avg_loss += losses[j];
    }

    avg_gain /= params.period1;
    avg_loss /= params.period1;

    double rs = (avg_loss == 0) ? 100.0 : (avg_gain / avg_loss);
    double rsi = 100.0 - (100.0 / (1.0 + rs));

    rsi_values.push_back(rsi);
  }

  // Prepare time values (dates) for plotting
  std::vector<double> time_values;
  time_values.reserve(rsi_values.size());
  for (size_t i = 0; i < rsi_values.size(); ++i) {
    time_values.push_back(candles[i + 1].timestamp); // RSI starts at second candle
  }

  // Plot RSI
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  std::string label = "RSI" + std::to_string(params.period1);
  ImPlot::PlotLine(label.c_str(), time_values.data(), rsi_values.data(),
                   static_cast<int>(rsi_values.size()));

  // Draw RSI levels (30 and 70) - Need to use actual time range
  ImPlotRect limits = ImPlot::GetPlotLimits();
  ImVec2 p1 = ImPlot::PlotToPixels(limits.X.Min, 30);
  ImVec2 p2 = ImPlot::PlotToPixels(limits.X.Max, 30);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

  p1 = ImPlot::PlotToPixels(limits.X.Min, 70);
  p2 = ImPlot::PlotToPixels(limits.X.Max, 70);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_macd(const std::string &symbol,
                                    RenderEngine::TimeFrame timeframe,
                                    const IndicatorParams &params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.size() < static_cast<size_t>(params.period2 + params.period3)) {
    return;
  }

  std::vector<double> macd_line;
  std::vector<double> signal_line;
  std::vector<double> histogram;

  // Calculate MACD line (EMA12 - EMA26)
  std::vector<double> ema12;
  std::vector<double> ema26;

  double multiplier12 = 2.0 / (params.period1 + 1);
  double multiplier26 = 2.0 / (params.period2 + 1);

  double ema12_val = candles[0].close;
  double ema26_val = candles[0].close;
  ema12.push_back(ema12_val);
  ema26.push_back(ema26_val);

  for (size_t i = 1; i < candles.size(); ++i) {
    ema12_val = (candles[i].close - ema12_val) * multiplier12 + ema12_val;
    ema26_val = (candles[i].close - ema26_val) * multiplier26 + ema26_val;
    ema12.push_back(ema12_val);
    ema26.push_back(ema26_val);
    macd_line.push_back(ema12_val - ema26_val);
  }

  // Calculate signal line (EMA9 of MACD)
  if (!macd_line.empty()) {
    double signal_val = macd_line[0];
    signal_line.push_back(signal_val);

    double multiplier9 = 2.0 / (params.period3 + 1);

    for (size_t i = 1; i < macd_line.size(); ++i) {
      signal_val = (macd_line[i] - signal_val) * multiplier9 + signal_val;
      signal_line.push_back(signal_val);
      histogram.push_back(macd_line[i] - signal_val);
    }
  }

  // Prepare time values (dates) for plotting
  std::vector<double> time_values;
  time_values.reserve(macd_line.size());
  for (size_t i = 1; i < candles.size(); ++i) { // MACD starts at second candle
    time_values.push_back(candles[i].timestamp);
  }

  // Plot MACD
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  ImPlot::PlotLine("MACD", time_values.data(), macd_line.data(),
                   static_cast<int>(macd_line.size()));
  ImPlot::PlotLine("Signal", time_values.data(), signal_line.data(),
                   static_cast<int>(signal_line.size()));

  // Plot histogram
  if (!histogram.empty()) {
    std::vector<double> macd_pos, macd_neg;
    macd_pos.resize(histogram.size());
    macd_neg.resize(histogram.size());

    for (size_t i = 0; i < histogram.size(); ++i) {
      if (histogram[i] > 0) {
        macd_pos[i] = histogram[i];
        macd_neg[i] = 0;
      } else {
        macd_pos[i] = 0;
        macd_neg[i] = histogram[i];
      }
    }

    std::vector<double> hist_time_values;
    hist_time_values.reserve(histogram.size());
    for (size_t i = 2; i < candles.size(); ++i) { // Histogram starts at third candle
      hist_time_values.push_back(candles[i].timestamp);
    }

    ImPlot::PlotBars("MACD Hist", hist_time_values.data(), macd_pos.data(),
                     static_cast<int>(macd_pos.size()), 0.67);
    ImPlot::PlotBars("MACD Hist", hist_time_values.data(), macd_neg.data(),
                     static_cast<int>(macd_neg.size()), 0.67);
  }

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_bollinger(const std::string &symbol,
                                         RenderEngine::TimeFrame timeframe,
                                         const IndicatorParams &params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.size() < static_cast<size_t>(params.period1)) {
    return;
  }

  std::vector<double> sma_values;
  std::vector<double> upper_band;
  std::vector<double> lower_band;

  for (size_t i = 0; i < candles.size(); ++i) {
    if (i < static_cast<size_t>(params.period1) - 1) {
      sma_values.push_back(0);
      upper_band.push_back(0);
      lower_band.push_back(0);
      continue;
    }

    double sum = 0;
    for (size_t j = i - params.period1 + 1; j <= i; ++j) {
      sum += candles[j].close;
    }

    double sma = sum / params.period1;
    sma_values.push_back(sma);

    // Calculate standard deviation
    double variance = 0;
    for (size_t j = i - params.period1 + 1; j <= i; ++j) {
      double diff = candles[j].close - sma;
      variance += diff * diff;
    }

    double std_dev = std::sqrt(variance / params.period1);

    upper_band.push_back(sma + params.std_dev * std_dev);
    lower_band.push_back(sma - params.std_dev * std_dev);
  }

  // Prepare time values (dates) for plotting
  std::vector<double> time_values;
  time_values.reserve(candles.size());
  for (const auto& candle : candles) {
    time_values.push_back(candle.timestamp);
  }

  // Plot Bollinger Bands
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  ImPlot::PlotLine("Bollinger Mid", time_values.data(), sma_values.data(),
                   static_cast<int>(sma_values.size()));
  ImPlot::PlotLine("Bollinger Upper", time_values.data(), upper_band.data(),
                   static_cast<int>(upper_band.size()));
  ImPlot::PlotLine("Bollinger Lower", time_values.data(), lower_band.data(),
                   static_cast<int>(lower_band.size()));

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_stochastic(const std::string &symbol,
                                          RenderEngine::TimeFrame timeframe,
                                          const IndicatorParams &params) {
  uint32_t symbol_id = getSymbolId(symbol);
  auto candles = processor_->getCandles(symbol_id, timeframe);
  if (candles.size() < static_cast<size_t>(params.period1 + params.period2)) {
    return;
  }

  std::vector<double> stochastic_k;
  std::vector<double> stochastic_d;

  for (size_t i = 0; i < candles.size(); ++i) {
    if (i < static_cast<size_t>(params.period1 - 1)) {
      stochastic_k.push_back(0);
      continue;
    }

    double high = candles[i].high;
    double low = candles[i].low;

    for (size_t j = i - params.period1 + 1; j <= i; ++j) {
      high = std::max(high, candles[j].high);
      low = std::min(low, candles[j].low);
    }

    if (high == low) {
      stochastic_k.push_back(0);
      continue;
    }

    double k = ((candles[i].close - low) / (high - low)) * 100.0;
    stochastic_k.push_back(k);
  }

  // Calculate Stochastic D (SMA of K)
  for (size_t i = 0; i < stochastic_k.size(); ++i) {
    if (i < static_cast<size_t>(params.period2 - 1)) {
      stochastic_d.push_back(0);
      continue;
    }

    double sum = 0;
    for (size_t j = i - params.period2 + 1; j <= i; ++j) {
      sum += stochastic_k[j];
    }

    stochastic_d.push_back(sum / params.period2);
  }

  // Prepare time values (dates) for plotting
  std::vector<double> time_values_k;
  time_values_k.reserve(stochastic_k.size());
  for (size_t i = params.period1 - 1; i < candles.size(); ++i) { // K starts after period1 candles
    time_values_k.push_back(candles[i].timestamp);
  }

  std::vector<double> time_values_d;
  time_values_d.reserve(stochastic_d.size());
  for (size_t i = params.period1 + params.period2 - 2; i < candles.size(); ++i) { // D starts after period1 + period2 -1 candles
    time_values_d.push_back(candles[i].timestamp);
  }

  // Plot Stochastic
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  ImPlot::PlotLine("Stochastic K", time_values_k.data(), stochastic_k.data(),
                   static_cast<int>(stochastic_k.size()));
  ImPlot::PlotLine("Stochastic D", time_values_d.data(), stochastic_d.data(),
                   static_cast<int>(stochastic_d.size()));

  // Draw stochastic levels (20 and 80) - Need to use actual time range
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

} // namespace BTQuant
