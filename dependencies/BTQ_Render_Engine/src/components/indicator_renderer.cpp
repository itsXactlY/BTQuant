#include "../../include/components/indicator_renderer.hpp"
#include "../../include/data_visualization_engine.hpp"
#include "../../include/market_data_processor.hpp"
#include "implot.h"
#include <cmath>
#include <iostream>

namespace BTQuant {

IndicatorRenderer::IndicatorRenderer(
    VulkanCore *vulkan_core,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : vulkan_core_(vulkan_core), processor_(processor) {
  // Initialize visualization engine (will be properly initialized later)
  visualization_engine_ = nullptr;

  // Create default indicator configuration
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

  std::cout << "[IndicatorRenderer] Initialized with "
            << default_indicators_.size() << " default indicators" << std::endl;
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

    case IndicatorType::WADDAH_ATTAR_EXPLOSION:
      render_waddah_attar_explosion(symbol, timeframe, params);
      break;

    default:
      break;
    }
  }
}

void IndicatorRenderer::render_sma(const std::string &symbol,
                                   RenderEngine::TimeFrame timeframe,
                                   const IndicatorParams &params) {
  auto candles =
      processor_->getCandles(0, timeframe); // TODO: Get actual symbol ID
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

  // Plot SMA
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  std::string label = "SMA" + std::to_string(params.period1);
  ImPlot::PlotLine(label.c_str(), sma_values.data(),
                   static_cast<int>(sma_values.size()));

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_ema(const std::string &symbol,
                                   RenderEngine::TimeFrame timeframe,
                                   const IndicatorParams &params) {
  auto candles =
      processor_->getCandles(0, timeframe); // TODO: Get actual symbol ID
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

  // Plot EMA
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  std::string label = "EMA" + std::to_string(params.period1);
  ImPlot::PlotLine(label.c_str(), ema_values.data(),
                   static_cast<int>(ema_values.size()));

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_rsi(const std::string &symbol,
                                   RenderEngine::TimeFrame timeframe,
                                   const IndicatorParams &params) {
  auto candles =
      processor_->getCandles(0, timeframe); // TODO: Get actual symbol ID
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

  // Plot RSI
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  std::string label = "RSI" + std::to_string(params.period1);
  ImPlot::PlotLine(label.c_str(), rsi_values.data(),
                   static_cast<int>(rsi_values.size()));

  // Draw RSI levels (30 and 70)
  ImVec2 p1 = ImPlot::PlotToPixels(0, 30);
  ImVec2 p2 =
      ImPlot::PlotToPixels(static_cast<double>(rsi_values.size() - 1), 30);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

  p1 = ImPlot::PlotToPixels(0, 70);
  p2 = ImPlot::PlotToPixels(static_cast<double>(rsi_values.size() - 1), 70);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_macd(const std::string &symbol,
                                    RenderEngine::TimeFrame timeframe,
                                    const IndicatorParams &params) {
  auto candles =
      processor_->getCandles(0, timeframe); // TODO: Get actual symbol ID
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

  // Plot MACD
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  ImPlot::PlotLine("MACD", macd_line.data(),
                   static_cast<int>(macd_line.size()));
  ImPlot::PlotLine("Signal", signal_line.data(),
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

    ImPlot::PlotBars("MACD Hist", macd_pos.data(),
                     static_cast<int>(macd_pos.size()));
    ImPlot::PlotBars("MACD Hist", macd_neg.data(),
                     static_cast<int>(macd_neg.size()));
  }

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_bollinger(const std::string &symbol,
                                         RenderEngine::TimeFrame timeframe,
                                         const IndicatorParams &params) {
  auto candles =
      processor_->getCandles(0, timeframe); // TODO: Get actual symbol ID
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

  // Plot Bollinger Bands
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  ImPlot::PlotLine("Bollinger Mid", sma_values.data(),
                   static_cast<int>(sma_values.size()));
  ImPlot::PlotLine("Bollinger Upper", upper_band.data(),
                   static_cast<int>(upper_band.size()));
  ImPlot::PlotLine("Bollinger Lower", lower_band.data(),
                   static_cast<int>(lower_band.size()));

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();
}

void IndicatorRenderer::render_stochastic(const std::string &symbol,
                                          RenderEngine::TimeFrame timeframe,
                                          const IndicatorParams &params) {
  auto candles =
      processor_->getCandles(0, timeframe); // TODO: Get actual symbol ID
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

  // Plot Stochastic
  ImPlot::PushStyleColor(ImPlotCol_Line, ImU32(ImColor(params.color)));
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, params.line_width);

  ImPlot::PlotLine("Stochastic K", stochastic_k.data(),
                   static_cast<int>(stochastic_k.size()));
  ImPlot::PlotLine("Stochastic D", stochastic_d.data(),
                   static_cast<int>(stochastic_d.size()));

  // Draw stochastic levels (20 and 80)
  ImVec2 p1 = ImPlot::PlotToPixels(0, 20);
  ImVec2 p2 =
      ImPlot::PlotToPixels(static_cast<double>(stochastic_k.size() - 1), 20);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);

  p1 = ImPlot::PlotToPixels(0, 80);
  p2 = ImPlot::PlotToPixels(static_cast<double>(stochastic_k.size() - 1), 80);
  ImPlot::GetPlotDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 0, 128), 1.0f);
}

void IndicatorRenderer::render_waddah_attar_explosion(
    const std::string &symbol, RenderEngine::TimeFrame timeframe,
    const IndicatorParams &params) {
  auto candles =
      processor_->getCandles(0, timeframe); // TODO: Get actual symbol ID
  if (candles.size() < static_cast<size_t>(std::max(
                           {params.period1, params.period2, params.period3})) +
                           1) {
    return;
  }

  size_t count = candles.size();
  std::vector<double> up_trend(count, 0.0);
  std::vector<double> down_trend(count, 0.0);
  std::vector<double> explosion(count, 0.0);
  std::vector<double> dead_zone(count, 0.0);

  // 1. Calculate MACD
  std::vector<double> macd(count, 0.0);
  double ema_fast = candles[0].close;
  double ema_slow = candles[0].close;
  double mult_fast = 2.0 / (params.period1 + 1);
  double mult_slow = 2.0 / (params.period2 + 1);

  for (size_t i = 1; i < count; ++i) {
    ema_fast = (candles[i].close - ema_fast) * mult_fast + ema_fast;
    ema_slow = (candles[i].close - ema_slow) * mult_slow + ema_slow;
    macd[i] = ema_fast - ema_slow;
  }

  // 2. Calculate Bollinger Bands & Trends
  for (size_t i = 1; i < count; ++i) {
    double t1 = (macd[i] - macd[i - 1]) *
                params.period3; // period3 is used as sensitivity here
    up_trend[i] = (t1 >= 0) ? t1 : 0.0;
    down_trend[i] = (t1 < 0) ? -t1 : 0.0;

    // Bollinger
    if (i >= static_cast<size_t>(params.period1)) {
      double sum = 0;
      for (size_t j = i - params.period1 + 1; j <= i; ++j)
        sum += candles[j].close;
      double sma = sum / params.period1;
      double var = 0;
      for (size_t j = i - params.period1 + 1; j <= i; ++j)
        var += pow(candles[j].close - sma, 2);
      double std_dev = sqrt(var / params.period1);
      explosion[i] =
          (sma + params.std_dev * std_dev) - (sma - params.std_dev * std_dev);
    }
  }

  // 3. Simple ATR for Dead Zone
  double atr = 0;
  for (size_t i = 1; i < count; ++i) {
    double tr = std::max({candles[i].high - candles[i].low,
                          std::abs(candles[i].high - candles[i - 1].close),
                          std::abs(candles[i].low - candles[i - 1].close)});
    if (i == 1)
      atr = tr;
    else
      atr = (atr * 49 + tr) / 50.0; // 50 period ATR
    dead_zone[i] = atr * 3.7; // Hardcoded dead zone mult for now or use a param
  }

  // Plotting in a separate subplot if possible, or just as lines
  if (ImPlot::BeginPlot("Waddah Attar Explosion", ImVec2(-1, 200))) {
    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoLabel);
    ImPlot::SetupAxis(ImAxis_Y1, "Trend/Explosion", ImPlotAxisFlags_AutoFit);

    ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0, 1, 0, 0.8f));
    ImPlot::PlotBars("Up Trend", up_trend.data(), (int)count);
    ImPlot::PopStyleColor();

    ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(1, 0, 0, 0.8f));
    ImPlot::PlotBars("Down Trend", down_trend.data(), (int)count);
    ImPlot::PopStyleColor();

    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1, 1, 0, 1.0f));
    ImPlot::PlotLine("Explosion", explosion.data(), (int)count);
    ImPlot::PopStyleColor();

    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1, 1, 1, 0.5f));
    ImPlot::PlotLine("Dead Zone", dead_zone.data(), (int)count);
    ImPlot::PopStyleColor();

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant
