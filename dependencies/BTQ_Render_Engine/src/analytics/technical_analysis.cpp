#include "analytics/technical_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace BTQuant {

// ============================================================================
// Technical Indicators Implementation
// ============================================================================

TechnicalIndicators::IndicatorResult TechnicalIndicators::simple_moving_average(
    const std::vector<OHLCV>& data, int period) {
  IndicatorResult result;
  result.name = "SMA";
  result.parameters["period"] = period;

  if (data.size() < static_cast<size_t>(period)) {
    return result;
  }

  result.values.reserve(data.size() - period + 1);
  result.timestamps.reserve(data.size() - period + 1);

  double sum = 0.0;
  for (int i = 0; i < period; ++i) {
    sum += data[i].close;
  }
  result.values.push_back(sum / period);
  result.timestamps.push_back(data[period - 1].timestamp);

  for (size_t i = period; i < data.size(); ++i) {
    sum += data[i].close - data[i - period].close;
    result.values.push_back(sum / period);
    result.timestamps.push_back(data[i].timestamp);
  }

  return result;
}

TechnicalIndicators::IndicatorResult TechnicalIndicators::exponential_moving_average(
    const std::vector<OHLCV>& data, int period) {
  IndicatorResult result;
  result.name = "EMA";
  result.parameters["period"] = period;

  if (data.size() < static_cast<size_t>(period)) {
    return result;
  }

  double multiplier = 2.0 / (period + 1);
  double sum = 0.0;

  // Calculate initial SMA
  for (int i = 0; i < period; ++i) {
    sum += data[i].close;
  }
  double ema = sum / period;

  result.values.push_back(ema);
  result.timestamps.push_back(data[period - 1].timestamp);

  // Calculate EMA for remaining data
  for (size_t i = period; i < data.size(); ++i) {
    ema = (data[i].close - ema) * multiplier + ema;
    result.values.push_back(ema);
    result.timestamps.push_back(data[i].timestamp);
  }

  return result;
}

std::vector<TechnicalIndicators::IndicatorResult> TechnicalIndicators::bollinger_bands(
    const std::vector<OHLCV>& data, int period, double std_dev) {
  std::vector<IndicatorResult> results;

  IndicatorResult middle = simple_moving_average(data, period);
  middle.name = "BB_Middle";

  IndicatorResult upper;
  upper.name = "BB_Upper";
  upper.parameters["period"] = period;
  upper.parameters["std_dev"] = std_dev;

  IndicatorResult lower;
  lower.name = "BB_Lower";
  lower.parameters["period"] = period;
  lower.parameters["std_dev"] = std_dev;

  if (data.size() < static_cast<size_t>(period)) {
    results.push_back(middle);
    results.push_back(upper);
    results.push_back(lower);
    return results;
  }

  upper.values.reserve(data.size() - period + 1);
  upper.timestamps.reserve(data.size() - period + 1);
  lower.values.reserve(data.size() - period + 1);
  lower.timestamps.reserve(data.size() - period + 1);

  for (size_t i = period - 1; i < data.size(); ++i) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int j = 0; j < period; ++j) {
      sum += data[i - period + 1 + j].close;
      sum_sq += data[i - period + 1 + j].close * data[i - period + 1 + j].close;
    }
    double mean = sum / period;
    double variance = (sum_sq / period) - (mean * mean);
    double std = std::sqrt(variance);

    upper.values.push_back(mean + (std_dev * std));
    upper.timestamps.push_back(data[i].timestamp);
    lower.values.push_back(mean - (std_dev * std));
    lower.timestamps.push_back(data[i].timestamp);
  }

  results.push_back(middle);
  results.push_back(upper);
  results.push_back(lower);

  return results;
}

TechnicalIndicators::IndicatorResult TechnicalIndicators::rsi(const std::vector<OHLCV>& data,
                                                              int period) {
  IndicatorResult result;
  result.name = "RSI";
  result.parameters["period"] = period;

  if (data.size() < static_cast<size_t>(period + 1)) {
    return result;
  }

  result.values.reserve(data.size() - period);
  result.timestamps.reserve(data.size() - period);

  std::vector<double> gains;
  std::vector<double> losses;

  for (size_t i = 1; i < data.size(); ++i) {
    double change = data[i].close - data[i - 1].close;
    gains.push_back(change > 0 ? change : 0);
    losses.push_back(change < 0 ? -change : 0);
  }

  // Calculate initial average gain and loss
  double avg_gain = std::accumulate(gains.begin(), gains.begin() + period, 0.0) / period;
  double avg_loss = std::accumulate(losses.begin(), losses.begin() + period, 0.0) / period;

  double rs = avg_loss > 0 ? avg_gain / avg_loss : 0;
  double rsi = 100.0 - (100.0 / (1.0 + rs));

  result.values.push_back(rsi);
  result.timestamps.push_back(data[period].timestamp);

  // Calculate RSI using smoothed averages
  for (size_t i = period; i < gains.size(); ++i) {
    avg_gain = (avg_gain * (period - 1) + gains[i]) / period;
    avg_loss = (avg_loss * (period - 1) + losses[i]) / period;

    rs = avg_loss > 0 ? avg_gain / avg_loss : 0;
    rsi = 100.0 - (100.0 / (1.0 + rs));

    result.values.push_back(rsi);
    result.timestamps.push_back(data[i + 1].timestamp);
  }

  return result;
}

std::vector<TechnicalIndicators::IndicatorResult> TechnicalIndicators::macd(
    const std::vector<OHLCV>& data, int fast_period, int slow_period, int signal_period) {
  std::vector<IndicatorResult> results;

  IndicatorResult fast_ema = exponential_moving_average(data, fast_period);
  IndicatorResult slow_ema = exponential_moving_average(data, slow_period);

  IndicatorResult macd_line;
  macd_line.name = "MACD_Line";
  macd_line.parameters["fast_period"] = fast_period;
  macd_line.parameters["slow_period"] = slow_period;

  IndicatorResult signal_line;
  signal_line.name = "MACD_Signal";
  signal_line.parameters["signal_period"] = signal_period;

  IndicatorResult histogram;
  histogram.name = "MACD_Histogram";

  // Calculate MACD line (fast EMA - slow EMA)
  size_t fast_offset = fast_ema.values.size() > slow_ema.values.size()
                           ? fast_ema.values.size() - slow_ema.values.size()
                           : 0;

  for (size_t i = 0; i < slow_ema.values.size(); ++i) {
    size_t fast_idx = i + fast_offset;
    if (fast_idx < fast_ema.values.size()) {
      double macd_val = fast_ema.values[fast_idx] - slow_ema.values[i];
      macd_line.values.push_back(macd_val);
      macd_line.timestamps.push_back(slow_ema.timestamps[i]);
    }
  }

  // Calculate signal line (EMA of MACD line)
  if (macd_line.values.size() >= static_cast<size_t>(signal_period)) {
    double sum =
        std::accumulate(macd_line.values.begin(), macd_line.values.begin() + signal_period, 0.0);
    double signal = sum / signal_period;
    signal_line.values.push_back(signal);
    signal_line.timestamps.push_back(macd_line.timestamps[signal_period - 1]);

    double multiplier = 2.0 / (signal_period + 1);
    for (size_t i = signal_period; i < macd_line.values.size(); ++i) {
      signal = (macd_line.values[i] - signal) * multiplier + signal;
      signal_line.values.push_back(signal);
      signal_line.timestamps.push_back(macd_line.timestamps[i]);
    }
  }

  // Calculate histogram
  size_t hist_start = signal_period - 1;
  for (size_t i = hist_start; i < macd_line.values.size(); ++i) {
    size_t sig_idx = i - hist_start;
    if (sig_idx < signal_line.values.size()) {
      double hist_val = macd_line.values[i] - signal_line.values[sig_idx];
      histogram.values.push_back(hist_val);
      histogram.timestamps.push_back(macd_line.timestamps[i]);
    }
  }

  results.push_back(macd_line);
  results.push_back(signal_line);
  results.push_back(histogram);

  return results;
}

std::vector<TechnicalIndicators::IndicatorResult> TechnicalIndicators::stochastic(
    const std::vector<OHLCV>& data, int k_period, int d_period) {
  std::vector<IndicatorResult> results;

  IndicatorResult k_line;
  k_line.name = "Stochastic_K";
  k_line.parameters["k_period"] = k_period;
  k_line.parameters["d_period"] = d_period;

  IndicatorResult d_line;
  d_line.name = "Stochastic_D";
  d_line.parameters["k_period"] = k_period;
  d_line.parameters["d_period"] = d_period;

  if (data.size() < static_cast<size_t>(k_period)) {
    results.push_back(k_line);
    results.push_back(d_line);
    return results;
  }

  // Calculate %K
  for (size_t i = k_period - 1; i < data.size(); ++i) {
    double highest = data[i - k_period + 1].high;
    double lowest = data[i - k_period + 1].low;

    for (int j = 1; j < k_period; ++j) {
      highest = std::max(highest, data[i - k_period + 1 + j].high);
      lowest = std::min(lowest, data[i - k_period + 1 + j].low);
    }

    double range = highest - lowest;
    if (range > 0) {
      double k = ((data[i].close - lowest) / range) * 100.0;
      k_line.values.push_back(k);
      k_line.timestamps.push_back(data[i].timestamp);
    } else {
      k_line.values.push_back(50.0);
      k_line.timestamps.push_back(data[i].timestamp);
    }
  }

  // Calculate %K (moving average of %K)
  if (k_line.values.size() >= static_cast<size_t>(d_period)) {
    for (size_t i = d_period - 1; i < k_line.values.size(); ++i) {
      double sum = std::accumulate(k_line.values.begin() + i - d_period + 1,
                                   k_line.values.begin() + i + 1, 0.0);
      double d = sum / d_period;
      d_line.values.push_back(d);
      d_line.timestamps.push_back(k_line.timestamps[i]);
    }
  }

  results.push_back(k_line);
  results.push_back(d_line);

  return results;
}

TechnicalIndicators::IndicatorResult TechnicalIndicators::calculate_vwap(
    const std::vector<OHLCV>& data, size_t start_index) {
  IndicatorResult result;
  result.name = "VWAP";
  result.parameters["start_index"] = static_cast<double>(start_index);

  if (data.empty() || start_index >= data.size()) {
    return result;
  }

  double cumulative_price_volume = 0.0;
  double cumulative_volume = 0.0;

  result.values.reserve(data.size() - start_index);
  result.timestamps.reserve(data.size() - start_index);

  for (size_t i = start_index; i < data.size(); ++i) {
    double typical_price = (data[i].high + data[i].low + data[i].close) / 3.0;
    double price_times_volume = typical_price * data[i].volume;

    cumulative_price_volume += price_times_volume;
    cumulative_volume += data[i].volume;

    if (cumulative_volume > 0) {
      double vwap = cumulative_price_volume / cumulative_volume;
      result.values.push_back(vwap);
      result.timestamps.push_back(data[i].timestamp);
    } else {
      // If volume is zero, use the typical price as VWAP
      result.values.push_back(typical_price);
      result.timestamps.push_back(data[i].timestamp);
    }
  }

  return result;
}

TechnicalIndicators::IndicatorResult TechnicalIndicators::calculate_vwap_standard_deviation(
    const std::vector<OHLCV>& data, size_t start_index) {
  IndicatorResult result;
  result.name = "VWAP_StdDev";
  result.parameters["start_index"] = static_cast<double>(start_index);

  if (data.empty() || start_index >= data.size()) {
    return result;
  }

  // First, calculate the VWAP to use as the reference point
  double cumulative_price_volume = 0.0;
  double cumulative_volume = 0.0;

  result.values.reserve(data.size() - start_index);
  result.timestamps.reserve(data.size() - start_index);

  for (size_t i = start_index; i < data.size(); ++i) {
    double typical_price = (data[i].high + data[i].low + data[i].close) / 3.0;
    double price_times_volume = typical_price * data[i].volume;

    cumulative_price_volume += price_times_volume;
    cumulative_volume += data[i].volume;

    if (cumulative_volume > 0) {
      double current_vwap = cumulative_price_volume / cumulative_volume;

      // Calculate the weighted variance around the current VWAP
      double weighted_sum_squared_diff = 0.0;
      double vol_sum = 0.0;

      // Calculate weighted variance for all data points from start_index to current index
      for (size_t j = start_index; j <= i; ++j) {
        double price = (data[j].high + data[j].low + data[j].close) / 3.0;
        double diff = price - current_vwap;
        weighted_sum_squared_diff += (diff * diff) * data[j].volume;
        vol_sum += data[j].volume;
      }

      if (vol_sum > 0) {
        double variance = weighted_sum_squared_diff / vol_sum;
        double std_dev = std::sqrt(variance);
        result.values.push_back(std_dev);
        result.timestamps.push_back(data[i].timestamp);
      } else {
        result.values.push_back(0.0);
        result.timestamps.push_back(data[i].timestamp);
      }
    } else {
      result.values.push_back(0.0);
      result.timestamps.push_back(data[i].timestamp);
    }
  }

  return result;
}

// ============================================================================
// Volume Profile Analyzer Implementation - Simplified
// ============================================================================

VolumeProfileAnalyzer::VolumeProfileAnalyzer(double tick_size) : tick_size_(tick_size) {}

VolumeProfileAnalyzer::VolumeProfile VolumeProfileAnalyzer::calculate_volume_profile(
    const std::vector<TechnicalIndicators::OHLCV>& /*candles*/,
    const std::vector<RenderEngine::TradeData>& /*trades*/) {
  VolumeProfile profile;
  // Note: TradeData is forward-declared, so we can't access its members
  // This is a placeholder implementation
  return profile;
}

void VolumeProfileAnalyzer::calculate_value_area(const std::vector<VolumeNode>& /*nodes*/,
                                                 VolumeProfile& /*profile*/) {
  // Placeholder implementation
}

std::vector<VolumeProfileAnalyzer::VolumeImbalance> VolumeProfileAnalyzer::detect_volume_imbalances(
    const VolumeProfile& /*profile*/, double /*threshold*/) {
  return {};
}

// ============================================================================
// Market Depth Analyzer Implementation - Simplified
// ============================================================================

MarketDepthAnalyzer::MarketDepthSnapshot MarketDepthAnalyzer::create_depth_snapshot(
    const OrderbookData& /*orderbook*/) {
  MarketDepthSnapshot snapshot;
  // Note: OrderbookData is forward-declared, so we can't access its members
  // This is a placeholder implementation
  return snapshot;
}

MarketDepthAnalyzer::DepthAnalysis MarketDepthAnalyzer::analyze_market_depth(
    const MarketDepthSnapshot& /*snapshot*/) {
  DepthAnalysis analysis;
  // Placeholder implementation
  return analysis;
}

double MarketDepthAnalyzer::find_support_level(const std::vector<DepthLevel>& /*bids*/) {
  return 0;
}

double MarketDepthAnalyzer::find_resistance_level(const std::vector<DepthLevel>& /*asks*/) {
  return 0;
}

double MarketDepthAnalyzer::calculate_liquidity_score(const MarketDepthSnapshot& /*snapshot*/) {
  return 0;
}

double MarketDepthAnalyzer::estimate_market_impact(const MarketDepthSnapshot& /*snapshot*/,
                                                   double /*order_value*/) {
  return 0;
}

std::vector<double> MarketDepthAnalyzer::find_significant_levels(
    const MarketDepthSnapshot& /*snapshot*/) {
  return {};
}

std::vector<MarketDepthAnalyzer::LiquidityGap> MarketDepthAnalyzer::detect_liquidity_gaps(
    const MarketDepthSnapshot& /*snapshot*/, double /*min_gap_size*/) {
  return {};
}

// ============================================================================
// Pattern Recognizer Implementation
// ============================================================================

std::vector<PatternRecognizer::Pattern> PatternRecognizer::detect_patterns(
    const std::vector<TechnicalIndicators::OHLCV>& data) {
  std::vector<Pattern> patterns;

  if (data.size() < 10) {
    return patterns;
  }

  auto double_tops = detect_double_top(data);
  auto double_bottoms = detect_double_bottom(data);
  auto head_shoulders = detect_head_and_shoulders(data);
  auto triangles = detect_triangles(data);
  auto candlesticks = detect_candlestick_patterns(data);

  patterns.insert(patterns.end(), double_tops.begin(), double_tops.end());
  patterns.insert(patterns.end(), double_bottoms.begin(), double_bottoms.end());
  patterns.insert(patterns.end(), head_shoulders.begin(), head_shoulders.end());
  patterns.insert(patterns.end(), triangles.begin(), triangles.end());
  patterns.insert(patterns.end(), candlesticks.begin(), candlesticks.end());

  return patterns;
}

std::vector<PatternRecognizer::Pattern> PatternRecognizer::detect_double_top(
    const std::vector<TechnicalIndicators::OHLCV>& data) {
  std::vector<Pattern> patterns;

  auto peaks = find_peaks(data, true);

  for (size_t i = 0; i + 2 < peaks.size(); ++i) {
    size_t peak1 = peaks[i];
    size_t trough = peaks[i + 1];
    size_t peak2 = peaks[i + 2];

    // Check if peaks are within 3% of each other
    double price_diff = std::abs(data[peak1].close - data[peak2].close);
    double avg_price = (data[peak1].close + data[peak2].close) / 2;

    if (price_diff / avg_price < 0.03) {
      Pattern pattern;
      pattern.type = PatternType::DoubleTop;
      pattern.name = "Double Top";
      pattern.confidence = calculate_double_top_confidence(data, peak1, trough, peak2);
      pattern.start_time = data[peak1].timestamp;
      pattern.end_time = data[peak2].timestamp;
      pattern.entry_price = data[trough].close;
      pattern.target_price = data[trough].close - (data[peak1].close - data[trough].close);
      pattern.stop_loss = data[peak1].close + (data[peak1].close - data[trough].close) * 0.1;
      pattern.key_points = {{static_cast<double>(peak1), data[peak1].close},
                            {static_cast<double>(trough), data[trough].close},
                            {static_cast<double>(peak2), data[peak2].close}};
      pattern.description = "Double top pattern indicating potential reversal";
      patterns.push_back(pattern);
    }
  }

  return patterns;
}

std::vector<PatternRecognizer::Pattern> PatternRecognizer::detect_double_bottom(
    const std::vector<TechnicalIndicators::OHLCV>& data) {
  std::vector<Pattern> patterns;

  auto troughs = find_peaks(data, false);

  for (size_t i = 0; i + 2 < troughs.size(); ++i) {
    size_t trough1 = troughs[i];
    size_t peak = troughs[i + 1];
    size_t trough2 = troughs[i + 2];

    // Check if troughs are within 3% of each other
    double price_diff = std::abs(data[trough1].close - data[trough2].close);
    double avg_price = (data[trough1].close + data[trough2].close) / 2;

    if (price_diff / avg_price < 0.03) {
      Pattern pattern;
      pattern.type = PatternType::DoubleBottom;
      pattern.name = "Double Bottom";
      pattern.confidence = calculate_double_bottom_confidence(data, trough1, peak, trough2);
      pattern.start_time = data[trough1].timestamp;
      pattern.end_time = data[trough2].timestamp;
      pattern.entry_price = data[peak].close;
      pattern.target_price = data[peak].close + (data[peak].close - data[trough1].close);
      pattern.stop_loss = data[trough1].close - (data[peak].close - data[trough1].close) * 0.1;
      pattern.key_points = {{static_cast<double>(trough1), data[trough1].close},
                            {static_cast<double>(peak), data[peak].close},
                            {static_cast<double>(trough2), data[trough2].close}};
      pattern.description = "Double bottom pattern indicating potential reversal";
      patterns.push_back(pattern);
    }
  }

  return patterns;
}

std::vector<PatternRecognizer::Pattern> PatternRecognizer::detect_head_and_shoulders(
    const std::vector<TechnicalIndicators::OHLCV>& data) {
  std::vector<Pattern> patterns;

  auto peaks = find_peaks(data, true);
  auto troughs = find_peaks(data, false);

  // Look for head and shoulders pattern
  for (size_t i = 0; i + 4 < peaks.size() && i + 3 < troughs.size(); ++i) {
    size_t left_shoulder = peaks[i];
    size_t left_neck = troughs[i];
    size_t head = peaks[i + 2];
    size_t right_neck = troughs[i + 1];
    size_t right_shoulder = peaks[i + 3];

    // Validate pattern structure
    double left_shoulder_high = data[left_shoulder].close;
    double right_shoulder_high = data[right_shoulder].close;
    double head_high = data[head].close;

    double left_neck_low = data[left_neck].close;
    double right_neck_low = data[right_neck].close;

    // Head should be higher than both shoulders
    bool head_valid = head_high > left_shoulder_high && head_high > right_shoulder_high;

    // Shoulders should be roughly equal (within 2%)
    bool shoulders_equal =
        std::abs(left_shoulder_high - right_shoulder_high) / left_shoulder_high < 0.02;

    // Necklines should be roughly equal
    bool necklines_equal = std::abs(left_neck_low - right_neck_low) / left_neck_low < 0.02;

    if (head_valid && shoulders_equal && necklines_equal) {
      Pattern pattern;
      pattern.type = PatternType::HeadAndShoulders;
      pattern.name = "Head and Shoulders";
      pattern.confidence =
          calculate_head_shoulders_confidence(data, left_shoulder, head, right_shoulder);
      pattern.start_time = data[left_shoulder].timestamp;
      pattern.end_time = data[right_shoulder].timestamp;
      pattern.entry_price = (left_neck_low + right_neck_low) / 2;
      pattern.target_price = pattern.entry_price - (head_high - pattern.entry_price);
      pattern.stop_loss = head_high + (head_high - pattern.entry_price) * 0.1;
      pattern.key_points = {{static_cast<double>(left_shoulder), left_shoulder_high},
                            {static_cast<double>(left_neck), left_neck_low},
                            {static_cast<double>(head), head_high},
                            {static_cast<double>(right_neck), right_neck_low},
                            {static_cast<double>(right_shoulder), right_shoulder_high}};
      pattern.description = "Head and shoulders pattern indicating bearish reversal";
      patterns.push_back(pattern);
    }
  }

  return patterns;
}

std::vector<PatternRecognizer::Pattern> PatternRecognizer::detect_triangles(
    const std::vector<TechnicalIndicators::OHLCV>& data) {
  std::vector<Pattern> patterns;

  if (data.size() < 20) {
    return patterns;
  }

  // Simple triangle detection based on converging highs and lows
  for (size_t i = 0; i + 20 < data.size(); ++i) {
    double high_start = data[i].high;
    double low_start = data[i].low;
    double high_end = data[i + 19].high;
    double low_end = data[i + 19].low;

    // Check if highs are decreasing and lows are increasing
    bool highs_declining = high_end < high_start;
    bool lows_rising = low_end > low_start;

    if (highs_declining && lows_rising) {
      Pattern pattern;
      pattern.type = PatternType::Triangle;
      pattern.name = "Symmetrical Triangle";
      pattern.confidence = 0.7;
      pattern.start_time = data[i].timestamp;
      pattern.end_time = data[i + 19].timestamp;
      pattern.entry_price = (high_start + low_start) / 2;
      pattern.key_points = {{static_cast<double>(i), high_start},
                            {static_cast<double>(i), low_start},
                            {static_cast<double>(i + 19), high_end},
                            {static_cast<double>(i + 19), low_end}};
      pattern.description = "Symmetrical triangle pattern - continuation pattern";
      patterns.push_back(pattern);
    }
  }

  return patterns;
}

std::vector<PatternRecognizer::Pattern> PatternRecognizer::detect_candlestick_patterns(
    const std::vector<TechnicalIndicators::OHLCV>& data) {
  std::vector<Pattern> patterns;

  for (size_t i = 1; i < data.size(); ++i) {
    const auto& prev = data[i - 1];
    const auto& curr = data[i];

    if (is_hammer(curr)) {
      Pattern pattern;
      pattern.type = PatternType::Hammer;
      pattern.name = "Hammer";
      pattern.confidence = 0.6;
      pattern.start_time = prev.timestamp;
      pattern.end_time = curr.timestamp;
      pattern.entry_price = curr.close;
      pattern.description = "Hammer candlestick - potential bullish reversal";
      patterns.push_back(pattern);
    }

    if (is_doji(curr)) {
      Pattern pattern;
      pattern.type = PatternType::Doji;
      pattern.name = "Doji";
      pattern.confidence = 0.5;
      pattern.start_time = prev.timestamp;
      pattern.end_time = curr.timestamp;
      pattern.entry_price = curr.close;
      pattern.description = "Doji candlestick - indecision";
      patterns.push_back(pattern);
    }

    if (is_bullish_engulfing(prev, curr)) {
      Pattern pattern;
      pattern.type = PatternType::EngulfingBullish;
      pattern.name = "Bullish Engulfing";
      pattern.confidence = 0.75;
      pattern.start_time = prev.timestamp;
      pattern.end_time = curr.timestamp;
      pattern.entry_price = curr.close;
      pattern.description = "Bullish engulfing pattern - strong bullish signal";
      patterns.push_back(pattern);
    }

    if (is_bearish_engulfing(prev, curr)) {
      Pattern pattern;
      pattern.type = PatternType::EngulfingBearish;
      pattern.name = "Bearish Engulfing";
      pattern.confidence = 0.75;
      pattern.start_time = prev.timestamp;
      pattern.end_time = curr.timestamp;
      pattern.entry_price = curr.close;
      pattern.description = "Bearish engulfing pattern - strong bearish signal";
      patterns.push_back(pattern);
    }
  }

  return patterns;
}

std::vector<size_t> PatternRecognizer::find_peaks(
    const std::vector<TechnicalIndicators::OHLCV>& data, bool find_highs) {
  std::vector<size_t> peaks;

  for (size_t i = 2; i + 2 < data.size(); ++i) {
    bool is_peak = find_highs ? is_local_high(data, i) : is_local_low(data, i);

    if (is_peak) {
      peaks.push_back(i);
    }
  }

  return peaks;
}

double PatternRecognizer::find_valley_between(const std::vector<TechnicalIndicators::OHLCV>& data,
                                              size_t start, size_t end) {
  double min_price = data[start].low;

  for (size_t i = start; i <= end && i < data.size(); ++i) {
    min_price = std::min(min_price, data[i].low);
  }

  return min_price;
}

bool PatternRecognizer::is_local_high(const std::vector<TechnicalIndicators::OHLCV>& data,
                                      size_t index) {
  double current_high = data[index].high;

  bool higher_than_left = true;
  bool higher_than_right = true;

  for (size_t i = 1; i <= 3 && i <= index; ++i) {
    if (data[index - i].high >= current_high) {
      higher_than_left = false;
      break;
    }
  }

  for (size_t i = 1; i <= 3 && index + i < data.size(); ++i) {
    if (data[index + i].high >= current_high) {
      higher_than_right = false;
      break;
    }
  }

  return higher_than_left && higher_than_right;
}

bool PatternRecognizer::is_local_low(const std::vector<TechnicalIndicators::OHLCV>& data,
                                     size_t index) {
  double current_low = data[index].low;

  bool lower_than_left = true;
  bool lower_than_right = true;

  for (size_t i = 1; i <= 3 && i <= index; ++i) {
    if (data[index - i].low <= current_low) {
      lower_than_left = false;
      break;
    }
  }

  for (size_t i = 1; i <= 3 && index + i < data.size(); ++i) {
    if (data[index + i].low <= current_low) {
      lower_than_right = false;
      break;
    }
  }

  return lower_than_left && lower_than_right;
}

std::pair<double, double> PatternRecognizer::calculate_trend_line(
    const std::vector<glm::vec2>& points) {
  if (points.size() < 2) {
    return {0, 0};
  }

  double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
  int n = static_cast<int>(points.size());

  for (const auto& p : points) {
    sum_x += p.x;
    sum_y += p.y;
    sum_xy += p.x * p.y;
    sum_xx += p.x * p.x;
  }

  double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
  double intercept = (sum_y - slope * sum_x) / n;

  return {slope, intercept};
}

bool PatternRecognizer::is_hammer(const TechnicalIndicators::OHLCV& candle) {
  double body_size = std::abs(candle.close - candle.open);
  double upper_wick = candle.high - std::max(candle.open, candle.close);
  double lower_wick = std::min(candle.open, candle.close) - candle.low;
  double range = candle.high - candle.low;

  if (range == 0) return false;

  // Hammer: small body at top, long lower wick, little or no upper wick
  bool small_body = body_size < range * 0.3;
  bool long_lower_wick = lower_wick > body_size * 2;
  bool small_upper_wick = upper_wick < body_size * 0.5;

  return small_body && long_lower_wick && small_upper_wick;
}

bool PatternRecognizer::is_doji(const TechnicalIndicators::OHLCV& candle) {
  double body_size = std::abs(candle.close - candle.open);
  double range = candle.high - candle.low;

  if (range == 0) return false;

  // Doji: very small body (less than 5% of range)
  return body_size < range * 0.05;
}

bool PatternRecognizer::is_bullish_engulfing(const TechnicalIndicators::OHLCV& prev,
                                             const TechnicalIndicators::OHLCV& curr) {
  bool prev_bearish = prev.close < prev.open;
  bool curr_bullish = curr.close > curr.open;

  bool engulfing = curr.open < prev.close && curr.close > prev.open;

  return prev_bearish && curr_bullish && engulfing;
}

bool PatternRecognizer::is_bearish_engulfing(const TechnicalIndicators::OHLCV& prev,
                                             const TechnicalIndicators::OHLCV& curr) {
  bool prev_bullish = prev.close > prev.open;
  bool curr_bearish = curr.close < curr.open;

  bool engulfing = curr.open > prev.close && curr.close < prev.open;

  return prev_bullish && curr_bearish && engulfing;
}

double PatternRecognizer::calculate_double_top_confidence(
    const std::vector<TechnicalIndicators::OHLCV>& data, size_t peak1, size_t trough,
    size_t peak2) {
  double base_confidence = 0.7;

  // Adjust based on distance between peaks
  double distance = static_cast<double>(peak2 - peak1);
  if (distance < 10)
    base_confidence -= 0.1;
  else if (distance > 50)
    base_confidence -= 0.15;

  // Adjust based on trough depth
  double trough_depth = (data[peak1].close - data[trough].close) / data[peak1].close;
  if (trough_depth > 0.05) base_confidence += 0.1;

  return std::min(base_confidence, 0.95);
}

double PatternRecognizer::calculate_double_bottom_confidence(
    const std::vector<TechnicalIndicators::OHLCV>& data, size_t trough1, size_t peak,
    size_t trough2) {
  double base_confidence = 0.7;

  // Adjust based on distance between troughs
  double distance = static_cast<double>(trough2 - trough1);
  if (distance < 10)
    base_confidence -= 0.1;
  else if (distance > 50)
    base_confidence -= 0.15;

  // Adjust based on peak height
  double peak_height = (data[peak].close - data[trough1].close) / data[trough1].close;
  if (peak_height > 0.05) base_confidence += 0.1;

  return std::min(base_confidence, 0.95);
}

double PatternRecognizer::calculate_head_shoulders_confidence(
    const std::vector<TechnicalIndicators::OHLCV>& data, size_t left, size_t head, size_t right) {
  double base_confidence = 0.75;

  // Check symmetry
  double left_height = data[left].close - data[left + 1].close;
  double right_height = data[right].close - data[right + 1].close;
  double symmetry =
      1.0 - std::abs(left_height - right_height) / std::abs(left_height + right_height + 0.001);

  base_confidence *= symmetry;

  // Adjust based on head prominence
  double head_prominence = (data[head].close - std::min(data[left].close, data[right].close)) /
                           (data[left].close - data[left + 1].close + 0.001);
  if (head_prominence > 1.5) base_confidence += 0.1;

  return std::min(base_confidence, 0.95);
}

}  // namespace BTQuant
