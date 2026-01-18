/**
 * BTQuant Advanced Analytics and Trading Tools
 *
 * Professional trading analytics including technical indicators, volume
 * analysis, market depth visualization, pattern recognition, and risk
 * management tools.
 */

#include "../../include/vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

namespace BTQuant {

// ============================================================================
// Technical Analysis Indicators
// ============================================================================

// Moving Averages
TechnicalIndicators::IndicatorResult
TechnicalIndicators::simple_moving_average(const std::vector<OHLCV> &data,
                                           int period) {
  IndicatorResult result;
  result.name = "SMA";
  result.parameters["period"] = period;

  if (data.size() < static_cast<size_t>(period))
    return result;

  for (size_t i = period - 1; i < data.size(); ++i) {
    double sum = 0.0;
    for (int j = 0; j < period; ++j) {
      sum += data[i - j].close;
    }
    result.values.push_back(sum / period);
    result.timestamps.push_back(data[i].timestamp);
  }

  return result;
}

TechnicalIndicators::IndicatorResult
TechnicalIndicators::exponential_moving_average(const std::vector<OHLCV> &data,
                                                int period) {
  IndicatorResult result;
  result.name = "EMA";
  result.parameters["period"] = period;

  if (data.empty())
    return result;

  double multiplier = 2.0 / (period + 1);
  double ema = data[0].close;

  result.values.push_back(ema);
  result.timestamps.push_back(data[0].timestamp);

  for (size_t i = 1; i < data.size(); ++i) {
    ema = (data[i].close * multiplier) + (ema * (1 - multiplier));
    result.values.push_back(ema);
    result.timestamps.push_back(data[i].timestamp);
  }

  return result;
}

// Bollinger Bands
std::vector<TechnicalIndicators::IndicatorResult>
TechnicalIndicators::bollinger_bands(const std::vector<OHLCV> &data, int period,
                                     double std_dev) {
  std::vector<IndicatorResult> results(3);

  // Middle band (SMA)
  results[0] = simple_moving_average(data, period);
  results[0].name = "BB_Middle";

  // Calculate standard deviation and bands
  IndicatorResult upper_band, lower_band;
  upper_band.name = "BB_Upper";
  lower_band.name = "BB_Lower";
  upper_band.parameters["period"] = period;
  upper_band.parameters["std_dev"] = std_dev;
  lower_band.parameters["period"] = period;
  lower_band.parameters["std_dev"] = std_dev;

  for (size_t i = period - 1; i < data.size(); ++i) {
    double sum = 0.0;
    double sum_sq = 0.0;

    for (int j = 0; j < period; ++j) {
      double price = data[i - j].close;
      sum += price;
      sum_sq += price * price;
    }

    double mean = sum / period;
    double variance = (sum_sq / period) - (mean * mean);
    double std_deviation = std::sqrt(variance);

    upper_band.values.push_back(mean + (std_dev * std_deviation));
    lower_band.values.push_back(mean - (std_dev * std_deviation));
    upper_band.timestamps.push_back(data[i].timestamp);
    lower_band.timestamps.push_back(data[i].timestamp);
  }

  results[1] = upper_band;
  results[2] = lower_band;

  return results;
}

// RSI (Relative Strength Index)
TechnicalIndicators::IndicatorResult
TechnicalIndicators::rsi(const std::vector<OHLCV> &data, int period) {
  IndicatorResult result;
  result.name = "RSI";
  result.parameters["period"] = period;

  if (data.size() < static_cast<size_t>(period + 1))
    return result;

  std::vector<double> gains, losses;

  // Calculate price changes
  for (size_t i = 1; i < data.size(); ++i) {
    double change = data[i].close - data[i - 1].close;
    gains.push_back(change > 0 ? change : 0);
    losses.push_back(change < 0 ? -change : 0);
  }

  // Calculate initial average gain and loss
  double avg_gain = 0.0, avg_loss = 0.0;
  for (int i = 0; i < period; ++i) {
    avg_gain += gains[i];
    avg_loss += losses[i];
  }
  avg_gain /= period;
  avg_loss /= period;

  // Calculate RSI
  for (size_t i = period; i < gains.size(); ++i) {
    avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period;
    avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period;

    double rs = avg_loss == 0 ? 100 : avg_gain / avg_loss;
    double rsi_value = 100 - (100 / (1 + rs));

    result.values.push_back(rsi_value);
    result.timestamps.push_back(data[i + 1].timestamp);
  }

  return result;
}

// MACD (Moving Average Convergence Divergence)
std::vector<TechnicalIndicators::IndicatorResult>
TechnicalIndicators::macd(const std::vector<OHLCV> &data, int fast_period,
                          int slow_period, int signal_period) {
  std::vector<IndicatorResult> results(3);

  auto fast_ema = exponential_moving_average(data, fast_period);
  auto slow_ema = exponential_moving_average(data, slow_period);

  // MACD Line
  IndicatorResult macd_line;
  macd_line.name = "MACD";
  macd_line.parameters["fast_period"] = fast_period;
  macd_line.parameters["slow_period"] = slow_period;

  size_t start_idx = slow_period - fast_period;
  for (size_t i = start_idx; i < fast_ema.values.size(); ++i) {
    double macd_value = fast_ema.values[i] - slow_ema.values[i - start_idx];
    macd_line.values.push_back(macd_value);
    macd_line.timestamps.push_back(fast_ema.timestamps[i]);
  }

  // Signal Line (EMA of MACD)
  IndicatorResult signal_line;
  signal_line.name = "MACD_Signal";
  signal_line.parameters["signal_period"] = signal_period;

  if (macd_line.values.size() >= static_cast<size_t>(signal_period)) {
    double multiplier = 2.0 / (signal_period + 1);
    double signal = macd_line.values[0];

    signal_line.values.push_back(signal);
    signal_line.timestamps.push_back(macd_line.timestamps[0]);

    for (size_t i = 1; i < macd_line.values.size(); ++i) {
      signal = (macd_line.values[i] * multiplier) + (signal * (1 - multiplier));
      signal_line.values.push_back(signal);
      signal_line.timestamps.push_back(macd_line.timestamps[i]);
    }
  }

  // Histogram
  IndicatorResult histogram;
  histogram.name = "MACD_Histogram";

  for (size_t i = 0;
       i < std::min(macd_line.values.size(), signal_line.values.size()); ++i) {
    histogram.values.push_back(macd_line.values[i] - signal_line.values[i]);
    histogram.timestamps.push_back(macd_line.timestamps[i]);
  }

  results[0] = macd_line;
  results[1] = signal_line;
  results[2] = histogram;

  return results;
}

// Stochastic Oscillator
std::vector<TechnicalIndicators::IndicatorResult>
TechnicalIndicators::stochastic(const std::vector<OHLCV> &data, int k_period,
                                int d_period) {
  std::vector<IndicatorResult> results(2);

  IndicatorResult k_line, d_line;
  k_line.name = "Stoch_K";
  d_line.name = "Stoch_D";
  k_line.parameters["k_period"] = k_period;
  d_line.parameters["d_period"] = d_period;

  // Calculate %K
  for (size_t i = k_period - 1; i < data.size(); ++i) {
    double highest_high = data[i].high;
    double lowest_low = data[i].low;

    for (int j = 1; j < k_period; ++j) {
      highest_high = std::max(highest_high, data[i - j].high);
      lowest_low = std::min(lowest_low, data[i - j].low);
    }

    double k_value =
        ((data[i].close - lowest_low) / (highest_high - lowest_low)) * 100;
    k_line.values.push_back(k_value);
    k_line.timestamps.push_back(data[i].timestamp);
  }

  // Calculate %D (SMA of %K)
  for (size_t i = d_period - 1; i < k_line.values.size(); ++i) {
    double sum = 0.0;
    for (int j = 0; j < d_period; ++j) {
      sum += k_line.values[i - j];
    }
    d_line.values.push_back(sum / d_period);
    d_line.timestamps.push_back(k_line.timestamps[i]);
  }

  results[0] = k_line;
  results[1] = d_line;

  return results;
}
// End of TechnicalIndicators implementation

// ============================================================================
// Volume Profile Analysis
// ============================================================================

VolumeProfileAnalyzer::VolumeProfileAnalyzer(double tick_size)
    : tick_size_(tick_size) {}

VolumeProfileAnalyzer::VolumeProfile
VolumeProfileAnalyzer::calculate_volume_profile(
    const std::vector<TechnicalIndicators::OHLCV> &data,
    const std::vector<RenderEngine::TradeData> &trades) {
  VolumeProfile profile;

  if (data.empty() || trades.empty())
    return profile;

  profile.start_time = data.front().timestamp;
  profile.end_time = data.back().timestamp;

  // Find price range
  double min_price = data[0].low;
  double max_price = data[0].high;

  for (const auto &candle : data) {
    min_price = std::min(min_price, candle.low);
    max_price = std::max(max_price, candle.high);
  }

  // Create price levels
  int num_levels = static_cast<int>((max_price - min_price) / tick_size_) + 1;
  std::vector<VolumeNode> nodes(num_levels);

  for (int i = 0; i < num_levels; ++i) {
    nodes[i].price_level = min_price + (i * tick_size_);
    nodes[i].volume = 0.0;
    nodes[i].buy_volume = 0.0;
    nodes[i].sell_volume = 0.0;
    nodes[i].trade_count = 0;
  }

  // Aggregate volume by price level
  for (const auto &trade : trades) {
    if (trade.timestamp >= profile.start_time &&
        trade.timestamp <= profile.end_time) {
      int level_index =
          static_cast<int>((trade.price - min_price) / tick_size_);
      if (level_index >= 0 && level_index < num_levels) {
        nodes[level_index].volume += trade.size;
        nodes[level_index].trade_count++;

        if (trade.is_buy) {
          nodes[level_index].buy_volume += trade.size;
        } else {
          nodes[level_index].sell_volume += trade.size;
        }
      }
    }
  }

  // Calculate total volume
  profile.total_volume = 0.0;
  for (const auto &node : nodes) {
    profile.total_volume += node.volume;
  }

  // Find Point of Control (highest volume level)
  double max_volume = 0.0;
  for (const auto &node : nodes) {
    if (node.volume > max_volume) {
      max_volume = node.volume;
      profile.poc_price = node.price_level;
    }
  }

  // Calculate Value Area (70% of volume)
  calculate_value_area(nodes, profile);

  profile.nodes = nodes;
  return profile;
}

std::vector<VolumeProfileAnalyzer::VolumeImbalance>
VolumeProfileAnalyzer::detect_volume_imbalances(
    const VolumeProfileAnalyzer::VolumeProfile &profile, double threshold) {
  std::vector<VolumeImbalance> imbalances;

  for (const auto &node : profile.nodes) {
    if (node.volume > 0) {
      double ratio = 0.0;
      if (node.sell_volume > 0) {
        ratio = node.buy_volume / node.sell_volume;
      } else if (node.buy_volume > 0) {
        ratio = std::numeric_limits<double>::infinity();
      }

      VolumeImbalance imbalance;
      imbalance.price_level = node.price_level;
      imbalance.imbalance_ratio = ratio;
      imbalance.buy_volume = node.buy_volume;
      imbalance.sell_volume = node.sell_volume;
      imbalance.is_significant =
          (ratio > threshold || ratio < (1.0 / threshold));

      if (imbalance.is_significant) {
        imbalances.push_back(imbalance);
      }
    }
  }

  return imbalances;
}

// End of VolumeProfileAnalyzer implementation

void VolumeProfileAnalyzer::calculate_value_area(
    const std::vector<VolumeProfileAnalyzer::VolumeNode> &nodes,
    VolumeProfileAnalyzer::VolumeProfile &profile) {
  // Find the 70% value area around POC
  double target_volume = profile.total_volume * 0.7;
  double accumulated_volume = 0.0;

  // Find POC index
  int poc_index = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].price_level == profile.poc_price) {
      poc_index = static_cast<int>(i);
      break;
    }
  }

  // Expand from POC until we reach 70% of volume
  int low_index = poc_index;
  int high_index = poc_index;
  accumulated_volume = nodes[poc_index].volume;

  while (accumulated_volume < target_volume &&
         (low_index > 0 || high_index < static_cast<int>(nodes.size()) - 1)) {
    double low_volume = (low_index > 0) ? nodes[low_index - 1].volume : 0.0;
    double high_volume = (high_index < static_cast<int>(nodes.size()) - 1)
                             ? nodes[high_index + 1].volume
                             : 0.0;

    if (low_volume >= high_volume && low_index > 0) {
      low_index--;
      accumulated_volume += nodes[low_index].volume;
    } else if (high_index < static_cast<int>(nodes.size()) - 1) {
      high_index++;
      accumulated_volume += nodes[high_index].volume;
    } else {
      break;
    }
  }

  profile.value_area_low = nodes[low_index].price_level;
  profile.value_area_high = nodes[high_index].price_level;
}
// End of VolumeProfileAnalyzer implementation

// ============================================================================
// Market Depth Visualization
// ============================================================================

MarketDepthAnalyzer::MarketDepthSnapshot
MarketDepthAnalyzer::create_depth_snapshot(
    const RenderEngine::OrderbookData &orderbook) {
  MarketDepthSnapshot snapshot;
  snapshot.timestamp = orderbook.timestamp;
  snapshot.mid_price =
      (orderbook.bids.empty() || orderbook.asks.empty())
          ? 0.0
          : (orderbook.bids[0].price + orderbook.asks[0].price) / 2.0;
  snapshot.spread = orderbook.spread;

  // Process bids
  double cumulative_bid = 0.0;
  for (size_t i = 0; i < orderbook.bids.size(); ++i) {
    const auto &level = orderbook.bids[i];

    DepthLevel depth_level;
    depth_level.price = level.price;
    depth_level.size = level.size;
    cumulative_bid += level.size;
    depth_level.cumulative_size = cumulative_bid;
    depth_level.order_count = 1; // Simplified
    depth_level.average_order_size = level.size;

    snapshot.bids.push_back(depth_level);
  }

  // Process asks
  double cumulative_ask = 0.0;
  for (size_t i = 0; i < orderbook.asks.size(); ++i) {
    const auto &level = orderbook.asks[i];

    DepthLevel depth_level;
    depth_level.price = level.price;
    depth_level.size = level.size;
    cumulative_ask += level.size;
    depth_level.cumulative_size = cumulative_ask;
    depth_level.order_count = 1; // Simplified
    depth_level.average_order_size = level.size;

    snapshot.asks.push_back(depth_level);
  }

  snapshot.total_bid_volume = cumulative_bid;
  snapshot.total_ask_volume = cumulative_ask;

  // Calculate imbalance ratio
  if (snapshot.total_ask_volume > 0) {
    snapshot.imbalance_ratio =
        snapshot.total_bid_volume / snapshot.total_ask_volume;
  } else {
    snapshot.imbalance_ratio = std::numeric_limits<double>::infinity();
  }

  return snapshot;
}

MarketDepthAnalyzer::DepthAnalysis MarketDepthAnalyzer::analyze_market_depth(
    const MarketDepthAnalyzer::MarketDepthSnapshot &snapshot) {
  DepthAnalysis analysis;

  // Find support and resistance levels
  analysis.support_level = find_support_level(snapshot.bids);
  analysis.resistance_level = find_resistance_level(snapshot.asks);

  // Calculate liquidity score
  analysis.liquidity_score = calculate_liquidity_score(snapshot);

  // Estimate market impact
  analysis.market_impact_estimate =
      estimate_market_impact(snapshot, 10000.0); // $10k order

  // Find significant levels
  analysis.significant_levels = find_significant_levels(snapshot);

  return analysis;
}

std::vector<MarketDepthAnalyzer::LiquidityGap>
MarketDepthAnalyzer::detect_liquidity_gaps(
    const MarketDepthAnalyzer::MarketDepthSnapshot &snapshot,
    double min_gap_size) {
  std::vector<LiquidityGap> gaps;

  // Check bid side gaps
  for (size_t i = 1; i < snapshot.bids.size(); ++i) {
    double gap = snapshot.bids[i - 1].price - snapshot.bids[i].price;
    if (gap > min_gap_size) {
      LiquidityGap liquidity_gap;
      liquidity_gap.price_start = snapshot.bids[i].price;
      liquidity_gap.price_end = snapshot.bids[i - 1].price;
      liquidity_gap.gap_size = gap;
      liquidity_gap.is_bid_side = true;
      gaps.push_back(liquidity_gap);
    }
  }

  // Check ask side gaps
  for (size_t i = 1; i < snapshot.asks.size(); ++i) {
    double gap = snapshot.asks[i].price - snapshot.asks[i - 1].price;
    if (gap > min_gap_size) {
      LiquidityGap liquidity_gap;
      liquidity_gap.price_start = snapshot.asks[i - 1].price;
      liquidity_gap.price_end = snapshot.asks[i].price;
      liquidity_gap.gap_size = gap;
      liquidity_gap.is_bid_side = false;
      gaps.push_back(liquidity_gap);
    }
  }

  return gaps;
}

// End of MarketDepthAnalyzer implementation
double MarketDepthAnalyzer::find_support_level(
    const std::vector<MarketDepthAnalyzer::DepthLevel> &bids) {
  if (bids.empty())
    return 0.0;
  double max_volume = 0.0;
  double support_price = bids[0].price;
  for (const auto &level : bids) {
    if (level.size > max_volume) {
      max_volume = level.size;
      support_price = level.price;
    }
  }
  return support_price;
}

double MarketDepthAnalyzer::find_resistance_level(
    const std::vector<MarketDepthAnalyzer::DepthLevel> &asks) {
  if (asks.empty())
    return 0.0;
  double max_volume = 0.0;
  double resistance_price = asks[0].price;
  for (const auto &level : asks) {
    if (level.size > max_volume) {
      max_volume = level.size;
      resistance_price = level.price;
    }
  }
  return resistance_price;
}

double MarketDepthAnalyzer::calculate_liquidity_score(
    const MarketDepthSnapshot &snapshot) {
  double volume_score =
      std::log(snapshot.total_bid_volume + snapshot.total_ask_volume + 1.0);
  double spread_score = 1.0 / (1.0 + snapshot.spread);
  double depth_score =
      std::min(snapshot.bids.size(), snapshot.asks.size()) / 20.0;
  return (volume_score * 0.5) + (spread_score * 0.3) + (depth_score * 0.2);
}

double
MarketDepthAnalyzer::estimate_market_impact(const MarketDepthSnapshot &snapshot,
                                            double order_value) {
  double remaining_value = order_value;
  double weighted_price = 0.0;
  double total_quantity = 0.0;
  for (const auto &level : snapshot.asks) {
    double level_value = level.price * level.size;
    if (remaining_value <= level_value) {
      double quantity = remaining_value / level.price;
      weighted_price += level.price * quantity;
      total_quantity += quantity;
      remaining_value = 0;
      break;
    } else {
      weighted_price += level.price * level.size;
      total_quantity += level.size;
      remaining_value -= level_value;
    }
  }
  if (total_quantity > 0) {
    double average_price = weighted_price / total_quantity;
    return (average_price - snapshot.mid_price) / snapshot.mid_price;
  }
  return 0.0;
}

std::vector<double> MarketDepthAnalyzer::find_significant_levels(
    const MarketDepthSnapshot &snapshot) {
  std::vector<double> levels;
  double avg_bid_volume = 0.0, avg_ask_volume = 0.0;
  for (const auto &level : snapshot.bids)
    avg_bid_volume += level.size;
  if (!snapshot.bids.empty())
    avg_bid_volume /= snapshot.bids.size();
  for (const auto &level : snapshot.asks)
    avg_ask_volume += level.size;
  if (!snapshot.asks.empty())
    avg_ask_volume /= snapshot.asks.size();
  for (const auto &level : snapshot.bids)
    if (level.size > avg_bid_volume * 2.0)
      levels.push_back(level.price);
  for (const auto &level : snapshot.asks)
    if (level.size > avg_ask_volume * 2.0)
      levels.push_back(level.price);
  return levels;
}

// ============================================================================
// Pattern Recognition System
// ============================================================================

std::vector<BTQuant::PatternRecognizer::Pattern>
PatternRecognizer::detect_patterns(
    const std::vector<BTQuant::TechnicalIndicators::OHLCV> &data) {
  std::vector<Pattern> patterns;

  if (data.size() < 10)
    return patterns;

  // Detect various patterns
  auto double_tops = detect_double_top(data);
  patterns.insert(patterns.end(), double_tops.begin(), double_tops.end());

  auto double_bottoms = detect_double_bottom(data);
  patterns.insert(patterns.end(), double_bottoms.begin(), double_bottoms.end());

  auto head_shoulders = detect_head_and_shoulders(data);
  patterns.insert(patterns.end(), head_shoulders.begin(), head_shoulders.end());

  auto triangles = detect_triangles(data);
  patterns.insert(patterns.end(), triangles.begin(), triangles.end());

  auto candlestick_patterns = detect_candlestick_patterns(data);
  patterns.insert(patterns.end(), candlestick_patterns.begin(),
                  candlestick_patterns.end());

  return patterns;
}

std::vector<PatternRecognizer::Pattern> PatternRecognizer::detect_double_top(
    const std::vector<TechnicalIndicators::OHLCV> &data) {
  std::vector<Pattern> patterns;

  // Find local maxima
  std::vector<size_t> peaks = find_peaks(data, true);

  if (peaks.size() < 2)
    return patterns;

  // Look for double top pattern
  for (size_t i = 1; i < peaks.size(); ++i) {
    size_t peak1_idx = peaks[i - 1];
    size_t peak2_idx = peaks[i];

    double peak1_price = data[peak1_idx].high;
    double peak2_price = data[peak2_idx].high;

    // Check if peaks are similar height (within 2%)
    double price_diff = std::abs(peak1_price - peak2_price) / peak1_price;
    if (price_diff < 0.02) {
      // Find valley between peaks
      double valley_price = std::numeric_limits<double>::max();
      size_t valley_idx = peak1_idx;

      for (size_t j = peak1_idx + 1; j < peak2_idx; ++j) {
        if (data[j].low < valley_price) {
          valley_price = data[j].low;
          valley_idx = j;
        }
      }

      // Check if valley is significantly lower (at least 3% below peaks)
      double valley_depth =
          (std::min(peak1_price, peak2_price) - valley_price) /
          std::min(peak1_price, peak2_price);

      if (valley_depth > 0.03) {
        Pattern pattern;
        pattern.type = PatternType::DoubleTop;
        pattern.name = "Double Top";
        pattern.confidence = calculate_double_top_confidence(
            data, peak1_idx, valley_idx, peak2_idx);
        pattern.start_time = data[peak1_idx].timestamp;
        pattern.end_time = data[peak2_idx].timestamp;
        pattern.entry_price = valley_price;
        pattern.target_price =
            valley_price - (std::max(peak1_price, peak2_price) - valley_price);
        pattern.stop_loss = std::max(peak1_price, peak2_price);

        pattern.key_points.push_back(glm::vec2(peak1_idx, peak1_price));
        pattern.key_points.push_back(glm::vec2(valley_idx, valley_price));
        pattern.key_points.push_back(glm::vec2(peak2_idx, peak2_price));

        pattern.description =
            "Bearish reversal pattern with two peaks at similar levels";

        patterns.push_back(pattern);
      }
    }
  }

  return patterns;
}

std::vector<PatternRecognizer::Pattern> PatternRecognizer::detect_double_bottom(
    const std::vector<TechnicalIndicators::OHLCV> &data) {
  std::vector<Pattern> patterns;

  // Find local minima
  std::vector<size_t> troughs = find_peaks(data, false);

  if (troughs.size() < 2)
    return patterns;

  // Look for double bottom pattern
  for (size_t i = 1; i < troughs.size(); ++i) {
    size_t trough1_idx = troughs[i - 1];
    size_t trough2_idx = troughs[i];

    double trough1_price = data[trough1_idx].low;
    double trough2_price = data[trough2_idx].low;

    // Check if troughs are similar depth (within 2%)
    double price_diff = std::abs(trough1_price - trough2_price) / trough1_price;
    if (price_diff < 0.02) {
      // Find peak between troughs
      double peak_price = 0.0;
      size_t peak_idx = trough1_idx;

      for (size_t j = trough1_idx + 1; j < trough2_idx; ++j) {
        if (data[j].high > peak_price) {
          peak_price = data[j].high;
          peak_idx = j;
        }
      }

      // Check if peak is significantly higher (at least 3% above troughs)
      double peak_height =
          (peak_price - std::max(trough1_price, trough2_price)) /
          std::max(trough1_price, trough2_price);

      if (peak_height > 0.03) {
        Pattern pattern;
        pattern.type = PatternType::DoubleBottom;
        pattern.name = "Double Bottom";
        pattern.confidence = calculate_double_bottom_confidence(
            data, trough1_idx, peak_idx, trough2_idx);
        pattern.start_time = data[trough1_idx].timestamp;
        pattern.end_time = data[trough2_idx].timestamp;
        pattern.entry_price = peak_price;
        pattern.target_price =
            peak_price + (peak_price - std::min(trough1_price, trough2_price));
        pattern.stop_loss = std::min(trough1_price, trough2_price);

        pattern.key_points.push_back(glm::vec2(trough1_idx, trough1_price));
        pattern.key_points.push_back(glm::vec2(peak_idx, peak_price));
        pattern.key_points.push_back(glm::vec2(trough2_idx, trough2_price));

        pattern.description =
            "Bullish reversal pattern with two troughs at similar levels";

        patterns.push_back(pattern);
      }
    }
  }

  return patterns;
}

std::vector<PatternRecognizer::Pattern>
PatternRecognizer::detect_head_and_shoulders(
    const std::vector<TechnicalIndicators::OHLCV> &data) {
  std::vector<Pattern> patterns;

  std::vector<size_t> peaks = find_peaks(data, true);
  if (peaks.size() < 3)
    return patterns;

  // Look for head and shoulders pattern (3 peaks with middle one highest)
  for (size_t i = 2; i < peaks.size(); ++i) {
    size_t left_shoulder = peaks[i - 2];
    size_t head = peaks[i - 1];
    size_t right_shoulder = peaks[i];

    double left_price = data[left_shoulder].high;
    double head_price = data[head].high;
    double right_price = data[right_shoulder].high;

    // Check if head is higher than shoulders
    if (head_price > left_price && head_price > right_price) {
      // Check if shoulders are similar height (within 5%)
      double shoulder_diff = std::abs(left_price - right_price) / left_price;
      if (shoulder_diff < 0.05) {
        // Find neckline (valleys between shoulders and head)
        double left_valley = find_valley_between(data, left_shoulder, head);
        double right_valley = find_valley_between(data, head, right_shoulder);
        double neckline = (left_valley + right_valley) / 2.0;

        Pattern pattern;
        pattern.type = PatternType::HeadAndShoulders;
        pattern.name = "Head and Shoulders";
        pattern.confidence = calculate_head_shoulders_confidence(
            data, left_shoulder, head, right_shoulder);
        pattern.start_time = data[left_shoulder].timestamp;
        pattern.end_time = data[right_shoulder].timestamp;
        pattern.entry_price = neckline;
        pattern.target_price = neckline - (head_price - neckline);
        pattern.stop_loss = head_price;

        pattern.key_points.push_back(glm::vec2(left_shoulder, left_price));
        pattern.key_points.push_back(glm::vec2(head, head_price));
        pattern.key_points.push_back(glm::vec2(right_shoulder, right_price));

        pattern.description =
            "Bearish reversal pattern with head higher than two shoulders";

        patterns.push_back(pattern);
      }
    }
  }

  return patterns;
}

std::vector<PatternRecognizer::Pattern> PatternRecognizer::detect_triangles(
    const std::vector<TechnicalIndicators::OHLCV> &data) {
  std::vector<Pattern> patterns;

  // Simplified triangle detection
  if (data.size() < 20)
    return patterns;

  for (size_t i = 10; i < data.size() - 10; ++i) {
    // Look for converging trend lines
    std::vector<glm::vec2> highs, lows;

    // Collect recent highs and lows
    for (size_t j = i - 10; j <= i + 10; ++j) {
      if (is_local_high(data, j)) {
        highs.push_back(glm::vec2(j, data[j].high));
      }
      if (is_local_low(data, j)) {
        lows.push_back(glm::vec2(j, data[j].low));
      }
    }

    if (highs.size() >= 2 && lows.size() >= 2) {
      // Calculate trend lines
      auto high_trend = calculate_trend_line(highs);
      auto low_trend = calculate_trend_line(lows);

      // Check if lines are converging
      if (std::abs(high_trend.second) > 0.001 &&
          std::abs(low_trend.second) > 0.001) {
        if ((high_trend.second < 0 && low_trend.second > 0) ||
            (high_trend.second > 0 && low_trend.second < 0)) {

          Pattern pattern;
          pattern.type = PatternType::Triangle;
          pattern.name = "Triangle";
          pattern.confidence = 0.7; // Simplified confidence
          pattern.start_time = data[i - 10].timestamp;
          pattern.end_time = data[i + 10].timestamp;

          pattern.description =
              "Consolidation pattern with converging trend lines";
          patterns.push_back(pattern);
        }
      }
    }
  }

  return patterns;
}

std::vector<PatternRecognizer::Pattern>
PatternRecognizer::detect_candlestick_patterns(
    const std::vector<TechnicalIndicators::OHLCV> &data) {
  std::vector<Pattern> patterns;

  for (size_t i = 1; i < data.size(); ++i) {
    // Hammer pattern
    if (is_hammer(data[i])) {
      Pattern pattern;
      pattern.type = PatternType::Hammer;
      pattern.name = "Hammer";
      pattern.confidence = 0.8;
      pattern.start_time = data[i].timestamp;
      pattern.end_time = data[i].timestamp;
      pattern.description = "Bullish reversal candlestick pattern";
      patterns.push_back(pattern);
    }

    // Doji pattern
    if (is_doji(data[i])) {
      Pattern pattern;
      pattern.type = PatternType::Doji;
      pattern.name = "Doji";
      pattern.confidence = 0.6;
      pattern.start_time = data[i].timestamp;
      pattern.end_time = data[i].timestamp;
      pattern.description = "Indecision candlestick pattern";
      patterns.push_back(pattern);
    }

    // Engulfing patterns
    if (i > 0) {
      if (is_bullish_engulfing(data[i - 1], data[i])) {
        Pattern pattern;
        pattern.type = PatternType::EngulfingBullish;
        pattern.name = "Bullish Engulfing";
        pattern.confidence = 0.85;
        pattern.start_time = data[i - 1].timestamp;
        pattern.end_time = data[i].timestamp;
        pattern.description = "Bullish reversal two-candle pattern";
        patterns.push_back(pattern);
      }

      if (is_bearish_engulfing(data[i - 1], data[i])) {
        Pattern pattern;
        pattern.type = PatternType::EngulfingBearish;
        pattern.name = "Bearish Engulfing";
        pattern.confidence = 0.85;
        pattern.start_time = data[i - 1].timestamp;
        pattern.end_time = data[i].timestamp;
        pattern.description = "Bearish reversal two-candle pattern";
        patterns.push_back(pattern);
      }
    }
  }

  return patterns;
}

// Helper functions
std::vector<size_t> PatternRecognizer::find_peaks(
    const std::vector<TechnicalIndicators::OHLCV> &data, bool find_highs) {
  std::vector<size_t> peaks;

  for (size_t i = 2; i < data.size() - 2; ++i) {
    if (find_highs) {
      if (data[i].high > data[i - 1].high && data[i].high > data[i - 2].high &&
          data[i].high > data[i + 1].high && data[i].high > data[i + 2].high) {
        peaks.push_back(i);
      }
    } else {
      if (data[i].low < data[i - 1].low && data[i].low < data[i - 2].low &&
          data[i].low < data[i + 1].low && data[i].low < data[i + 2].low) {
        peaks.push_back(i);
      }
    }
  }

  return peaks;
}

double PatternRecognizer::find_valley_between(
    const std::vector<TechnicalIndicators::OHLCV> &data, size_t start,
    size_t end) {
  double min_price = std::numeric_limits<double>::max();
  for (size_t i = start + 1; i < end; ++i) {
    min_price = std::min(min_price, data[i].low);
  }
  return min_price;
}

bool PatternRecognizer::is_local_high(
    const std::vector<TechnicalIndicators::OHLCV> &data, size_t index) {
  if (index < 2 || index >= data.size() - 2)
    return false;
  return data[index].high > data[index - 1].high &&
         data[index].high > data[index + 1].high;
}

bool PatternRecognizer::is_local_low(
    const std::vector<TechnicalIndicators::OHLCV> &data, size_t index) {
  if (index < 2 || index >= data.size() - 2)
    return false;
  return data[index].low < data[index - 1].low &&
         data[index].low < data[index + 1].low;
}

std::pair<double, double>
PatternRecognizer::calculate_trend_line(const std::vector<glm::vec2> &points) {
  if (points.size() < 2)
    return {0.0, 0.0};

  // Simple linear regression
  double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
  int n = points.size();

  for (const auto &point : points) {
    sum_x += point.x;
    sum_y += point.y;
    sum_xy += point.x * point.y;
    sum_x2 += point.x * point.x;
  }

  double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
  double intercept = (sum_y - slope * sum_x) / n;

  return {intercept, slope};
}

bool PatternRecognizer::is_hammer(const TechnicalIndicators::OHLCV &candle) {
  double body_size = std::abs(candle.close - candle.open);
  double lower_shadow = std::min(candle.open, candle.close) - candle.low;
  double upper_shadow = candle.high - std::max(candle.open, candle.close);

  return (lower_shadow > body_size * 2) && (upper_shadow < body_size * 0.5);
}

bool PatternRecognizer::is_doji(const TechnicalIndicators::OHLCV &candle) {
  double body_size = std::abs(candle.close - candle.open);
  double total_range = candle.high - candle.low;

  return (body_size / total_range) < 0.1;
}

bool PatternRecognizer::is_bullish_engulfing(
    const TechnicalIndicators::OHLCV &prev,
    const TechnicalIndicators::OHLCV &curr) {
  bool prev_bearish = prev.close < prev.open;
  bool curr_bullish = curr.close > curr.open;
  bool engulfing = curr.open < prev.close && curr.close > prev.open;

  return prev_bearish && curr_bullish && engulfing;
}

bool PatternRecognizer::is_bearish_engulfing(
    const TechnicalIndicators::OHLCV &prev,
    const TechnicalIndicators::OHLCV &curr) {
  bool prev_bullish = prev.close > prev.open;
  bool curr_bearish = curr.close < curr.open;
  bool engulfing = curr.open > prev.close && curr.close < prev.open;

  return prev_bullish && curr_bearish && engulfing;
}

double PatternRecognizer::calculate_double_top_confidence(
    const std::vector<TechnicalIndicators::OHLCV> &data, size_t peak1,
    size_t valley, size_t peak2) {
  // Simplified confidence calculation
  double peak_similarity =
      1.0 - std::abs(data[peak1].high - data[peak2].high) / data[peak1].high;
  double valley_depth =
      (std::min(data[peak1].high, data[peak2].high) - data[valley].low) /
      std::min(data[peak1].high, data[peak2].high);

  return std::min(1.0, peak_similarity * valley_depth * 2.0);
}

double PatternRecognizer::calculate_double_bottom_confidence(
    const std::vector<TechnicalIndicators::OHLCV> &data, size_t trough1,
    size_t peak, size_t trough2) {
  // Simplified confidence calculation
  double trough_similarity =
      1.0 - std::abs(data[trough1].low - data[trough2].low) / data[trough1].low;
  double peak_height =
      (data[peak].high - std::max(data[trough1].low, data[trough2].low)) /
      std::max(data[trough1].low, data[trough2].low);

  return std::min(1.0, trough_similarity * peak_height * 2.0);
}

double PatternRecognizer::calculate_head_shoulders_confidence(
    const std::vector<TechnicalIndicators::OHLCV> &data, size_t left,
    size_t head, size_t right) {
  // Simplified confidence calculation
  double shoulder_similarity =
      1.0 - std::abs(data[left].high - data[right].high) / data[left].high;
  double head_prominence =
      (data[head].high - std::max(data[left].high, data[right].high)) /
      data[head].high;

  return std::min(1.0, shoulder_similarity * head_prominence * 3.0);
}
// End of PatternRecognizer implementation

} // namespace BTQuant