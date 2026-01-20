#include "../../include/data/candle_aggregator.hpp"
#include "../../include/market_data_processor.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace BTQuant {

double
CandleAggregator::get_interval_seconds(RenderEngine::TimeFrame tf) const {
  using TF = RenderEngine::TimeFrame;
  switch (tf) {
  case TF::TF_TICK:
    return 0.0; // Special: no bucketing
  case TF::TF_100MS:
    return 0.1;
  case TF::TF_200MS:
    return 0.2;
  case TF::TF_500MS:
    return 0.5;
  case TF::TF_1SEC:
    return 1.0;
  case TF::TF_5SEC:
    return 5.0;
  case TF::TF_15SEC:
    return 15.0;
  case TF::TF_30SEC:
    return 30.0;
  case TF::TF_1MIN:
    return 60.0;
  case TF::TF_2MIN:
    return 120.0;
  case TF::TF_3MIN:
    return 180.0;
  case TF::TF_5MIN:
    return 300.0;
  case TF::TF_15MIN:
    return 900.0;
  case TF::TF_30MIN:
    return 1800.0;
  case TF::TF_1HOUR:
    return 3600.0;
  case TF::TF_2HOUR:
    return 7200.0;
  case TF::TF_4HOUR:
    return 14400.0;
  case TF::TF_6HOUR:
    return 21600.0;
  case TF::TF_12HOUR:
    return 43200.0;
  case TF::TF_1DAY:
    return 86400.0;
  case TF::TF_1WEEK:
    return 604800.0;
  default:
    return 60.0; // Default to 1 minute
  }
}

double CandleAggregator::get_bucket_start(double timestamp,
                                          RenderEngine::TimeFrame tf) const {
  double interval = get_interval_seconds(tf);

  // For tick data, return exact timestamp
  if (interval == 0.0) {
    return timestamp;
  }

  // Floor to bucket boundary
  return std::floor(timestamp / interval) * interval;
}

void CandleAggregator::update_bucket(RenderEngine::TimeFrame tf,
                                     double bucket_start, float price,
                                     float size) {
  auto &candle = buckets_[tf][bucket_start];

  if (candle.volume == 0.0f) {
    // First trade in this bucket - initialize
    candle.timestamp = bucket_start;
    candle.open = price;
    candle.high = price;
    candle.low = price;
    candle.close = price;
    candle.volume = size;
  } else {
    // Update existing candle
    candle.high = std::max(candle.high, price);
    candle.low = std::min(candle.low, price);
    candle.close = price; // Latest trade is close
    candle.volume += size;
  }
}

void CandleAggregator::add_trade(double timestamp, float price, float size) {
  using TF = RenderEngine::TimeFrame;

  // DEBUG: Log first few trades
  static int trade_count = 0;
  if (trade_count++ < 5) {
    std::cout << "[CandleAggregator] Trade #" << trade_count
              << " ts=" << std::fixed << std::setprecision(2) << timestamp
              << " price=" << price << " size=" << size << std::endl;
  }

  // Add to all relevant timeframes
  static const std::vector<TF> all_timeframes = {
      TF::TF_TICK,  TF::TF_100MS, TF::TF_200MS, TF::TF_500MS,  TF::TF_1SEC,
      TF::TF_5SEC,  TF::TF_15SEC, TF::TF_30SEC, TF::TF_1MIN,   TF::TF_2MIN,
      TF::TF_3MIN,  TF::TF_5MIN,  TF::TF_15MIN, TF::TF_30MIN,  TF::TF_1HOUR,
      TF::TF_2HOUR, TF::TF_4HOUR, TF::TF_6HOUR, TF::TF_12HOUR, TF::TF_1DAY,
      TF::TF_1WEEK};

  for (auto tf : all_timeframes) {
    double bucket_start = get_bucket_start(timestamp, tf);

    // DEBUG: Log bucket for 1MIN timeframe
    if (tf == TF::TF_1MIN && trade_count <= 5) {
      std::cout << "[CandleAggregator]   TF_1MIN bucket_start=" << std::fixed
                << std::setprecision(2) << bucket_start << std::endl;
    }

    update_bucket(tf, bucket_start, price, size);
  }
}

std::vector<CandleAggregator::Candle>
CandleAggregator::get_candles(RenderEngine::TimeFrame tf, int limit) const {
  std::vector<Candle> result;

  auto it = buckets_.find(tf);
  if (it == buckets_.end()) {
    return result;
  }

  const auto &bucket_map = it->second;

  // Get last 'limit' candles
  int start_idx = std::max(0, static_cast<int>(bucket_map.size()) - limit);
  int current_idx = 0;

  for (const auto &[timestamp, candle] : bucket_map) {
    if (current_idx++ >= start_idx) {
      result.push_back(candle);
    }
  }

  return result;
}

void CandleAggregator::clear() { buckets_.clear(); }

} // namespace BTQuant
