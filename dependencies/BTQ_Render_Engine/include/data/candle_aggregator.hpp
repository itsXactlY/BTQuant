#pragma once

#include "../timeframe.hpp"
#include <map>
#include <vector>

namespace BTQuant {

class CandleAggregator {
public:
  struct Candle {
    double timestamp; // Bucket start time in seconds
    float open;
    float high;
    float low;
    float close;
    float volume;

    Candle() : timestamp(0), open(0), high(0), low(0), close(0), volume(0) {}
  };

  CandleAggregator() = default;

  // Add a trade to all timeframes
  void add_trade(double timestamp, float price, float size);

  // Get candles for a specific timeframe
  std::vector<Candle> get_candles(RenderEngine::TimeFrame tf,
                                  int limit = 1000) const;

  // Clear all data
  void clear();

private:
  // Map: TimeFrame -> (BucketStartTime -> Candle)
  std::map<RenderEngine::TimeFrame, std::map<double, Candle>> buckets_;

  // Get bucket start time for a given timestamp and timeframe
  double get_bucket_start(double timestamp, RenderEngine::TimeFrame tf) const;

  // Get interval in seconds for a timeframe
  double get_interval_seconds(RenderEngine::TimeFrame tf) const;

  // Update or create candle in bucket
  void update_bucket(RenderEngine::TimeFrame tf, double bucket_start,
                     float price, float size);
};

} // namespace BTQuant
