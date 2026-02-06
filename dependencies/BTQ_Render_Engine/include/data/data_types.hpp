#pragma once

#include <vector>
#include <cstdint>

// Price level for order book data
struct PriceLevel {
  double price;
  double size;
  uint64_t timestamp = 0;  // Optional timestamp for individual price levels
};

struct VolumeProfileLevel {
  double price;
  double total_volume;
  double buy_volume;
  double sell_volume;
};

namespace BTQuant {
namespace RenderEngine {

// Time frame definitions for OHLCV aggregation
// Extended to include higher timeframes for multi-timeframe analysis
enum class TimeFrame {
  TF_1MS,    // 1 millisecond
  TF_10MS,   // 10 milliseconds
  TF_100MS,  // 100 milliseconds
  TF_500MS,  // 500 milliseconds
  TF_1SEC,   // 1 second
  TF_3SEC,   // 3 seconds
  TF_5SEC,   // 5 seconds
  TF_15SEC,  // 15 seconds
  TF_30SEC,  // 30 seconds
  TF_1MIN,   // 1 minute
  TF_2MIN,   // 2 minutes
  TF_5MIN,   // 5 minutes
  TF_15MIN,  // 15 minutes
  TF_30MIN,  // 30 minutes
  TF_1HOUR,  // 1 hour
  TF_2HOUR,  // 2 hours
  TF_4HOUR,  // 4 hours
  TF_6HOUR,  // 6 hours
  TF_12HOUR, // 12 hours
  TF_1DAY,   // 1 day
  TF_1WEEK   // 1 week
};

// OHLCV Candle Data for Charting
struct OHLCVCandle {
  uint64_t timestamp;  // Start time of the candle in microseconds
  double open;
  double high;
  double low;
  double close;
  double volume;
  uint64_t trade_count;
};

} // namespace RenderEngine
} // namespace BTQuant