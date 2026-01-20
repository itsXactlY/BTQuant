#pragma once

namespace BTQuant {
namespace RenderEngine {

// Time frame definitions for OHLCV aggregation
enum class TimeFrame {
  // Sub-second
  TF_TICK,  // Raw tick data (no aggregation)
  TF_100MS, // 100 milliseconds
  TF_200MS, // 200 milliseconds
  TF_500MS, // 500 milliseconds

  // Seconds
  TF_1SEC,  // 1 second
  TF_5SEC,  // 5 seconds
  TF_15SEC, // 15 seconds
  TF_30SEC, // 30 seconds

  // Minutes
  TF_1MIN,  // 1 minute
  TF_2MIN,  // 2 minutes
  TF_3MIN,  // 3 minutes
  TF_5MIN,  // 5 minutes
  TF_15MIN, // 15 minutes
  TF_30MIN, // 30 minutes

  // Hours
  TF_1HOUR,  // 1 hour
  TF_2HOUR,  // 2 hours
  TF_4HOUR,  // 4 hours
  TF_6HOUR,  // 6 hours
  TF_12HOUR, // 12 hours

  // Days+
  TF_1DAY, // 1 day
  TF_1WEEK // 1 week
};

} // namespace RenderEngine
} // namespace BTQuant
