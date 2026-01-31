#include "detectors/stop_hunt_detector.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace BTQuant {

StopHuntDetector::StopHuntDetector(HotSpineExtendedReader &reader)
    : reader_(reader) {}

double StopHuntDetector::calculate_median(std::vector<double> &values) {
  if (values.empty())
    return 0.0;

  std::sort(values.begin(), values.end());
  size_t n = values.size();

  if (n % 2 == 0) {
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
  } else {
    return values[n / 2];
  }
}

std::optional<StopHuntSignal>
StopHuntDetector::detect(const std::string &symbol) {
  // Get prices from all exchanges
  auto prices = reader_.get_all_exchange_prices(symbol);

  if (prices.size() < min_exchanges_) {
    return std::nullopt;
  }

  // Calculate median price (excluding outliers)
  std::vector<double> price_values;
  price_values.reserve(prices.size());
  for (const auto &[exchange, price] : prices) {
    price_values.push_back(price);
  }

  double median_price = calculate_median(price_values);
  if (median_price == 0.0) {
    return std::nullopt;
  }

  // Check each exchange for deviation
  for (const auto &[exchange, price] : prices) {
    double deviation_pct = ((price - median_price) / median_price) * 100.0;

    if (std::abs(deviation_pct) > threshold_pct_) {
      // Found an outlier - this could be a stop hunt
      StopHuntSignal signal;
      signal.symbol = symbol;
      signal.hunt_exchange = exchange;
      signal.hunt_price = price;
      signal.median_price = median_price;
      signal.hunt_deviation_pct = deviation_pct;
      signal.timestamp_us = reader_.get_current_time_us();
      signal.is_long_signal =
          deviation_pct < 0; // Negative = dump = long opportunity

      // Add stable exchanges
      for (const auto &[other_exchange, other_price] : prices) {
        if (other_exchange != exchange) {
          signal.stable_exchanges.emplace_back(other_exchange, other_price);
        }
      }

      detections_++;
      return signal;
    }
  }

  return std::nullopt;
}

std::string StopHuntSignal::to_string() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "StopHunt(symbol=" << symbol << ", exchange=" << hunt_exchange
      << ", price=" << hunt_price << ", median=" << median_price
      << ", deviation=" << hunt_deviation_pct << "%"
      << ", signal=" << (is_long_signal ? "LONG" : "SHORT") << ")";
  return oss.str();
}

} // namespace BTQuant