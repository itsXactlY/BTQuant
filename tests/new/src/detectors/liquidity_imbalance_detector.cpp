#include "detectors/liquidity_imbalance_detector.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace BTQuant {

LiquidityImbalanceDetector::LiquidityImbalanceDetector(
    HotSpineExtendedReader &reader)
    : reader_(reader) {}

double
LiquidityImbalanceDetector::calculate_orderbook_depth(const OrderbookData &ob,
                                                      double bps_range) {

  double mid = ob.mid_price();
  if (mid == 0.0)
    return 0.0;

  double bid_threshold = mid * (1.0 - bps_range / 10000.0);
  double ask_threshold = mid * (1.0 + bps_range / 10000.0);

  double bid_depth = 0.0;
  for (const auto &[price, size] : ob.bids) {
    if (price >= bid_threshold) {
      bid_depth += size;
    }
  }

  double ask_depth = 0.0;
  for (const auto &[price, size] : ob.asks) {
    if (price <= ask_threshold) {
      ask_depth += size;
    }
  }

  return bid_depth + ask_depth;
}

double LiquidityImbalanceDetector::calculate_bid_ask_imbalance(
    const OrderbookData &ob) {
  double mid = ob.mid_price();
  if (mid == 0.0)
    return 0.0;

  double bps_range = 100.0;
  double bid_threshold = mid * (1.0 - bps_range / 10000.0);
  double ask_threshold = mid * (1.0 + bps_range / 10000.0);

  double bid_depth = 0.0;
  for (const auto &[price, size] : ob.bids) {
    if (price >= bid_threshold) {
      bid_depth += size;
    }
  }

  double ask_depth = 0.0;
  for (const auto &[price, size] : ob.asks) {
    if (price <= ask_threshold) {
      ask_depth += size;
    }
  }

  double total = bid_depth + ask_depth;
  if (total == 0.0)
    return 0.0;

  return (bid_depth - ask_depth) / total;
}

std::optional<LiquiditySignal>
LiquidityImbalanceDetector::detect(const std::string &symbol) {
  // Get orderbooks from all exchanges
  auto orderbooks = reader_.get_all_exchange_orderbooks(symbol);

  if (orderbooks.size() < 2) {
    return std::nullopt;
  }

  // Calculate depths for each exchange
  struct ExchangeDepth {
    std::string exchange;
    double depth;
    double spread_bps;
    const OrderbookData *ob;
  };

  std::vector<ExchangeDepth> depths;
  depths.reserve(orderbooks.size());

  for (const auto &[exchange, ob] : orderbooks) {
    double depth = calculate_orderbook_depth(ob);
    double spread = ob.spread_bps();
    depths.push_back({exchange, depth, spread, &ob});
  }

  // Find thinnest and thickest
  auto min_it =
      std::min_element(depths.begin(), depths.end(),
                       [](const ExchangeDepth &a, const ExchangeDepth &b) {
                         return a.depth < b.depth;
                       });

  auto max_it =
      std::max_element(depths.begin(), depths.end(),
                       [](const ExchangeDepth &a, const ExchangeDepth &b) {
                         return a.depth < b.depth;
                       });

  if (min_it->depth == 0.0 || max_it->depth == 0.0) {
    return std::nullopt;
  }

  double depth_ratio = max_it->depth / min_it->depth;

  // Check if ratio exceeds threshold
  if (depth_ratio > depth_ratio_threshold_) {
    LiquiditySignal signal;
    signal.symbol = symbol;
    signal.thin_exchange = min_it->exchange;
    signal.thick_exchange = max_it->exchange;
    signal.thin_depth = min_it->depth;
    signal.thick_depth = max_it->depth;
    signal.depth_ratio = depth_ratio;
    signal.thin_spread_bps = min_it->spread_bps;
    signal.thick_spread_bps = max_it->spread_bps;
    signal.timestamp_us = reader_.get_current_time_us();

    // Determine direction from bid/ask imbalance on thin exchange
    double imbalance = calculate_bid_ask_imbalance(*min_it->ob);

    if (imbalance > imbalance_threshold_) {
      signal.direction =
          ImbalanceDirection::SHORT; // Heavy bids = potential dump
    } else if (imbalance < -imbalance_threshold_) {
      signal.direction =
          ImbalanceDirection::LONG; // Heavy asks = potential pump
    } else {
      signal.direction = ImbalanceDirection::NEUTRAL;
    }

    detections_++;
    return signal;
  }

  return std::nullopt;
}

std::string LiquiditySignal::to_string() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "LiquidityImbalance(symbol=" << symbol << ", thin=" << thin_exchange
      << "(depth=" << thin_depth << ")"
      << ", thick=" << thick_exchange << "(depth=" << thick_depth << ")"
      << ", ratio=" << depth_ratio << "x"
      << ", direction=";

  switch (direction) {
  case ImbalanceDirection::LONG:
    oss << "LONG";
    break;
  case ImbalanceDirection::SHORT:
    oss << "SHORT";
    break;
  case ImbalanceDirection::NEUTRAL:
    oss << "NEUTRAL";
    break;
  }

  oss << ")";
  return oss.str();
}

} // namespace BTQuant