#include "detectors/whale_frontrun_detector.hpp"
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace BTQuant {

WhaleFrontRunDetector::WhaleFrontRunDetector(HotSpineExtendedReader &reader)
    : reader_(reader), whale_threshold_usd_(100000.0),
      min_lagging_exchanges_(2), reaction_threshold_bps_(50.0),
      reaction_window_us_(1000000), detections_(0) {}

std::string WhaleFrontRunDetector::make_key(const std::string &exchange,
                                            const std::string &symbol) const {
  return exchange + ":" + symbol;
}

void WhaleFrontRunDetector::PriceHistory::add(uint64_t timestamp_us,
                                              double price) {
  prices.push_back({timestamp_us, price});
  if (prices.size() > MAX_HISTORY) {
    prices.pop_front();
  }
}

double
WhaleFrontRunDetector::PriceHistory::get_volatility_bps(uint64_t lookback_us) {
  if (prices.size() < 2)
    return 0.0;

  uint64_t cutoff = prices.back().first - lookback_us;
  std::vector<double> returns;

  for (size_t i = 1; i < prices.size(); ++i) {
    if (prices[i - 1].first >= cutoff) {
      double ret = (prices[i].second / prices[i - 1].second - 1.0) * 10000.0;
      returns.push_back(ret);
    }
  }

  if (returns.empty())
    return 0.0;

  double mean =
      std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
  double sq_sum = 0.0;
  for (double r : returns) {
    sq_sum += (r - mean) * (r - mean);
  }

  return std::sqrt(sq_sum / returns.size());
}

std::optional<WhaleSignal>
WhaleFrontRunDetector::detect(const std::string &symbol) {
  uint64_t now = reader_.get_current_time_us();

  // Get recent trades from all exchanges
  auto exchanges = SymbolRegistry::instance().get_exchanges();

  for (const auto &exchange : exchanges) {
    auto recent_trades = reader_.get_recent_trades(exchange, symbol, 10);

    // Update price history
    for (const auto &trade : recent_trades) {
      auto &history = price_histories_[make_key(exchange, symbol)];
      history.add(trade.ts_local, trade.price);
    }

    // Check for whale trades
    for (const auto &trade : recent_trades) {
      // Skip old trades (>5 seconds)
      if (now - trade.ts_local > 5'000'000)
        continue;

      double trade_size_usd = trade.price * trade.size;

      if (trade_size_usd >= whale_threshold_usd_) {
        // Found a whale trade - check if other exchanges have reacted
        std::vector<std::string> lagging;

        for (const auto &other_exchange : exchanges) {
          if (other_exchange == exchange)
            continue;

          auto other_price = reader_.get_latest_price(other_exchange, symbol);
          if (!other_price)
            continue;

          double price_diff_bps =
              std::abs((*other_price - trade.price) / trade.price) * 10000.0;

          // If price difference is small, exchange hasn't reacted yet
          if (price_diff_bps < reaction_threshold_bps_) {
            lagging.push_back(other_exchange);
          }
        }

        // If enough exchanges are lagging, we have a front-run opportunity
        if (lagging.size() >= min_lagging_exchanges_) {
          WhaleSignal signal;
          signal.symbol = symbol;
          signal.whale_exchange = exchange;
          signal.whale_price = trade.price;
          signal.whale_size = trade.size;
          signal.whale_size_usd = trade_size_usd;
          signal.is_buy = trade.is_buy();
          signal.lagging_exchanges = lagging;
          signal.timestamp_us = now;

          // Estimate expected move based on recent volatility
          auto &history = price_histories_[make_key(exchange, symbol)];
          double vol_bps = history.get_volatility_bps(60'000'000); // 1 minute
          signal.expected_move_bps = vol_bps * 0.5; // Conservative estimate

          detections_++;
          return signal;
        }
      }
    }
  }

  return std::nullopt;
}

std::string WhaleSignal::to_string() const {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  oss << "WhaleDetected(symbol=" << symbol << ", exchange=" << whale_exchange
      << ", size=$" << whale_size_usd << ", side=" << (is_buy ? "BUY" : "SELL")
      << ", lagging=" << lagging_exchanges.size() << " exchanges"
      << ", expected_move=" << expected_move_bps << "bps"
      << ")";
  return oss.str();
}

} // namespace BTQuant