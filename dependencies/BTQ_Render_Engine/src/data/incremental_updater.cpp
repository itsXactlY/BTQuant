#include "data/incremental_updater.hpp"

// Include market_data_processor.hpp for full type definitions of SymbolAnalytics and TradeData
#include <algorithm>
#include <cmath>
#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "market_data_processor.hpp"

namespace BTQuant {
namespace RenderEngine {

/**
 * @brief Process a single trade incrementally
 * Only updates affected analytics without recalculating everything
 * @param symbol_data The symbol analytics to update
 * @param trade The trade data to process
 * @param dirty_flag Optional dirty flag to set on state change
 */
void processTradeIncrementally(SymbolAnalytics& symbol_data, const BTQuant::TradeData& trade,
                               HeatmapDirtyFlag* dirty_flag) {
  // Track if state changed for dirty flag
  bool state_changed = false;

  // Update basic trade metrics incrementally
  symbol_data.trade_count++;
  symbol_data.last_trade_price = trade.price;
  symbol_data.last_trade_size = trade.volume;  // volume instead of size
  symbol_data.last_trade_time = trade.timestamp_us;  // timestamp_us instead of timestamp
  state_changed = true;  // Trade update always changes state

  // Update buy/sell counts
  if (trade.is_buy()) {  // Use is_buy() method
    symbol_data.buy_count++;
  } else {
    symbol_data.sell_count++;
  }

  // Update buy/sell ratio
  if (symbol_data.buy_count + symbol_data.sell_count > 0) {
    symbol_data.buy_sell_ratio = static_cast<double>(symbol_data.buy_count) /
                                 (symbol_data.buy_count + symbol_data.sell_count);
  }

  // Update VWAP incrementally using running totals
  symbol_data.running_total_price_volume += trade.price * trade.volume;
  symbol_data.running_total_volume += trade.volume;

  // Calculate VWAP from running totals
  if (symbol_data.running_total_volume > 0.0) {
    double new_vwap = symbol_data.running_total_price_volume / symbol_data.running_total_volume;
    symbol_data.vwap = new_vwap;
    symbol_data.vwap_deviation =
        symbol_data.last_trade_price > 0.0
            ? ((symbol_data.last_trade_price - symbol_data.vwap) / symbol_data.vwap) * 100.0
            : 0.0;
  }

  // Update volume metrics incrementally
  uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch())
                        .count();

  if (now_us - symbol_data.last_update_time < 60000000)  // 1 minute
    symbol_data.volume_1m += trade.volume;
  if (now_us - symbol_data.last_update_time < 300000000)  // 5 minutes
    symbol_data.volume_5m += trade.volume;
  if (now_us - symbol_data.last_update_time < 900000000)  // 15 minutes
    symbol_data.volume_15m += trade.volume;

  // Update buy/sell volumes
  if (trade.is_buy()) {
    symbol_data.buy_volume += trade.volume;
  } else {
    symbol_data.sell_volume += trade.volume;
  }

  // Update price ranges incrementally
  if (symbol_data.price_min == 0.0 || trade.price < symbol_data.price_min) {
    symbol_data.price_min = trade.price;
  }
  if (trade.price > symbol_data.price_max) {
    symbol_data.price_max = trade.price;
  }

  // Update price position
  if (symbol_data.price_max > symbol_data.price_min) {
    symbol_data.price_position =
        ((trade.price - symbol_data.price_min) / (symbol_data.price_max - symbol_data.price_min)) *
        100.0;
  }

  // Update trade size metrics incrementally
  symbol_data.avg_trade_size =
      (symbol_data.avg_trade_size * (symbol_data.trade_count - 1) + trade.volume) /
      symbol_data.trade_count;

  // Check for large trades
  if (trade.volume > 2.0 * symbol_data.avg_trade_size) {
    symbol_data.large_trade_count++;
  }

  // Update OHLCV candles incrementally
  // Update candles for all timeframes incrementally
  for (auto timeframe :
       {TimeFrame::TF_1MS,   TimeFrame::TF_10MS,  TimeFrame::TF_100MS,  TimeFrame::TF_500MS,
        TimeFrame::TF_1SEC,  TimeFrame::TF_3SEC,  TimeFrame::TF_5SEC,   TimeFrame::TF_15SEC,
        TimeFrame::TF_30SEC, TimeFrame::TF_1MIN,  TimeFrame::TF_2MIN,   TimeFrame::TF_5MIN,
        TimeFrame::TF_15MIN, TimeFrame::TF_30MIN, TimeFrame::TF_1HOUR,  TimeFrame::TF_2HOUR,
        TimeFrame::TF_4HOUR, TimeFrame::TF_6HOUR, TimeFrame::TF_12HOUR, TimeFrame::TF_1DAY,
        TimeFrame::TF_1WEEK}) {
    uint64_t duration_us = MarketDataProcessor::getTimeFrameDuration(timeframe);
    uint64_t candle_start = (trade.timestamp_us / duration_us) * duration_us;

    auto& current_candle = symbol_data.current_candles[timeframe];

    // Check if we need a new candle
    if (current_candle.timestamp == 0 || candle_start != current_candle.timestamp) {
      // Save previous candle if it exists
      if (current_candle.timestamp != 0) {
        symbol_data.candles[timeframe].push_back(current_candle);
      }

      // Create new candle
      current_candle.timestamp = candle_start;
      current_candle.open = trade.price;
      current_candle.high = trade.price;
      current_candle.low = trade.price;
      current_candle.close = trade.price;
      current_candle.volume = trade.volume;
      current_candle.trade_count = 1;
    } else {
      // Update existing candle incrementally
      current_candle.high = std::max(current_candle.high, trade.price);
      current_candle.low = std::min(current_candle.low, trade.price);
      current_candle.close = trade.price;
      current_candle.volume += trade.volume;
      current_candle.trade_count++;
    }
  }

  // Update volume profile incrementally
  // Update the volume profile level for this price incrementally
  auto& vp_level = symbol_data.session_volume_profile[trade.price];
  vp_level.price = trade.price;
  vp_level.total_volume += trade.volume;
  if (trade.is_buy()) {
    vp_level.buy_volume += trade.volume;
  } else {
    vp_level.sell_volume += trade.volume;
  }

  // Update momentum incrementally using a sliding window
  // Add the new trade to the rolling window
  symbol_data.momentum_prices.push_back(trade.price);

  // Maintain window size
  if (symbol_data.momentum_prices.size() > symbol_data.momentum_window_size) {
    symbol_data.momentum_prices.pop_front();
  }

  // Calculate momentum if we have enough data
  if (symbol_data.momentum_prices.size() >= 2) {
    double first = symbol_data.momentum_prices.front();
    double last = symbol_data.momentum_prices.back();
    symbol_data.momentum = ((last - first) / first) * 100.0;

    // Calculate momentum strength (volatility of momentum)
    if (symbol_data.momentum_prices.size() >= 2) {
      double mean = std::accumulate(symbol_data.momentum_prices.begin(),
                                    symbol_data.momentum_prices.end(), 0.0) /
                    symbol_data.momentum_prices.size();
      double variance = 0.0;
      for (double price : symbol_data.momentum_prices) {
        variance += std::pow(price - mean, 2);
      }
      variance /= symbol_data.momentum_prices.size();
      symbol_data.momentum_strength = std::sqrt(variance);
    }
  }

  // Update volatility incrementally using a sliding window
  // Add the new trade to the rolling window if we have previous trades
  if (!symbol_data.recent_trades.empty()) {
    double prev_price = symbol_data.recent_trades.back().price;
    double current_return = std::log(trade.price / prev_price);

    symbol_data.log_returns.push_back(current_return);

    // Maintain window size
    if (symbol_data.log_returns.size() > symbol_data.volatility_window_size) {
      symbol_data.log_returns.pop_front();
    }
  }

  // Calculate volatility if we have enough data
  if (symbol_data.log_returns.size() >= 2) {
    double mean =
        std::accumulate(symbol_data.log_returns.begin(), symbol_data.log_returns.end(), 0.0) /
        symbol_data.log_returns.size();
    double variance = 0.0;
    for (double ret : symbol_data.log_returns) {
      variance += std::pow(ret - mean, 2);
    }
    variance /= symbol_data.log_returns.size();
    symbol_data.volatility = std::sqrt(variance) * std::sqrt(252 * 24 * 60 * 60);  // Annualized
    symbol_data.sharpe_ratio = mean / std::sqrt(variance);  // Simplified Sharpe ratio
  }

  // Set dirty flag if state changed and flag is provided
  if (state_changed && dirty_flag) {
    dirty_flag->set();
  }
}

}  // namespace RenderEngine
}  // namespace BTQuant