#include "market_data_processor.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>
namespace BTQuant {
namespace RenderEngine {

MarketDataProcessor::MarketDataProcessor()
    : vwap_window_size_(5000), momentum_window_size_(500),
      volatility_window_size_(1000), spread_analysis_window_(1000),
      parallel_processing_enabled_(true), stop_workers_(false) {
  // Create worker threads
  const size_t num_threads = std::max(std::thread::hardware_concurrency(), 2u);
  for (size_t i = 0; i < num_threads; ++i) {
    worker_threads_.emplace_back([this]() { workerThread(); });
  }
}

MarketDataProcessor::~MarketDataProcessor() {
  stop_workers_ = true;
  task_queue_cv_.notify_all();

  for (auto &thread : worker_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void MarketDataProcessor::workerThread() {
  while (!stop_workers_) {
    std::function<void()> task;

    {
      std::unique_lock<std::mutex> lock(task_queue_mutex_);
      task_queue_cv_.wait(
          lock, [this]() { return stop_workers_ || !task_queue_.empty(); });

      if (stop_workers_ && task_queue_.empty()) {
        return;
      }

      task = std::move(task_queue_.front());
      task_queue_.pop();
    }

    task();
  }
}

void MarketDataProcessor::submitTask(std::function<void()> task) {
  if (!parallel_processing_enabled_) {
    task();
    return;
  }

  std::unique_lock<std::mutex> lock(task_queue_mutex_);
  task_queue_.push(std::move(task));
  task_queue_cv_.notify_one();
}

void MarketDataProcessor::processTradeUpdate(const MarketDataUpdate &update) {
  if (update.type != MarketDataType::TRADE) {
    return;
  }

  // Create a copy of the update for async processing
  MarketDataUpdate update_copy = update;

  submitTask([this, update_copy]() {
    std::unique_lock<std::shared_mutex> lock(data_mutex_);

    auto &symbol_data = symbol_analytics_[update_copy.symbol_id];

    // Update basic trade data
    TradeData trade;
    trade.timestamp = update_copy.timestamp;
    trade.price = update_copy.price;
    trade.size = update_copy.size;
    trade.is_buy = (update_copy.side == "buy");

    symbol_data.recent_trades.push_back(trade);

    // Maintain window size
    if (symbol_data.recent_trades.size() > vwap_window_size_ * 2) {
      symbol_data.recent_trades.erase(symbol_data.recent_trades.begin(),
                                      symbol_data.recent_trades.begin() +
                                          vwap_window_size_);
    }

    // Update OHLCV candles
    updateCandles(symbol_data, trade);

    // Clear indicator cache for this symbol
    auto cache_it = indicator_caches_.find(update_copy.symbol_id);
    if (cache_it != indicator_caches_.end()) {
      std::lock_guard<std::mutex> cache_lock(cache_it->second.mutex);
      cache_it->second.cache.clear();
    }

    // Update analytics
    updateVWAP(symbol_data);
    updateMomentum(symbol_data);
    updateVolatility(symbol_data);
    updateTradingMetrics(symbol_data, trade);

    // Update performance metrics
    performance_metrics_.total_trades_processed++;
    performance_metrics_.last_update_time =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
  });
}

void MarketDataProcessor::processTradeUpdates(
    const std::vector<MarketDataUpdate> &updates) {
  if (updates.empty())
    return;

  std::unique_lock<std::shared_mutex> lock(data_mutex_);

  for (const auto &update : updates) {
    if (update.type != MarketDataType::TRADE)
      continue;

    auto &symbol_data = symbol_analytics_[update.symbol_id];
    symbol_data.symbol_id = update.symbol_id;

    // Minimal trade data for candle aggregation only
    TradeData trade;
    trade.timestamp = update.timestamp;
    trade.price = update.price;
    trade.size = update.size;
    trade.is_buy = (update.side == "buy");

    // ONLY update candles - skip all other analytics for speed
    updateCandles(symbol_data, trade);

    performance_metrics_.total_trades_processed++;
    trade_count_delta_++;

    // Simple latency metric: current - exchange timestamp
    uint64_t now_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
    double latency = (now_us - trade.timestamp) / 1000.0;
    if (latency > 0 && latency < 5000) { // Filter outliers
      double old_avg = performance_metrics_.avg_latency_ms.load();
      performance_metrics_.avg_latency_ms.store(old_avg * 0.99 +
                                                latency * 0.01);
    }
  }
  // Skip VWAP/Momentum/Volatility/indicator cache - too slow for real-time
}

void MarketDataProcessor::processOrderbookUpdate(
    const MarketDataUpdate &update) {
  if (update.type != MarketDataType::ORDERBOOK) {
    return;
  }

  // Create a copy of the update for async processing
  MarketDataUpdate update_copy = update;

  submitTask([this, update_copy]() {
    std::unique_lock<std::shared_mutex> lock(data_mutex_);

    auto &symbol_data = symbol_analytics_[update_copy.symbol_id];

    // Aggregation Logic for L2 Updates
    // -------------------------------------------------------------------------
    // 1. Update Persistent State (Price -> Size)
    // -------------------------------------------------------------------------
    auto update_level_map = [](auto &map,
                               const std::vector<PriceLevel> &levels) {
      for (const auto &level : levels) {
        if (level.size > 0) {
          map[level.price] = level.size;
        } else {
          map.erase(level.price);
        }
      }
    };

    // Pruning Logic: "Smart Accumulation"
    // If we receive a new Top Bid of 100, then any known Bid > 100 must be
    // gone. If we receive a new Top Ask of 101, then any known Ask < 101 must
    // be gone.
    if (!update_copy.bids.empty()) {
      double best_bid =
          update_copy.bids.front().price; // Assumed sorted descending
      // Remove known bids strictly greater than valid best bid
      auto it = symbol_data.consolidated_bids.lower_bound(best_bid);
      // lower_bound with active greater comparator returns first element <= key
      // So begin() to lower_bound() covers elements > key
      symbol_data.consolidated_bids.erase(symbol_data.consolidated_bids.begin(),
                                          it);
    }
    if (!update_copy.asks.empty()) {
      double best_ask =
          update_copy.asks.front().price; // Assumed sorted ascending
      // Remove known asks strictly less than valid best ask
      auto it = symbol_data.consolidated_asks.lower_bound(best_ask);
      // lower_bound with active less comparator returns first element >= key
      // So begin() to lower_bound() covers elements < key
      symbol_data.consolidated_asks.erase(symbol_data.consolidated_asks.begin(),
                                          it);
    }

    update_level_map(symbol_data.consolidated_bids, update_copy.bids);
    update_level_map(symbol_data.consolidated_asks, update_copy.asks);

    // -------------------------------------------------------------------------
    // 2. Generate Snapshot from Persistent State
    // -------------------------------------------------------------------------
    OrderbookData orderbook;
    orderbook.timestamp = update_copy.timestamp;

    // Flatten maps to vectors for the UI (limit to top 50 for performance)
    size_t count = 0;
    for (const auto &[price, size] : symbol_data.consolidated_bids) {
      orderbook.bids.push_back({price, size});
      if (++count >= 50)
        break;
    }

    count = 0;
    for (const auto &[price, size] : symbol_data.consolidated_asks) {
      orderbook.asks.push_back({price, size});
      if (++count >= 50)
        break;
    }

    // -------------------------------------------------------------------------
    // 3. Analytics
    // -------------------------------------------------------------------------
    if (!orderbook.bids.empty() && !orderbook.asks.empty()) {
      orderbook.spread = orderbook.asks[0].price - orderbook.bids[0].price;
      orderbook.spread_percent =
          (orderbook.spread / orderbook.bids[0].price) * 100.0;

      // Calculate market depth
      orderbook.bid_depth = calculateMarketDepth(orderbook.bids);
      orderbook.ask_depth = calculateMarketDepth(orderbook.asks);
      orderbook.total_depth = orderbook.bid_depth + orderbook.ask_depth;

      // Calculate imbalance
      orderbook.imbalance = (orderbook.bid_depth - orderbook.ask_depth) /
                            std::max(orderbook.total_depth, 0.001);
    }

    // DEBUG: Periodic logging to diagnose data flow
    static uint64_t update_counter = 0;
    if (update_counter++ % 100 == 0) {
      std::cout << "[MDP] Update #" << update_counter
                << " SymID: " << update_copy.symbol_id
                << " InBids: " << update_copy.bids.size()
                << " InAsks: " << update_copy.asks.size()
                << " ConsBids: " << symbol_data.consolidated_bids.size()
                << " ConsAsks: " << symbol_data.consolidated_asks.size()
                << " OutBids: " << orderbook.bids.size() << std::endl;
    }

    symbol_data.recent_orderbooks.push_back(orderbook);

    // Maintain window size
    if (symbol_data.recent_orderbooks.size() > 1) {
      // Keep only the latest snapshot for display to save memory?
      // Or keep history for charts? "spread_analysis_window_" likely used for
      // charts.
      if (symbol_data.recent_orderbooks.size() > spread_analysis_window_ * 2) {
        symbol_data.recent_orderbooks.erase(
            symbol_data.recent_orderbooks.begin(),
            symbol_data.recent_orderbooks.end() - spread_analysis_window_);
      }
    }

    // Update spread analytics
    updateSpreadAnalysis(symbol_data);

    // Update performance metrics
    performance_metrics_.total_orderbooks_processed++;
    book_count_delta_++;
  });
}

SymbolAnalytics
MarketDataProcessor::getSymbolAnalytics(uint32_t symbol_id) const {
  std::shared_lock<std::shared_mutex> lock(data_mutex_);

  auto it = symbol_analytics_.find(symbol_id);
  if (it != symbol_analytics_.end()) {
    return it->second;
  }

  return SymbolAnalytics{}; // Return empty analytics if not found
}

std::vector<uint32_t> MarketDataProcessor::getActiveSymbols() const {
  std::shared_lock<std::shared_mutex> lock(data_mutex_);

  std::vector<uint32_t> symbols;
  symbols.reserve(symbol_analytics_.size());

  for (const auto &[symbol_id, _] : symbol_analytics_) {
    symbols.push_back(symbol_id);
  }

  return symbols;
}

ProcessorPerformanceMetrics MarketDataProcessor::getPerformanceMetrics() const {
  uint64_t now_us =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();

  // Update rates every second
  if (now_us - last_performance_update_us_ >= 1000000) {
    double dt = (now_us - last_performance_update_us_) / 1000000.0;
    performance_metrics_.trades_per_second.store(
        trade_count_delta_.exchange(0) / dt);
    performance_metrics_.orderbooks_per_second.store(
        book_count_delta_.exchange(0) / dt);

    last_performance_update_us_ = now_us;
  }

  return performance_metrics_.toNonAtomic();
}

void MarketDataProcessor::clearSymbolData(uint32_t symbol_id) {
  std::unique_lock<std::shared_mutex> lock(data_mutex_);
  symbol_analytics_.erase(symbol_id);
  indicator_caches_.erase(symbol_id);
}

void MarketDataProcessor::clearAllData() {
  std::unique_lock<std::shared_mutex> lock(data_mutex_);
  symbol_analytics_.clear();
  indicator_caches_.clear();
  performance_metrics_.fromNonAtomic(ProcessorPerformanceMetrics{});
}

void MarketDataProcessor::clearIndicatorCache(
    uint32_t symbol_id, const std::string &indicator_name) {
  std::unique_lock<std::shared_mutex> lock(data_mutex_);

  auto cache_it = indicator_caches_.find(symbol_id);
  if (cache_it != indicator_caches_.end()) {
    std::lock_guard<std::mutex> cache_lock(cache_it->second.mutex);
    cache_it->second.cache.erase(indicator_name);
  }
}

void MarketDataProcessor::clearAllIndicatorCaches() {
  std::unique_lock<std::shared_mutex> lock(data_mutex_);
  indicator_caches_.clear();
}

void MarketDataProcessor::setParallelProcessingEnabled(bool enabled) {
  parallel_processing_enabled_ = enabled;
  std::cout << "[MarketDataProcessor] Parallel processing "
            << (enabled ? "enabled" : "disabled") << std::endl;
}

bool MarketDataProcessor::isParallelProcessingEnabled() const {
  return parallel_processing_enabled_;
}

void MarketDataProcessor::setVWAPWindow(size_t window_size) {
  vwap_window_size_ =
      std::max(size_t(10), std::min(window_size, size_t(100000)));
}

void MarketDataProcessor::setMomentumWindow(size_t window_size) {
  momentum_window_size_ =
      std::max(size_t(5), std::min(window_size, size_t(100)));
}

void MarketDataProcessor::setVolatilityWindow(size_t window_size) {
  volatility_window_size_ =
      std::max(size_t(10), std::min(window_size, size_t(200)));
}

void MarketDataProcessor::updateVWAP(SymbolAnalytics &symbol_data) {
  if (symbol_data.recent_trades.empty()) {
    return;
  }

  // Calculate VWAP over the window
  double total_volume = 0.0;
  double total_value = 0.0;

  size_t start_idx = symbol_data.recent_trades.size() > vwap_window_size_
                         ? symbol_data.recent_trades.size() - vwap_window_size_
                         : 0;

  for (size_t i = start_idx; i < symbol_data.recent_trades.size(); ++i) {
    const auto &trade = symbol_data.recent_trades[i];
    total_volume += trade.size;
    total_value += trade.price * trade.size;
  }

  if (total_volume > 0) {
    symbol_data.vwap = total_value / total_volume;

    // Calculate VWAP deviation
    double current_price = symbol_data.recent_trades.back().price;
    symbol_data.vwap_deviation =
        ((current_price - symbol_data.vwap) / symbol_data.vwap) * 100.0;
  }

  // Update volume metrics
  symbol_data.volume_1m = calculateVolumeInWindow(symbol_data.recent_trades,
                                                  60 * 1000000ULL); // 1 minute
  symbol_data.volume_5m = calculateVolumeInWindow(
      symbol_data.recent_trades, 5 * 60 * 1000000ULL); // 5 minutes
  symbol_data.volume_15m = calculateVolumeInWindow(
      symbol_data.recent_trades, 15 * 60 * 1000000ULL); // 15 minutes
}

void MarketDataProcessor::updateMomentum(SymbolAnalytics &symbol_data) {
  if (symbol_data.recent_trades.size() < 2) {
    return;
  }

  // Calculate price changes over the momentum window
  std::vector<double> price_changes;
  size_t start_idx =
      symbol_data.recent_trades.size() > momentum_window_size_
          ? symbol_data.recent_trades.size() - momentum_window_size_
          : 0;

  for (size_t i = start_idx + 1; i < symbol_data.recent_trades.size(); ++i) {
    double prev_price = symbol_data.recent_trades[i - 1].price;
    double curr_price = symbol_data.recent_trades[i].price;

    if (prev_price > 0) {
      double change = ((curr_price - prev_price) / prev_price) * 100.0;
      price_changes.push_back(change);
    }
  }

  if (!price_changes.empty()) {
    // Calculate momentum as average price change
    symbol_data.momentum =
        std::accumulate(price_changes.begin(), price_changes.end(), 0.0) /
        price_changes.size();

    // Calculate momentum strength (standard deviation of changes)
    double mean = symbol_data.momentum;
    double variance = 0.0;
    for (double change : price_changes) {
      variance += (change - mean) * (change - mean);
    }
    symbol_data.momentum_strength = std::sqrt(variance / price_changes.size());
  }

  // Update price extremes
  auto minmax = std::minmax_element(
      symbol_data.recent_trades.begin() + start_idx,
      symbol_data.recent_trades.end(),
      [](const TradeData &a, const TradeData &b) { return a.price < b.price; });

  if (minmax.first != symbol_data.recent_trades.end()) {
    symbol_data.price_min = minmax.first->price;
    symbol_data.price_max = minmax.second->price;

    double current_price = symbol_data.recent_trades.back().price;
    if (symbol_data.price_max > symbol_data.price_min) {
      symbol_data.price_position =
          ((current_price - symbol_data.price_min) /
           (symbol_data.price_max - symbol_data.price_min)) *
          100.0;
    }
  }
}

void MarketDataProcessor::updateVolatility(SymbolAnalytics &symbol_data) {
  if (symbol_data.recent_trades.size() < volatility_window_size_) {
    return;
  }

  // Calculate returns over the volatility window
  std::vector<double> returns;
  size_t start_idx = symbol_data.recent_trades.size() - volatility_window_size_;

  for (size_t i = start_idx + 1; i < symbol_data.recent_trades.size(); ++i) {
    double prev_price = symbol_data.recent_trades[i - 1].price;
    double curr_price = symbol_data.recent_trades[i].price;

    if (prev_price > 0) {
      double return_val = std::log(curr_price / prev_price);
      returns.push_back(return_val);
    }
  }

  if (returns.size() >= 2) {
    // Calculate mean return
    double mean_return =
        std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    // Calculate variance
    double variance = 0.0;
    for (double ret : returns) {
      variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= (returns.size() - 1);

    // Annualized volatility (assuming trades are roughly evenly spaced)
    symbol_data.volatility =
        std::sqrt(variance) * std::sqrt(252 * 24 * 60); // Rough annualization

    // Calculate Sharpe-like ratio (return/volatility)
    if (symbol_data.volatility > 0) {
      symbol_data.sharpe_ratio = mean_return / symbol_data.volatility;
    }
  }
}

void MarketDataProcessor::updateTradingMetrics(SymbolAnalytics &symbol_data,
                                               const TradeData &trade) {
  // Update trade counts
  if (trade.is_buy) {
    symbol_data.buy_volume += trade.size;
    symbol_data.buy_count++;
  } else {
    symbol_data.sell_volume += trade.size;
    symbol_data.sell_count++;
  }

  // Calculate buy/sell ratio
  double total_volume = symbol_data.buy_volume + symbol_data.sell_volume;
  if (total_volume > 0) {
    symbol_data.buy_sell_ratio = symbol_data.buy_volume / total_volume;
  }

  // Update trade size statistics
  symbol_data.avg_trade_size =
      (symbol_data.avg_trade_size * (symbol_data.trade_count - 1) +
       trade.size) /
      symbol_data.trade_count;
  symbol_data.trade_count++;

  // Track large trades (> 2x average)
  if (trade.size > symbol_data.avg_trade_size * 2.0) {
    symbol_data.large_trade_count++;
  }

  // Update last trade info
  symbol_data.last_trade_price = trade.price;
  symbol_data.last_trade_size = trade.size;
  symbol_data.last_trade_time = trade.timestamp;
}

void MarketDataProcessor::updateSpreadAnalysis(SymbolAnalytics &symbol_data) {
  if (symbol_data.recent_orderbooks.empty()) {
    return;
  }

  // Calculate average spread over window
  double total_spread = 0.0;
  double total_spread_percent = 0.0;
  double total_imbalance = 0.0;
  size_t valid_books = 0;

  size_t start_idx =
      symbol_data.recent_orderbooks.size() > spread_analysis_window_
          ? symbol_data.recent_orderbooks.size() - spread_analysis_window_
          : 0;

  for (size_t i = start_idx; i < symbol_data.recent_orderbooks.size(); ++i) {
    const auto &book = symbol_data.recent_orderbooks[i];
    if (book.spread > 0) {
      total_spread += book.spread;
      total_spread_percent += book.spread_percent;
      total_imbalance += book.imbalance;
      valid_books++;
    }
  }

  if (valid_books > 0) {
    symbol_data.avg_spread = total_spread / valid_books;
    symbol_data.avg_spread_percent = total_spread_percent / valid_books;
    symbol_data.avg_imbalance = total_imbalance / valid_books;

    // Current spread from latest orderbook
    const auto &latest = symbol_data.recent_orderbooks.back();
    symbol_data.current_spread = latest.spread;
    symbol_data.current_spread_percent = latest.spread_percent;
    symbol_data.current_imbalance = latest.imbalance;

    // Market depth
    symbol_data.market_depth = latest.total_depth;
  }
}

double MarketDataProcessor::calculateMarketDepth(
    const std::vector<PriceLevel> &levels) const {
  double total_depth = 0.0;

  // Calculate depth for first 5 levels (or all if less than 5)
  size_t max_levels = std::min(levels.size(), size_t(5));
  for (size_t i = 0; i < max_levels; ++i) {
    total_depth += levels[i].price * levels[i].size;
  }

  return total_depth;
}

double MarketDataProcessor::calculateVolumeInWindow(
    const std::vector<TradeData> &trades, uint64_t window_us) const {
  if (trades.empty()) {
    return 0.0;
  }

  uint64_t current_time = trades.back().timestamp;
  uint64_t window_start = current_time - window_us;

  double volume = 0.0;
  for (auto it = trades.rbegin(); it != trades.rend(); ++it) {
    if (it->timestamp < window_start) {
      break;
    }
    volume += it->size;
  }

  return volume;
}

std::vector<SymbolRanking>
MarketDataProcessor::getRankings(RankingCriteria criteria, size_t limit) const {
  std::lock_guard lock(data_mutex_);

  std::vector<SymbolRanking> rankings;
  rankings.reserve(symbol_analytics_.size());

  // Create rankings based on criteria
  for (const auto &[symbol_id, analytics] : symbol_analytics_) {
    SymbolRanking ranking;
    ranking.symbol_id = symbol_id;

    switch (criteria) {
    case RankingCriteria::VOLUME:
      ranking.value = analytics.volume_1m;
      ranking.label = "Volume (1m)";
      break;
    case RankingCriteria::MOMENTUM:
      ranking.value = std::abs(analytics.momentum);
      ranking.label = "Momentum";
      break;
    case RankingCriteria::VOLATILITY:
      ranking.value = analytics.volatility;
      ranking.label = "Volatility";
      break;
    case RankingCriteria::SPREAD:
      ranking.value = analytics.avg_spread_percent;
      ranking.label = "Spread %";
      break;
    case RankingCriteria::IMBALANCE:
      ranking.value = std::abs(analytics.current_imbalance);
      ranking.label = "Imbalance";
      break;
    }

    rankings.push_back(ranking);
  }

  // Sort by value (descending)
  std::sort(rankings.begin(), rankings.end(),
            [](const SymbolRanking &a, const SymbolRanking &b) {
              return a.value > b.value;
            });

  // Limit results
  if (limit > 0 && rankings.size() > limit) {
    rankings.resize(limit);
  }

  return rankings;
}

MarketSummary MarketDataProcessor::getMarketSummary() const {
  std::lock_guard lock(data_mutex_);

  MarketSummary summary;
  summary.total_symbols = symbol_analytics_.size();

  if (symbol_analytics_.empty()) {
    return summary;
  }

  // Aggregate statistics
  double total_volume = 0.0;
  double total_momentum = 0.0;
  double total_volatility = 0.0;
  size_t active_symbols = 0;

  for (const auto &[symbol_id, analytics] : symbol_analytics_) {
    if (analytics.trade_count > 0) {
      active_symbols++;
      total_volume += analytics.volume_1m;
      total_momentum += analytics.momentum;
      total_volatility += analytics.volatility;

      // Count trending symbols
      if (std::abs(analytics.momentum) > 1.0) { // > 1% momentum
        if (analytics.momentum > 0) {
          summary.trending_up++;
        } else {
          summary.trending_down++;
        }
      }
    }
  }

  summary.active_symbols = active_symbols;

  if (active_symbols > 0) {
    summary.avg_volume = total_volume / active_symbols;
    summary.avg_momentum = total_momentum / active_symbols;
    summary.avg_volatility = total_volatility / active_symbols;
  }

  summary.last_update = std::chrono::high_resolution_clock::time_point(
      std::chrono::high_resolution_clock::duration(
          performance_metrics_.last_update_time));

  return summary;
}

uint64_t MarketDataProcessor::getTimeFrameDuration(TimeFrame timeframe) {
  switch (timeframe) {
  case TimeFrame::TF_1MS:
    return 1000ULL; // 1ms in microseconds
  case TimeFrame::TF_10MS:
    return 10000ULL; // 10ms
  case TimeFrame::TF_100MS:
    return 100000ULL; // 100ms
  case TimeFrame::TF_500MS:
    return 500000ULL; // 500ms
  case TimeFrame::TF_1SEC:
    return 1000000ULL; // 1 second
  case TimeFrame::TF_3SEC:
    return 3000000ULL; // 3 seconds
  case TimeFrame::TF_5SEC:
    return 5000000ULL; // 5 seconds
  case TimeFrame::TF_15SEC:
    return 15000000ULL; // 15 seconds
  default:
    return 1000000ULL; // Default to 1 second
  }
}

std::vector<OHLCVCandle>
MarketDataProcessor::getCandles(uint32_t symbol_id, TimeFrame timeframe) const {
  std::lock_guard lock(data_mutex_);

  std::vector<OHLCVCandle> result;

  auto it = symbol_analytics_.find(symbol_id);
  if (it != symbol_analytics_.end()) {
    const auto &candles_it = it->second.candles.find(timeframe);
    if (candles_it != it->second.candles.end()) {
      result = candles_it->second; // Copy completed candles
    }

    // Append current aggregating candle for real-time visualization
    const auto &current_it = it->second.current_candles.find(timeframe);
    if (current_it != it->second.current_candles.end()) {
      result.push_back(current_it->second);
    }
  }

  return result;
}

std::optional<OHLCVCandle>
MarketDataProcessor::getCurrentCandle(uint32_t symbol_id,
                                      TimeFrame timeframe) const {
  std::lock_guard lock(data_mutex_);

  auto it = symbol_analytics_.find(symbol_id);
  if (it != symbol_analytics_.end()) {
    const auto &current_it = it->second.current_candles.find(timeframe);
    if (current_it != it->second.current_candles.end()) {
      return current_it->second;
    }
  }

  return std::nullopt;
}

std::optional<OrderbookData>
MarketDataProcessor::getOrderbookData(uint32_t symbol_id) const {
  std::lock_guard lock(data_mutex_);

  auto it = symbol_analytics_.find(symbol_id);
  if (it != symbol_analytics_.end() && !it->second.recent_orderbooks.empty()) {
    return it->second.recent_orderbooks.back();
  }

  return std::nullopt;
}

OHLCVCandle MarketDataProcessor::createNewCandle(uint64_t timestamp,
                                                 double price,
                                                 double size) const {
  OHLCVCandle candle;
  candle.timestamp = timestamp;
  candle.open = price;
  candle.high = price;
  candle.low = price;
  candle.close = price;
  candle.volume = size;
  candle.trade_count = 1;
  return candle;
}

bool MarketDataProcessor::isTradeInCurrentCandle(const OHLCVCandle &candle,
                                                 uint64_t trade_timestamp,
                                                 TimeFrame timeframe) const {
  uint64_t duration = getTimeFrameDuration(timeframe);
  return trade_timestamp < candle.timestamp + duration;
}

void MarketDataProcessor::updateCandle(OHLCVCandle &candle, double price,
                                       double size) const {
  if (price > candle.high) {
    candle.high = price;
  }
  if (price < candle.low) {
    candle.low = price;
  }
  candle.close = price;
  candle.volume += size;
  candle.trade_count++;
}

void MarketDataProcessor::updateCandles(SymbolAnalytics &symbol_data,
                                        const TradeData &trade) {
  // Process all sub-second timeframes (1ms-15sec only)
  static const std::vector<TimeFrame> timeframes = {
      TimeFrame::TF_1MS,   TimeFrame::TF_10MS, TimeFrame::TF_100MS,
      TimeFrame::TF_500MS, TimeFrame::TF_1SEC, TimeFrame::TF_3SEC,
      TimeFrame::TF_5SEC,  TimeFrame::TF_15SEC};

  for (TimeFrame tf : timeframes) {
    uint64_t duration = getTimeFrameDuration(tf);
    uint64_t candle_start = (trade.timestamp / duration) * duration;

    auto current_candle_it = symbol_data.current_candles.find(tf);

    if (current_candle_it != symbol_data.current_candles.end()) {
      if (isTradeInCurrentCandle(current_candle_it->second, trade.timestamp,
                                 tf)) {
        updateCandle(current_candle_it->second, trade.price, trade.size);
      } else {
        // Finalize old candle - NO LIMIT, keep all candles
        symbol_data.candles[tf].push_back(current_candle_it->second);
        // Start new candle
        symbol_data.current_candles[tf] =
            createNewCandle(candle_start, trade.price, trade.size);
      }
    } else {
      symbol_data.current_candles[tf] =
          createNewCandle(candle_start, trade.price, trade.size);
    }
  }
}

} // namespace RenderEngine
} // namespace BTQuant