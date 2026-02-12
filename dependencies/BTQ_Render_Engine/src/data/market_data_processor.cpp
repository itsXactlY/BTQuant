#include "market_data_processor.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <map>
#include <numeric>
#include <ranges>
#include <stdexcept>

#include "analytics/cluster_engine.hpp"
#include "cache_manager.hpp"
#include "trading/HotspineData.h"

namespace BTQuant {
namespace RenderEngine {

MarketDataProcessor::MarketDataProcessor()
    : vwap_window_size_(100),
      momentum_window_size_(50),
      volatility_window_size_(100),
      spread_analysis_window_(50),
      parallel_processing_enabled_(true) {
  // Initialize shards
  for (size_t i = 0; i < NUM_SHARDS; ++i) {
    shards_.emplace_back(std::make_unique<Shard>());
  }

  // Initialize cache manager
  cache_manager_ = std::make_shared<CacheManager>();

  // Start worker threads
  for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
    workers_.emplace_back(&MarketDataProcessor::processQueueLoop, this);
  }

  // REDUNDANT: Polling is now handled by HotSpineDataBridge
  // polling_thread_ = std::thread(&MarketDataProcessor::pollingLoop, this);

  // Initialize Cluster Engine with a default tick size of 0.01
  cluster_engine_ = std::make_unique<Analytics::ClusterEngine>(0.01);

  // Initialize Orderbook Snapshot Manager
  orderbook_snapshot_manager_ = std::make_unique<OrderbookSnapshotManager>();

  std::cout << "[MarketDataProcessor] Initialized with " << NUM_SHARDS << " shards, "
            << workers_.size() << " worker threads, and polling thread" << std::endl;
}

MarketDataProcessor::~MarketDataProcessor() {
  running_ = false;
  update_queue_.enqueue(MarketDataUpdate{});  // Wake up workers

  // Wait for workers to finish
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }

  // Wait for polling thread to finish
  if (polling_thread_.joinable()) {
    polling_thread_.join();
  }

  // Clean up Orderbook Snapshot Manager
  orderbook_snapshot_manager_.reset();

  std::cout << "[MarketDataProcessor] Shutdown complete" << std::endl;
}

void MarketDataProcessor::processTradeUpdate(const MarketDataUpdate& update) {
  update_queue_.enqueue(update);
}

void MarketDataProcessor::processTradeUpdates(const std::vector<MarketDataUpdate>& updates) {
  for (const auto& update : updates) {
    update_queue_.enqueue(update);
  }
}

void MarketDataProcessor::processOrderbookUpdate(const MarketDataUpdate& update) {
  update_queue_.enqueue(update);
}

SymbolAnalytics MarketDataProcessor::getSymbolAnalytics(uint32_t symbol_id) const {
  auto& shard = getShard(symbol_id);
  std::shared_lock lock(shard.mutex);

  auto it = shard.data.find(symbol_id);
  if (it != shard.data.end()) {
    SymbolAnalytics result = it->second;
    // The recent_trades_db is already handled by the copy constructor
    return result;
  }

  return SymbolAnalytics{};
}

std::vector<uint32_t> MarketDataProcessor::getActiveSymbols() const {
  std::vector<uint32_t> active_symbols;

  for (auto& shard_ptr : shards_) {
    std::shared_lock lock(shard_ptr->mutex);
    for (const auto& pair : shard_ptr->data) {
      auto now = std::chrono::high_resolution_clock::now();
      auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
          now.time_since_epoch() - std::chrono::seconds(pair.second.last_update_time / 1000000));
      if (time_diff.count() < 300) {  // Active if updated within 5 minutes
        active_symbols.push_back(pair.first);
      }
    }
  }

  return active_symbols;
}

void MarketDataProcessor::clearHistory() {
  for (auto& shard_ptr : shards_) {
    std::unique_lock lock(shard_ptr->mutex);
    for (auto& pair : shard_ptr->data) {
      pair.second.candles.clear();
      // Clear the double-buffered state for recent trades
      {
        auto& back_buffer = pair.second.recent_trades_db.write();
        back_buffer.clear();
        pair.second.recent_trades_db.swap();
      }
      pair.second.recent_orderbooks.clear();
      pair.second.consolidated_bids.clear();
      pair.second.consolidated_asks.clear();
    }
  }
}

ProcessorPerformanceMetrics MarketDataProcessor::getPerformanceMetrics() const {
  return performance_metrics_.toNonAtomic();
}

std::vector<SymbolRanking> MarketDataProcessor::getRankings(RankingCriteria criteria,
                                                            size_t limit) const {
  std::vector<SymbolRanking> rankings;

  for (auto& shard_ptr : shards_) {
    std::shared_lock lock(shard_ptr->mutex);
    for (const auto& pair : shard_ptr->data) {
      double value = 0.0;
      std::string label;

      switch (criteria) {
        case RankingCriteria::VOLUME:
          value = pair.second.volume_15m;
          label = "Volume";
          break;
        case RankingCriteria::MOMENTUM:
          value = pair.second.momentum;
          label = "Momentum";
          break;
        case RankingCriteria::VOLATILITY:
          value = pair.second.volatility;
          label = "Volatility";
          break;
        case RankingCriteria::SPREAD:
          value = pair.second.current_spread_percent;
          label = "Spread %";
          break;
        case RankingCriteria::IMBALANCE:
          value = std::abs(pair.second.current_imbalance);
          label = "Imbalance";
          break;
      }

      rankings.push_back({pair.first, value, label});
    }
  }

  // Sort by value descending
  std::sort(rankings.begin(), rankings.end(),
            [](const SymbolRanking& a, const SymbolRanking& b) { return a.value > b.value; });

  if (limit > 0 && rankings.size() > limit) {
    rankings.resize(limit);
  }

  return rankings;
}

std::vector<OHLCVCandle> MarketDataProcessor::getCandles(uint32_t symbol_id,
                                                         TimeFrame timeframe) const {
  auto& shard = getShard(symbol_id);
  std::shared_lock lock(shard.mutex);

  auto it = shard.data.find(symbol_id);
  if (it != shard.data.end()) {
    auto tf_it = it->second.candles.find(timeframe);
    if (tf_it != it->second.candles.end()) {
      return tf_it->second;
    }
  }

  return {};
}

std::optional<OHLCVCandle> MarketDataProcessor::getCurrentCandle(uint32_t symbol_id,
                                                                 TimeFrame timeframe) const {
  auto& shard = getShard(symbol_id);
  std::shared_lock lock(shard.mutex);

  auto it = shard.data.find(symbol_id);
  if (it != shard.data.end()) {
    auto tf_it = it->second.current_candles.find(timeframe);
    if (tf_it != it->second.current_candles.end()) {
      return tf_it->second;
    }
  }

  return std::nullopt;
}

std::optional<OrderbookData> MarketDataProcessor::getOrderbookData(uint32_t symbol_id) const {
  auto& shard = getShard(symbol_id);
  std::shared_lock lock(shard.mutex);

  auto it = shard.data.find(symbol_id);
  if (it != shard.data.end() && !it->second.recent_orderbooks.empty()) {
    return it->second.recent_orderbooks.back();
  }

  return std::nullopt;
}

std::vector<OrderbookData> MarketDataProcessor::getHistoricalOrderbooks(uint32_t symbol_id,
                                                                        size_t count) const {
  auto& shard = getShard(symbol_id);
  std::shared_lock lock(shard.mutex);

  auto it = shard.data.find(symbol_id);
  if (it != shard.data.end()) {
    const auto& history = it->second.recent_orderbooks;
    if (count == 0 || count >= history.size()) {
      return {history.begin(), history.end()};
    }
    return {history.end() - count, history.end()};
  }
  return {};
}

std::vector<VolumeProfileLevel> MarketDataProcessor::getVolumeProfile(uint32_t symbol_id,
                                                                      TimeFrame timeframe) const {
  // Try to get from cache first
  if (cache_manager_) {
    // For now, we'll use a fixed timestamp (0) for session profiles, but in a real implementation
    // this would use the appropriate bar timestamp
    auto cached_result = cache_manager_->getCachedProfileData(symbol_id, timeframe, 0);
    if (cached_result.has_value()) {
      return cached_result.value();
    }
  }

  auto& shard = getShard(symbol_id);
  std::shared_lock lock(shard.mutex);

  auto it = shard.data.find(symbol_id);
  if (it != shard.data.end()) {
    std::vector<VolumeProfileLevel> result;
    result.reserve(it->second.session_volume_profile.size());
    for (const auto& [price, level] : it->second.session_volume_profile) {
      result.push_back(level);
    }

    // Cache the result if cache manager is available
    if (cache_manager_) {
      cache_manager_->cacheProfileData(symbol_id, timeframe, 0, result);
    }

    return result;
  }

  return {};
}

MarketSummary MarketDataProcessor::getMarketSummary() const {
  MarketSummary summary;
  auto now = std::chrono::high_resolution_clock::now();

  for (auto& shard_ptr : shards_) {
    std::shared_lock lock(shard_ptr->mutex);
    for (const auto& pair : shard_ptr->data) {
      summary.total_symbols++;
      summary.last_update = now;

      if (pair.second.momentum > 0.01)
        summary.trending_up++;
      else if (pair.second.momentum < -0.01)
        summary.trending_down++;

      summary.avg_volume += pair.second.volume_15m;
      summary.avg_momentum += pair.second.momentum;
      summary.avg_volatility += pair.second.volatility;

      auto time_diff = std::chrono::duration_cast<std::chrono::minutes>(
          now - std::chrono::high_resolution_clock::time_point(
                    std::chrono::microseconds(pair.second.last_update_time)));
      if (time_diff.count() < 5) {
        summary.active_symbols++;
      }
    }
  }

  if (summary.total_symbols > 0) {
    summary.avg_volume /= summary.total_symbols;
    summary.avg_momentum /= summary.total_symbols;
    summary.avg_volatility /= summary.total_symbols;
  }

  return summary;
}

uint64_t MarketDataProcessor::getTimeFrameDuration(TimeFrame timeframe) {
  switch (timeframe) {
    case TimeFrame::TF_1MS:
      return 1000;
    case TimeFrame::TF_10MS:
      return 10000;
    case TimeFrame::TF_100MS:
      return 100000;
    case TimeFrame::TF_500MS:
      return 500000;
    case TimeFrame::TF_1SEC:
      return 1000000;
    case TimeFrame::TF_3SEC:
      return 3000000;
    case TimeFrame::TF_5SEC:
      return 5000000;
    case TimeFrame::TF_15SEC:
      return 15000000;
    case TimeFrame::TF_30SEC:
      return 30000000;
    case TimeFrame::TF_1MIN:
      return 60000000;
    case TimeFrame::TF_2MIN:
      return 120000000;
    case TimeFrame::TF_5MIN:
      return 300000000;
    case TimeFrame::TF_15MIN:
      return 900000000;
    case TimeFrame::TF_30MIN:
      return 1800000000;
    case TimeFrame::TF_1HOUR:
      return 3600000000;
    case TimeFrame::TF_2HOUR:
      return 7200000000;
    case TimeFrame::TF_4HOUR:
      return 14400000000;
    case TimeFrame::TF_6HOUR:
      return 21600000000;
    case TimeFrame::TF_12HOUR:
      return 43200000000;
    case TimeFrame::TF_1DAY:
      return 86400000000;
    case TimeFrame::TF_1WEEK:
      return 604800000000;
    default:
      return 1000000;
  }
}

void MarketDataProcessor::clearSymbolData(uint32_t symbol_id) {
  auto& shard = getShard(symbol_id);
  std::unique_lock lock(shard.mutex);

  auto it = shard.data.find(symbol_id);
  if (it != shard.data.end()) {
    // Create a new SymbolAnalytics object with the double-buffered state cleared
    SymbolAnalytics new_data;
    new_data.symbol_id = symbol_id;
    // The double-buffered state is initialized with an empty vector by default
    it->second = std::move(new_data);
  }
}

void MarketDataProcessor::clearAllData() {
  for (auto& shard_ptr : shards_) {
    std::unique_lock lock(shard_ptr->mutex);
    shard_ptr->data.clear();
  }
}

void MarketDataProcessor::setVWAPWindow(size_t window_size) { vwap_window_size_ = window_size; }

void MarketDataProcessor::setMomentumWindow(size_t window_size) {
  momentum_window_size_ = window_size;
}

void MarketDataProcessor::setVolatilityWindow(size_t window_size) {
  volatility_window_size_ = window_size;
}

void MarketDataProcessor::clearIndicatorCache(uint32_t symbol_id,
                                              const std::string& indicator_name) {
  std::unique_lock lock(indicator_cache_mutex_);
  auto it = indicator_caches_.find(symbol_id);
  if (it != indicator_caches_.end()) {
    it->second.cache.erase(indicator_name);
  }
}

void MarketDataProcessor::clearAllIndicatorCaches() {
  std::unique_lock lock(indicator_cache_mutex_);
  indicator_caches_.clear();
}

void MarketDataProcessor::setParallelProcessingEnabled(bool enabled) {
  parallel_processing_enabled_ = enabled;
}

bool MarketDataProcessor::isParallelProcessingEnabled() const {
  return parallel_processing_enabled_;
}

// C++26 Push Notification System Implementation

uint64_t MarketDataProcessor::subscribe(uint32_t symbol_id, NotificationType filter,
                                        SymbolCallback callback) {
  uint64_t id = next_subscription_id_.fetch_add(1, std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(subscribers_mutex_);
  // Copy-on-write: create new list with subscription added
  auto new_list = std::make_shared<SubscriberList>(*subscribers_);
  new_list->push_back({id, symbol_id, filter, std::move(callback)});
  subscribers_ = new_list;

  return id;
}

void MarketDataProcessor::unsubscribe(uint64_t subscription_id) {
  std::lock_guard<std::mutex> lock(subscribers_mutex_);
  // Copy-on-write: create new list without subscription
  auto new_list = std::make_shared<SubscriberList>();
  new_list->reserve(subscribers_->size());

  for (const auto& sub : *subscribers_) {
    if (sub.id != subscription_id) {
      new_list->push_back(sub);
    }
  }
  subscribers_ = new_list;
}

void MarketDataProcessor::notifySubscribers(uint32_t symbol_id, NotificationType type) const {
  // Lock-free read: copy shared_ptr under lock, then iterate without lock
  std::shared_ptr<SubscriberList> current_subs;
  {
    std::lock_guard<std::mutex> lock(subscribers_mutex_);
    current_subs = subscribers_;
  }

  // Now iterate without holding any lock - zero contention on hot path
  for (const auto& sub : *current_subs) {
    // Match if subscriber wants all symbols (0) or this specific symbol
    // AND subscriber wants this notification type
    if ((sub.symbol_id == 0 || sub.symbol_id == symbol_id) && sub.filter == type) {
      try {
        sub.callback(symbol_id, type);
      } catch (...) {
        // Don't let subscriber exceptions kill the processor
      }
    }
  }
}

// Private methods implementation

void MarketDataProcessor::updateVWAP(SymbolAnalytics& symbol_data) {
  const auto& recent_trades = symbol_data.recent_trades_db.read();
  if (recent_trades.empty()) return;

  double total_volume = 0.0;
  double total_price_volume = 0.0;

  size_t count = std::min(vwap_window_size_, recent_trades.size());
  for (size_t i = recent_trades.size() - count; i < recent_trades.size(); ++i) {
    const auto& trade = recent_trades[i];
    total_volume += trade.size;
    total_price_volume += trade.price * trade.size;
  }

  if (total_volume > 0.0) {
    symbol_data.vwap = total_price_volume / total_volume;
    symbol_data.vwap_deviation =
        symbol_data.last_trade_price > 0.0
            ? ((symbol_data.last_trade_price - symbol_data.vwap) / symbol_data.vwap) * 100.0
            : 0.0;
  }
}

void MarketDataProcessor::updateMomentum(SymbolAnalytics& symbol_data) {
  const auto& recent_trades = symbol_data.recent_trades_db.read();
  if (recent_trades.size() < 2) return;

  size_t count = std::min(momentum_window_size_, recent_trades.size());
  std::vector<double> prices;
  prices.reserve(count);

  for (size_t i = recent_trades.size() - count; i < recent_trades.size(); ++i) {
    prices.push_back(recent_trades[i].price);
  }

  if (prices.size() >= 2) {
    double first = prices.front();
    double last = prices.back();
    symbol_data.momentum = ((last - first) / first) * 100.0;

    // Calculate momentum strength (volatility of momentum)
    double mean = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
    double variance = 0.0;
    for (double price : prices) {
      variance += std::pow(price - mean, 2);
    }
    variance /= prices.size();
    symbol_data.momentum_strength = std::sqrt(variance);
  }
}

void MarketDataProcessor::updateVolatility(SymbolAnalytics& symbol_data) {
  const auto& recent_trades = symbol_data.recent_trades_db.read();
  if (recent_trades.size() < 2) return;

  size_t count = std::min(volatility_window_size_, recent_trades.size());
  std::vector<double> returns;
  returns.reserve(count - 1);

  for (size_t i = recent_trades.size() - count + 1; i < recent_trades.size(); ++i) {
    double prev = recent_trades[i - 1].price;
    double curr = recent_trades[i].price;
    returns.push_back(std::log(curr / prev));
  }

  if (!returns.empty()) {
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double ret : returns) {
      variance += std::pow(ret - mean, 2);
    }
    variance /= returns.size();
    symbol_data.volatility = std::sqrt(variance) * std::sqrt(252 * 24 * 60 * 60);  // Annualized
    symbol_data.sharpe_ratio = mean / std::sqrt(variance);  // Simplified Sharpe ratio
  }
}

void MarketDataProcessor::updateTradingMetrics(SymbolAnalytics& symbol_data,
                                               const TradeData& trade) {
  symbol_data.trade_count++;
  symbol_data.last_trade_price = trade.price;
  symbol_data.last_trade_size = trade.size;
  symbol_data.last_trade_time = trade.timestamp;

  // Update volume metrics
  uint64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch())
                        .count();

  if (now_us - symbol_data.last_update_time < 60000000) {  // 1 minute
    symbol_data.volume_1m += trade.size;
  }
  if (now_us - symbol_data.last_update_time < 300000000) {  // 5 minutes
    symbol_data.volume_5m += trade.size;
  }
  if (now_us - symbol_data.last_update_time < 900000000) {  // 15 minutes
    symbol_data.volume_15m += trade.size;
  }

  // Update buy/sell metrics
  if (trade.is_buy) {
    symbol_data.buy_volume += trade.size;
    symbol_data.buy_count++;
  } else {
    symbol_data.sell_volume += trade.size;
    symbol_data.sell_count++;
  }

  symbol_data.buy_sell_ratio = (symbol_data.buy_count + symbol_data.sell_count) > 0
                                   ? static_cast<double>(symbol_data.buy_count) /
                                         (symbol_data.buy_count + symbol_data.sell_count)
                                   : 0.5;

  // Update price ranges
  if (symbol_data.price_min == 0.0 || trade.price < symbol_data.price_min) {
    symbol_data.price_min = trade.price;
  }
  if (trade.price > symbol_data.price_max) {
    symbol_data.price_max = trade.price;
  }

  if (symbol_data.price_max > symbol_data.price_min) {
    symbol_data.price_position =
        ((trade.price - symbol_data.price_min) / (symbol_data.price_max - symbol_data.price_min)) *
        100.0;
  }

  // Update average trade size
  symbol_data.avg_trade_size =
      (symbol_data.avg_trade_size * (symbol_data.trade_count - 1) + trade.size) /
      symbol_data.trade_count;

  // Check for large trades
  if (trade.size > 2.0 * symbol_data.avg_trade_size) {
    symbol_data.large_trade_count++;
  }

  // Update Volume Profile
  auto& vp_level = symbol_data.session_volume_profile[trade.price];
  vp_level.price = trade.price;
  vp_level.total_volume += trade.size;
  if (trade.is_buy) {
    vp_level.buy_volume += trade.size;
  } else {
    vp_level.sell_volume += trade.size;
  }
}

void MarketDataProcessor::updateSpreadAnalysis(SymbolAnalytics& symbol_data) {
  if (symbol_data.recent_orderbooks.empty()) return;

  const auto& latest = symbol_data.recent_orderbooks.back();
  symbol_data.current_spread = latest.spread;
  symbol_data.current_spread_percent = latest.spread_percent;
  symbol_data.current_imbalance = latest.imbalance;
  symbol_data.market_depth = latest.total_depth;

  // Calculate averages
  double total_spread = 0.0;
  double total_spread_percent = 0.0;
  double total_imbalance = 0.0;

  size_t count = std::min(spread_analysis_window_, symbol_data.recent_orderbooks.size());
  for (size_t i = symbol_data.recent_orderbooks.size() - count;
       i < symbol_data.recent_orderbooks.size(); ++i) {
    total_spread += symbol_data.recent_orderbooks[i].spread;
    total_spread_percent += symbol_data.recent_orderbooks[i].spread_percent;
    total_imbalance += symbol_data.recent_orderbooks[i].imbalance;
  }

  symbol_data.avg_spread = total_spread / count;
  symbol_data.avg_spread_percent = total_spread_percent / count;
  symbol_data.avg_imbalance = total_imbalance / count;
}

void MarketDataProcessor::updateCandles(SymbolAnalytics& symbol_data, const TradeData& trade) {
  for (auto timeframe :
       {TimeFrame::TF_1MS,   TimeFrame::TF_10MS,  TimeFrame::TF_100MS,  TimeFrame::TF_500MS,
        TimeFrame::TF_1SEC,  TimeFrame::TF_3SEC,  TimeFrame::TF_5SEC,   TimeFrame::TF_15SEC,
        TimeFrame::TF_30SEC, TimeFrame::TF_1MIN,  TimeFrame::TF_2MIN,   TimeFrame::TF_5MIN,
        TimeFrame::TF_15MIN, TimeFrame::TF_30MIN, TimeFrame::TF_1HOUR,  TimeFrame::TF_2HOUR,
        TimeFrame::TF_4HOUR, TimeFrame::TF_6HOUR, TimeFrame::TF_12HOUR, TimeFrame::TF_1DAY,
        TimeFrame::TF_1WEEK}) {
    updateCandleForTimeframe(symbol_data, trade, timeframe);
  }
}

void MarketDataProcessor::updateCandleForTimeframe(SymbolAnalytics& symbol_data,
                                                   const TradeData& trade, TimeFrame timeframe) {
  uint64_t duration_us = getTimeFrameDuration(timeframe);
  uint64_t candle_start = (trade.timestamp / duration_us) * duration_us;

  auto& candles = symbol_data.candles[timeframe];
  auto& current_candle = symbol_data.current_candles[timeframe];

  // Check if we need a new candle
  if (current_candle.timestamp == 0 || candle_start != current_candle.timestamp) {
    // Save previous candle if it exists
    if (current_candle.timestamp != 0) {
      // Keep FULL candle history (no limit)
      candles.push_back(current_candle);
    }

    // Create new candle
    current_candle = createNewCandle(candle_start, trade.price, trade.size);
  } else {
    // Update existing candle
    updateCandle(current_candle, trade.price, trade.size);
  }
}

OHLCVCandle MarketDataProcessor::createNewCandle(uint64_t timestamp, double price,
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

bool MarketDataProcessor::isTradeInCurrentCandle(const OHLCVCandle& candle,
                                                 uint64_t trade_timestamp,
                                                 TimeFrame timeframe) const {
  uint64_t duration_us = getTimeFrameDuration(timeframe);
  uint64_t candle_start = candle.timestamp;
  uint64_t candle_end = candle_start + duration_us;
  return trade_timestamp >= candle_start && trade_timestamp < candle_end;
}

void MarketDataProcessor::updateCandle(OHLCVCandle& candle, double price, double size) const {
  candle.high = std::max(candle.high, price);
  candle.low = std::min(candle.low, price);
  candle.close = price;
  candle.volume += size;
  candle.trade_count++;
}

double MarketDataProcessor::calculateMarketDepth(const std::vector<PriceLevel>& levels) const {
  double depth = 0.0;
  for (const auto& level : levels) {
    depth += level.size;
  }
  return depth;
}

double MarketDataProcessor::calculateVolumeInWindow(const std::vector<TradeData>& trades,
                                                    uint64_t window_us) const {
  if (trades.empty()) return 0.0;

  uint64_t now = trades.back().timestamp;
  uint64_t window_start = now - window_us;

  double volume = 0.0;
  for (auto it = trades.rbegin(); it != trades.rend(); ++it) {
    if (it->timestamp < window_start) break;
    volume += it->size;
  }

  return volume;
}

void MarketDataProcessor::processQueueLoop() {
  MarketDataUpdate update;

  while (running_) {
    if (update_queue_.try_dequeue(update)) {
      if (!running_) break;

      auto start_time = std::chrono::high_resolution_clock::now();

      // Process the update
      processUpdate(update);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

      // Update performance metrics
      performance_metrics_.processing_latency_us.store(latency.count());
      performance_metrics_.avg_latency_ms.store(
          (performance_metrics_.avg_latency_ms.load() + latency.count() / 1000.0) / 2.0);

      if (update.type == MarketDataType::TRADE) {
        performance_metrics_.total_trades_processed.fetch_add(1);
        trade_count_delta_.fetch_add(1);
      } else {
        performance_metrics_.total_orderbooks_processed.fetch_add(1);
        book_count_delta_.fetch_add(1);
      }

      // Update rates periodically
      auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch())
                        .count();

      if (now_us - last_performance_update_us_ > 1000000) {  // Every second
        double trades_per_sec = trade_count_delta_.exchange(0) * 1.0;
        double books_per_sec = book_count_delta_.exchange(0) * 1.0;

        performance_metrics_.trades_per_second.store(trades_per_sec);
        performance_metrics_.orderbooks_per_second.store(books_per_sec);
        performance_metrics_.last_update_time.store(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        last_performance_update_us_ = now_us;
      }
    } else {
      // Sleep briefly if no work
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
}

void MarketDataProcessor::processUpdate(const MarketDataUpdate& update) {
  auto& shard = getShard(update.symbol_id);
  std::unique_lock lock(shard.mutex);

  auto& symbol_data = shard.data[update.symbol_id];
  symbol_data.symbol_id = update.symbol_id;
  symbol_data.last_update_time = update.timestamp;

  if (update.type == MarketDataType::TRADE) {
    TradeData trade;
    trade.symbol_id = update.symbol_id;
    trade.timestamp = update.timestamp;
    trade.price = update.price;
    trade.size = update.size;
    trade.is_buy = (update.side == "buy");

    // Use incremental updater to update analytics efficiently
    // The processTradeIncrementally method handles adding the trade to the double-buffered state
    processTradeIncrementally(symbol_data, trade);

  } else if (update.type == MarketDataType::ORDERBOOK) {
    OrderbookData orderbook;
    orderbook.symbol_id = update.symbol_id;
    orderbook.timestamp = update.timestamp;
    orderbook.bids = update.bids;
    orderbook.asks = update.asks;

    // Calculate spread and depth
    if (!update.bids.empty() && !update.asks.empty()) {
      double best_bid = update.bids.front().price;
      double best_ask = update.asks.front().price;
      orderbook.spread = best_ask - best_bid;
      orderbook.spread_percent = (orderbook.spread / best_bid) * 100.0;
    }

    orderbook.bid_depth = calculateMarketDepth(update.bids);
    orderbook.ask_depth = calculateMarketDepth(update.asks);
    orderbook.total_depth = orderbook.bid_depth + orderbook.ask_depth;
    orderbook.imbalance = orderbook.total_depth > 0.0
                              ? (orderbook.bid_depth - orderbook.ask_depth) / orderbook.total_depth
                              : 0.0;

    // Update consolidated orderbook
    symbol_data.consolidated_bids.clear();
    symbol_data.consolidated_asks.clear();

    for (const auto& level : update.bids) {
      symbol_data.consolidated_bids[level.price] += level.size;
    }
    for (const auto& level : update.asks) {
      symbol_data.consolidated_asks[level.price] += level.size;
    }

    // Limit orderbook history to prevent memory growth
    if (symbol_data.recent_orderbooks.size() >= 1000) {
      symbol_data.recent_orderbooks.erase(symbol_data.recent_orderbooks.begin());
    }
    symbol_data.recent_orderbooks.push_back(orderbook);

    // Update the atomic orderbook snapshot for renderer access (double-buffering)
    // via the dedicated OrderbookSnapshotManager
    if (orderbook_snapshot_manager_) {
      orderbook_snapshot_manager_->updateSnapshot(update.symbol_id, orderbook);
    }
  }

  // Invalidate cache for this symbol and all timeframes when new data arrives
  if (cache_manager_) {
    cache_manager_->invalidateCacheForSymbol(update.symbol_id);
  }

  // For trades, analytics are updated incrementally in processTradeIncrementally
  // For orderbooks, update spread analysis periodically
  if (update.type == MarketDataType::ORDERBOOK) {
    static uint64_t update_counter = 0;
    if (++update_counter % 5 == 0) {  // Update spread analysis every 5 updates
      updateSpreadAnalysis(symbol_data);
    }
  }
  // Publish snapshot at most ~60 fps per symbol (every 16ms)
  {
    auto now = std::chrono::steady_clock::now();
    auto& last_pub = shard.last_publish_time[update.symbol_id];
    if (now - last_pub >= std::chrono::milliseconds(16)) {
      last_pub = now;
      publishSnapshot(update.symbol_id, symbol_data);
    }
  }

  // Release lock before notifying subscribers (avoid holding while calling
  // callbacks)
  NotificationType notify_type = (update.type == MarketDataType::TRADE)
                                     ? NotificationType::TRADE
                                     : NotificationType::ORDERBOOK;
  uint32_t notify_symbol_id = update.symbol_id;
  lock.unlock();

  // Push notification to all subscribers (lock-free iteration)
  notifySubscribers(notify_symbol_id, notify_type);
}

void MarketDataProcessor::processTradeIncrementally(SymbolAnalytics& symbol_data,
                                                    const TradeData& trade) {
  // Update basic trade analytics
  symbol_data.last_trade_price = trade.price;
  symbol_data.last_trade_size = trade.size;
  symbol_data.last_trade_time = trade.timestamp;
  symbol_data.trade_count++;

  // Update volume analytics
  if (trade.is_buy) {
    symbol_data.buy_volume += trade.size;
    symbol_data.buy_count++;
  } else {
    symbol_data.sell_volume += trade.size;
    symbol_data.sell_count++;
  }

  // Update price range
  if (symbol_data.price_min == 0.0 || trade.price < symbol_data.price_min) {
    symbol_data.price_min = trade.price;
  }
  if (symbol_data.price_max == 0.0 || trade.price > symbol_data.price_max) {
    symbol_data.price_max = trade.price;
  }

  // Update the back buffer with the new trade data
  {
    auto& back_buffer = symbol_data.recent_trades_db.write();
    // Limit trade history to prevent memory growth
    if (back_buffer.size() >= 10000) {
      back_buffer.erase(back_buffer.begin());
    }
    back_buffer.push_back(trade);
    // Swap the buffers to make the updated data available for readers
    symbol_data.recent_trades_db.swap();
  }

  // Update OHLCV candles every trade (O(1) per timeframe)
  updateCandles(symbol_data, trade);

  // Update trading metrics every trade (O(1))
  updateTradingMetrics(symbol_data, trade);

  // Throttle expensive O(n) analytics to every 100th trade
  // to reduce lock hold time and prevent render thread starvation
  if (symbol_data.trade_count % 100 == 0) {
    updateVWAP(symbol_data);
    updateMomentum(symbol_data);
    updateVolatility(symbol_data);
    updateSpreadAnalysis(symbol_data);
  }

  // Update atomic indicators periodically (e.g., every 50 trades to balance performance)
  if (symbol_data.trade_count % 50 == 0) {
    // Note: We need to call this with the symbol_id, but we don't have it here
    // So we'll update atomic indicators in the publishSnapshot method where we have the symbol_id
  }

  // Update performance metrics
  auto now = std::chrono::high_resolution_clock::now();
  auto time_since_last = std::chrono::duration_cast<std::chrono::microseconds>(
      now - std::chrono::high_resolution_clock::time_point(
                std::chrono::high_resolution_clock::duration(symbol_data.last_update_time)));
  symbol_data.last_update_time = now.time_since_epoch().count();
}

// Define constants for the polling loop
constexpr size_t MAX_BATCH_SIZE = 4096;

// Main polling loop that reads from shared memory ring buffer
void MarketDataProcessor::pollingLoop() {
  while (running_.load(std::memory_order_acquire)) {
    poll_hotspine();

    // Brief sleep to prevent 100% CPU usage when no data is available
    std::this_thread::sleep_for(std::chrono::microseconds(10));  // 10 microsecond delay
  }
}

// Poll the shared memory ring buffer for new events
void MarketDataProcessor::pollSharedMemoryRingBuffer() {
  // Check if we have a valid HotSpineDataBridge
  if (!hotspine_bridge_) {
    return;  // No bridge available, nothing to poll
  }

  // Get the current write head from the shared memory header via the bridge
  uint64_t shared_write_head =
      hotspine_bridge_->getHeader()->write_head.load(std::memory_order_acquire);

  // If no new data is available, return early
  if (local_read_tail_ >= shared_write_head) {
    return;
  }

  // Calculate how many events we need to process
  uint64_t events_to_process =
      std::min(static_cast<uint64_t>(MAX_BATCH_SIZE), shared_write_head - local_read_tail_);

  // Process up to MAX_BATCH_SIZE events per cycle using pointer arithmetic
  for (uint64_t i = 0; i < events_to_process; ++i) {
    // Use the new indexing pattern: base_ptr[index & mask]
    uint64_t current_index = (local_read_tail_ + i) & (HotSpine::V3::RING_BUFFER_MASK);

    // Access the event directly using pointer arithmetic: base_ptr[index & mask]
    const auto& event_ref = hotspine_bridge_->getBasePtr()[current_index];

    // Check if this is a warmup event
    if (event_ref.flags & HotSpine::V3::HotspineData::IS_WARMUP) {
      // For warmup events, just touch memory to keep cache hot, but skip processing
      continue;
    }

    // Convert HotspineData to MarketDataUpdate and process
    MarketDataUpdate update;
    update.timestamp = event_ref.timestamp;
    update.symbol_id = event_ref.symbol_id;
    update.price = event_ref.price;
    update.size = event_ref.volume;

    // Determine event type based on eventType
    if (event_ref.event_type == 0) {  // Assuming 0 is TRADE
      update.type = MarketDataType::TRADE;
      update.side = (event_ref.flags & 0x04) ? "buy" : "sell";  // Assuming bit 2 indicates side
    } else {  // Assuming other values are ORDERBOOK
      update.type = MarketDataType::ORDERBOOK;
      // Note: bids/asks would need to be reconstructed from the payload
    }

    // Process the actual market data event
    processUpdate(update);
  }

  // Update our local read tail to reflect the processed events
  local_read_tail_ += events_to_process;
}

// Implementation of the poll_hotspine() function as specified in architect.md
void MarketDataProcessor::poll_hotspine() {
  // Check if we have a valid HotSpineDataBridge
  if (!hotspine_bridge_) {
    return;  // No bridge available, nothing to poll
  }

  // Get the current write head from the shared memory header via the bridge
  uint64_t shared_write_head =
      hotspine_bridge_->getHeader()->write_head.load(std::memory_order_acquire);

  // If no new data is available, return early
  if (local_read_tail_ >= shared_write_head) {
    return;
  }

  // Calculate how many events we need to process (max 50,000 per frame for UI responsiveness)
  const uint64_t MAX_EVENTS_PER_FRAME = 50000;
  uint64_t events_to_process = std::min(MAX_EVENTS_PER_FRAME, shared_write_head - local_read_tail_);

  // Process events using atomic storage updates with relaxed memory ordering
  for (uint64_t i = 0; i < events_to_process; ++i) {
    // Use the indexing pattern: base_ptr[index & mask]
    uint64_t current_index = (local_read_tail_ + i) & (HotSpine::V3::RING_BUFFER_MASK);

    // Access the event directly using pointer arithmetic: base_ptr[index & mask]
    const auto& event_ref = hotspine_bridge_->getBasePtr()[current_index];

    // Check if this is a warmup event
    if (event_ref.flags & HotSpine::V3::HotspineData::IS_WARMUP) {
      // For warmup events, just touch memory to keep cache hot, but skip processing
      continue;
    }

    // Update atomic storage using memory_order_relaxed for performance
    // Get the atomic symbol info for this symbol_id
    auto* atomic_info = atomic_registry_.get_mutable_atomic_snapshot(event_ref.symbol_id);
    if (atomic_info) {
      // Update the atomic values with relaxed memory ordering for performance
      atomic_info->price.store(event_ref.price, std::memory_order_relaxed);
      atomic_info->volume.store(event_ref.volume, std::memory_order_relaxed);
      atomic_info->timestamp.store(event_ref.timestamp, std::memory_order_relaxed);
      atomic_info->last_update_time.store(event_ref.timestamp, std::memory_order_relaxed);

      // Update last trade price specifically for trade events
      if (event_ref.event_type == 0) {  // Assuming 0 is TRADE
        atomic_info->last_trade_price.store(event_ref.price, std::memory_order_relaxed);

        // Update high/low prices - this would normally be updated periodically, not per trade
        // For now, we'll update them per trade for simplicity
        double current_high = atomic_info->high_price.load(std::memory_order_relaxed);
        double current_low = atomic_info->low_price.load(std::memory_order_relaxed);

        if (current_high == 0.0 || event_ref.price > current_high) {
          atomic_info->high_price.store(event_ref.price, std::memory_order_relaxed);
        }
        if (current_low == 0.0 || event_ref.price < current_low) {
          atomic_info->low_price.store(event_ref.price, std::memory_order_relaxed);
        }

        // Update buy/sell volume based on flags
        if (event_ref.flags & 0x04) {  // Assuming bit 2 indicates buy
          atomic_info->buy_volume.fetch_add(event_ref.volume, std::memory_order_relaxed);
        } else {
          atomic_info->sell_volume.fetch_add(event_ref.volume, std::memory_order_relaxed);
        }
      }

      // Increment trade count
      atomic_info->trade_count.fetch_add(1, std::memory_order_relaxed);
    }

    // Feed raw trades into ClusterEngine *after* updating the atomic price
    // Use a thread-local buffer for ClusterEngine updates to avoid locking the main atomic storage
    if (cluster_engine_ && event_ref.event_type == 0) {  // Assuming 0 is TRADE
      // Use thread-local buffer to collect trades before batch processing
      thread_local std::vector<MarketData::Trade> trade_buffer;

      // Create a temporary MarketData::Trade object from the HotspineData
      MarketData::Trade trade;
      trade.price = event_ref.price;
      trade.quantity = event_ref.volume;
      trade.timestamp_us = event_ref.timestamp;
      trade.is_buyer_maker = (event_ref.flags & 0x04) != 0;  // Assuming bit 2 indicates buyer maker

      // Add trade to thread-local buffer
      trade_buffer.push_back(trade);

      // Process the buffer in batches to minimize lock contention on ClusterEngine
      if (trade_buffer.size() >= 100) {  // Process in batches of 100
        cluster_engine_->process_trade_batch(trade_buffer);
        trade_buffer.clear();  // Clear the buffer after processing
      }
    }
  }

  // Update our local read tail to reflect the processed events
  local_read_tail_ += events_to_process;
}

void MarketDataProcessor::publishSnapshot(uint32_t symbol_id, const SymbolAnalytics& analytics) {
  auto& shard = getShard(symbol_id);
  // Create per-symbol buffer on first use (caller holds unique_lock on shard)
  auto& buf_ptr = shard.snapshot_buffers[symbol_id];
  if (!buf_ptr) {
    buf_ptr = std::make_unique<TripleBuffer<RenderSnapshot>>();
  }
  auto& snap = buf_ptr->write_buffer();

  // Clear and reuse allocated memory
  snap.clear();
  snap.symbol_id = symbol_id;
  snap.last_update_time = analytics.last_update_time;
  snap.last_price = analytics.last_trade_price;
  snap.vwap = analytics.vwap;
  snap.momentum = analytics.momentum;
  snap.volatility = analytics.volatility;

  // --- Pre-flatten orderbook (maps → flat vectors) ---
  snap.bids.reserve(analytics.consolidated_bids.size());
  for (const auto& [price, size] : analytics.consolidated_bids) {
    snap.bids.push_back({static_cast<float>(price), static_cast<float>(size)});
  }
  snap.asks.reserve(analytics.consolidated_asks.size());
  for (const auto& [price, size] : analytics.consolidated_asks) {
    snap.asks.push_back({static_cast<float>(price), static_cast<float>(size)});
  }

  // Compute price range for LOB
  if (!snap.bids.empty()) {
    snap.ob_max_price = snap.bids.front().price;  // Highest bid
    snap.ob_min_price = snap.bids.back().price;   // Lowest bid
  }
  if (!snap.asks.empty()) {
    snap.ob_min_price = std::min(snap.ob_min_price, snap.asks.front().price);
    snap.ob_max_price = std::max(snap.ob_max_price, snap.asks.back().price);
  }

  // --- Pre-convert recent trades to GPU-ready format ---
  const auto& trades = analytics.recent_trades_db.read();
  size_t trade_count = std::min(static_cast<size_t>(1000), trades.size());
  snap.trade_ticks.reserve(trade_count);
  for (size_t i = trades.size() - trade_count; i < trades.size(); ++i) {
    const auto& t = trades[i];
    snap.trade_ticks.emplace_back(t.timestamp, static_cast<float>(t.price),
                                  static_cast<float>(t.size), t.symbol_id, t.is_buy);
  }

  // Atomically publish — render thread can now see this snapshot
  buf_ptr->publish();

  // Update atomic indicator values for fast UI polling
  update_atomic_indicators(symbol_id, analytics);

  // Also update the atomic orderbook snapshot if we have recent orderbook data
  if (!analytics.recent_orderbooks.empty() && orderbook_snapshot_manager_) {
    const auto& latest_orderbook = analytics.recent_orderbooks.back();
    orderbook_snapshot_manager_->updateSnapshot(symbol_id, latest_orderbook);
  }
}

// Indicator calculation implementations
std::vector<double> MarketDataProcessor::calculate_sma(const std::vector<double>& prices,
                                                       int period) const {
  std::vector<double> sma;
  if (prices.size() < static_cast<size_t>(period)) {
    sma.resize(prices.size());
    return sma;
  }

  sma.resize(prices.size());

  // Calculate first SMA value
  double sum = 0.0;
  for (int i = 0; i < period; ++i) {
    sum += prices[i];
  }
  sma[period - 1] = sum / period;

  // Calculate subsequent SMA values using rolling window
  for (size_t i = period; i < prices.size(); ++i) {
    sum = sum - prices[i - period] + prices[i];
    sma[i] = sum / period;
  }

  return sma;
}

std::vector<double> MarketDataProcessor::calculate_ema(const std::vector<double>& prices,
                                                       int period) const {
  std::vector<double> ema;
  if (prices.empty()) return ema;

  ema.resize(prices.size());

  // Smoothing factor
  double multiplier = 2.0 / (period + 1.0);

  // First EMA value is the same as first SMA
  if (prices.size() >= static_cast<size_t>(period)) {
    double sum = 0.0;
    for (int i = 0; i < period; ++i) {
      sum += prices[i];
    }
    ema[period - 1] = sum / period;

    // Calculate subsequent EMA values
    for (size_t i = period; i < prices.size(); ++i) {
      ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1];
    }
  } else {
    // If we don't have enough data for the full period, just copy the prices
    for (size_t i = 0; i < prices.size(); ++i) {
      ema[i] = prices[i];
    }
  }

  return ema;
}

std::vector<double> MarketDataProcessor::calculate_rsi(const std::vector<double>& prices,
                                                       int period) const {
  std::vector<double> rsi;
  if (prices.size() <= static_cast<size_t>(period)) {
    rsi.resize(prices.size());
    return rsi;
  }

  rsi.resize(prices.size());

  // Initialize with zeros for the first 'period' elements
  for (int i = 0; i < period; ++i) {
    rsi[i] = 50.0;  // Neutral value
  }

  // Calculate price changes
  std::vector<double> gains(period, 0.0);
  std::vector<double> losses(period, 0.0);

  // Calculate initial average gain and loss
  double avg_gain = 0.0;
  double avg_loss = 0.0;

  for (int i = 1; i <= period; ++i) {
    double change = prices[i] - prices[i - 1];
    if (change >= 0) {
      gains[i % period] = change;
      avg_gain += change;
    } else {
      losses[i % period] = -change;
      avg_loss -= change;
    }
  }

  avg_gain /= period;
  avg_loss /= period;

  // Calculate RSI for the rest of the data points
  for (size_t i = period + 1; i < prices.size(); ++i) {
    double change = prices[i] - prices[i - 1];
    double current_gain = (change >= 0) ? change : 0.0;
    double current_loss = (change < 0) ? -change : 0.0;

    // Update averages using Wilder's smoothing method
    avg_gain = (avg_gain * (period - 1) + current_gain) / period;
    avg_loss = (avg_loss * (period - 1) + current_loss) / period;

    // Calculate RSI
    if (avg_loss == 0.0) {
      rsi[i] = 100.0;
    } else {
      double rs = avg_gain / avg_loss;
      rsi[i] = 100.0 - (100.0 / (1.0 + rs));
    }
  }

  return rsi;
}

std::vector<double> MarketDataProcessor::calculate_macd_line(const std::vector<double>& prices,
                                                             int fast_period,
                                                             int slow_period) const {
  auto fast_ema = calculate_ema(prices, fast_period);
  auto slow_ema = calculate_ema(prices, slow_period);

  std::vector<double> macd_line;
  macd_line.resize(std::min(fast_ema.size(), slow_ema.size()));

  for (size_t i = 0; i < macd_line.size(); ++i) {
    macd_line[i] = fast_ema[i] - slow_ema[i];
  }

  return macd_line;
}

std::vector<double> MarketDataProcessor::calculate_macd_signal(const std::vector<double>& macd_line,
                                                               int signal_period) const {
  return calculate_ema(macd_line, signal_period);
}

std::vector<double> MarketDataProcessor::calculate_macd_histogram(
    const std::vector<double>& macd_line, const std::vector<double>& signal_line) const {
  std::vector<double> histogram;
  histogram.resize(std::min(macd_line.size(), signal_line.size()));

  for (size_t i = 0; i < histogram.size(); ++i) {
    histogram[i] = macd_line[i] - signal_line[i];
  }

  return histogram;
}

std::vector<double> MarketDataProcessor::calculate_bollinger_bands(
    const std::vector<double>& prices, int period, double std_dev, std::vector<double>& upper_band,
    std::vector<double>& middle_band, std::vector<double>& lower_band) const {
  auto sma = calculate_sma(prices, period);

  middle_band = sma;
  upper_band.resize(prices.size());
  lower_band.resize(prices.size());

  for (size_t i = 0; i < prices.size(); ++i) {
    if (i < static_cast<size_t>(period - 1)) {
      // Not enough data to calculate standard deviation
      upper_band[i] = prices[i];
      lower_band[i] = prices[i];
    } else {
      // Calculate standard deviation for the current window
      double sum = 0.0;
      for (int j = 0; j < period; ++j) {
        size_t idx = i - static_cast<size_t>(period - 1) + j;
        double diff = prices[idx] - sma[i];
        sum += diff * diff;
      }
      double variance = sum / period;
      double std_dev_val = std::sqrt(variance);

      upper_band[i] = sma[i] + (std_dev * std_dev_val);
      lower_band[i] = sma[i] - (std_dev * std_dev_val);
    }
  }

  return sma;  // Return middle band as well
}

std::vector<double> MarketDataProcessor::calculate_stochastic_k(const std::vector<double>& highs,
                                                                const std::vector<double>& lows,
                                                                const std::vector<double>& closes,
                                                                int k_period) const {
  std::vector<double> stoch_k;
  stoch_k.resize(closes.size());

  for (size_t i = 0; i < closes.size(); ++i) {
    if (i < static_cast<size_t>(k_period - 1)) {
      // Not enough data to calculate
      stoch_k[i] = 50.0;  // Neutral value
    } else {
      // Find highest high and lowest low in the period
      double highest_high = highs[i];
      double lowest_low = lows[i];

      for (int j = 0; j < k_period; ++j) {
        size_t idx = i - static_cast<size_t>(k_period - 1) + j;
        if (idx < highs.size() && highs[idx] > highest_high) {
          highest_high = highs[idx];
        }
        if (idx < lows.size() && lows[idx] < lowest_low) {
          lowest_low = lows[idx];
        }
      }

      // Calculate stochastic K
      if (highest_high != lowest_low) {
        stoch_k[i] = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100.0;
      } else {
        stoch_k[i] = 50.0;  // Neutral value when high equals low
      }
    }
  }

  return stoch_k;
}

std::vector<double> MarketDataProcessor::calculate_atr(const std::vector<double>& highs,
                                                       const std::vector<double>& lows,
                                                       const std::vector<double>& closes,
                                                       int period) const {
  std::vector<double> tr;
  tr.reserve(std::max({highs.size(), lows.size(), closes.size()}));

  // Calculate True Range values
  for (size_t i = 0; i < highs.size() && i < lows.size() && i < closes.size(); ++i) {
    double h_minus_l = highs[i] - lows[i];
    double h_minus_pc = (i > 0) ? std::abs(highs[i] - closes[i - 1]) : 0.0;
    double l_minus_pc = (i > 0) ? std::abs(lows[i] - closes[i - 1]) : 0.0;

    double true_range = std::max({h_minus_l, h_minus_pc, l_minus_pc});
    tr.push_back(true_range);
  }

  // Calculate ATR using Wilder's smoothing method
  std::vector<double> atr(tr.size());

  if (tr.size() < static_cast<size_t>(period)) {
    return atr;
  }

  // Calculate initial ATR (simple average of first 'period' TR values)
  double sum = 0.0;
  for (int i = 0; i < period; ++i) {
    sum += tr[i];
  }
  atr[period - 1] = sum / period;

  // Calculate subsequent ATR values using Wilder's smoothing
  for (size_t i = period; i < tr.size(); ++i) {
    atr[i] = ((atr[i - 1] * (period - 1)) + tr[i]) / period;
  }

  return atr;
}

void MarketDataProcessor::update_atomic_indicators(uint32_t symbol_id,
                                                   const SymbolAnalytics& analytics) {
  // Extract price data from candles for indicator calculations
  auto candles_it = analytics.candles.find(TimeFrame::TF_1MIN);
  if (candles_it == analytics.candles.end() || candles_it->second.empty()) {
    return;  // No candles available for this timeframe
  }

  const auto& candles = candles_it->second;

  // Extract closing prices
  std::vector<double> closes;
  std::vector<double> highs;
  std::vector<double> lows;

  closes.reserve(candles.size());
  highs.reserve(candles.size());
  lows.reserve(candles.size());

  for (const auto& candle : candles) {
    closes.push_back(candle.close);
    highs.push_back(candle.high);
    lows.push_back(candle.low);
  }

  // Create atomic indicator values structure
  AtomicIndicatorValues atomic_values;

  // Calculate and update SMA values
  if (closes.size() >= 9) {
    auto sma_9_vals = calculate_sma(closes, 9);
    if (!sma_9_vals.empty())
      atomic_values.sma_9.store(sma_9_vals.back(), std::memory_order_relaxed);
  }

  if (closes.size() >= 10) {
    auto sma_10_vals = calculate_sma(closes, 10);
    if (!sma_10_vals.empty())
      atomic_values.sma_10.store(sma_10_vals.back(), std::memory_order_relaxed);
  }

  if (closes.size() >= 20) {
    auto sma_20_vals = calculate_sma(closes, 20);
    if (!sma_20_vals.empty())
      atomic_values.sma_20.store(sma_20_vals.back(), std::memory_order_relaxed);
  }

  if (closes.size() >= 50) {
    auto sma_50_vals = calculate_sma(closes, 50);
    if (!sma_50_vals.empty())
      atomic_values.sma_50.store(sma_50_vals.back(), std::memory_order_relaxed);
  }

  if (closes.size() >= 200) {
    auto sma_200_vals = calculate_sma(closes, 200);
    if (!sma_200_vals.empty())
      atomic_values.sma_200.store(sma_200_vals.back(), std::memory_order_relaxed);
  }

  // Calculate and update EMA values
  if (closes.size() >= 9) {
    auto ema_9_vals = calculate_ema(closes, 9);
    if (!ema_9_vals.empty())
      atomic_values.ema_9.store(ema_9_vals.back(), std::memory_order_relaxed);
  }

  if (closes.size() >= 10) {
    auto ema_10_vals = calculate_ema(closes, 10);
    if (!ema_10_vals.empty())
      atomic_values.ema_10.store(ema_10_vals.back(), std::memory_order_relaxed);
  }

  if (closes.size() >= 21) {
    auto ema_21_vals = calculate_ema(closes, 21);
    if (!ema_21_vals.empty())
      atomic_values.ema_21.store(ema_21_vals.back(), std::memory_order_relaxed);
  }

  if (closes.size() >= 50) {
    auto ema_50_vals = calculate_ema(closes, 50);
    if (!ema_50_vals.empty())
      atomic_values.ema_50.store(ema_50_vals.back(), std::memory_order_relaxed);
  }

  if (closes.size() >= 200) {
    auto ema_200_vals = calculate_ema(closes, 200);
    if (!ema_200_vals.empty())
      atomic_values.ema_200.store(ema_200_vals.back(), std::memory_order_relaxed);
  }

  // Calculate and update RSI
  if (closes.size() >= 14) {  // Standard RSI period
    auto rsi_vals = calculate_rsi(closes, 14);
    if (!rsi_vals.empty()) atomic_values.rsi.store(rsi_vals.back(), std::memory_order_relaxed);
  }

  // Calculate and update MACD
  if (closes.size() >= 26) {                                    // Need at least 26 periods for MACD
    auto macd_line_vals = calculate_macd_line(closes, 12, 26);  // Standard MACD (12, 26)
    if (!macd_line_vals.empty()) {
      atomic_values.macd_line.store(macd_line_vals.back(), std::memory_order_relaxed);

      auto macd_signal_vals = calculate_macd_signal(macd_line_vals, 9);  // Standard signal line (9)
      if (!macd_signal_vals.empty()) {
        atomic_values.macd_signal.store(macd_signal_vals.back(), std::memory_order_relaxed);

        auto macd_hist_vals = calculate_macd_histogram(macd_line_vals, macd_signal_vals);
        if (!macd_hist_vals.empty()) {
          atomic_values.macd_histogram.store(macd_hist_vals.back(), std::memory_order_relaxed);
        }
      }
    }
  }

  // Calculate and update Bollinger Bands
  if (closes.size() >= 20) {  // Standard Bollinger Bands period
    std::vector<double> upper_band, middle_band, lower_band;
    calculate_bollinger_bands(closes, 20, 2.0, upper_band, middle_band, lower_band);

    if (!upper_band.empty())
      atomic_values.bollinger_upper.store(upper_band.back(), std::memory_order_relaxed);
    if (!middle_band.empty())
      atomic_values.bollinger_middle.store(middle_band.back(), std::memory_order_relaxed);
    if (!lower_band.empty())
      atomic_values.bollinger_lower.store(lower_band.back(), std::memory_order_relaxed);
  }

  // Calculate and update Stochastic
  if (closes.size() >= 14 && highs.size() >= 14 &&
      lows.size() >= 14) {  // Standard stochastic period
    auto stoch_k_vals = calculate_stochastic_k(highs, lows, closes, 14);
    if (!stoch_k_vals.empty()) {
      atomic_values.stochastic_k.store(stoch_k_vals.back(), std::memory_order_relaxed);

      // Calculate D line (3-period SMA of K line)
      if (stoch_k_vals.size() >= 3) {
        auto stoch_d_vals = calculate_sma(stoch_k_vals, 3);
        if (!stoch_d_vals.empty()) {
          atomic_values.stochastic_d.store(stoch_d_vals.back(), std::memory_order_relaxed);
        }
      }
    }
  }

  // Calculate and update ATR
  if (closes.size() >= 14 && highs.size() >= 14 && lows.size() >= 14) {  // Standard ATR period
    auto atr_vals = calculate_atr(highs, lows, closes, 14);
    if (!atr_vals.empty()) {
      atomic_values.atr.store(atr_vals.back(), std::memory_order_relaxed);
    }
  }

  // Update timestamp
  atomic_values.last_updated.store(std::chrono::duration_cast<std::chrono::microseconds>(
                                       std::chrono::high_resolution_clock::now().time_since_epoch())
                                       .count(),
                                   std::memory_order_relaxed);

  // Update the atomic indicator values in the shard directly (lock is already held by caller)
  auto& shard = getShard(symbol_id);
  shard.atomic_indicator_values[symbol_id] = atomic_values;
}

}  // namespace RenderEngine
}  // namespace BTQuant