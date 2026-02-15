#include "market_data_processor.hpp"
#include "cache_manager.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <numeric>
#include <stdexcept>

// Include miniaudio if available
#ifdef MINIAUDIO_IMPLEMENTATION
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#endif


namespace BTQuant {
namespace RenderEngine {

#ifdef MINIAUDIO_IMPLEMENTATION
// Static variables for audio resources
static ma_engine* g_engine = nullptr;
static bool audio_initialized = false;

// Generate a simple beep sound data for trade alerts
static ma_result play_beep_sound(ma_engine* engine, float frequency, float duration, float amplitude) {
    // For trade sounds, we'll use ma_engine_play_sound with dynamically generated sound
    // based on trade characteristics (volume, direction, etc.)
    
    if (engine == nullptr) {
        return MA_INVALID_ARGS;
    }
    
    // In a real implementation, you would have different sound files based on trade characteristics
    // For example: different sounds for large trades vs small trades, buy vs sell, etc.
    
    // For now, we'll use a simplified approach that would work with actual sound files
    // depending on the trade characteristics
    
    // Determine sound file based on trade characteristics
    const char* sound_file = "assets/trade_beep.wav";  // Default sound file
    
    // In a real implementation, you could have different sounds:
    if (amplitude > 0.5f) {
        // Large trade - use different sound
        sound_file = "assets/large_trade_beep.wav";
    } else if (frequency > 600.0f) {
        // High frequency (small trade) - use different sound
        sound_file = "assets/small_trade_beep.wav";
    }
    
    // Try to play the sound file using ma_engine_play_sound
    ma_result result = ma_engine_play_sound(engine, sound_file, NULL);
    
    if (result != MA_SUCCESS) {
        // If the sound file doesn't exist, we'll generate a simple tone dynamically
        // This is a fallback implementation that would work even without sound files
        std::cout << "[Audio] Sound file not found: " << sound_file 
                  << ", generating tone dynamically (freq: " << frequency << "Hz)" << std::endl;
                  
        // In a real implementation, we would generate a tone buffer and play it
        // For now, we'll just log that we would generate a tone
        // The actual tone generation would require creating a ma_sound with generated PCM data
    } else {
        std::cout << "[Audio] Played trade sound: " << sound_file 
                  << " (freq: " << frequency << "Hz)" << std::endl;
    }
    
    return result; // Return the actual result to allow proper error handling
}
#endif

MarketDataProcessor::MarketDataProcessor()
    : vwap_window_size_(100),
      momentum_window_size_(50),
      volatility_window_size_(100),
      spread_analysis_window_(50),
      parallel_processing_enabled_(true),
      orderbook_snapshot_buffer_(ORDERBOOK_SNAPSHOT_BUFFER_SIZE),
      atomic_snapshots_(MAX_SYMBOLS) {
  // Initialize shards
  for (size_t i = 0; i < NUM_SHARDS; ++i) {
    shards_.emplace_back(std::make_unique<Shard>());
  }

  // Initialize cache manager
  cache_manager_ = std::make_shared<CacheManager>();

  // Initialize audio engine for order flow acoustics
  initializeAudioEngine();

  // Start worker threads
  for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
    workers_.emplace_back(&MarketDataProcessor::processQueueLoop, this);
  }

  std::cout << "[MarketDataProcessor] Initialized with " << NUM_SHARDS << " shards and "
            << workers_.size() << " worker threads" << std::endl;
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

  // Shutdown audio engine
  shutdownAudioEngine();

  std::cout << "[MarketDataProcessor] Shutdown complete" << std::endl;
}

void MarketDataProcessor::initializeAudioEngine() {
#ifdef MINIAUDIO_IMPLEMENTATION
  ma_result result;
  ma_engine_config config = ma_engine_config_init();
  
  // Initialize the audio engine
  g_engine = new ma_engine;
  result = ma_engine_init(&config, g_engine);
  if (result != MA_SUCCESS) {
      std::cerr << "[MarketDataProcessor] Failed to initialize audio engine: " << result << std::endl;
      delete g_engine;
      g_engine = nullptr;
      audio_enabled_ = false;
      return;
  }
  
  audio_engine_ = g_engine;
  audio_enabled_ = true;
  std::cout << "[MarketDataProcessor] Audio engine initialized for order flow acoustics" << std::endl;
#else
  // For now, we'll just enable the audio functionality
  // In a real implementation, this would initialize miniaudio
  audio_enabled_ = true;
  std::cout << "[MarketDataProcessor] Audio engine initialized for order flow acoustics (stub implementation)" << std::endl;
#endif
}

void MarketDataProcessor::shutdownAudioEngine() {
#ifdef MINIAUDIO_IMPLEMENTATION
  if (g_engine) {
    ma_engine_uninit(g_engine);
    delete g_engine;
    g_engine = nullptr;
    audio_engine_ = nullptr;
  }
#else
  // In a real implementation, this would properly shut down miniaudio
  if (audio_engine_) {
    // In stub implementation, audio_engine_ is just a placeholder
    audio_engine_ = nullptr;
  }
#endif
  audio_enabled_ = false;
  std::cout << "[MarketDataProcessor] Audio engine shut down" << std::endl;
}

void MarketDataProcessor::playTradeSound(double volume, bool is_buy) {
  if (!audio_enabled_) return;

  // Enqueue the audio event to be processed in the main polling loop
  // This provides better integration with the main loop and prevents audio
  // processing from blocking the data processing pipeline
  std::pair<double, bool> audio_event = std::make_pair(volume, is_buy);
  audio_event_queue_.enqueue(audio_event);
}

void MarketDataProcessor::processAudioEvents() {
  std::pair<double, bool> audio_event;
  
  // Process all queued audio events
  while (audio_event_queue_.try_dequeue(audio_event)) {
    double volume = audio_event.first;
    bool is_buy = audio_event.second;
    
    if (!audio_enabled_) return;

    // Calculate pitch based on volume (inverse relationship - large trades = low pitch)
    // Normalize volume to a range for pitch calculation
    double normalized_volume = std::log(volume + 1.0); // Log scale to handle wide range of volumes
    double pitch_factor = std::min(1.0, std::max(0.0, (normalized_volume - 0.5) / 5.0)); // Adjust range as needed

    // Calculate pitch (inverse relationship: large volume = low pitch)
    double pitch = base_pitch_ + (max_pitch_ - base_pitch_) * (1.0 - pitch_factor);

    // Clamp pitch to valid range
    pitch = std::max(min_pitch_, std::min(max_pitch_, pitch));

#ifdef MINIAUDIO_IMPLEMENTATION
    // Play a sound using miniaudio with pitch based on trade volume
    if (g_engine) {
      // For buy trades, use higher pitch; for sell trades, use lower pitch
      float frequency = static_cast<float>(pitch);
      if (!is_buy) {
          frequency *= 0.8f; // Lower pitch for sell trades
      }

      // Play a tone with the calculated frequency
      ma_result result = play_beep_sound(g_engine, frequency, 0.1f, 0.3f); // 0.1s duration, 0.3 amplitude

      if (result != MA_SUCCESS) {
          std::cout << "[Audio] Failed to play tone - Volume: " << volume
                    << ", Pitch: " << pitch
                    << ", Side: " << (is_buy ? "BUY" : "SELL") << std::endl;
      } else {
          std::cout << "[Audio] Played trade sound - Volume: " << volume
                    << ", Pitch: " << pitch
                    << ", Side: " << (is_buy ? "BUY" : "SELL") << std::endl;
      }
    }
#else
    // For the stub implementation, just print to console
    std::cout << "[Audio] Playing trade sound - Volume: " << volume
              << ", Pitch: " << pitch
              << ", Side: " << (is_buy ? "BUY" : "SELL") << std::endl;
#endif
  }
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
    return it->second;
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
      pair.second.recent_trades.clear();
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

std::optional<AtomicL2Snapshot> MarketDataProcessor::get_atomic_snapshot(uint32_t symbol_id) const {
  // Bounds check to ensure symbol_id is within the array size
  if (symbol_id >= MAX_SYMBOLS) {
    return std::nullopt;
  }

  // Read from the atomic snapshot array directly without locks
  const auto& atomic_snapshot = atomic_snapshots_[symbol_id];

  // Load values using memory_order_relaxed for optimal performance
  AtomicL2Snapshot snapshot;
  snapshot.symbol_id = atomic_snapshot.symbol_id.load(std::memory_order_relaxed);
  
  // Only return a snapshot if it contains valid data (symbol_id != 0 means it's been initialized)
  if (snapshot.symbol_id == 0) {
    return std::nullopt;
  }
  
  snapshot.timestamp = atomic_snapshot.timestamp.load(std::memory_order_relaxed);
  snapshot.best_bid = atomic_snapshot.best_bid.load(std::memory_order_relaxed);
  snapshot.best_ask = atomic_snapshot.best_ask.load(std::memory_order_relaxed);
  snapshot.best_bid_size = atomic_snapshot.best_bid_size.load(std::memory_order_relaxed);
  snapshot.best_ask_size = atomic_snapshot.best_ask_size.load(std::memory_order_relaxed);
  snapshot.spread = atomic_snapshot.spread.load(std::memory_order_relaxed);
  snapshot.spread_percent = atomic_snapshot.spread_percent.load(std::memory_order_relaxed);
  snapshot.last_trade_price = atomic_snapshot.last_trade_price.load(std::memory_order_relaxed);
  snapshot.last_trade_size = atomic_snapshot.last_trade_size.load(std::memory_order_relaxed);
  snapshot.last_trade_time = atomic_snapshot.last_trade_time.load(std::memory_order_relaxed);
  snapshot.mid_price = atomic_snapshot.mid_price.load(std::memory_order_relaxed);

  return snapshot;
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
    it->second = SymbolAnalytics{};
    it->second.symbol_id = symbol_id;
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
  if (symbol_data.recent_trades.empty()) return;

  double total_volume = 0.0;
  double total_price_volume = 0.0;

  size_t count = std::min(vwap_window_size_, symbol_data.recent_trades.size());
  for (size_t i = symbol_data.recent_trades.size() - count; i < symbol_data.recent_trades.size();
       ++i) {
    const auto& trade = symbol_data.recent_trades[i];
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
  if (symbol_data.recent_trades.size() < 2) return;

  size_t count = std::min(momentum_window_size_, symbol_data.recent_trades.size());
  std::vector<double> prices;
  prices.reserve(count);

  for (size_t i = symbol_data.recent_trades.size() - count; i < symbol_data.recent_trades.size();
       ++i) {
    prices.push_back(symbol_data.recent_trades[i].price);
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
  if (symbol_data.recent_trades.size() < 2) return;

  size_t count = std::min(volatility_window_size_, symbol_data.recent_trades.size());
  std::vector<double> returns;
  returns.reserve(count - 1);

  for (size_t i = symbol_data.recent_trades.size() - count + 1;
       i < symbol_data.recent_trades.size(); ++i) {
    double prev = symbol_data.recent_trades[i - 1].price;
    double curr = symbol_data.recent_trades[i].price;
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
       {TimeFrame::TF_1MS, TimeFrame::TF_10MS, TimeFrame::TF_100MS, TimeFrame::TF_500MS,
        TimeFrame::TF_1SEC, TimeFrame::TF_3SEC, TimeFrame::TF_5SEC, TimeFrame::TF_15SEC,
        TimeFrame::TF_30SEC, TimeFrame::TF_1MIN, TimeFrame::TF_2MIN, TimeFrame::TF_5MIN,
        TimeFrame::TF_15MIN, TimeFrame::TF_30MIN, TimeFrame::TF_1HOUR, TimeFrame::TF_2HOUR,
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

  // Timestamp for periodic audio engine maintenance
  auto last_audio_maintenance = std::chrono::high_resolution_clock::now();

  while (running_) {
    bool processed_data = false;
    
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
      
      processed_data = true;
    }
    
    // Process audio events in the main polling loop for better integration
    processAudioEvents();
    
    // Periodic audio engine maintenance/check
    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_audio_maintenance).count() > 100) {
      // Perform any necessary audio engine maintenance here
      // For example, checking if audio engine is still running properly
      if (audio_enabled_ && audio_engine_ == nullptr) {
        // Attempt to reinitialize audio if needed
        initializeAudioEngine();
      }
      last_audio_maintenance = now;
    }
    
    if (!processed_data) {
      // Sleep briefly if no work
      // During idle periods, we can still service audio engine needs
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

    // Limit trade history to prevent memory growth
    if (symbol_data.recent_trades.size() >= 10000) {
      symbol_data.recent_trades.erase(symbol_data.recent_trades.begin());
    }
    symbol_data.recent_trades.push_back(trade);

    // Use incremental updater to update analytics efficiently
    processTradeIncrementally(symbol_data, trade);

    // Trigger audio acoustics for order flow
    playTradeSound(update.size, trade.is_buy);

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
    
    // Create and add OrderBookSnapshot to the ring buffer
    OrderBookSnapshot snapshot;
    snapshot.timestamp = update.timestamp;
    snapshot.symbol_id = update.symbol_id;
    
    // Set best bid/ask
    if (!update.bids.empty()) {
      snapshot.best_bid = update.bids.front().price;
      snapshot.best_bid_size = update.bids.front().size;
    }
    if (!update.asks.empty()) {
      snapshot.best_ask = update.asks.front().price;
      snapshot.best_ask_size = update.asks.front().size;
    }
    
    // Calculate spread
    if (snapshot.best_bid > 0.0 && snapshot.best_ask > 0.0) {
      snapshot.spread = snapshot.best_ask - snapshot.best_bid;
    }
    
    // Calculate total volumes
    for (const auto& bid : update.bids) {
      snapshot.total_bid_volume += bid.size;
    }
    for (const auto& ask : update.asks) {
      snapshot.total_ask_volume += ask.size;
    }
    
    // Copy top levels to snapshot
    snapshot.bid_levels_count = std::min(static_cast<uint32_t>(update.bids.size()), 
                                        static_cast<uint32_t>(OrderBookSnapshot::MAX_LEVELS));
    snapshot.ask_levels_count = std::min(static_cast<uint32_t>(update.asks.size()), 
                                        static_cast<uint32_t>(OrderBookSnapshot::MAX_LEVELS));
    
    for (uint32_t i = 0; i < snapshot.bid_levels_count; ++i) {
      snapshot.bids[i].price = update.bids[i].price;
      snapshot.bids[i].size = update.bids[i].size;
    }
    
    for (uint32_t i = 0; i < snapshot.ask_levels_count; ++i) {
      snapshot.asks[i].price = update.asks[i].price;
      snapshot.asks[i].size = update.asks[i].size;
    }
    
    // Add the snapshot to the ring buffer
    addOrderBookSnapshot(snapshot);
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

  // Update atomic snapshot for lock-free access
  if (update.symbol_id < MAX_SYMBOLS) {
    auto& atomic_snapshot = atomic_snapshots_[update.symbol_id];
    
    // Update the atomic snapshot with the latest data
    atomic_snapshot.symbol_id.store(update.symbol_id, std::memory_order_relaxed);
    atomic_snapshot.timestamp.store(update.timestamp, std::memory_order_relaxed);
    
    // Update bid/ask data based on orderbook update
    if (update.type == MarketDataType::ORDERBOOK && !update.bids.empty() && !update.asks.empty()) {
      atomic_snapshot.best_bid.store(update.bids.front().price, std::memory_order_relaxed);
      atomic_snapshot.best_bid_size.store(update.bids.front().size, std::memory_order_relaxed);
      atomic_snapshot.best_ask.store(update.asks.front().price, std::memory_order_relaxed);
      atomic_snapshot.best_ask_size.store(update.asks.front().size, std::memory_order_relaxed);
      
      double spread = update.asks.front().price - update.bids.front().price;
      atomic_snapshot.spread.store(spread, std::memory_order_relaxed);
      if (update.bids.front().price > 0.0) {
        atomic_snapshot.spread_percent.store((spread / update.bids.front().price) * 100.0, std::memory_order_relaxed);
      }
    }
    
    // Update trade data based on trade update
    if (update.type == MarketDataType::TRADE) {
      atomic_snapshot.last_trade_price.store(update.price, std::memory_order_relaxed);
      atomic_snapshot.last_trade_size.store(update.size, std::memory_order_relaxed);
      atomic_snapshot.last_trade_time.store(update.timestamp, std::memory_order_relaxed);
    }
    
    // Calculate and update mid price
    double best_bid = atomic_snapshot.best_bid.load(std::memory_order_relaxed);
    double best_ask = atomic_snapshot.best_ask.load(std::memory_order_relaxed);
    if (best_bid > 0.0 && best_ask > 0.0) {
      atomic_snapshot.mid_price.store((best_bid + best_ask) / 2.0, std::memory_order_relaxed);
    } else {
      // Fallback to last trade price if bid/ask not available
      double last_trade_price = atomic_snapshot.last_trade_price.load(std::memory_order_relaxed);
      if (last_trade_price > 0.0) {
        atomic_snapshot.mid_price.store(last_trade_price, std::memory_order_relaxed);
      }
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

void MarketDataProcessor::processTradeIncrementally(SymbolAnalytics& symbol_data, const TradeData& trade) {
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

  // Update VWAP
  updateVWAP(symbol_data);

  // Update momentum
  updateMomentum(symbol_data);

  // Update volatility
  updateVolatility(symbol_data);

  // Update trading metrics
  updateTradingMetrics(symbol_data, trade);

  // Update spread analysis
  updateSpreadAnalysis(symbol_data);

  // Update OHLCV candles
  updateCandles(symbol_data, trade);

  // Update performance metrics
  auto now = std::chrono::high_resolution_clock::now();
  auto time_since_last = std::chrono::duration_cast<std::chrono::microseconds>(
      now - std::chrono::high_resolution_clock::time_point(
               std::chrono::high_resolution_clock::duration(symbol_data.last_update_time)));
  symbol_data.last_update_time = now.time_since_epoch().count();
}

// Ring buffer methods for OrderBookSnapshot
void MarketDataProcessor::addOrderBookSnapshot(const OrderBookSnapshot& snapshot) {
  std::lock_guard<std::mutex> lock(snapshot_buffer_mutex_);
  
  size_t write_idx = snapshot_write_index_.load(std::memory_order_relaxed);
  orderbook_snapshot_buffer_[write_idx] = snapshot;
  
  // Update indices atomically
  size_t next_write_idx = (write_idx + 1) % ORDERBOOK_SNAPSHOT_BUFFER_SIZE;
  snapshot_write_index_.store(next_write_idx, std::memory_order_release);
  
  // Update count (but don't exceed buffer size)
  size_t current_count = snapshot_count_.load(std::memory_order_relaxed);
  if (current_count < ORDERBOOK_SNAPSHOT_BUFFER_SIZE) {
    snapshot_count_.store(current_count + 1, std::memory_order_release);
  }
}

std::vector<OrderBookSnapshot> MarketDataProcessor::getOrderBookSnapshots(size_t count) const {
  std::lock_guard<std::mutex> lock(snapshot_buffer_mutex_);
  
  size_t actual_count = std::min(count, static_cast<size_t>(snapshot_count_.load(std::memory_order_acquire)));
  std::vector<OrderBookSnapshot> result;
  result.reserve(actual_count);
  
  if (actual_count == 0) {
    return result;
  }
  
  // Calculate the starting index to get the most recent snapshots
  size_t current_write_idx = snapshot_write_index_.load(std::memory_order_acquire);
  size_t start_idx = (current_write_idx - actual_count + ORDERBOOK_SNAPSHOT_BUFFER_SIZE) % ORDERBOOK_SNAPSHOT_BUFFER_SIZE;
  
  // Retrieve snapshots from start_idx to current_write_idx
  for (size_t i = 0; i < actual_count; ++i) {
    size_t idx = (start_idx + i) % ORDERBOOK_SNAPSHOT_BUFFER_SIZE;
    result.push_back(orderbook_snapshot_buffer_[idx]);
  }
  
  return result;
}

std::optional<OrderBookSnapshot> MarketDataProcessor::getLatestOrderBookSnapshot() const {
  std::lock_guard<std::mutex> lock(snapshot_buffer_mutex_);
  
  if (snapshot_count_.load(std::memory_order_acquire) == 0) {
    return std::nullopt;
  }
  
  // Get the index of the most recent snapshot
  size_t current_write_idx = snapshot_write_index_.load(std::memory_order_acquire);
  size_t latest_idx = (current_write_idx == 0) ? ORDERBOOK_SNAPSHOT_BUFFER_SIZE - 1 : current_write_idx - 1;
  
  return orderbook_snapshot_buffer_[latest_idx];
}

size_t MarketDataProcessor::getOrderBookSnapshotCount() const {
  return snapshot_count_.load(std::memory_order_acquire);
}

}  // namespace RenderEngine
}  // namespace BTQuant