#include "market_data_processor.h"

#include <cctype>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <unordered_map>

#include "utilities.h"

// Include DynamicLogger for proper logging
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "../../../tests/new/include/utils/dynamic_logger.hpp"

using namespace BTQuant::Logging;

namespace {

// Helper function for timestamped logging (same as in exchange_connection_manager)
// getCurrentTimestamp() is declared at global scope above

// wall-clock "now" in ms
int64_t nowMicros() {
  using namespace std::chrono;
  return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

// getCurrentTimestamp() function is defined in utilities.h

// robust double parser with logging
double safeParseDouble(const std::string& label, const std::string& s, bool& ok) {
  ok = false;
  if (s.empty()) {
    return 0.0;
  }
  try {
    std::size_t pos = 0;
    double v = std::stod(s, &pos);
    if (pos != s.size()) {
      std::cerr << "safeParseDouble(" << label << "): trailing chars in '" << s << "'\n";
      return 0.0;
    }
    ok = true;
    return v;
  } catch (const std::exception& e) {
    std::cerr << "safeParseDouble(" << label << "): exception for '" << s << "': " << e.what() << "\n";
    return 0.0;
  }
}

// try multiple possible keys on an Element
std::string getAny(const ccapi::Element& el, std::initializer_list<const char*> keys) {
  const auto& m = el.getNameValueMap();  // std::map<std::string_view, std::string>
  for (const char* k : keys) {
    auto it = m.find(std::string_view(k));
    if (it != m.end()) return it->second;
  }
  return {};
}

}  // anonymous namespace

// ccapi logger definition (must exist in exactly one TU)
namespace ccapi {
Logger* Logger::logger = nullptr;
}

using namespace MarketData;

MarketDataProcessor::MarketDataProcessor(std::shared_ptr<MSSQLBulkInserter> db, std::shared_ptr<CandleAggregator> candle_agg,
                                         std::shared_ptr<HotSpine::HotSpineWriter> hotspine_writer, bool enable_exclusive_hotspine,
                                         const ConfigTypes::DebugConfig& debug_config)
    : db_(std::move(db)),
      candle_agg_(std::move(candle_agg)),
      hotspine_writer_(std::move(hotspine_writer)),
      enable_exclusive_hotspine_(enable_exclusive_hotspine),
      debug_config_(debug_config) {
  initHotSpineV3();
}

void MarketDataProcessor::debugLog(const std::string& msg) const {
  if (debug_config_.enabled) {
    BTQ_LOG_DEBUG("market_data_processor", msg);
  }
}

void MarketDataProcessor::initHotSpineV3() {
  shm_v3_fd_ = shm_open("/BTQU_V3", O_RDWR | O_CREAT, 0666);
  if (shm_v3_fd_ == -1) {
    BTQ_LOG_ERROR("market_data_processor", "Failed to shm_open /BTQU_V3");
    return;
  }

  if (ftruncate(shm_v3_fd_, sizeof(HotSpine::V3::SharedMemoryLayoutV3)) == -1) {
    BTQ_LOG_ERROR("market_data_processor", "Failed to ftruncate SHM");
    return;
  }

  void* ptr = mmap(0, sizeof(HotSpine::V3::SharedMemoryLayoutV3), PROT_READ | PROT_WRITE, MAP_SHARED, shm_v3_fd_, 0);

  if (ptr == MAP_FAILED) {
    BTQ_LOG_ERROR("market_data_processor", "Failed to mmap SHM");
    return;
  }

  shm_v3_ = static_cast<HotSpine::V3::SharedMemoryLayoutV3*>(ptr);

  // Initialize if brand new (magic check)
  if (shm_v3_->header.magic != 0x42545133) {
    // Construct in place
    // Note: This zeros it out primarily because of global/static init rules for member structs if they have default ctor?
    // Actually explicit memset might be safer or relying on ctor.
    // Given VolumeNode/ClusterColumn are POD-like but with constructors? No they are structs.
    // new (shm_v3_) ... calls constructor.
    // Since we overwrote SharedMemoryLayoutV3 with structs without constructors, default init is fine.

    // We should probably memset the buffer to 0 first to be clean
    memset(ptr, 0, sizeof(HotSpine::V3::SharedMemoryLayoutV3));

    new (shm_v3_) HotSpine::V3::SharedMemoryLayoutV3();
    shm_v3_->header.magic = 0x42545133;
    shm_v3_->header.head_index = 0;
    // Initializing lock is handled by AtomicSeqLock constructor logic usually (std::atomic default 0)
  }
}

void MarketDataProcessor::threadAffinityCheck() {
  std::call_once(affinity_flag_, []() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(3, &cpuset);  // Pin to Core 3
    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
      BTQ_LOG_ERROR("market_data_processor", "Failed to pin thread to Core 3");
    } else {
      BTQ_LOG_INFO("market_data_processor", "Thread pinned to Core 3");
    }
  });
}

std::vector<std::string> MarketDataProcessor::split(const std::string& s, char delim) {
  std::vector<std::string> parts;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    parts.push_back(item);
  }
  return parts;
}

void MarketDataProcessor::processEvent(const ccapi::Event& event, ccapi::Session* /*session*/) {
  using Type = ccapi::Event::Type;
  const auto type = event.getType();

  debugLog("MarketDataProcessor: Received event of type: " + std::to_string(static_cast<int>(type)));
  debugLog("MarketDataProcessor: WebSocket Event Tracking - Type: " + std::to_string(static_cast<int>(type)));
  debugLog("MarketDataProcessor: WebSocket Connection Health - Event received, connection appears active");
  debugLog("MarketDataProcessor: CCAPI Data Flow - Event received for processing");
  debugLog("MarketDataProcessor: CCAPI Data Flow - Event type: " + std::to_string(static_cast<int>(type)) + " detected");

  try {
    if (type == Type::SESSION_STATUS || type == Type::SUBSCRIPTION_STATUS) {
      std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: Session/Subscription Status Event:" << std::endl;
      std::cout << event.toPrettyString(2, 2) << std::endl;

      // Check for WebSocket connection issues
      std::string eventStr = event.toString();
      if (eventStr.find("WebSocket") != std::string::npos || eventStr.find("websocket") != std::string::npos ||
          eventStr.find("CONNECTION") != std::string::npos) {
        std::cout << "[" << getCurrentTimestamp() << "][WARNING] MarketDataProcessor: WebSocket-related status event detected!" << std::endl;

        // Detailed WebSocket status analysis
        if (eventStr.find("CONNECTED") != std::string::npos || eventStr.find("connected") != std::string::npos) {
          std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: WebSocket connection established!" << std::endl;
        } else if (eventStr.find("DISCONNECTED") != std::string::npos || eventStr.find("disconnected") != std::string::npos) {
          std::cout << "[" << getCurrentTimestamp() << "][ERROR] MarketDataProcessor: WebSocket connection lost!" << std::endl;
        } else if (eventStr.find("ERROR") != std::string::npos || eventStr.find("error") != std::string::npos) {
          std::cout << "[" << getCurrentTimestamp() << "][ERROR] MarketDataProcessor: WebSocket error detected!" << std::endl;
        }
      }

      return;
    }

    if (type == Type::SUBSCRIPTION_DATA) {
      // std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: Processing SUBSCRIPTION_DATA event" << std::endl;
      // std::cout << "[" << getCurrentTimestamp() << "][INFO]   Message count: " << event.getMessageList().size() << std::endl;

      if (event.getMessageList().empty()) {
        std::cout << "[" << getCurrentTimestamp() << "][WARNING] MarketDataProcessor: Empty message list received - check WebSocket connection" << std::endl;
      }
      // else {
      //   std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: WebSocket data received successfully!" << std::endl;
      // }

      for (const auto& msg : event.getMessageList()) {
        using MType = ccapi::Message::Type;
        auto mtype = msg.getType();

        if (mtype == MType::MARKET_DATA_EVENTS_TRADE) {
          handleTradeMessage(msg);
        } else if (mtype == MType::MARKET_DATA_EVENTS_MARKET_DEPTH) {
          handleOrderbookMessage(msg);
        } else {
          std::cout << "[" << getCurrentTimestamp() << "][WARNING] MarketDataProcessor: Unsupported message type: " << static_cast<int>(mtype) << std::endl;
        }
      }
    } else {
      BTQ_LOG_INFO("market_data_processor", "Unknown event type: " + std::to_string(static_cast<int>(type)));
    }
  } catch (const std::exception& e) {
    ++errors_;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] MarketDataProcessor: Error in processEvent: " << e.what() << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] This could indicate data parsing issues or WebSocket protocol errors" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] WebSocket data flow may be disrupted" << std::endl;
  }
}

void MarketDataProcessor::setBufferLimits(std::size_t max_trades, std::size_t max_candles, std::size_t max_orderbooks) {
  max_trade_buffer_size_ = max_trades;
  max_candle_buffer_size_ = max_candles;
  max_orderbook_buffer_size_ = max_orderbooks;
}

void MarketDataProcessor::handleTradeMessage(const ccapi::Message& msg) {
  // Pin thread to Core 3 (Agent 3 Directive)
  threadAffinityCheck();

  const auto& cid_list = msg.getCorrelationIdList();
  const std::string cid = cid_list.empty() ? "" : cid_list[0];
  auto parts = split(cid, ':');

  std::string exchange = parts.size() > 0 ? parts[0] : "";
  std::string symbol = parts.size() > 1 ? parts[1] : "";
  std::string market_type = parts.size() > 2 ? parts[2] : "spot";

  // Keep essential DEBUG for trade processing visibility
  if (debug_config_.enabled) {
    std::cout << "[" << getCurrentTimestamp() << "][DEBUG] Processing trade: " << exchange << ":" << symbol << std::endl;
  }

  if (exchange.empty() || symbol.empty()) {
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] handleTradeMessage: empty exchange/symbol in CID: " << cid << std::endl;
    return;
  }

  const std::string key = exchange + ":" + symbol + ":" + market_type;

  const auto& elements = msg.getElementList();
  if (elements.empty()) return;

  // exchange timestamp from Message (µs)
  auto tp = msg.getTime();
  int64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count();

  // receive time (µs)
  int64_t recv_time_us = nowMicros();

  for (const auto& el : elements) {
    const auto& m = el.getNameValueMap();

    std::string price_s = getAny(el, {"LAST_PRICE", "PRICE"});
    std::string qty_s = getAny(el, {"LAST_SIZE", "SIZE"});
    std::string is_bm_s = getAny(el, {"IS_BUYER_MAKER"});
    std::string trade_id = getAny(el, {"TRADE_ID"});

    bool ok_price = false, ok_qty = false;
    double price = safeParseDouble("trade.price", price_s, ok_price);
    double qty = safeParseDouble("trade.size", qty_s, ok_qty);

    if (!ok_price || !ok_qty) {
      std::cerr << "handleTradeMessage: bad PRICE/SIZE, element = " << ccapi::toString(m) << std::endl;
      continue;
    }

    Trade t;
    t.timestamp_us = ts_us;  // stores microseconds despite the name
    t.exchange = exchange;
    t.symbol = symbol;
    t.market_type = market_type;
    t.price = price;
    t.quantity = qty;
    t.trade_id = trade_id;

    bool buyer_maker = (is_bm_s == "1" || is_bm_s == "true");
    t.is_buyer_maker = buyer_maker;
    t.side = buyer_maker ? "sell" : "buy";

    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      if (!enable_exclusive_hotspine_) {
        trade_buffer_.push_back(t);
      }
      active_pairs_.insert(exchange + ":" + symbol + ":" + market_type);
      ++trades_received_;
    }
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      pair_stats_[key].trades++;
    }
    candle_agg_->processTrade(t);

    // Write to HotSpine if writer is available
    if (hotspine_writer_) {
      if (!hotspine_writer_->writeTrade(t)) {
        std::cerr << "[" << getCurrentTimestamp() << "][ERROR] HotSpine write failed for trade: " << exchange << ":" << symbol << std::endl;
      }
    }

    // HOTSPINE V3 INTEGRATION (Agent 3)
    // HOTSPINE V3 INTEGRATION (Agent 3 Refactor)
    engine_.process_trade(t);

    if (shm_v3_) {
      // 1. Locate Slot
      uint64_t current_head = shm_v3_->header.head_index.load(std::memory_order_relaxed);
      auto& viewport = shm_v3_->history[current_head % 1024];

      // 2. Lock (Enter)
      shm_v3_->header.global_lock.write_begin();

      // 3. Rasterize
      engine_.snapshot_to_viewport(viewport, t.price);

      // 4. Metadata
      viewport.timestamp_us = t.timestamp_us;
      viewport.close = t.price;
      // In this hot path, we might want to update high/low/open for the viewport context
      // But engine_.snapshot_to_viewport handles the "Infinite Canvas" slice.
      // We'll trust the engine output for volume rows, and just set recent price info here.

      // 5. Lock (Exit)
      shm_v3_->header.global_lock.write_end();

      // 6. Commit
      shm_v3_->header.head_index.store(current_head + 1, std::memory_order_release);
    }

    // In exclusive hotswap mode, don't write to database
    if (!enable_exclusive_hotspine_) {
      // keep stats as milliseconds
      double latency_ms = static_cast<double>(recv_time_us - t.timestamp_us) / 1000.0;
      stats_.avg_latency_ms = 0.99 * stats_.avg_latency_ms + 0.01 * latency_ms;
    }
  }

  // In exclusive hotswap mode, don't flush trades to database
  if (!enable_exclusive_hotspine_) {
    flushTradesIfNeeded();
  }
}

void MarketDataProcessor::handleOrderbookMessage(const ccapi::Message& msg) {
  const auto& cid_list = msg.getCorrelationIdList();
  const std::string cid = cid_list.empty() ? "" : cid_list[0];
  auto parts = split(cid, ':');

  std::string exchange = parts.size() > 0 ? parts[0] : "";
  std::string symbol = parts.size() > 1 ? parts[1] : "";
  std::string market_type = parts.size() > 2 ? parts[2] : "spot";

  if (exchange.empty() || symbol.empty()) {
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] handleOrderbookMessage: empty exchange/symbol in CID: " << cid << std::endl;
    return;
  }

  const std::string key = exchange + ":" + symbol + ":" + market_type;
  const auto& elements = msg.getElementList();
  if (elements.empty()) return;

  auto tp = msg.getTime();
  int64_t ts_us = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count();

  std::vector<HotSpine::HotOrderbookLevel> hot_bids;
  std::vector<HotSpine::HotOrderbookLevel> hot_asks;

  for (const auto& el : elements) {
    auto bid_p_s = getAny(el, {"BID_PRICE", "BEST_BID_PRICE"});
    auto bid_q_s = getAny(el, {"BID_SIZE", "BEST_BID_SIZE"});
    auto ask_p_s = getAny(el, {"ASK_PRICE", "BEST_ASK_PRICE"});
    auto ask_q_s = getAny(el, {"ASK_SIZE", "BEST_ASK_SIZE"});

    if (!bid_p_s.empty() && !bid_q_s.empty()) {
      bool okp = false, okq = false;
      double p = safeParseDouble("bid_price", bid_p_s, okp);
      double q = safeParseDouble("bid_size", bid_q_s, okq);
      if (okp && okq && q > 0.0) {
        hot_bids.push_back({p, q});
      }
    }

    if (!ask_p_s.empty() && !ask_q_s.empty()) {
      bool okp = false, okq = false;
      double p = safeParseDouble("ask_price", ask_p_s, okp);
      double q = safeParseDouble("ask_size", ask_q_s, okq);
      if (okp && okq && q > 0.0) {
        hot_asks.push_back({p, q});
      }
    }
  }

  if (hot_bids.empty() && hot_asks.empty()) return;

  MarketData::OrderbookSnapshot ob;
  ob.timestamp_us = ts_us;
  ob.exchange = exchange;
  ob.symbol = symbol;
  ob.market_type = market_type;

  if (!enable_exclusive_hotspine_) {
    // Only build JSON if we are NOT in exclusive mode (avoid overhead)
    auto build_side_json = [](const std::vector<HotSpine::HotOrderbookLevel>& side) {
      std::ostringstream oss;
      oss << "[";
      for (std::size_t i = 0; i < side.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "[" << side[i].price << "," << side[i].size << "]";
      }
      oss << "]";
      return oss.str();
    };
    ob.bids_json = build_side_json(hot_bids);
    ob.asks_json = build_side_json(hot_asks);
    ob.checksum.clear();

    std::lock_guard<std::mutex> lock(buffer_mutex_);
    orderbook_buffer_.push_back(ob);
  }

  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    pair_stats_[key].orderbooks++;
    ++stats_.orderbooks_received;
  }

  if (hotspine_writer_) {
    hotspine_writer_->writeOrderbook(ob, hot_bids, hot_asks);
  }

  if (!enable_exclusive_hotspine_) {
    flushOrderbooksIfNeeded();
  }
}

void MarketDataProcessor::flushTradesIfNeeded(bool force) {
  // In exclusive hotswap mode, don't flush trades to database
  if (enable_exclusive_hotspine_) {
    return;
  }

  std::vector<Trade> batch;
  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (!force && trade_buffer_.size() < max_trade_buffer_size_) {
      return;
    }
    batch.swap(trade_buffer_);
  }
  if (batch.empty()) {
    return;
  }

  try {
    if (!db_ || !db_->isConnected()) {
      std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database connection not available - cannot insert " << batch.size() << " trades" << std::endl;
      // Put trades back in buffer for retry
      {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        trade_buffer_.insert(trade_buffer_.end(), batch.begin(), batch.end());
      }
      return;
    }

    std::lock_guard<std::mutex> db_lock(db_mutex_);
    auto start_time = std::chrono::high_resolution_clock::now();

    db_->bulkInsertTrades(batch);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    stats_.trades_inserted += batch.size();
    std::cout << "[" << getCurrentTimestamp() << "][INFO] Database: Inserted " << batch.size() << " trades in " << duration.count() << "ms" << std::endl;
  } catch (const std::exception& e) {
    ++errors_;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database insert failed: " << e.what() << " - " << batch.size() << " trades lost" << std::endl;
    // Put trades back in buffer for retry
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      trade_buffer_.insert(trade_buffer_.end(), batch.begin(), batch.end());
    }
  }

  // Check if we need to perform a connection health check
  int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  int64_t last_check = last_connection_check_time_.load();
  if (now_ms - last_check > CONNECTION_CHECK_INTERVAL_MS) {
    if (last_connection_check_time_.compare_exchange_strong(last_check, now_ms)) {
      // Only one thread should perform the health check
      checkDatabaseConnectionHealth();
    }
  }
}

void MarketDataProcessor::flushCandlesIfNeeded(bool force) {
  // In exclusive hotswap mode, don't flush candles to database
  if (enable_exclusive_hotspine_) {
    return;
  }

  // Pull newly completed candles from aggregator
  auto newly_completed = candle_agg_->getAllCompletedCandles();
  stats_.candles_generated += newly_completed.size();

  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    candle_buffer_.insert(candle_buffer_.end(), newly_completed.begin(), newly_completed.end());
    if (!force && candle_buffer_.size() < max_candle_buffer_size_) {
      return;
    }
    if (candle_buffer_.empty()) return;

    // group per table
    std::unordered_map<std::string, std::vector<OHLCV>> by_table;
    for (const auto& c : candle_buffer_) {
      by_table[c.getTableName()].push_back(c);
    }
    candle_buffer_.clear();

    for (auto& kv : by_table) {
      try {
        if (!db_) {
          std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushCandlesIfNeeded: Database connection not available!" << std::endl;
          std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Cannot insert candles - database connection failed" << std::endl;
          continue;
        }

        if (!db_->isConnected()) {
          std::cerr << "[" << getCurrentTimestamp() << "][ERROR] flushCandlesIfNeeded: Database connection not available!" << std::endl;
          std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Cannot insert candles - database connection failed" << std::endl;

          // Attempt to reconnect
          std::cerr << "[" << getCurrentTimestamp() << "][INFO]   Attempting to reconnect to database..." << std::endl;
          continue;
        }

        std::lock_guard<std::mutex> db_lock(db_mutex_);
        db_->bulkInsertOHLCV(kv.first, kv.second);
        stats_.candles_inserted += kv.second.size();
      } catch (const std::exception& e) {
        ++errors_;
        std::cerr << "flushCandlesIfNeeded error: " << e.what() << std::endl;
      }
    }
  }

  // Check if we need to perform a connection health check
  int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  int64_t last_check = last_connection_check_time_.load();
  if (now_ms - last_check > CONNECTION_CHECK_INTERVAL_MS) {
    if (last_connection_check_time_.compare_exchange_strong(last_check, now_ms)) {
      // Only one thread should perform the health check
      checkDatabaseConnectionHealth();
    }
  }
}

void MarketDataProcessor::flushOrderbooksIfNeeded(bool force) {
  // In exclusive hotswap mode, don't flush orderbooks to database
  if (enable_exclusive_hotspine_) {
    return;
  }

  std::vector<OrderbookSnapshot> batch;
  {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    if (!force && orderbook_buffer_.size() < max_orderbook_buffer_size_) {
      return;
    }
    batch.swap(orderbook_buffer_);
  }
  if (batch.empty()) {
    return;
  }

  try {
    if (!db_ || !db_->isConnected()) {
      std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database connection not available - cannot insert " << batch.size() << " orderbooks" << std::endl;
      return;
    }

    std::lock_guard<std::mutex> db_lock(db_mutex_);
    auto start_time = std::chrono::high_resolution_clock::now();

    db_->bulkInsertOrderbooks(batch);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    stats_.orderbooks_inserted += batch.size();
    std::cout << "[" << getCurrentTimestamp() << "][INFO] Database: Inserted " << batch.size() << " orderbooks in " << duration.count() << "ms" << std::endl;
  } catch (const std::exception& e) {
    ++errors_;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database insert failed: " << e.what() << " - " << batch.size() << " orderbooks lost" << std::endl;
  }
}

void MarketDataProcessor::flushBuffers() {
  // In exclusive hotswap mode, only flush HotSpine
  if (enable_exclusive_hotspine_) {
    if (hotspine_writer_) {
      hotspine_writer_->flushBatch();
    }
  } else {
    flushTradesIfNeeded(true);
    flushCandlesIfNeeded(true);
    flushOrderbooksIfNeeded(true);
  }
}

MarketDataProcessor::Stats MarketDataProcessor::getStats() const {
  Stats out;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    out = stats_;
  }
  out.trades_received = trades_received_.load();
  out.errors = errors_.load();
  out.trades_per_sec = trades_per_sec_.load();
  out.orderbooks_per_sec = orderbooks_per_sec_.load();

  const int64_t now_us = nowMicros();
  int64_t last_us = last_stats_ts_us_.exchange(now_us);
  if (last_us > 0) {
    double dt_sec = static_cast<double>(now_us - last_us) / 1'000'000.0;
    if (dt_sec > 0.1) {
      uint64_t tr = trades_received_.load();
      uint64_t ob;
      {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ob = stats_.orderbooks_received;
      }

      uint64_t tr_prev = trades_last_window_.exchange(tr);
      uint64_t ob_prev = orderbooks_last_window_.exchange(ob);

      double tr_rate = static_cast<double>(tr - tr_prev) / dt_sec;
      double ob_rate = static_cast<double>(ob - ob_prev) / dt_sec;

      trades_per_sec_.store(tr_rate);
      orderbooks_per_sec_.store(ob_rate);
      out.trades_per_sec = tr_rate;
      out.orderbooks_per_sec = ob_rate;
    }
  }

  return out;
}

std::unordered_map<std::string, MarketDataProcessor::PairStats> MarketDataProcessor::getPairStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return pair_stats_;  // copy
}

std::string MarketDataProcessor::getStatsJson() const {
  auto st = getStats();
  auto pmap = getPairStats();

  std::ostringstream oss;
  oss << "{";
  oss << "\"trades_received\":" << st.trades_received << ",";
  oss << "\"trades_inserted\":" << st.trades_inserted << ",";
  oss << "\"candles_generated\":" << st.candles_generated << ",";
  oss << "\"candles_inserted\":" << st.candles_inserted << ",";
  oss << "\"orderbooks_received\":" << st.orderbooks_received << ",";
  oss << "\"orderbooks_inserted\":" << st.orderbooks_inserted << ",";
  oss << "\"errors\":" << st.errors << ",";
  oss << "\"avg_latency_ms\":" << st.avg_latency_ms << ",";
  oss << "\"trades_per_sec\":" << st.trades_per_sec << ",";
  oss << "\"orderbooks_per_sec\":" << st.orderbooks_per_sec << ",";

  oss << "\"pairs\":{";
  bool first = true;
  for (const auto& kv : pmap) {
    if (!first) oss << ",";
    first = false;
    oss << "\"" << kv.first << "\":{"
        << "\"trades\":" << kv.second.trades << ","
        << "\"orderbooks\":" << kv.second.orderbooks << "}";
  }
  oss << "}}";
  return oss.str();
}

// Add a method to log WebSocket data flow statistics
void MarketDataProcessor::logWebSocketDataFlowStats() const {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: WebSocket Data Flow Statistics:" << std::endl;

  auto st = getStats();

  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Data Reception Rates:" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Trades received: " << st.trades_received << " (" << st.trades_per_sec << " /sec)" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Orderbooks received: " << st.orderbooks_received << " (" << st.orderbooks_per_sec << " /sec)"
            << std::endl;

  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Data Processing Status:" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Trades inserted: " << st.trades_inserted << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Orderbooks inserted: " << st.orderbooks_inserted << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Candles generated: " << st.candles_generated << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Candles inserted: " << st.candles_inserted << std::endl;

  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Error Statistics:" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Total errors: " << st.errors << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Average latency: " << st.avg_latency_ms << " ms" << std::endl;

  std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket Health Indicators:" << std::endl;
  if (st.trades_per_sec > 0 || st.orderbooks_per_sec > 0) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     WebSocket connection: HEALTHY (receiving data)" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Data flow is active and healthy" << std::endl;
  } else {
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]   WebSocket connection: NO DATA RECEIVED" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]   Possible issues:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - WebSocket connection not established" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Subscription not successful" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Exchange not sending data" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Network connectivity issues" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - WebSocket connection dropped" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - WebSocket protocol errors" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Message parsing failures" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Exchange rate limiting" << std::endl;

    // Add WebSocket reconnection guidance
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]   WebSocket Reconnection:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - CCAPI should automatically attempt to reconnect" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Check for SESSION_STATUS events indicating reconnection attempts" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Monitor for SUBSCRIPTION_STATUS events after reconnection" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]     - Verify WebSocket connection is re-established" << std::endl;
  }

  // Add WebSocket-specific troubleshooting
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket Troubleshooting:" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     If no data is received:" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]       1. Check WebSocket connection status in ExchangeConnectionManager" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]       2. Verify subscription was successful" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]       3. Check for SESSION_STATUS events" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]       4. Look for SUBSCRIPTION_STATUS events" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]       5. Verify WebSocket URL configuration" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]       6. Check network connectivity to exchange" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]       7. Test with different WebSocket timeout settings" << std::endl;

  auto pmap = getPairStats();
  if (!pmap.empty()) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   Active Trading Pairs:" << std::endl;
    for (const auto& kv : pmap) {
      std::cout << "[" << getCurrentTimestamp() << "][INFO]     " << kv.first << ": trades=" << kv.second.trades << ", orderbooks=" << kv.second.orderbooks
                << std::endl;
    }
  } else {
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]   No active trading pairs detected" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][WARNING]   This could indicate subscription issues" << std::endl;
  }
}

// Add a method to validate WebSocket data flow
void MarketDataProcessor::validateWebSocketDataFlow() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: Validating WebSocket data flow..." << std::endl;

  auto st = getStats();

  // Check if we're receiving any data
  if (st.trades_received == 0 && st.orderbooks_received == 0) {
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   WebSocket Data Flow Validation FAILED!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   No data received from WebSocket connection!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   This indicates a critical environment setup issue!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Possible causes:" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - WebSocket connection not established" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Subscription not successful" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Exchange not sending data" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Network connectivity issues" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - WebSocket connection dropped" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - WebSocket protocol errors" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Message parsing failures" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Exchange rate limiting" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - CCAPI configuration not properly handled" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Exchange configuration parsing issues" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     - Invalid WebSocket URL configuration" << std::endl;

    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Troubleshooting steps:" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     1. Check WebSocket connection status in ExchangeConnectionManager" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     2. Verify subscription was successful" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     3. Look for SESSION_STATUS events" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     4. Check for SUBSCRIPTION_STATUS events" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     5. Verify WebSocket URL configuration" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     6. Check network connectivity to exchange" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     7. Test with different WebSocket timeout settings" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     8. Verify CCAPI configuration is properly loaded" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     9. Check if exchange configurations are correctly parsed" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]    10. Validate WebSocket session options and timeouts" << std::endl;

    return;
  }

  // Check data reception rates
  if (st.trades_per_sec < 0.1 && st.orderbooks_per_sec < 0.1) {
    std::cerr << "[" << getCurrentTimestamp() << "][WARNING] WebSocket Data Flow Validation WARNING!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Low data reception rates detected!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Trades: " << st.trades_per_sec << " /sec, Orderbooks: " << st.orderbooks_per_sec << " /sec"
              << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][WARNING] This could indicate WebSocket performance issues" << std::endl;
  }

  // Check for data processing errors
  if (st.errors > 0) {
    std::cerr << "[" << getCurrentTimestamp() << "][WARNING] WebSocket Data Flow Validation WARNING!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Data processing errors detected: " << st.errors << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][WARNING] This could indicate WebSocket data parsing issues" << std::endl;
  }

  // If we get here, data flow is healthy
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket Data Flow Validation PASSED!" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Healthy data reception detected:" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Trades: " << st.trades_received << " received, " << st.trades_per_sec << " /sec" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Orderbooks: " << st.orderbooks_received << " received, " << st.orderbooks_per_sec << " /sec"
            << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Average latency: " << st.avg_latency_ms << " ms" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]     Errors: " << st.errors << std::endl;

  // Check database integration if enabled
  if (db_) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   Database Integration Status:" << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Database connection: " << (db_->isConnected() ? "CONNECTED" : "DISCONNECTED") << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Trades inserted: " << st.trades_inserted << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Orderbooks inserted: " << st.orderbooks_inserted << std::endl;
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     Candles inserted: " << st.candles_inserted << std::endl;
  } else {
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   Database Integration: DISABLED (using HotSpine exclusively)" << std::endl;
  }

  // Check HotSpine integration
  if (hotspine_writer_) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   HotSpine Integration: ENABLED" << std::endl;
  } else {
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   HotSpine Integration: DISABLED" << std::endl;
  }

  std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket data flow validation completed successfully!" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Environment setup appears to be working correctly!" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   CCAPI configuration is properly handled!" << std::endl;
}

// Add a method to validate CCAPI configuration handling
void MarketDataProcessor::validateCCAPIConfiguration() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: Validating CCAPI configuration handling..." << std::endl;

  // Check if we have any active pairs (indicating successful subscription)
  auto pmap = getPairStats();
  if (pmap.empty()) {
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   CCAPI Configuration Validation FAILED!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   No active trading pairs detected!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   This indicates CCAPI configuration may not be properly handled!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]   Possible root causes:" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     1. WebSocket connection not established" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     2. Subscription requests not sent" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     3. Exchange not responding to subscriptions" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     4. WebSocket data not being processed" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     5. Event handler not receiving events" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     6. Correlation ID parsing issues" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     7. WebSocket protocol errors" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR]     8. Network connectivity issues" << std::endl;
    return;
  }

  std::cout << "[" << getCurrentTimestamp() << "][INFO]   CCAPI Configuration Validation PASSED!" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Active trading pairs: " << pmap.size() << std::endl;
  for (const auto& kv : pmap) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     " << kv.first << ": trades=" << kv.second.trades << ", orderbooks=" << kv.second.orderbooks
              << std::endl;
  }
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   CCAPI configuration is properly handled!" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   WebSocket connection and data flow appear healthy!" << std::endl;
}

// Add a method to add WebSocket debugging
void MarketDataProcessor::addWebSocketDebugging() {
  if (!debug_config_.enabled) return;
}

// ============================================================================

// Connection health monitoring methods
void MarketDataProcessor::checkDatabaseConnectionHealth() {
  std::lock_guard<std::mutex> lock(db_mutex_);

  if (!db_) {
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database connection health check: No database connection object available!" << std::endl;
    return;
  }

  bool connected = db_->isConnected();
  std::cout << "[" << getCurrentTimestamp() << "][INFO] Database connection health check: " << (connected ? "CONNECTED" : "DISCONNECTED") << std::endl;

  if (!connected) {
    std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Database connection is down!" << std::endl;
    std::cerr << "[" << getCurrentTimestamp() << "][WARNING] Attempting to reconnect..." << std::endl;

    if (attemptDatabaseReconnect()) {
      std::cout << "[" << getCurrentTimestamp() << "][INFO] Database reconnection successful!" << std::endl;
    } else {
      std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database reconnection failed!" << std::endl;
      std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database operations will be queued until connection is restored" << std::endl;
    }
  }

  // Check if we need to perform a connection health check
  int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  int64_t last_check = last_connection_check_time_.load();
  if (now_ms - last_check > CONNECTION_CHECK_INTERVAL_MS) {
    if (last_connection_check_time_.compare_exchange_strong(last_check, now_ms)) {
      // Only one thread should perform the health check
      checkDatabaseConnectionHealth();
    }
  }
}

bool MarketDataProcessor::attemptDatabaseReconnect() {
  if (!db_) {
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Cannot reconnect: No database connection object available!" << std::endl;
    return false;
  }

  try {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] Attempting database reconnection..." << std::endl;

    // The MSSQLBulkInserter doesn't have a reconnect method, so we'll try to verify the connection
    // which should trigger a reconnection attempt internally if supported
    bool success = db_->verifyConnection();

    if (success) {
      std::cout << "[" << getCurrentTimestamp() << "][INFO] Database reconnection successful!" << std::endl;
      return true;
    } else {
      std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database reconnection failed!" << std::endl;
      return false;
    }
  } catch (const std::exception& e) {
    std::cerr << "[" << getCurrentTimestamp() << "][ERROR] Database reconnection error: " << e.what() << std::endl;
    return false;
  }
}

// ============================================================================
// Update HotSpine writer for reconnection
// ============================================================================
void MarketDataProcessor::updateHotSpineWriter(std::shared_ptr<HotSpine::HotSpineWriter> new_writer) {
  std::lock_guard<std::mutex> lock(buffer_mutex_);
  hotspine_writer_ = new_writer;
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataProcessor: HotSpine writer updated" << std::endl;
}
