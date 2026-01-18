#include "hotspine_data_bridge.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace BTQuant {
namespace RenderEngine {

HotSpineDataBridge::HotSpineDataBridge(const std::string &shm_name,
                                       const std::string &symbols_file)
    : shm_name_(shm_name), symbols_file_(symbols_file), running_(false),
      last_trade_count_(0), last_orderbook_count_(0),
      total_trades_processed_(0), total_orderbooks_processed_(0),
      data_latency_us_(0), update_frequency_hz_(0.0) {
  // Initialize symbol registry
  symbol_registry_ = &BTQuant::SymbolRegistry::instance();

  // Load symbol mappings
  if (!symbols_file_.empty()) {
    symbol_registry_->load_from_file(symbols_file_);
  }

  // Initialize HotSpine reader
  try {
    hotspine_reader_ = std::make_unique<HotSpine::HotSpineReader>(shm_name_);
    if (!hotspine_reader_->isAttached()) {
      throw std::runtime_error("Failed to attach to HotSpine shared memory");
    }
    std::cout << "[HotSpineDataBridge] Successfully connected to " << shm_name_
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "[HotSpineDataBridge] Failed to initialize: " << e.what()
              << std::endl;
    throw;
  }

  // Initialize performance monitoring
  last_update_time_ = std::chrono::high_resolution_clock::now();

  // Pre-allocate data structures for performance
  trade_buffer_.reserve(1000);
  orderbook_buffer_.reserve(100);

  std::cout << "[HotSpineDataBridge] Initialized successfully" << std::endl;
}

HotSpineDataBridge::~HotSpineDataBridge() { stop(); }

bool HotSpineDataBridge::start() {
  if (running_) {
    return true;
  }

  if (!isConnected()) {
    std::cerr << "[HotSpineDataBridge] Cannot start - not connected to HotSpine"
              << std::endl;
    return false;
  }

  running_ = true;

  // Start data processing thread
  data_thread_ = std::thread(&HotSpineDataBridge::dataProcessingLoop, this);

  // Start performance monitoring thread
  perf_thread_ =
      std::thread(&HotSpineDataBridge::performanceMonitoringLoop, this);

  std::cout << "[HotSpineDataBridge] Started data processing threads"
            << std::endl;
  return true;
}

void HotSpineDataBridge::stop() {
  if (!running_) {
    return;
  }

  running_ = false;

  // Wait for threads to finish
  if (data_thread_.joinable()) {
    data_thread_.join();
  }

  if (perf_thread_.joinable()) {
    perf_thread_.join();
  }

  std::cout << "[HotSpineDataBridge] Stopped data processing" << std::endl;
}

bool HotSpineDataBridge::isConnected() const {
  return hotspine_reader_ && hotspine_reader_->isAttached();
}

std::vector<MarketDataUpdate> HotSpineDataBridge::getLatestUpdates() {
  std::lock_guard lock(data_mutex_);

  std::vector<MarketDataUpdate> updates;
  updates.reserve(trade_buffer_.size() + orderbook_buffer_.size());

  // Convert trades to market data updates
  for (const auto &trade : trade_buffer_) {
    MarketDataUpdate update;
    update.type = MarketDataType::TRADE;
    update.symbol_id = trade.symbol_id;
    update.timestamp = trade.ts_exchange;
    update.local_timestamp = trade.ts_local;
    update.price = trade.price;
    update.size = trade.size;
    update.side = (trade.side == 0) ? "buy" : "sell";

    // Get symbol info for display
    if (auto symbol_info = symbol_registry_->get_symbol_info(trade.symbol_id)) {
      update.exchange = symbol_info->exchange;
      update.symbol = symbol_info->symbol;
    }

    updates.push_back(update);
  }

  // Convert orderbooks to market data updates
  for (const auto &ob : orderbook_buffer_) {
    MarketDataUpdate update;
    update.type = MarketDataType::ORDERBOOK;
    update.symbol_id = ob.symbol_id;
    update.timestamp = ob.ts_exchange;
    update.local_timestamp = ob.ts_local;

    // Get symbol info for display
    if (auto symbol_info = symbol_registry_->get_symbol_info(ob.symbol_id)) {
      update.exchange = symbol_info->exchange;
      update.symbol = symbol_info->symbol;
    }

    // Copy bid/ask levels (full depth from snapshot)
    for (int i = 0; i < ob.bids_count && i < 20; ++i) {
      update.bids.push_back({ob.bids[i].price, ob.bids[i].size});
    }
    for (int i = 0; i < ob.asks_count && i < 20; ++i) {
      update.asks.push_back({ob.asks[i].price, ob.asks[i].size});
    }

    updates.push_back(update);
  }

  // Clear buffers after copying
  trade_buffer_.clear();
  orderbook_buffer_.clear();

  return updates;
}

std::vector<SymbolData> HotSpineDataBridge::getAllSymbols() const {
  std::vector<SymbolData> symbols;

  auto all_symbols = symbol_registry_->get_all_symbols();
  symbols.reserve(all_symbols.size());

  for (const auto &symbol_info : all_symbols) {
    SymbolData data;
    data.symbol_id = symbol_info.id;
    data.exchange = symbol_info.exchange;
    data.symbol = symbol_info.symbol;
    data.full_symbol = symbol_info.full_symbol();

    // Get latest market data for this symbol
    std::lock_guard lock(symbol_data_mutex_);
    auto it = symbol_market_data_.find(symbol_info.id);
    if (it != symbol_market_data_.end()) {
      data.last_price = it->second.last_price;
      data.price_change = it->second.price_change;
      data.price_change_percent = it->second.price_change_percent;
      data.volume_24h = it->second.volume_24h;
      data.high_24h = it->second.high_24h;
      data.low_24h = it->second.low_24h;
      data.bid_price = it->second.bid_price;
      data.ask_price = it->second.ask_price;
      data.spread = it->second.spread;
      data.last_update_time = it->second.last_update_time;
    }

    symbols.push_back(data);
  }

  return symbols;
}

PerformanceMetrics HotSpineDataBridge::getPerformanceMetrics() const {
  std::lock_guard lock(perf_mutex_);
  return performance_metrics_;
}

void HotSpineDataBridge::dataProcessingLoop() {
  std::cout << "[HotSpineDataBridge] Data processing loop started" << std::endl;

  auto last_stats_time = std::chrono::high_resolution_clock::now();
  const auto stats_interval = std::chrono::seconds(5);

  while (running_) {
    bool data_processed = false;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Process trades
    HotSpine::HotTrade trade;
    while (hotspine_reader_->pollTrade(trade)) {
      processTrade(trade);
      data_processed = true;
      total_trades_processed_++;
      // Don't process more than 1000 per loop to keep UI responsive
      if (total_trades_processed_ % 1000 == 0)
        break;
    }

    // Process orderbooks
    HotSpine::HotOrderbookSnapshot orderbook;
    while (hotspine_reader_->pollOrderbook(orderbook)) {
      processOrderbook(orderbook);
      data_processed = true;
      total_orderbooks_processed_++;
      if (total_orderbooks_processed_ % 100 == 0)
        break;
    }

    if (data_processed) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - start_time);

      std::lock_guard lock(perf_mutex_);
      data_latency_us_ = latency.count();
    }

    // Print periodic statistics
    auto now = std::chrono::high_resolution_clock::now();
    if (now - last_stats_time >= stats_interval) {
      printStatistics();
      last_stats_time = now;
    }

    // Small sleep to prevent busy waiting
    if (!data_processed) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  std::cout << "[HotSpineDataBridge] Data processing loop stopped" << std::endl;
}

void HotSpineDataBridge::processTrade(const HotSpine::HotTrade &trade) {
  auto now = std::chrono::high_resolution_clock::now();
  auto local_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                             now.time_since_epoch())
                             .count();

  // Add to trade buffer
  {
    std::lock_guard lock(data_mutex_);
    trade_buffer_.push_back(trade);

    // Limit buffer size to prevent memory growth
    if (trade_buffer_.size() > MAX_BUFFER_SIZE) {
      trade_buffer_.erase(trade_buffer_.begin(),
                          trade_buffer_.begin() +
                              (trade_buffer_.size() - MAX_BUFFER_SIZE));
    }
  }

  // Update symbol market data
  updateSymbolMarketData(trade.symbol_id, trade.price, trade.size,
                         local_timestamp);

  // Calculate data-to-processing latency
  // Remove ts_local processing as it's not available in stub
}

void HotSpineDataBridge::processOrderbook(
    const HotSpine::HotOrderbookSnapshot &orderbook) {
  auto now = std::chrono::high_resolution_clock::now();
  auto local_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                             now.time_since_epoch())
                             .count();

  // Add to orderbook buffer
  {
    std::lock_guard lock(data_mutex_);
    orderbook_buffer_.push_back(orderbook);

    // Limit buffer size
    if (orderbook_buffer_.size() >
        MAX_BUFFER_SIZE / 10) { // Smaller buffer for orderbooks
      orderbook_buffer_.erase(
          orderbook_buffer_.begin(),
          orderbook_buffer_.begin() +
              (orderbook_buffer_.size() - MAX_BUFFER_SIZE / 10));
    }
  }

  // Update symbol market data with bid/ask prices
  if (orderbook.bids_count > 0 && orderbook.asks_count > 0) {
    updateSymbolOrderbookData(orderbook.symbol_id, orderbook.bids[0].price,
                              orderbook.asks[0].price, local_timestamp);
  }
}

void HotSpineDataBridge::updateSymbolMarketData(uint32_t symbol_id,
                                                double price, double size,
                                                uint64_t timestamp) {
  std::lock_guard lock(symbol_data_mutex_);

  auto &data = symbol_market_data_[symbol_id];

  // Update price data
  double old_price = data.last_price;
  data.last_price = price;
  data.last_update_time = timestamp;

  // Calculate price change
  if (old_price > 0) {
    data.price_change = price - old_price;
    data.price_change_percent = (data.price_change / old_price) * 100.0;
  }

  // Update 24h high/low
  if (data.high_24h == 0 || price > data.high_24h) {
    data.high_24h = price;
  }
  if (data.low_24h == 0 || price < data.low_24h) {
    data.low_24h = price;
  }

  // Update volume
  data.volume_24h += size;

  // Calculate momentum (simple moving average of price changes)
  data.momentum_history.push_back(data.price_change_percent);
  if (data.momentum_history.size() > 20) { // Keep last 20 changes
    data.momentum_history.pop_front();
  }

  // Calculate average momentum
  double total_momentum = 0.0;
  for (double momentum : data.momentum_history) {
    total_momentum += momentum;
  }
  data.momentum = data.momentum_history.empty()
                      ? 0.0
                      : total_momentum / data.momentum_history.size();
}

void HotSpineDataBridge::updateSymbolOrderbookData(uint32_t symbol_id,
                                                   double bid_price,
                                                   double ask_price,
                                                   uint64_t timestamp) {
  std::lock_guard lock(symbol_data_mutex_);

  auto &data = symbol_market_data_[symbol_id];
  data.bid_price = bid_price;
  data.ask_price = ask_price;
  data.spread = ask_price - bid_price;
  data.spread_percent =
      (bid_price > 0) ? (data.spread / bid_price) * 100.0 : 0.0;
  data.last_update_time = timestamp;
}

void HotSpineDataBridge::performanceMonitoringLoop() {
  std::cout << "[HotSpineDataBridge] Performance monitoring loop started"
            << std::endl;

  auto last_time = std::chrono::high_resolution_clock::now();
  uint64_t last_trades = total_trades_processed_;
  uint64_t last_orderbooks = total_orderbooks_processed_;

  while (running_) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time);

    if (elapsed.count() > 0) {
      uint64_t current_trades = total_trades_processed_;
      uint64_t current_orderbooks = total_orderbooks_processed_;

      double trades_per_sec =
          (current_trades - last_trades) * 1000.0 / elapsed.count();
      double orderbooks_per_sec =
          (current_orderbooks - last_orderbooks) * 1000.0 / elapsed.count();

      std::lock_guard lock(perf_mutex_);
      performance_metrics_.trades_per_second = trades_per_sec;
      performance_metrics_.orderbooks_per_second = orderbooks_per_sec;
      performance_metrics_.total_trades_processed = current_trades;
      performance_metrics_.total_orderbooks_processed = current_orderbooks;
      performance_metrics_.connection_healthy = isConnected();

      // Update buffer status
      // Remove buffer status as it's not available in stub

      last_time = now;
      last_trades = current_trades;
      last_orderbooks = current_orderbooks;
    }
  }

  std::cout << "[HotSpineDataBridge] Performance monitoring loop stopped"
            << std::endl;
}

void HotSpineDataBridge::printStatistics() {
  std::lock_guard lock(perf_mutex_);

  std::cout << "[HotSpineDataBridge] Statistics:" << std::endl;
  std::cout << "  Trades processed: "
            << performance_metrics_.total_trades_processed << std::endl;
  std::cout << "  Orderbooks processed: "
            << performance_metrics_.total_orderbooks_processed << std::endl;
  std::cout << "  Trades/sec: " << performance_metrics_.trades_per_second
            << std::endl;
  std::cout << "  Orderbooks/sec: "
            << performance_metrics_.orderbooks_per_second << std::endl;
  std::cout << "  Avg processing latency: "
            << performance_metrics_.avg_processing_latency_us << " µs"
            << std::endl;
  std::cout << "  Buffer utilization: "
            << performance_metrics_.buffer_utilization_percent << "%"
            << std::endl;
  std::cout << "  Connection healthy: "
            << (performance_metrics_.connection_healthy ? "Yes" : "No")
            << std::endl;
}

bool HotSpineDataBridge::reconnect() {
  std::cout << "[HotSpineDataBridge] Attempting to reconnect..." << std::endl;

  // Stop current processing
  bool was_running = running_;
  if (was_running) {
    stop();
  }

  // Recreate HotSpine reader
  try {
    hotspine_reader_ = std::make_unique<HotSpine::HotSpineReader>(shm_name_);
    if (!hotspine_reader_->isAttached()) {
      std::cerr << "[HotSpineDataBridge] Reconnection failed - could not "
                   "connect to shared memory"
                << std::endl;
      return false;
    }

    std::cout << "[HotSpineDataBridge] Reconnected successfully" << std::endl;

    // Restart if it was running before
    if (was_running) {
      return start();
    }

    return true;
  } catch (const std::exception &e) {
    std::cerr << "[HotSpineDataBridge] Reconnection failed: " << e.what()
              << std::endl;
    return false;
  }
}

void HotSpineDataBridge::reloadSymbolMappings() {
  if (!symbols_file_.empty()) {
    std::cout << "[HotSpineDataBridge] Reloading symbol mappings from "
              << symbols_file_ << std::endl;
    symbol_registry_->load_from_file(symbols_file_);
  }
}

std::vector<std::string> HotSpineDataBridge::getAvailableExchanges() const {
  return symbol_registry_->get_exchanges();
}

std::vector<SymbolData>
HotSpineDataBridge::getExchangeSymbols(const std::string &exchange) const {
  std::vector<SymbolData> symbols;

  auto all_symbols = symbol_registry_->get_all_symbols();
  std::vector<SymbolInfo> exchange_symbols;
  for (const auto &symbol_info : all_symbols) {
    if (symbol_info.exchange == exchange) {
      exchange_symbols.push_back(symbol_info);
    }
  }
  symbols.reserve(exchange_symbols.size());

  for (const auto &symbol_info : exchange_symbols) {
    SymbolData data;
    data.symbol_id = symbol_info.id;
    data.exchange = symbol_info.exchange;
    data.symbol = symbol_info.symbol;
    data.full_symbol = symbol_info.full_symbol();

    // Get latest market data
    std::lock_guard lock(symbol_data_mutex_);
    auto it = symbol_market_data_.find(symbol_info.id);
    if (it != symbol_market_data_.end()) {
      data.last_price = it->second.last_price;
      data.price_change = it->second.price_change;
      data.price_change_percent = it->second.price_change_percent;
      data.volume_24h = it->second.volume_24h;
      data.high_24h = it->second.high_24h;
      data.low_24h = it->second.low_24h;
      data.bid_price = it->second.bid_price;
      data.ask_price = it->second.ask_price;
      data.spread = it->second.spread;
      data.momentum = it->second.momentum;
      data.last_update_time = it->second.last_update_time;
    }

    symbols.push_back(data);
  }

  return symbols;
}

} // namespace RenderEngine
} // namespace BTQuant