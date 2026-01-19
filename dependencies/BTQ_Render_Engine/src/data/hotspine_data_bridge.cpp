#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "stubs/symbol_registry.hpp"
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace BTQuant {

HotSpineDataBridge::HotSpineDataBridge(const std::string &shm_path)
    : m_shm_path(shm_path) {

  // Load symbol registry from shared memory file
  SymbolRegistry::instance().load_from_file("/dev/shm/btquant_symbols.json");

  // Force simulation mode for testing
  std::cerr << "[HotSpine] Using SIMULATION mode for testing" << std::endl;
  m_is_simulated = true;
  init_simulation();
}

HotSpineDataBridge::~HotSpineDataBridge() {
  stop();
  if (m_shm_ptr && m_shm_ptr != MAP_FAILED) {
    munmap(m_shm_ptr, m_shm_size);
  }
  if (m_shm_fd != -1) {
    close(m_shm_fd);
  }
}

bool HotSpineDataBridge::start() {
  m_running = true;
  return true;
}

void HotSpineDataBridge::stop() { m_running = false; }

void HotSpineDataBridge::poll() {
  if (!m_running)
    return;

  if (m_is_simulated) {
    poll_simulated();
  } else {
    poll_shm();
  }
}

std::shared_ptr<MarketInstrument>
HotSpineDataBridge::get_instrument(uint32_t symbol_id) {
  std::lock_guard<std::mutex> lock(m_map_mutex);

  auto it = m_id_map.find(symbol_id);
  if (it != m_id_map.end())
    return it->second;

  // Lazy Registration
  char buf[64];
  snprintf(buf, sizeof(buf), "Unknown-%u", symbol_id);
  std::string sym = buf;
  std::string exchange = "UNKNOWN";

  // Get symbol information from registry
  auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
  if (symbol_info) {
    sym = symbol_info->symbol;
    exchange = symbol_info->exchange;
  }

  auto inst = std::make_shared<MarketInstrument>();
  inst->symbol = sym;
  inst->exchange = exchange;
  inst->symbol_id = symbol_id;

  m_instruments[inst->symbol] = inst;
  m_id_map[symbol_id] = inst;

  return inst;
}

void HotSpineDataBridge::poll_shm() {
  if (!m_header)
    return;

  // --- Process Trades ---
  uint64_t write_idx =
      __atomic_load_n(&m_header->write_index, __ATOMIC_ACQUIRE);
  uint64_t capacity = m_header->capacity;

  while (m_last_read_idx < write_idx) {
    const HotTrade &trade = m_trades[m_last_read_idx % capacity];

    auto inst = get_instrument(trade.symbol_id);
    inst->push_trade(trade);

    // Send to MarketDataProcessor for OHLCV aggregation
    if (m_data_processor) {
      RenderEngine::MarketDataUpdate update;
      update.type = RenderEngine::MarketDataType::TRADE;
      update.symbol_id = trade.symbol_id;
      update.timestamp = trade.ts_exchange;
      update.price = trade.price;
      update.size = trade.size;
      update.side = trade.side == 0 ? "buy" : "sell";
      m_data_processor->processTradeUpdate(update);
    }

    m_last_read_idx++;
    if (write_idx - m_last_read_idx > 1000) { // Catch up if behind
      if (write_idx - m_last_read_idx > capacity)
        m_last_read_idx = write_idx;
      break;
    }
  }

  // --- Process Orderbooks ---
  uint64_t book_write_idx =
      __atomic_load_n(&m_header->orderbook_write_index, __ATOMIC_ACQUIRE);
  uint64_t book_capacity = m_header->orderbook_capacity;

  while (m_last_book_read_idx < book_write_idx) {
    const HotOrderbookSnapshot &snap =
        m_books[m_last_book_read_idx % book_capacity];

    auto inst = get_instrument(snap.symbol_id);
    inst->update_book(snap);

    m_last_book_read_idx++;
    if (book_write_idx - m_last_book_read_idx > 100)
      break;
  }

  // Update Reader Index
  __atomic_store_n(&m_header->read_index, m_last_read_idx, __ATOMIC_RELEASE);
  __atomic_store_n(&m_header->orderbook_read_index, m_last_book_read_idx,
                   __ATOMIC_RELEASE);
}

void HotSpineDataBridge::init_simulation() {
  std::lock_guard<std::mutex> lock(m_map_mutex);

  auto create_inst = [&](const std::string &symbol, uint32_t id) {
    auto inst = std::make_shared<MarketInstrument>();
    inst->symbol = symbol;
    inst->exchange = "BINANCE";
    inst->symbol_id = id;
    m_instruments[symbol] = inst;
    m_id_map[id] = inst;
  };

  create_inst("BTC-USDT", 1);
  create_inst("ETH-USDT", 2);
  create_inst("SOL-USDT", 3);
}

void HotSpineDataBridge::poll_simulated() {
  static std::mt19937 rng(std::random_device{}());
  static std::uniform_real_distribution<double> dist(0.0, 1.0);

  // Simulate high-frequency updates
  double now = (double)std::chrono::system_clock::to_time_t(
                   std::chrono::system_clock::now()) +
               (double)std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                       .count() /
                   1e6;

  // Simulate 10kHz bursty traffic
  for (auto &[sym, inst] : m_instruments) {
    if (dist(rng) < 0.1) { // 10% chance update per poll
      HotTrade trade;
      trade.ts_exchange = (uint64_t)(now * 1000000.0);

      double last_price = 45000.0;
      if (inst->size > 0)
        last_price = inst->closes[(inst->write_idx - 1 +
                                   MarketInstrument::HISTORY_CAPACITY) %
                                  MarketInstrument::HISTORY_CAPACITY];

      double change = (dist(rng) - 0.5) * 10.0;
      trade.price = last_price + change;
      trade.size = dist(rng) * 2.0;
      trade.symbol_id = inst->symbol_id;
      trade.side = (change > 0) ? 1 : 2;

      inst->push_trade(trade);

      // Send to MarketDataProcessor for OHLCV aggregation
      if (m_data_processor) {
        RenderEngine::MarketDataUpdate update;
        update.type = RenderEngine::MarketDataType::TRADE;
        update.symbol_id = trade.symbol_id;
        update.timestamp = trade.ts_exchange;
        update.price = trade.price;
        update.size = trade.size;
        update.side = trade.side == 1 ? "buy" : "sell";
        m_data_processor->processTradeUpdate(update);
      }

      // Periodically update simulated book
      if (dist(rng) < 0.05) {
        HotOrderbookSnapshot snap;
        snap.ts_exchange = trade.ts_exchange;
        snap.symbol_id = inst->symbol_id;
        snap.bids_count = 20;
        snap.asks_count = 20;
        for (int i = 0; i < 20; ++i) {
          snap.bids[i] = {trade.price - (i + 1) * 0.5, dist(rng)};
          snap.asks[i] = {trade.price + (i + 1) * 0.5, dist(rng)};
        }
        inst->update_book(snap);
      }
    }
  }

  // std::this_thread::sleep_for(std::chrono::microseconds(100)); // Simulate
  // work
}

} // namespace BTQuant
