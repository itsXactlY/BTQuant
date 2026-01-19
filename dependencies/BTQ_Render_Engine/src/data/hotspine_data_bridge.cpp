#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "symbol_registry.hpp"
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

  // Detect if shared memory is valid/available
  bool shm_valid = (m_header != nullptr && m_header->magic == 0x55515442);

  if (!shm_valid) {
    poll_simulated();
  } else {
    poll_shm();
  }
}

std::shared_ptr<InstrumentStore>
HotSpineDataBridge::get_instrument(uint32_t symbol_id) {
  std::lock_guard<std::mutex> lock(m_map_mutex);

  // Search by ID
  for (auto &[sym, inst] : m_instruments) {
    if (inst->symbol_id == symbol_id)
      return inst;
  }

  // Dynamic Discovery
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

  auto inst = std::make_shared<InstrumentStore>();
  inst->symbol = sym;
  inst->exchange = exchange;
  inst->symbol_id = symbol_id;
  m_instruments[sym] = inst;

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
    // Cast raw bytes directly (No Mocks)
    const HotTrade &trade = m_trades[m_last_read_idx % capacity];

    auto inst = get_instrument(trade.symbol_id);
    {
      std::lock_guard<std::mutex> lock(inst->data_mutex);
      double ts = (double)trade.ts_exchange / 1000000.0;

      // SoA Update
      inst->timestamps.push_back(ts);
      inst->opens.push_back(trade.price);
      inst->highs.push_back(trade.price);
      inst->lows.push_back(trade.price);
      inst->closes.push_back(trade.price);
      inst->volumes.push_back(trade.size);

      // Volume Profile Accumulation (Price rounded to 0.5 tick)
      double tick_size = 0.5;
      double rounded_price = std::round(trade.price / tick_size) * tick_size;
      inst->m_vol_profile[rounded_price] += trade.size;

      // Keep a reasonable history (HFT density management)
      if (inst->timestamps.size() > 10000) {
        inst->timestamps.erase(inst->timestamps.begin());
        inst->opens.erase(inst->opens.begin());
        inst->highs.erase(inst->highs.begin());
        inst->lows.erase(inst->lows.begin());
        inst->closes.erase(inst->closes.begin());
        inst->volumes.erase(inst->volumes.begin());
      }
    }

    m_last_read_idx++;
  }

  // --- Process Orderbooks ---
  uint64_t book_write_idx =
      __atomic_load_n(&m_header->orderbook_write_index, __ATOMIC_ACQUIRE);
  uint64_t book_capacity = m_header->orderbook_capacity;

  while (m_last_book_read_idx < book_write_idx) {
    const HotOrderbookSnapshot &snap =
        m_books[m_last_book_read_idx % book_capacity];

    auto inst = get_instrument(snap.symbol_id);
    {
      std::lock_guard<std::mutex> lock(inst->data_mutex);
      inst->latest_snapshot = snap;
    }

    m_last_book_read_idx++;
  }

  // Update Reader Index
  __atomic_store_n(&m_header->read_index, m_last_read_idx, __ATOMIC_RELEASE);
  __atomic_store_n(&m_header->orderbook_read_index, m_last_book_read_idx,
                   __ATOMIC_RELEASE);
}

void HotSpineDataBridge::init_simulation() {
  std::lock_guard<std::mutex> lock(m_map_mutex);

  auto create_inst = [&](const std::string &symbol, uint32_t id) {
    auto inst = std::make_shared<InstrumentStore>();
    inst->symbol = symbol;
    inst->exchange = "BINANCE";
    inst->symbol_id = id;
    m_instruments[symbol] = inst;
  };

  create_inst("BTC-USDT", 1);
  create_inst("ETH-USDT", 2);
  create_inst("SOL-USDT", 3);
}

void HotSpineDataBridge::poll_simulated() {
  static double t = 0;
  t += 0.01;

  std::lock_guard<std::mutex> lock(m_map_mutex);
  for (auto &[sym, inst] : m_instruments) {
    std::lock_guard<std::mutex> data_lock(inst->data_mutex);

    auto now_ns = std::chrono::high_resolution_clock::now();
    double now = std::chrono::duration_cast<std::chrono::microseconds>(
                     now_ns.time_since_epoch())
                     .count() /
                 1000000.0;
    double base_price = inst->symbol_id * 1000.0;
    double sine_val = std::sin(t + inst->symbol_id) * 50.0;
    double current_price = base_price + sine_val;

    // SoA Update
    inst->timestamps.push_back(now);
    inst->opens.push_back(current_price - 1.0);
    inst->highs.push_back(current_price + 2.0);
    inst->lows.push_back(current_price - 3.0);
    inst->closes.push_back(current_price);
    inst->volumes.push_back(100.0 + std::abs(sine_val));

    // Volume Profile Accumulation for Simulation
    double tick_size = 0.5;
    double rounded_price = std::round(current_price / tick_size) * tick_size;
    inst->m_vol_profile[rounded_price] += 10.0;

    // Keep history manageable
    if (inst->timestamps.size() > 500) {
      inst->timestamps.erase(inst->timestamps.begin());
      inst->opens.erase(inst->opens.begin());
      inst->highs.erase(inst->highs.begin());
      inst->lows.erase(inst->lows.begin());
      inst->closes.erase(inst->closes.begin());
      inst->volumes.erase(inst->volumes.begin());
    }
  }
}

} // namespace BTQuant
