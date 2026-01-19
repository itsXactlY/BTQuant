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

  // Attempt to open shared memory
  m_shm_fd = shm_open(m_shm_path.c_str(), O_RDONLY, 0666);
  if (m_shm_fd == -1) {
    std::cerr << "[HotSpineDataBridge] Failed to open SHM: " << m_shm_path
              << ". Falling back to simulation." << std::endl;
    m_is_simulated = true;
    // init_simulation();
    return true;
  }

  // Get SHM size
  struct stat st;
  if (fstat(m_shm_fd, &st) == -1) {
    std::cerr << "[HotSpineDataBridge] fstat failed" << std::endl;
    m_is_simulated = true;
    // init_simulation();
    return true;
  }
  m_shm_size = st.st_size;

  // Map SHM
  m_shm_ptr = mmap(nullptr, m_shm_size, PROT_READ, MAP_SHARED, m_shm_fd, 0);
  if (m_shm_ptr == MAP_FAILED) {
    std::cerr << "[HotSpineDataBridge] mmap failed" << std::endl;
    m_is_simulated = true;
    // init_simulation();
    return true;
  }

  m_header = static_cast<SharedMemoryHeader *>(m_shm_ptr);
  if (m_header->magic != 0x55515442) {
    std::cerr << "[HotSpineDataBridge] Invalid magic number: " << std::hex
              << m_header->magic << std::dec << ". Using simulation."
              << std::endl;
    m_is_simulated = true;
    // init_simulation();
    return true;
  }

  // Set up pointers
  m_trades = (HotTrade *)((uint8_t *)m_shm_ptr + sizeof(SharedMemoryHeader));
  m_books = (HotOrderbookSnapshot *)((uint8_t *)m_trades +
                                     (m_header->capacity * sizeof(HotTrade)));

  m_last_read_idx = m_header->read_index;
  m_last_book_read_idx = m_header->orderbook_read_index;

  std::cout << "[HotSpineDataBridge] Connected to SHM " << m_shm_path
            << " (Capacity: " << m_header->capacity << ")" << std::endl;

  return true;
}

void HotSpineDataBridge::stop() { m_running = false; }

void HotSpineDataBridge::poll() {
  if (!m_running)
    return;

  // if (m_is_simulated) {
  //   poll_simulated();
  // } else {
    poll_shm();
  // }
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

    // --- Pass to MarketDataProcessor ---
    if (m_data_processor) {
      RenderEngine::MarketDataUpdate update{};
      update.type = RenderEngine::MarketDataType::TRADE;
      update.symbol_id = trade.symbol_id;
      update.timestamp = trade.ts_exchange; // Microseconds
      update.price = trade.price;
      update.size = trade.size;
      update.side = (trade.side == 0) ? "buy" : "sell";
      m_data_processor->processTradeUpdate(update);
    }

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

    // --- Pass to MarketDataProcessor ---
    if (m_data_processor) {
      RenderEngine::MarketDataUpdate update{};
      update.type = RenderEngine::MarketDataType::ORDERBOOK;
      update.symbol_id = snap.symbol_id;
      update.timestamp = snap.ts_exchange;

      // Convert HotOrderbookLevel to PriceLevel
      for (int i = 0; i < snap.bids_count; ++i) {
        RenderEngine::PriceLevel level;
        level.price = snap.bids[i].price;
        level.size = snap.bids[i].size;
        update.bids.push_back(level);
      }
      for (int i = 0; i < snap.asks_count; ++i) {
        RenderEngine::PriceLevel level;
        level.price = snap.asks[i].price;
        level.size = snap.asks[i].size;
        update.asks.push_back(level);
      }
      m_data_processor->processOrderbookUpdate(update);
    }

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


} // namespace BTQuant
