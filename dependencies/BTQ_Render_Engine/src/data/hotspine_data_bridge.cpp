#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "symbol_registry.hpp"
#include <chrono>
#include <cmath>
#include <cstring> // For strerror
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
  // Open shared memory
  m_shm_fd = shm_open(m_shm_path.c_str(), O_RDWR, 0666);
  if (m_shm_fd == -1) {
    std::cerr << "[HotSpineDataBridge] ERROR: Failed to open shared memory '"
              << m_shm_path << "': " << strerror(errno) << std::endl;
    std::cerr << "  Make sure HotSpine data feed is running!" << std::endl;
    return false; // FAIL - require real data
  }

  // Get the size
  struct stat sb;
  if (fstat(m_shm_fd, &sb) == -1) {
    std::cerr << "[HotSpineDataBridge] ERROR: Failed to fstat shared memory"
              << std::endl;
    close(m_shm_fd);
    m_shm_fd = -1;
    return false;
  }
  m_shm_size = sb.st_size;

  // Map it
  m_shm_ptr =
      mmap(NULL, m_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_shm_fd, 0);
  if (m_shm_ptr == MAP_FAILED) {
    std::cerr << "[HotSpineDataBridge] ERROR: Failed to mmap shared memory"
              << std::endl;
    close(m_shm_fd);
    m_shm_fd = -1;
    return false;
  }

  // Initialize pointers
  m_header = reinterpret_cast<SharedMemoryHeader *>(m_shm_ptr);

  // Verify magic number - "UQTB" = 0x42545155 in little-endian uint32_t
  // or check ASCII bytes directly: 'U'(55) 'Q'(51) 'T'(54) 'B'(42)
  uint32_t expected_magic_le = 0x42545155; // Little-endian representation
  if (m_header->magic != expected_magic_le) {
    std::cerr
        << "[HotSpineDataBridge] ERROR: Invalid magic number in shared memory"
        << std::endl;
    std::cerr << "  Expected: 0x" << std::hex << expected_magic_le
              << " (UQTB), Got: 0x" << m_header->magic << std::dec << std::endl;
    std::cerr << "  Cannot proceed without valid shared memory!" << std::endl;
    munmap(m_shm_ptr, m_shm_size);
    m_shm_ptr = nullptr;
    m_header = nullptr;
    close(m_shm_fd);
    m_shm_fd = -1;
    return false; // FAIL - don't fall back to simulation
  }

  // Calculate ring buffer positions
  // CRITICAL: HotSpine reserves 4096 bytes for header, NOT
  // sizeof(SharedMemoryHeader)!
  constexpr size_t HOTSPINE_HEADER_SIZE = 4096;
  m_trades = reinterpret_cast<HotTrade *>(static_cast<char *>(m_shm_ptr) +
                                          HOTSPINE_HEADER_SIZE);

  size_t trades_size = m_header->capacity * sizeof(HotTrade);
  m_books = reinterpret_cast<HotOrderbookSnapshot *>(
      static_cast<char *>(m_shm_ptr) + HOTSPINE_HEADER_SIZE + trades_size);

  std::cout << "[HotSpineDataBridge] Connected to shared memory: " << m_shm_path
            << " (size=" << m_shm_size << " bytes, "
            << "trades_capacity=" << m_header->capacity << ", "
            << "books_capacity=" << m_header->orderbook_capacity << ")"
            << std::endl;

  m_running = true;
  return true;
}

void HotSpineDataBridge::stop() { m_running = false; }

void HotSpineDataBridge::poll() {
  static int poll_count = 0;
  if (poll_count++ % 100 == 0) {
    std::cout << "[HotSpineDataBridge] poll() call " << poll_count << std::endl;
  }

  if (!m_running || !m_header) {
    std::cout << "[HotSpineDataBridge] poll() - not running or no header"
              << std::endl;
    return;
  }

  // Always use real shared memory data
  poll_shm();

  if (poll_count % 100 == 1) {
    std::cout << "[HotSpineDataBridge] poll_shm() completed" << std::endl;
  }
}

std::shared_ptr<InstrumentStore>
HotSpineDataBridge::get_instrument(uint32_t symbol_id) {
  std::lock_guard<std::mutex> lock(m_map_mutex);

  // Search by ID first
  for (auto &[sym, inst] : m_instruments) {
    if (inst->symbol_id == symbol_id)
      return inst;
  }

  // Check if symbol is in registry - ONLY create if known
  auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
  if (!symbol_info) {
    // Unknown symbol - ignore it
    return nullptr;
  }

  // Create instrument for KNOWN symbol
  auto inst = std::make_shared<InstrumentStore>();
  inst->symbol = symbol_info->symbol;
  inst->exchange = symbol_info->exchange;
  inst->symbol_id = symbol_id;
  m_instruments[inst->symbol] = inst;

  std::cout << "[HotSpineDataBridge] Discovered: " << inst->symbol
            << " (ID=" << symbol_id << ", Exchange=" << inst->exchange << ")"
            << std::endl;

  return inst;
}

void HotSpineDataBridge::poll_shm() {
  if (!m_header)
    return;

  // --- Process Trades ---
  uint64_t write_idx =
      __atomic_load_n(&m_header->write_index, __ATOMIC_ACQUIRE);
  uint64_t capacity = m_header->capacity;

  uint64_t to_process =
      (write_idx > m_last_read_idx) ? (write_idx - m_last_read_idx) : 0;

  // CRITICAL: Limit trades per poll to avoid hanging
  const uint64_t MAX_PER_POLL = 100;
  uint64_t processed = 0;

  // Skip old history but keep recent data for initial chart population
  if (m_last_read_idx == 0 && to_process > 5000) {
    m_last_read_idx = write_idx - 5000; // Keep last 5000 trades
  }

  while (m_last_read_idx < write_idx && processed < MAX_PER_POLL) {
    processed++;
    // Cast raw bytes directly (No Mocks)
    const HotTrade &trade = m_trades[m_last_read_idx % capacity];

    auto inst = get_instrument(trade.symbol_id);
    if (!inst) {
      m_last_read_idx++;
      continue; // Skip unknown symbols
    }

    // Retrieve symbol_info for the trade's symbol_id for validation
    auto symbol_info =
        SymbolRegistry::instance().get_symbol_info(trade.symbol_id);
    if (!symbol_info) {
      m_last_read_idx++;
      continue; // Should not happen if get_instrument returned a valid inst
    }

    {
      std::lock_guard<std::mutex> lock(inst->data_mutex);
      double ts = (double)trade.ts_exchange / 1000000.0;

      // CRITICAL: Validate trade belongs to this instrument
      static int mismatch_count = 0;
      if (inst->symbol != symbol_info->symbol && mismatch_count++ < 5) {
        std::cout << "[ERROR] Symbol mismatch! InstrumentStore=" << inst->symbol
                  << " but trade is for " << symbol_info->symbol << std::endl;
      }

      // Use aggregator instead of naive approach
      inst->add_trade(ts, static_cast<float>(trade.price),
                      static_cast<float>(trade.size));

      // Volume Profile Accumulation (Price rounded to 0.5 tick)
      double tick_size = 0.5;
      double rounded_price = std::round(trade.price / tick_size) * tick_size;
      inst->m_vol_profile[rounded_price] += trade.size;
    }

    m_last_read_idx++;
  }

  // --- Process Orderbooks ---
  uint64_t book_write_idx =
      __atomic_load_n(&m_header->orderbook_write_index, __ATOMIC_ACQUIRE);
  uint64_t book_capacity = m_header->orderbook_capacity;

  uint64_t books_to_process = (book_write_idx > m_last_book_read_idx)
                                  ? (book_write_idx - m_last_book_read_idx)
                                  : 0;

  // Skip historical orderbooks on first run
  if (m_last_book_read_idx == 0 && books_to_process > 100) {
    m_last_book_read_idx = book_write_idx;
    return;
  }

  // Limit orderbooks per poll
  const uint64_t MAX_BOOKS_PER_POLL = 50;
  uint64_t books_processed = 0;

  while (m_last_book_read_idx < book_write_idx &&
         books_processed < MAX_BOOKS_PER_POLL) {
    books_processed++;
    const HotOrderbookSnapshot &snap =
        m_books[m_last_book_read_idx % book_capacity];

    auto inst = get_instrument(snap.symbol_id);
    if (!inst)
      continue; // Skip unknown symbols

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
