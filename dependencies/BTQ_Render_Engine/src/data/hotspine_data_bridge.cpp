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

  if (!m_running) {
    std::cout << "[HotSpineDataBridge] poll() - not running" << std::endl;
    return;
  }

  // ONLY real shared memory - NO simulation
  poll_shm();

  if (poll_count % 100 == 1) {
    std::cout << "[HotSpineDataBridge] poll completed" << std::endl;
  }
}

std::string HotSpineDataBridge::getSymbolName(uint32_t symbol_id) const {
  auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
  return symbol_info ? symbol_info->symbol : "UNKNOWN";
}

std::string HotSpineDataBridge::getExchangeName(uint32_t symbol_id) const {
  auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
  return symbol_info ? symbol_info->exchange : "UNKNOWN";
}

std::vector<uint32_t> HotSpineDataBridge::getActiveSymbols() const {
  if (!m_data_processor) {
    return {};
  }
  return m_data_processor->getActiveSymbols();
}

void HotSpineDataBridge::poll_shm() {
  if (!m_header || !m_data_processor) {
    static int warn_count = 0;
    if (warn_count++ % 100 == 0) {
      std::cout << "[poll_shm] WARNING: m_header=" << m_header
                << " m_data_processor=" << m_data_processor.get() << std::endl;
    }
    return;
  }

  // --- Process Trades ---
  uint64_t write_idx =
      __atomic_load_n(&m_header->write_index, __ATOMIC_ACQUIRE);
  uint64_t capacity = m_header->capacity;

  uint64_t to_process =
      (write_idx > m_last_read_idx) ? (write_idx - m_last_read_idx) : 0;

  // DEBUG: Show trade processing stats
  static int debug_count = 0;
  if (debug_count++ % 100 == 0) {
    std::cout << "[poll_shm] write_idx=" << write_idx
              << " last_read=" << m_last_read_idx
              << " to_process=" << to_process << std::endl;
  }

  // CRITICAL: Limit trades per poll to avoid hanging
  const uint64_t MAX_PER_POLL = 1000; // Increased from 100!
  uint64_t processed = 0;

  // Skip old history but keep recent data for initial chart population
  if (m_last_read_idx == 0 && to_process > 5000) {
    m_last_read_idx = write_idx - 5000; // Keep last 5000 trades
  }

  while (m_last_read_idx < write_idx && processed < MAX_PER_POLL) {

    processed++;
    const HotTrade &trade = m_trades[m_last_read_idx % capacity];

    // Direkte Weitergabe an MarketDataProcessor
    RenderEngine::MarketDataUpdate update;
    update.type = RenderEngine::MarketDataType::TRADE;
    update.symbol_id = trade.symbol_id;
    update.timestamp = trade.ts_exchange;
    update.price = trade.price;
    update.size = trade.size;
    update.side = (trade.side == 0) ? "buy" : "sell";

    m_data_processor->processTradeUpdate(update);
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

    // Direkte Weitergabe an MarketDataProcessor
    RenderEngine::MarketDataUpdate update;
    update.type = RenderEngine::MarketDataType::ORDERBOOK;
    update.symbol_id = snap.symbol_id;
    update.timestamp = snap.ts_exchange;

    // Konvertiere Orderbook-Ebenen
    for (int i = 0; i < snap.bids_count; ++i) {
      PriceLevel level;
      level.price = snap.bids[i].price;
      level.size = snap.bids[i].size;
      update.bids.push_back(level);
    }

    for (int i = 0; i < snap.asks_count; ++i) {
      PriceLevel level;
      level.price = snap.asks[i].price;
      level.size = snap.asks[i].size;
      update.asks.push_back(level);
    }

    m_data_processor->processOrderbookUpdate(update);
    m_last_book_read_idx++;
  }

  // Update Reader Index
  __atomic_store_n(&m_header->read_index, m_last_read_idx, __ATOMIC_RELEASE);
  __atomic_store_n(&m_header->orderbook_read_index, m_last_book_read_idx,
                   __ATOMIC_RELEASE);
}

} // namespace BTQuant
