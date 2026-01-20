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

void HotSpineDataBridge::sync() {
  if (!m_running) {
    return;
  }

  // ONLY real shared memory - NO simulation
  sync_shm();
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

void HotSpineDataBridge::sync_shm() {
  if (!m_header || !m_data_processor) {
    static int warn_count = 0;
    if (warn_count++ % 100 == 0) {
      std::cout << "[sync_shm] WARNING: m_header=" << m_header
                << " m_data_processor=" << m_data_processor.get() << std::endl;
    }
    return;
  }

  // --- Process Trades ---
  uint64_t write_idx =
      __atomic_load_n(&m_header->write_index, __ATOMIC_ACQUIRE);
  uint64_t capacity = m_header->capacity;

  // Initial catch-up: If reader is starting fresh, it should process all
  // Catch-up logic for trades.
  // If we are starting from zero but there is already data in the buffer,
  // skip all but the most recent 10k trades to avoid ancient history pollution.
  if (m_last_read_idx == 0 && write_idx > 100000) {
    m_last_read_idx = write_idx - 100000;
    std::cout << "[sync_shm] Catching up to recent trades: " << m_last_read_idx
              << std::endl;
  }

  if (m_last_read_idx == 0 && write_idx > capacity) {
    // Shared memory is a ring buffer. If it has wrapped, start at the oldest
    // available entry.
    m_last_read_idx = write_idx - capacity;
    std::cout
        << "[sync_shm] Ring buffer wrapped. Starting from oldest available: "
        << m_last_read_idx << std::endl;
  }

  // Batch processing for maximum performance
  std::vector<RenderEngine::MarketDataUpdate> trade_batch;
  const uint64_t BATCH_SIZE = 100000;
  trade_batch.reserve(BATCH_SIZE);

  while (m_last_read_idx < write_idx) {
    const HotTrade &trade = m_trades[m_last_read_idx % capacity];

    // SANITY CHECK: Skip uninitialized or corrupt trades
    if (trade.ts_exchange == 0) {
      m_last_read_idx++;
      continue;
    }

    RenderEngine::MarketDataUpdate update;
    update.type = RenderEngine::MarketDataType::TRADE;
    update.symbol_id = trade.symbol_id;
    update.timestamp = trade.ts_exchange;
    update.price = trade.price;
    update.size = trade.size;
    update.side = (trade.side == 0) ? "buy" : "sell";

    trade_batch.push_back(std::move(update));
    m_last_read_idx++;

    if (trade_batch.size() >= BATCH_SIZE) {
      m_data_processor->processTradeUpdates(trade_batch);
      static uint64_t batch_count = 0;
      if (++batch_count % 10 == 0) {
        std::cout << "[HotSpineDataBridge] Processed batch of "
                  << trade_batch.size()
                  << " trades. Last Read Index: " << m_last_read_idx
                  << std::endl;
      }
      trade_batch.clear();
    }
  }

  // Finalize remaining trades in the batch
  if (!trade_batch.empty()) {
    m_data_processor->processTradeUpdates(trade_batch);
  }

  // CRITICAL: Update shared memory read_index so producer knows we've consumed
  // it
  __atomic_store_n(&m_header->read_index, m_last_read_idx, __ATOMIC_RELEASE);

  // --- Process Orderbooks ---
  uint64_t book_write_idx =
      __atomic_load_n(&m_header->orderbook_write_index, __ATOMIC_ACQUIRE);
  uint64_t book_capacity = m_header->orderbook_capacity;

  uint64_t books_to_process = (book_write_idx > m_last_book_read_idx)
                                  ? (book_write_idx - m_last_book_read_idx)
                                  : 0;

  // Catch-up logic for books
  if (m_last_book_read_idx == 0 && books_to_process > 100) {
    m_last_book_read_idx = book_write_idx - 20;
    std::cout << "[sync_shm] Book catch-up to " << m_last_book_read_idx
              << std::endl;
  }

  if (books_to_process > 500) {
    m_last_book_read_idx = book_write_idx - 50;
  }

  // Process ALL available orderbooks in the buffer
  while (m_last_book_read_idx < book_write_idx) {
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
