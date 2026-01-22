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
    throw std::runtime_error("Failed to open shared memory: " +
                             std::string(strerror(errno)));
  }

  // Get the size
  struct stat sb;
  if (fstat(m_shm_fd, &sb) == -1) {
    std::cerr << "[HotSpineDataBridge] ERROR: Failed to fstat shared memory"
              << std::endl;
    close(m_shm_fd);
    m_shm_fd = -1;
    throw std::runtime_error("Failed to fstat shared memory");
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
    throw std::runtime_error("Failed to mmap shared memory");
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
    throw std::runtime_error("Invalid magic number in shared memory");
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

  // Initial catch-up: Process ENTIRE ring buffer on first sync
  // This ensures we have all available historical data
  if (m_last_read_idx == 0 && write_idx > 0) {
    // For ring buffer: start at oldest valid position
    if (write_idx > capacity) {
      m_last_read_idx = write_idx - capacity; // Buffer wrapped, start at oldest
    } else {
      m_last_read_idx = 0; // Buffer not full, process from beginning
    }
    std::cout << "[sync_shm] Processing full ring buffer. Start: "
              << m_last_read_idx << " End: " << write_idx << std::endl;
  }

  // Periodic debug: Show sync progress every 5 seconds
  static uint64_t last_debug_time = 0;
  uint64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  if (now - last_debug_time >= 5) {
    std::cout << "[sync_shm] Trade W=" << write_idx << " R=" << m_last_read_idx
              << " Book W="
              << __atomic_load_n(&m_header->orderbook_write_index,
                                 __ATOMIC_ACQUIRE)
              << " R=" << m_last_book_read_idx << std::endl;
    last_debug_time = now;
  }

  // Batch processing
  std::vector<RenderEngine::MarketDataUpdate> trade_batch;
  const uint64_t BATCH_SIZE = 100000;
  trade_batch.reserve(BATCH_SIZE);

  // Timestamp reasonable filter: Jan 1st 2024 = 1704067200 sec -> 1.704e15
  // micros
  const uint64_t MIN_VALID_TS = 1704067200000000ULL;

  while (m_last_read_idx < write_idx) {
    const HotTrade &trade = m_trades[m_last_read_idx % capacity];

    // SANITY CHECK: Skip uninitialized or corrupt trades
    if (trade.ts_exchange < MIN_VALID_TS) {
      m_last_read_idx++;
      continue;
    }

    // Validate trade data
    if (trade.price <= 0 || trade.size <= 0) {
      std::cerr << "[HotSpineDataBridge] WARNING: Invalid trade data - price: "
                << trade.price << ", size: " << trade.size << std::endl;
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

  // Catch-up logic for books: Process ALL available history on first run
  if (m_last_book_read_idx == 0 && book_write_idx > 0) {
    if (book_write_idx > book_capacity) {
      m_last_book_read_idx = book_write_idx - book_capacity;
    } else {
      m_last_book_read_idx = 0;
    }
    std::cout << "[sync_shm] Orderbook Full Sync from " << m_last_book_read_idx
              << " to " << book_write_idx << std::endl;
  }

  // CRITICAL FIX: Detect ring buffer wraparound
  // If our read pointer is ahead of the write pointer, the buffer has wrapped
  if (m_last_book_read_idx > book_write_idx) {
    // Reset read pointer to catch up with the wrapped write pointer
    if (book_write_idx > book_capacity / 4) {
      m_last_book_read_idx = book_write_idx - (book_capacity / 4);
    } else {
      m_last_book_read_idx = 0;
    }
    std::cout << "[sync_shm] Book buffer wraparound detected. Reset R="
              << m_last_book_read_idx << " W=" << book_write_idx << std::endl;
  }

  // Process ALL available orderbooks in the buffer
  while (m_last_book_read_idx < book_write_idx) {
    const HotOrderbookSnapshot &snap =
        m_books[m_last_book_read_idx % book_capacity];

    // LOG snapshots occasionally to verify data is arriving
    static uint64_t snap_processed = 0;
    if (snap_processed++ % 5000 == 0) {
      std::cout << "[OrderbookBridge] Sync: Sym=" << snap.symbol_id
                << " Bids=" << (int)snap.bids_count
                << " Asks=" << (int)snap.asks_count << std::endl;
    }

    // Timestamp reasonable filter
    // Timestamp handling with fallback
    uint64_t final_timestamp = snap.ts_exchange;
    const uint64_t MIN_VALID_TS = 1704067200000000ULL;

    if (final_timestamp < MIN_VALID_TS) {
      // Fallback to system time
      final_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    }

    // Validate symbol ID (0 is often uninitialized in SHM)
    if (snap.symbol_id == 0) {
      m_last_book_read_idx++;
      continue;
    }

    // Direkte Weitergabe an MarketDataProcessor
    RenderEngine::MarketDataUpdate update;
    update.type = RenderEngine::MarketDataType::ORDERBOOK;
    update.symbol_id = snap.symbol_id;
    update.timestamp = final_timestamp;

    // Konvertiere Orderbook-Ebenen
    int safe_bids_count = std::min((int)snap.bids_count, 20);
    for (int i = 0; i < safe_bids_count; ++i) {
      if (snap.bids[i].price <= 0 || snap.bids[i].size <= 0)
        continue;

      PriceLevel level;
      level.price = snap.bids[i].price;
      level.size = snap.bids[i].size;
      update.bids.push_back(level);
    }

    int safe_asks_count = std::min((int)snap.asks_count, 20);
    for (int i = 0; i < safe_asks_count; ++i) {
      if (snap.asks[i].price <= 0 || snap.asks[i].size <= 0)
        continue;

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
