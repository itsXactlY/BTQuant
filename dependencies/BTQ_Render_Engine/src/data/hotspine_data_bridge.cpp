#include "../../include/hotspine_data_bridge.hpp"
#include "../../include/market_data_processor.hpp"
#include "../../include/symbol_registry.hpp"
#include <chrono>
#include <cstring> // For strerror
#include <expected>
#include <fcntl.h>
#include <iostream>
#include <print>
#include <pthread.h> // For thread priority
#include <sched.h>   // For real-time scheduling
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

std::expected<void, std::string> HotSpineDataBridge::start() {
  // Open shared memory
  m_shm_fd = shm_open(m_shm_path.c_str(), O_RDWR, 0666);
  if (m_shm_fd == -1) [[unlikely]] {
    return std::unexpected(std::format("Failed to open shared memory '{}': {}",
                                       m_shm_path, strerror(errno)));
  }

  // Get the size
  struct stat sb;
  if (fstat(m_shm_fd, &sb) == -1) [[unlikely]] {
    close(m_shm_fd);
    m_shm_fd = -1;
    return std::unexpected("Failed to fstat shared memory");
  }
  m_shm_size = sb.st_size;

  // Map it
  m_shm_ptr = mmap(nullptr, m_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   m_shm_fd, 0);
  if (m_shm_ptr == MAP_FAILED) [[unlikely]] {
    close(m_shm_fd);
    m_shm_fd = -1;
    return std::unexpected("Failed to mmap shared memory");
  }

  // Initialize pointers
  m_header = reinterpret_cast<SharedMemoryHeader *>(m_shm_ptr);

  // Verify magic number - "UQTB" = 0x42545155
  constexpr uint32_t expected_magic_le = 0x42545155;
  if (m_header->magic != expected_magic_le) [[unlikely]] {
    munmap(m_shm_ptr, m_shm_size);
    m_shm_ptr = nullptr;
    m_header = nullptr;
    close(m_shm_fd);
    m_shm_fd = -1;
    return std::unexpected("Invalid magic number in shared memory");
  }

  // Calculate ring buffer positions
  constexpr size_t HOTSPINE_HEADER_SIZE = 4096;
  m_trades = reinterpret_cast<HotTrade *>(static_cast<char *>(m_shm_ptr) +
                                          HOTSPINE_HEADER_SIZE);

  size_t trades_size = m_header->capacity * sizeof(HotTrade);
  m_books = reinterpret_cast<HotOrderbookSnapshot *>(
      static_cast<char *>(m_shm_ptr) + HOTSPINE_HEADER_SIZE + trades_size);

  std::println("[HotSpineDataBridge] Connected to SHM: {} (capacity={})",
               m_shm_path, m_header->capacity);

  m_running.store(true, std::memory_order_release);

  // Start real-time sync thread using std::jthread
  m_sync_thread =
      std::jthread([this](std::stop_token stoken) { this->sync_loop(); });

  return {};
}

void HotSpineDataBridge::stop() {
  m_running = false;
  if (m_sync_thread.joinable()) {
    m_sync_thread.request_stop();
    m_sync_thread.join();
  }
}

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
  return m_data_processor ? m_data_processor->getActiveSymbols()
                          : std::vector<uint32_t>{};
}

std::span<const HotTrade> HotSpineDataBridge::getTradeBuffer() const {
  if (!m_trades || !m_header)
    return {};
  return std::span<const HotTrade>(m_trades, m_header->capacity);
}

std::span<const HotOrderbookSnapshot>
HotSpineDataBridge::getBookBuffer() const {
  if (!m_books || !m_header)
    return {};
  return std::span<const HotOrderbookSnapshot>(m_books,
                                               m_header->orderbook_capacity);
}

void HotSpineDataBridge::sync_loop() {
  // Set real-time scheduling priority for minimal latency
  sched_param param;
  param.sched_priority =
      sched_get_priority_max(SCHED_FIFO) - 10; // High priority but not max
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
    std::println(
        stderr,
        "[HotSpineDataBridge] WARNING: Failed to set real-time priority");
  }

  // Lock memory to prevent paging
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    std::println(stderr, "[HotSpineDataBridge] WARNING: Failed to lock memory");
  }

  const auto sync_interval =
      std::chrono::microseconds(100); // 10kHz sync rate for ultra-low latency
  auto next_sync = std::chrono::steady_clock::now() + sync_interval;

  while (!m_sync_thread.get_stop_token().stop_requested() && m_running) {
    sync_shm();

    // Busy-wait with yield for precise timing
    auto now = std::chrono::steady_clock::now();
    if (now < next_sync) {
      std::this_thread::sleep_until(next_sync);
    }
    next_sync += sync_interval;
  }
}

void HotSpineDataBridge::sync_shm() {
  if (!m_header || !m_data_processor) {
    static int warn_count = 0;
    if (warn_count++ % 100 == 0) {
      std::println(stderr,
                   "[sync_shm] WARNING: m_header={} m_data_processor={}",
                   (void *)m_header, (void *)m_data_processor.get());
    }
    return;
  }

  // --- Process Trades ---
  uint64_t write_idx =
      __atomic_load_n(&m_header->write_index, __ATOMIC_ACQUIRE);
  uint64_t capacity = m_header->capacity;

  // Initial catch-up: Process ENTIRE ring buffer on first sync
  // This ensures we have all available historical data
  uint64_t last_read = m_last_read_idx.load(std::memory_order_acquire);
  if (last_read == 0 && write_idx > 0) {
    // For ring buffer: start at oldest valid position
    if (write_idx > capacity) {
      last_read = write_idx - capacity; // Buffer wrapped, start at oldest
    } else {
      last_read = 0; // Buffer not full, process from beginning
    }
    m_last_read_idx.store(last_read, std::memory_order_release);
    std::println("[sync_shm] Processing full ring buffer. Start: {} End: {}",
                 last_read, write_idx);
  }

  // Periodic debug: Show sync progress every 5 seconds
  static uint64_t last_debug_time = 0;
  uint64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  if (now - last_debug_time >= 5) {
    std::println(
        "[sync_shm] Trade W={} R={} Book W={} R={}", write_idx,
        m_last_read_idx.load(),
        __atomic_load_n(&m_header->orderbook_write_index, __ATOMIC_ACQUIRE),
        m_last_book_read_idx.load());
    last_debug_time = now;
  }

  // Batch processing
  std::vector<RenderEngine::MarketDataUpdate> trade_batch;
  const uint64_t BATCH_SIZE = 100000;
  trade_batch.reserve(BATCH_SIZE);

  // Timestamp reasonable filter: Jan 1st 2024 = 1704067200 sec -> 1.704e15
  // micros
  const uint64_t MIN_VALID_TS = 1704067200000000ULL;

  while (last_read < write_idx) {
    const HotTrade &trade = m_trades[last_read % capacity];

    // SANITY CHECK: Skip uninitialized or corrupt trades
    if (trade.ts_exchange < MIN_VALID_TS) {
      last_read++;
      continue;
    }

    // Validate trade data
    if (trade.price <= 0 || trade.size <= 0) {
      std::println(stderr,
                   "[HotSpineDataBridge] WARNING: Invalid trade data - price: "
                   "{}, size: {}",
                   trade.price, trade.size);
      last_read++;
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
    last_read++;

    if (trade_batch.size() >= BATCH_SIZE) {
      m_data_processor->processTradeUpdates(trade_batch);
      static uint64_t batch_count = 0;
      if (++batch_count % 10 == 0) {
        std::println("[HotSpineDataBridge] Processed batch of {} trades. Last "
                     "Read Index: {}",
                     trade_batch.size(), last_read);
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
  __atomic_store_n(&m_header->read_index, last_read, __ATOMIC_RELEASE);
  m_last_read_idx.store(last_read, std::memory_order_release);

  // --- Process Orderbooks ---
  uint64_t book_write_idx =
      __atomic_load_n(&m_header->orderbook_write_index, __ATOMIC_ACQUIRE);
  uint64_t book_capacity = m_header->orderbook_capacity;

  // Catch-up logic for books: Process ALL available history on first run
  uint64_t last_book_read =
      m_last_book_read_idx.load(std::memory_order_acquire);
  if (last_book_read == 0 && book_write_idx > 0) {
    if (book_write_idx > book_capacity) {
      last_book_read = book_write_idx - book_capacity;
    } else {
      last_book_read = 0;
    }
    m_last_book_read_idx.store(last_book_read, std::memory_order_release);
    std::println("[sync_shm] Orderbook Full Sync from {} to {}", last_book_read,
                 book_write_idx);
  }

  // CRITICAL FIX: Detect ring buffer wraparound
  // If our read pointer is ahead of the write pointer, the buffer has wrapped
  if (last_book_read > book_write_idx) {
    // Reset read pointer to catch up with the wrapped write pointer
    if (book_write_idx > book_capacity / 4) {
      last_book_read = book_write_idx - (book_capacity / 4);
    } else {
      last_book_read = 0;
    }
    m_last_book_read_idx.store(last_book_read, std::memory_order_release);
    std::println("[sync_shm] Book buffer wraparound detected. Reset R={} W={}",
                 last_book_read, book_write_idx);
  }

  // Process ALL available orderbooks in the buffer
  while (last_book_read < book_write_idx) {
    const HotOrderbookSnapshot &snap = m_books[last_book_read % book_capacity];

    // LOG snapshots occasionally to verify data is arriving
    static uint64_t snap_processed = 0;
    if (snap_processed++ % 5000 == 0) {
      std::println("[OrderbookBridge] Sync: Sym={} Bids={} Asks={}",
                   snap.symbol_id, (int)snap.bids_count, (int)snap.asks_count);
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
      last_book_read++;
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
    last_book_read++;
  }

  // Update Reader Index
  __atomic_store_n(&m_header->read_index, last_read, __ATOMIC_RELEASE);
  __atomic_store_n(&m_header->orderbook_read_index, last_book_read,
                   __ATOMIC_RELEASE);
  m_last_book_read_idx.store(last_book_read, std::memory_order_release);
}

} // namespace BTQuant
