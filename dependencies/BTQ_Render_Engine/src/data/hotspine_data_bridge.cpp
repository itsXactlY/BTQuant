#include "../../include/hotspine_data_bridge.hpp"
#include "../../include/market_data_processor.hpp"
#include "../../include/symbol_registry.hpp"
#include <chrono>
#include <cstring>
#include <expected>
#include <fcntl.h>
#include <format>
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

  // Verify version number - Must match HOTSPINE_VERSION
  // Assuming HOTSPINE_VERSION is 3 now
  if (m_header->version != 3) [[unlikely]] {
    munmap(m_shm_ptr, m_shm_size);
    m_shm_ptr = nullptr;
    m_header = nullptr;
    close(m_shm_fd);
    m_shm_fd = -1;
    return std::unexpected(
        std::format("Shared memory version mismatch: expected 3, found {}",
                    m_header->version));
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
    if (warn_count++ % 1000 == 0) {
      std::println(stderr,
                   "[sync_shm] WARNING: m_header={} m_data_processor={}",
                   (void *)m_header, (void *)m_data_processor.get());
    }
    return;
  }

  // --- Adaptive Validation Logic ---
  uint64_t trade_write_idx =
      __atomic_load_n(&m_header->write_index, __ATOMIC_ACQUIRE);
  uint64_t trade_read_idx = m_last_read_idx.load(std::memory_order_relaxed);
  uint64_t trade_lag = (trade_write_idx > trade_read_idx)
                           ? (trade_write_idx - trade_read_idx)
                           : 0;

  ValidationLevel current_level = ValidationLevel::FULL;
  if (trade_lag > 5000) {
    current_level = ValidationLevel::MINIMAL;
  } else if (trade_lag > 1000) {
    current_level = ValidationLevel::ADAPTIVE;
  }
  m_validation_level.store(current_level, std::memory_order_relaxed);

  // Measure start time for overhead tracking
  auto start_time = std::chrono::high_resolution_clock::now();

  // --- Process Trades ---
  uint64_t capacity = m_header->capacity;

  // Initial catch-up logic
  if (trade_read_idx == 0 && trade_write_idx > 0) {
    if (trade_write_idx > capacity) {
      trade_read_idx = trade_write_idx - capacity;
    } else {
      trade_read_idx = 0;
    }
    m_last_read_idx.store(trade_read_idx, std::memory_order_release);
    std::println("[sync_shm] Trade Catch-up: Start={} End={} Lag={}",
                 trade_read_idx, trade_write_idx, trade_lag);
  }

  // Debug logging
  static uint64_t last_debug_time = 0;
  uint64_t now_sec = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  if (now_sec - last_debug_time >= 10) {
    std::string level_str = "FULL";
    if (current_level == ValidationLevel::ADAPTIVE)
      level_str = "ADAPTIVE";
    if (current_level == ValidationLevel::MINIMAL)
      level_str = "MINIMAL";

    std::println("[sync_shm] Trade W={} R={} Lag={} Level={}", trade_write_idx,
                 trade_read_idx, trade_lag, level_str);
    last_debug_time = now_sec;
  }

  std::vector<RenderEngine::MarketDataUpdate> trade_batch;
  const uint64_t BATCH_SIZE = 100000;
  trade_batch.reserve(BATCH_SIZE);

  constexpr uint64_t MIN_VALID_TS = 1704067200000000ULL;

  // Optimized Loop
  while (trade_read_idx < trade_write_idx) {
    const HotTrade &trade = m_trades[trade_read_idx % capacity];
    bool valid = true;

    // Validation strategy based on level
    if (current_level == ValidationLevel::FULL) {
      if (trade.ts_exchange < MIN_VALID_TS || trade.price <= 0 ||
          trade.size <= 0) {
        valid = false;
      }
    } else if (current_level == ValidationLevel::ADAPTIVE) {
      // Skip price/size checks, only check timestamp
      if (trade.ts_exchange < MIN_VALID_TS) {
        valid = false;
      }
    }
    // MINIMAL: Assume valid (no checks)

    if (valid) {
      RenderEngine::MarketDataUpdate update;
      update.type = RenderEngine::MarketDataType::TRADE;
      update.symbol_id = trade.symbol_id;
      update.timestamp = trade.ts_exchange;
      update.price = trade.price;
      update.size = trade.size;
      update.side = (trade.side == 0) ? "buy" : "sell";
      trade_batch.push_back(std::move(update));
    }

    trade_read_idx++;

    if (trade_batch.size() >= BATCH_SIZE) {
      m_data_processor->processTradeUpdates(trade_batch);
      trade_batch.clear();
      // Update read index periodically during large batches to allow recovery
      m_last_read_idx.store(trade_read_idx, std::memory_order_relaxed);
    }
  }

  if (!trade_batch.empty()) {
    m_data_processor->processTradeUpdates(trade_batch);
  }

  __atomic_store_n(&m_header->read_index, trade_read_idx, __ATOMIC_RELEASE);
  m_last_read_idx.store(trade_read_idx, std::memory_order_release);

  // --- Process Orderbooks ---
  uint64_t book_write_idx =
      __atomic_load_n(&m_header->orderbook_write_index, __ATOMIC_ACQUIRE);
  uint64_t book_capacity = m_header->orderbook_capacity;
  uint64_t book_read_idx = m_last_book_read_idx.load(std::memory_order_relaxed);

  // Catch-up logic for books
  if (book_read_idx == 0 && book_write_idx > 0) {
    if (book_write_idx > book_capacity) {
      book_read_idx = book_write_idx - book_capacity;
    } else {
      book_read_idx = 0;
    }
    std::println("[sync_shm] Book Catch-up: Start={} End={}", book_read_idx,
                 book_write_idx);
  }

  // Wraparound check
  if (book_read_idx > book_write_idx) {
    book_read_idx = (book_write_idx > book_capacity / 4)
                        ? (book_write_idx - book_capacity / 4)
                        : 0;
    std::println("[sync_shm] Book Wraparound: Reset to {}", book_read_idx);
  }

  while (book_read_idx < book_write_idx) {
    const HotOrderbookSnapshot &snap = m_books[book_read_idx % book_capacity];
    bool skip_snapshot = false;

    // Fast-fail on symbol ID
    if (snap.symbol_id == 0) {
      book_read_idx++;
      continue;
    }

    // Adaptive timestamp check (only in FULL/ADAPTIVE)
    uint64_t final_timestamp = snap.ts_exchange;
    if (current_level != ValidationLevel::MINIMAL &&
        final_timestamp < MIN_VALID_TS) {
      final_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    }

    RenderEngine::MarketDataUpdate update;
    update.type = RenderEngine::MarketDataType::ORDERBOOK;
    update.symbol_id = snap.symbol_id;
    update.timestamp = final_timestamp;

    // Optimized Level Processing
    int safe_bids = std::min((int)snap.bids_count, 200);
    int safe_asks = std::min((int)snap.asks_count, 200);

    // Reserve to avoid reallocations
    update.bids.reserve(safe_bids);
    update.asks.reserve(safe_asks);

    if (current_level == ValidationLevel::FULL) {
      for (int i = 0; i < safe_bids; ++i) {
        if (snap.bids[i].price > 0 && snap.bids[i].size > 0) {
          update.bids.push_back({snap.bids[i].price, snap.bids[i].size});
        }
      }
      for (int i = 0; i < safe_asks; ++i) {
        if (snap.asks[i].price > 0 && snap.asks[i].size > 0) {
          update.asks.push_back({snap.asks[i].price, snap.asks[i].size});
        }
      }
    } else {
      // ADAPTIVE & MINIMAL: Trust the producer, skip per-level checks for speed
      // This effectively vectorizes better as we just copy
      for (int i = 0; i < safe_bids; ++i) {
        update.bids.push_back({snap.bids[i].price, snap.bids[i].size});
      }
      for (int i = 0; i < safe_asks; ++i) {
        update.asks.push_back({snap.asks[i].price, snap.asks[i].size});
      }
    }

    m_data_processor->processOrderbookUpdate(update);
    book_read_idx++;
  }

  __atomic_store_n(&m_header->orderbook_read_index, book_read_idx,
                   __ATOMIC_RELEASE);
  m_last_book_read_idx.store(book_read_idx, std::memory_order_release);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();
  m_validation_overhead_us.store(duration, std::memory_order_relaxed);
}

} // namespace BTQuant