#include "../../include/hotspine_data_bridge.hpp"

#include <fcntl.h>
#include <pthread.h>  // For thread priority
#include <sched.h>    // For real-time scheduling
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <iostream>
#include <print>

#include "../../include/structured_logger.hpp"
#include "../../include/market_data_processor.hpp"
#include "../../include/symbol_registry.hpp"

namespace BTQuant {

HotSpineDataBridge::HotSpineDataBridge(const std::string& shm_path) : shm_path_(shm_path) {
  // Load symbol registry from shared memory file
  SymbolRegistry::instance().load_from_file("/dev/shm/btquant_symbols.json");
}

HotSpineDataBridge::~HotSpineDataBridge() {
  stop();
  if (shm_ptr_ && shm_ptr_ != MAP_FAILED) {
    munmap(shm_ptr_, shm_size_);
  }
  if (shm_fd_ != -1) {
    close(shm_fd_);
  }
}

std::expected<void, std::string> HotSpineDataBridge::start() {
  // Open shared memory
  shm_fd_ = shm_open(shm_path_.c_str(), O_RDWR, 0666);
  if (shm_fd_ == -1) [[unlikely]] {
    std::string error_msg =
        std::format("Failed to open shared memory '{}': {}", shm_path_, strerror(errno));
    BTQ_LOG_ERROR_EX(error_msg, "shm_path", shm_path_, "errno", errno);
    return std::unexpected(error_msg);
  }

  // Get the size
  struct stat sb;
  if (fstat(shm_fd_, &sb) == -1) [[unlikely]] {
    std::string error_msg = "Failed to fstat shared memory";
    BTQ_LOG_ERROR_EX(error_msg, "shm_path", shm_path_, "errno", errno);
    close(shm_fd_);
    shm_fd_ = -1;
    return std::unexpected(error_msg);
  }
  shm_size_ = sb.st_size;

  // Map it
  shm_ptr_ = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  if (shm_ptr_ == MAP_FAILED) [[unlikely]] {
    std::string error_msg = "Failed to mmap shared memory";
    BTQ_LOG_ERROR(error_msg);
    close(shm_fd_);
    shm_fd_ = -1;
    return std::unexpected(error_msg);
  }

  // Initialize pointers
  header_ = reinterpret_cast<SharedMemoryHeader*>(shm_ptr_);

  // Verify magic number - "UQTB" = 0x42545155
  constexpr uint32_t expected_magic_le = 0x42545155;
  if (header_->magic != expected_magic_le) [[unlikely]] {
    std::string error_msg = "Invalid magic number in shared memory";
    BTQ_LOG_ERROR(error_msg);
    munmap(shm_ptr_, shm_size_);
    shm_ptr_ = nullptr;
    header_ = nullptr;
    close(shm_fd_);
    shm_fd_ = -1;
    return std::unexpected(error_msg);
  }

  // Calculate ring buffer positions
  constexpr size_t HOTSPINE_HEADER_SIZE = 4096;
  trades_ = reinterpret_cast<HotTrade*>(static_cast<char*>(shm_ptr_) + HOTSPINE_HEADER_SIZE);

  size_t trades_size = header_->capacity * sizeof(HotTrade);
  books_ = reinterpret_cast<HotOrderbookSnapshot*>(static_cast<char*>(shm_ptr_) +
                                                    HOTSPINE_HEADER_SIZE + trades_size);

  std::string success_msg = std::format("[HotSpineDataBridge] Connected to SHM: {} (capacity={})",
                                        shm_path_, header_->capacity);
  BTQ_LOG_INFO(success_msg);

  running_.store(true, std::memory_order_release);

  // Start real-time sync thread using std::jthread
  sync_thread_ = std::jthread([this](std::stop_token /*stoken*/) { this->sync_loop(); });

  BTQ_LOG_INFO("HotSpineDataBridge started successfully");
  return {};
}

void HotSpineDataBridge::stop() {
  running_ = false;
  if (sync_thread_.joinable()) {
    sync_thread_.request_stop();
    sync_thread_.join();
  }
}

void HotSpineDataBridge::sync() {
  if (!running_) {
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
  return data_processor_ ? data_processor_->getActiveSymbols() : std::vector<uint32_t>{};
}

std::span<const HotTrade> HotSpineDataBridge::getTradeBuffer() const {
  if (!trades_ || !header_) return {};
  return std::span<const HotTrade>(trades_, header_->capacity);
}

std::span<const HotOrderbookSnapshot> HotSpineDataBridge::getBookBuffer() const {
  if (!books_ || !header_) return {};
  return std::span<const HotOrderbookSnapshot>(books_, header_->orderbook_capacity);
}

void HotSpineDataBridge::sync_loop() {
  // Set real-time scheduling priority for minimal latency
  sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 10;  // High priority but not max
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
    BTQ_LOG_WARNING("[HotSpineDataBridge] Failed to set real-time priority");
  }

  // Lock memory to prevent paging
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    BTQ_LOG_WARNING("[HotSpineDataBridge] Failed to lock memory");
  }

  const auto sync_interval =
      std::chrono::microseconds(100);  // 10kHz sync rate for ultra-low latency
  auto next_sync = std::chrono::steady_clock::now() + sync_interval;

  BTQ_LOG_INFO("HotSpineDataBridge sync loop started");

  while (!sync_thread_.get_stop_token().stop_requested() && running_) {
    sync_shm();

    // Busy-wait with yield for precise timing
    auto now = std::chrono::steady_clock::now();
    if (now < next_sync) {
      std::this_thread::sleep_until(next_sync);
    }
    next_sync += sync_interval;
  }

  BTQ_LOG_INFO("HotSpineDataBridge sync loop stopped");
}

void HotSpineDataBridge::sync_shm() {
  if (!header_ || !data_processor_) {
    static int warn_count = 0;
    if (warn_count++ % 100 == 0) {
      BTQ_LOG_WARNING(std::format("header_={} data_processor_={}", (void*)header_,
                                  (void*)data_processor_.get()));
    }
    return;
  }

  // --- Process Trades ---
  uint64_t write_idx = __atomic_load_n(&header_->write_index, __ATOMIC_ACQUIRE);
  uint64_t capacity = header_->capacity;

  // Initial catch-up: Process ENTIRE ring buffer on first sync
  // This ensures we have all available historical data
  uint64_t last_read = last_read_idx_.load(std::memory_order_acquire);
  if (last_read == 0 && write_idx > 0) {
    // For ring buffer: start at oldest valid position
    if (write_idx > capacity) {
      last_read = write_idx - capacity;  // Buffer wrapped, start at oldest
    } else {
      last_read = 0;  // Buffer not full, process from beginning
    }
    last_read_idx_.store(last_read, std::memory_order_release);
    BTQ_LOG_INFO(
        std::format("Processing full ring buffer. Start: {} End: {}", last_read, write_idx));
  }

  // Periodic debug: Show sync progress every 5 seconds
  static uint64_t last_debug_time = 0;
  uint64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
  if (now - last_debug_time >= 5) {
    BTQ_LOG_INFO(std::format("Trade W={} R={} Book W={} R={}", write_idx, last_read_idx_.load(),
                             __atomic_load_n(&header_->orderbook_write_index, __ATOMIC_ACQUIRE),
                             last_book_read_idx_.load()));
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
    const HotTrade& trade = trades_[last_read % capacity];

    // SANITY CHECK: Skip uninitialized or corrupt trades
    if (trade.ts_exchange < MIN_VALID_TS) {
      last_read++;
      continue;
    }

    // Validate trade data
    if (trade.price <= 0 || trade.size <= 0) {
      BTQ_LOG_WARNING(
          std::format("Invalid trade data - price: {}, size: {}", trade.price, trade.size));
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
      data_processor_->processTradeUpdates(trade_batch);
      static uint64_t batch_count = 0;
      if (++batch_count % 10 == 0) {
        BTQ_LOG_INFO(std::format("Processed batch of {} trades. Last Read Index: {}",
                                 trade_batch.size(), last_read));
      }
      trade_batch.clear();
    }
  }

  // Finalize remaining trades in the batch
  if (!trade_batch.empty()) {
    data_processor_->processTradeUpdates(trade_batch);
  }

  // CRITICAL: Update shared memory read_index so producer knows we've consumed
  // it
  __atomic_store_n(&header_->read_index, last_read, __ATOMIC_RELEASE);
  last_read_idx_.store(last_read, std::memory_order_release);

  // --- Process Orderbooks ---
  uint64_t book_write_idx = __atomic_load_n(&header_->orderbook_write_index, __ATOMIC_ACQUIRE);
  uint64_t book_capacity = header_->orderbook_capacity;

  // Catch-up logic for books: Process ALL available history on first run
  uint64_t last_book_read = last_book_read_idx_.load(std::memory_order_acquire);
  if (last_book_read == 0 && book_write_idx > 0) {
    if (book_write_idx > book_capacity) {
      last_book_read = book_write_idx - book_capacity;
    } else {
      last_book_read = 0;
    }
    last_book_read_idx_.store(last_book_read, std::memory_order_release);
    BTQ_LOG_INFO(std::format("Orderbook Full Sync from {} to {}", last_book_read, book_write_idx));
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
    last_book_read_idx_.store(last_book_read, std::memory_order_release);
    BTQ_LOG_INFO(std::format("Book buffer wraparound detected. Reset R={} W={}", last_book_read,
                             book_write_idx));
  }

  // Process ALL available orderbooks in the buffer
  while (last_book_read < book_write_idx) {
    const HotOrderbookSnapshot& snap = books_[last_book_read % book_capacity];

    // LOG snapshots occasionally to verify data is arriving
    static uint64_t snap_processed = 0;
    if (snap_processed++ % 5000 == 0) {
      BTQ_LOG_INFO(std::format("Sync: Sym={} Bids={} Asks={}", snap.symbol_id, (int)snap.bids_count,
                               (int)snap.asks_count));
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
    int safe_bids_count = std::min((int)snap.bids_count, 200);
    for (int i = 0; i < safe_bids_count; ++i) {
      if (snap.bids[i].price <= 0 || snap.bids[i].size <= 0) continue;

      RenderEngine::PriceLevel level;
      level.price = snap.bids[i].price;
      level.size = snap.bids[i].size;
      update.bids.push_back(level);
    }

    int safe_asks_count = std::min((int)snap.asks_count, 200);
    for (int i = 0; i < safe_asks_count; ++i) {
      if (snap.asks[i].price <= 0 || snap.asks[i].size <= 0) continue;

      RenderEngine::PriceLevel level;
      level.price = snap.asks[i].price;
      level.size = snap.asks[i].size;
      update.asks.push_back(level);
    }

    data_processor_->processOrderbookUpdate(update);
    last_book_read++;
  }

  // Update Reader Index
  __atomic_store_n(&header_->read_index, last_read, __ATOMIC_RELEASE);
  __atomic_store_n(&header_->orderbook_read_index, last_book_read, __ATOMIC_RELEASE);
  last_book_read_idx_.store(last_book_read, std::memory_order_release);
}

}  // namespace BTQuant
