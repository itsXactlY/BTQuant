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

#include "../../include/market_data_processor.hpp"
#include "../../include/structured_logger.hpp"
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
  auto result = connect();
  if (!result.has_value()) {
    return result;
  }

  running_.store(true, std::memory_order_release);

  // Start real-time sync thread using std::jthread
  sync_thread_ = std::jthread([this](std::stop_token /*stoken*/) { this->sync_loop(); });

  BTQ_LOG_INFO("HotSpineDataBridge started successfully");
  return {};
}

std::expected<void, std::string> HotSpineDataBridge::connect() {
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

  // Verify magic number and version in HotSpineLayoutV3
  // Magic number should be "BTQ3" = 0x42545155
  constexpr uint32_t EXPECTED_MAGIC = 0x42545155;  // "BTQ3"
  constexpr uint32_t EXPECTED_VERSION = 2;         // Version 3

  if (header_->magic != EXPECTED_MAGIC) [[unlikely]] {
    std::string error_msg =
        std::format("Invalid magic number in shared memory. Expected: 0x{:X}, Got: 0x{:X}",
                    EXPECTED_MAGIC, header_->magic);
    BTQ_LOG_ERROR(error_msg);
    munmap(shm_ptr_, shm_size_);
    shm_ptr_ = nullptr;
    header_ = nullptr;
    close(shm_fd_);
    shm_fd_ = -1;
    return std::unexpected(error_msg);
  }

  if (header_->version != EXPECTED_VERSION) [[unlikely]] {
    std::string error_msg = std::format("Invalid version in shared memory. Expected: {}, Got: {}",
                                        EXPECTED_VERSION, header_->version);
    BTQ_LOG_ERROR(error_msg);
    munmap(shm_ptr_, shm_size_);
    shm_ptr_ = nullptr;
    header_ = nullptr;
    close(shm_fd_);
    shm_fd_ = -1;
    return std::unexpected(error_msg);
  }

  // The ring buffer data (HotTrade entries) starts right after the header
  char* buffer_start = reinterpret_cast<char*>(header_) + sizeof(SharedMemoryHeader);

  // The SHM layout is: RingBufferHeader + HotTrade[RING_BUFFER_SIZE]
  // No orderbook data is stored in SHM — orderbook panels use other data sources
  size_t trade_capacity = HotSpine::V3::RING_BUFFER_SIZE;

  trades_ = reinterpret_cast<HotTrade*>(buffer_start);
  books_ = nullptr;  // No orderbook data in SHM

  std::string success_msg = std::format(
      "[HotSpineDataBridge] Connected to SHM: {} (magic=0x{:X}, version={}, trade_capacity={})",
      shm_path_, header_->magic, header_->version, trade_capacity);
  BTQ_LOG_INFO(success_msg);

  BTQ_LOG_INFO("HotSpineDataBridge connected successfully");
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
  // Return a span based on the available data in the ring buffer
  uint64_t available_count = header_->get_available_count();
  // Limit to a reasonable size to avoid returning huge spans
  size_t count = std::min(static_cast<size_t>(available_count),
                          static_cast<size_t>(HotSpine::V3::RING_BUFFER_SIZE));
  return std::span<const HotTrade>(trades_, count);
}

std::span<const HotOrderbookSnapshot> HotSpineDataBridge::getBookBuffer() const {
  if (!books_ || !header_) return {};
  // For now, return a span based on the ring buffer size
  // In a real implementation, orderbooks would be stored separately or interleaved
  return std::span<const HotOrderbookSnapshot>(books_, HotSpine::V3::RING_BUFFER_SIZE / 2);
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
      std::chrono::microseconds(1000);  // 1kHz sync rate — sub-ms latency, less CPU pressure
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

  // Get current write index from the new ring buffer header
  uint64_t current_write_idx = header_->write_head.load(std::memory_order_acquire);
  uint64_t capacity = HotSpine::V3::RING_BUFFER_SIZE;  // Use the new constant

  // Initial catch-up: Process ENTIRE ring buffer on first sync
  uint64_t last_read = last_read_idx_.load(std::memory_order_acquire);
  if (last_read == 0 && current_write_idx > 0) {
    // For ring buffer: start at oldest valid position
    if (current_write_idx > capacity) {
      last_read = current_write_idx - capacity;  // Buffer wrapped, start at oldest
    } else {
      last_read = 0;  // Buffer not full, process from beginning
    }
    last_read_idx_.store(last_read, std::memory_order_release);
    BTQ_LOG_INFO(std::format("Processing full ring buffer. Start: {} End: {}", last_read,
                             current_write_idx));
  }

  // Batch processing - lock-free approach
  // Use a pre-allocated vector to avoid dynamic allocation during processing
  thread_local static std::vector<RenderEngine::MarketDataUpdate> trade_batch;
  trade_batch.clear(); // Clear instead of creating new vector each time
  const uint64_t BATCH_SIZE = 100000;
  trade_batch.reserve(BATCH_SIZE);

  // Timestamp filter: Relaxed to allow replay/simulation data (since epoch)
  const uint64_t MIN_VALID_TS = 1000ULL;

  while (last_read < current_write_idx) {
    // Use the new ring buffer mask for indexing
    const HotTrade& trade = trades_[(last_read & HotSpine::V3::RING_BUFFER_MASK)];

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
      // Send batch to processor - this uses lock-free queue internally
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

  // Update the read tail using the new atomic approach
  header_->read_tail.store(last_read, std::memory_order_release);
  last_read_idx_.store(last_read, std::memory_order_release);
}

}  // namespace BTQuant
