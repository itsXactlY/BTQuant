#include "hotspine_bridge.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>

#include "data/core_types.hpp"
#include "market_data_processor.hpp"

// ============================================================================
// V2 Shared Memory Structs — inlined to avoid header conflicts
// Must match Python ctypes structure exactly
// ============================================================================
namespace {

struct HotTrade {
  uint64_t ts_exchange;
  uint64_t ts_local;
  double price;
  double size;
  uint32_t symbol_id;
  uint8_t side;  // 0=Buy, 1=Sell
  uint8_t padding[3];
};
static_assert(sizeof(HotTrade) == 40, "HotTrade must be 40 bytes");

struct __attribute__((packed)) ShmHeader {
  uint32_t magic;    // "BTQU"
  uint32_t version;  // 2
  uint64_t capacity;
  uint64_t write_index;
  uint64_t read_index;
  uint64_t lost_count;
  uint64_t orderbook_write_index;
  uint64_t orderbook_read_index;
  uint64_t orderbook_lost_count;
  uint64_t orderbook_capacity;
  uint8_t padding[24];  // Total: 4+4 + 8*8 + 24 = 96
};
static_assert(sizeof(ShmHeader) == 96, "ShmHeader must be 96 bytes");

}  // anonymous namespace

namespace BTQuant {

// ============================================================================
// Lifecycle
// ============================================================================

HotspineBridge::~HotspineBridge() { stop(); }

bool HotspineBridge::start(RenderEngine::MarketDataProcessor* processor,
                           const std::string& shm_path, const std::string& symbols_path) {
  if (running_.load()) return false;
  if (!processor) return false;
  processor_ = processor;

  // ---- Load symbol mappings ----
  {
    std::ifstream ifs(symbols_path);
    if (!ifs.is_open()) {
      std::cerr << "[HotspineBridge] Cannot open " << symbols_path << "\n";
      return false;
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    uint32_t internal_idx = 0;
    size_t pos = 0;
    while ((pos = content.find("\"id\":", pos)) != std::string::npos) {
      pos += 5;
      while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) ++pos;
      uint32_t ext_id = 0;
      while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9') {
        ext_id = ext_id * 10 + (content[pos] - '0');
        ++pos;
      }
      if (internal_idx < MAX_REMAP && ext_id > 0) {
        remap_[internal_idx] = ext_id;
        external_to_internal_[ext_id] = internal_idx;
        std::cerr << "[HotspineBridge] Remap: external " << ext_id << " -> internal "
                  << internal_idx << "\n";
        ++internal_idx;
      }
    }
    std::cerr << "[HotspineBridge] Loaded " << internal_idx << " symbol mappings\n";
  }

  // ---- mmap shared memory ----
  int fd = ::open(shm_path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "[HotspineBridge] Cannot open " << shm_path << ": " << strerror(errno) << "\n";
    return false;
  }

  struct stat st;
  if (::fstat(fd, &st) < 0) {
    ::close(fd);
    return false;
  }
  mapped_size_ = static_cast<size_t>(st.st_size);

  mapped_ = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);

  if (mapped_ == MAP_FAILED) {
    std::cerr << "[HotspineBridge] mmap failed: " << strerror(errno) << "\n";
    mapped_ = nullptr;
    return false;
  }

  // Verify magic — raw uint32 comparison (BTQU = 0x42545155 on this platform)
  auto* header = static_cast<const ShmHeader*>(mapped_);
  constexpr uint32_t BTQU_MAGIC = 0x42545155;
  if (header->magic != BTQU_MAGIC) {
    std::cerr << "[HotspineBridge] Bad magic: 0x" << std::hex << header->magic << std::dec
              << " (expected 0x" << std::hex << BTQU_MAGIC << std::dec << ")\n";
    ::munmap(mapped_, mapped_size_);
    mapped_ = nullptr;
    return false;
  }

  uint64_t capacity = header->capacity;
  std::cerr << "[HotspineBridge] Attached to SHM (" << (mapped_size_ / 1024 / 1024)
            << " MB), trade capacity=" << capacity << ", version=" << header->version << "\n";

  // Start from current write position (don't replay history)
  // Can't take address of packed member, so use memcpy
  uint64_t initial_write;
  std::memcpy(&initial_write, &header->write_index, sizeof(uint64_t));
  last_head_ = initial_write;
  std::cerr << "[HotspineBridge] Starting from write_index=" << last_head_ << "\n";

  // ---- Start poll thread ----
  running_.store(true, std::memory_order_release);
  poll_thread_ = std::thread(&HotspineBridge::poll_loop, this);

  return true;
}

void HotspineBridge::stop() {
  running_.store(false, std::memory_order_release);
  if (poll_thread_.joinable()) poll_thread_.join();

  if (mapped_) {
    ::munmap(mapped_, mapped_size_);
    mapped_ = nullptr;
  }
}

// ============================================================================
// Poll Loop — Background Thread
// Layout: [ShmHeader][HotTrade * capacity][OB data...]
// ============================================================================

void HotspineBridge::poll_loop() {
  auto* header = static_cast<const ShmHeader*>(mapped_);
  uint64_t capacity = header->capacity;
  if (capacity == 0) {
    std::cerr << "[HotspineBridge] Zero capacity, exiting poll\n";
    return;
  }

  // Trade array starts just after the header
  auto* trades =
      reinterpret_cast<const HotTrade*>(static_cast<const uint8_t*>(mapped_) + sizeof(ShmHeader));

  // Compute offset of write_index field for atomic read
  const uint8_t* base = static_cast<const uint8_t*>(mapped_);
  constexpr size_t WRITE_IDX_OFFSET = 16;  // offset of write_index in header

  while (running_.load(std::memory_order_acquire)) {
    // Read write_index atomically via memcpy (packed struct, can't take address)
    uint64_t current_write;
    std::memcpy(&current_write, base + WRITE_IDX_OFFSET, sizeof(uint64_t));
    std::atomic_thread_fence(std::memory_order_acquire);

    if (current_write > last_head_) {
      uint64_t count = current_write - last_head_;
      if (count > capacity) count = capacity;  // Wraparound safety

      for (uint64_t i = last_head_; i < last_head_ + count; ++i) {
        const auto& ht = trades[i % capacity];

        // Convert HotTrade → TradeData
        TradeData trade{};
        trade.timestamp_us = ht.ts_exchange;
        trade.price = ht.price;
        trade.volume = ht.size;
        trade.side = (ht.side == 0) ? TradeSide::BUY : TradeSide::SELL;

        // Remap external symbol_id → internal
        auto it = external_to_internal_.find(ht.symbol_id);
        if (it != external_to_internal_.end()) {
          trade.symbol_id = it->second;
        } else {
          trade.symbol_id = 0;
        }

        processor_->enqueue_trade(trade);
        trades_ingested_.fetch_add(1, std::memory_order_relaxed);
      }

      last_head_ = last_head_ + count;
    }

    // Sleep ~500μs between polls
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
}

}  // namespace BTQuant
