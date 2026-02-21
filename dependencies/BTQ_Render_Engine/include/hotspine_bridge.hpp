#pragma once

/**
 * @file hotspine_bridge.hpp
 * @brief Reads SharedMemoryLayoutV3 from /dev/shm/btquant_hotspine
 *        and injects trades into MarketDataProcessor via enqueue_trade().
 */

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <unordered_map>

#include "market_data_processor.hpp"

namespace BTQuant {

class HotspineBridge {
 public:
  HotspineBridge() = default;
  ~HotspineBridge();

  // Non-copyable
  HotspineBridge(const HotspineBridge&) = delete;
  HotspineBridge& operator=(const HotspineBridge&) = delete;

  /// Open shared memory, load symbol map, start poll thread
  bool start(RenderEngine::MarketDataProcessor* processor,
             const std::string& shm_path = "/dev/shm/btquant_hotspine",
             const std::string& symbols_path = "/dev/shm/btquant_symbols.json");

  /// Stop the poll thread and unmap memory
  void stop();

  bool is_running() const { return running_.load(std::memory_order_acquire); }
  uint64_t trades_ingested() const { return trades_ingested_.load(std::memory_order_relaxed); }

 private:
  void poll_loop();

  RenderEngine::MarketDataProcessor* processor_ = nullptr;
  void* mapped_ = nullptr;
  size_t mapped_size_ = 0;

  std::thread poll_thread_;
  std::atomic<bool> running_{false};
  std::atomic<uint64_t> trades_ingested_{0};

  uint64_t last_head_ = 0;  // Last seen head_index in the ring buffer

  // External symbol_id (10000+) → internal (0-99)
  static constexpr uint32_t MAX_REMAP = 100;
  uint32_t remap_[MAX_REMAP] = {};  // remap_[internal] = external
  std::unordered_map<uint32_t, uint32_t> external_to_internal_;
};

}  // namespace BTQuant
