#pragma once

#include "hotspine_layout_v3.hpp"  // Updated to use V3 layout with atomic operations
#include "hotspine_layout.hpp"     // For HotOrderbookSnapshot compatibility
#include <memory>
#include <string>
#include <utility>

namespace HotSpine {

/**
 * HotSpineReader - Reads market data from shared memory
 *
 * This class provides a real implementation for reading market data
 * from the HotSpine shared memory protocol.
 */
class HotSpineReader {
public:
  /**
   * Construct a reader attached to shared memory
   * @param shm_name Shared memory name (e.g., "/btquant_hotspine")
   */
  explicit HotSpineReader(const std::string &shm_name);

  ~HotSpineReader();

  // Non-copyable, non-movable
  HotSpineReader(const HotSpineReader &) = delete;
  HotSpineReader &operator=(const HotSpineReader &) = delete;
  HotSpineReader(HotSpineReader &&) = delete;
  HotSpineReader &operator=(HotSpineReader &&) = delete;

  /**
   * Check if successfully attached to shared memory
   */
  bool isAttached() const;

  /**
   * Poll for next trade data
   * @param trade Output structure to fill with trade data
   * @return true if trade was read, false if no new data
   */
  bool pollTrade(HotSpine::V3::HotspineData &trade);

  /**
   * Poll for next orderbook snapshot
   * @param snapshot Output structure to fill with orderbook data
   * @return true if snapshot was read, false if no new data
   */
  bool pollOrderbook(HotOrderbookSnapshot &snapshot);

  /**
   * Get the shared memory name
   */
  const std::string &getShmName() const;

  /**
   * Reattach to shared memory (for recovery)
   * @return true if successfully reattached
   */
  bool reattach();

  /**
   * Get current buffer status
   * @return pair of (used, capacity)
   */
  std::pair<uint64_t, uint64_t> get_buffer_status() const;

  /**
   * Check if shared memory is healthy
   */
  bool is_healthy() const;

private:
  void attach_to_shm();
  void detach_from_shm();
  bool validate_header();

  std::string shm_name_;
  bool attached_ = false;
  int fd_ = -1;
  void *mapped_region_ = nullptr;
  size_t mapped_size_ = 0;
  // Removed mutex for lock-free operation in hot path
  // Using atomic operations for thread safety where needed
};

} // namespace HotSpine
