#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace BTQuant {

/**
 * Lock-free triple buffer for single-producer / single-consumer data sharing.
 *
 * Writer (data thread) calls write_buffer() + publish().
 * Reader (render thread) calls consume() + read().
 *
 * No locks. No copies on the read path. Just atomic index swaps.
 *
 * Memory ordering:
 *   publish() uses acq_rel on the index swap and release on new_data_ flag.
 *   consume() uses acq_rel on the index swap and acquire on new_data_ flag.
 *   This guarantees that all writes to back buffer are visible after consume().
 */
template <typename T>
class TripleBuffer {
 public:
  TripleBuffer() = default;

  // Non-copyable, non-movable (contains atomics)
  TripleBuffer(const TripleBuffer&) = delete;
  TripleBuffer& operator=(const TripleBuffer&) = delete;
  TripleBuffer(TripleBuffer&&) = delete;
  TripleBuffer& operator=(TripleBuffer&&) = delete;

  /**
   * Get the back buffer for writing (writer thread only).
   * The writer owns this buffer exclusively — no synchronization needed.
   */
  T& write_buffer() { return buffers_[back_.load(std::memory_order_relaxed)]; }

  /**
   * Publish the back buffer: swap back ↔ middle atomically.
   * After this call, the data written to write_buffer() is available for the reader.
   * Writer thread only.
   */
  void publish() {
    // Swap back and middle indices atomically
    int back = back_.load(std::memory_order_relaxed);
    int middle = middle_.exchange(back, std::memory_order_acq_rel);
    back_.store(middle, std::memory_order_relaxed);
    new_data_.store(true, std::memory_order_release);
  }

  /**
   * Consume: if new data is available, swap middle ↔ front atomically.
   * Returns true if new data was consumed (front buffer updated).
   * Reader thread only.
   */
  bool consume() {
    if (!new_data_.exchange(false, std::memory_order_acquire)) {
      return false;  // No new data since last consume
    }
    // Swap front and middle indices atomically
    int front = front_.load(std::memory_order_relaxed);
    int middle = middle_.exchange(front, std::memory_order_acq_rel);
    front_.store(middle, std::memory_order_relaxed);
    return true;
  }

  /**
   * Read the front buffer (reader thread only).
   * This is the most recent published snapshot.
   * Zero-copy — returns a const reference to the internal buffer.
   */
  const T& read() const { return buffers_[front_.load(std::memory_order_relaxed)]; }

  /**
   * Check if there's new data without consuming it.
   */
  bool has_new_data() const { return new_data_.load(std::memory_order_relaxed); }

 private:
  std::array<T, 3> buffers_{};
  std::atomic<int> back_{0};    // Writer's current buffer index
  std::atomic<int> middle_{1};  // Staging buffer index (published, not yet consumed)
  std::atomic<int> front_{2};   // Reader's current buffer index
  std::atomic<bool> new_data_{false};
};

}  // namespace BTQuant
