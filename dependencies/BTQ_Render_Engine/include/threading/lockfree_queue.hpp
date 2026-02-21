#pragma once

#include <atomic>
#include <cstddef>
#include <type_traits>

namespace BTQuant {

template <typename T, size_t Capacity>
class SpscRingBuffer {
  static_assert(std::is_trivial_v<T>, "T MUST be a trivial type to prevent memory leaks");
  static_assert((Capacity != 0) && ((Capacity & (Capacity - 1)) == 0),
                "Capacity MUST be a power of 2");

 public:
  SpscRingBuffer() : head_(0), tail_(0) {}

  // Disable copy/move
  SpscRingBuffer(const SpscRingBuffer&) = delete;
  SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

  bool push(const T& item) {
    const size_t current_tail = tail_.load(std::memory_order_relaxed);
    const size_t next_tail = (current_tail + 1) & (Capacity - 1);

    if (next_tail == head_.load(std::memory_order_acquire)) {
      return false;  // Queue voll
    }

    buffer_[current_tail] = item;
    tail_.store(next_tail, std::memory_order_release);
    return true;
  }

  bool pop(T& item) {
    const size_t current_head = head_.load(std::memory_order_relaxed);

    if (current_head == tail_.load(std::memory_order_acquire)) {
      return false;  // Queue leer
    }

    item = buffer_[current_head];
    head_.store((current_head + 1) & (Capacity - 1), std::memory_order_release);
    return true;
  }

  size_t size() const {
    const size_t h = head_.load(std::memory_order_acquire);
    const size_t t = tail_.load(std::memory_order_acquire);
    return (t - h + Capacity) & (Capacity - 1);
  }

  /// Non-consuming read of the last N written items.
  /// Thread-safe for single reader. Copies into out[0..return_value-1].
  /// out[0] = oldest of the returned items, out[count-1] = most recent.
  /// Does NOT advance tail — process_queues() can still drain independently.
  size_t peek(size_t n, T* out) const noexcept {
    const size_t h = head_.load(std::memory_order_acquire);
    const size_t t = tail_.load(std::memory_order_acquire);
    const size_t available = (t - h + Capacity) & (Capacity - 1);
    const size_t count = (n < available) ? n : available;

    // Copy the most recent `count` items (tail - count .. tail - 1)
    for (size_t i = 0; i < count; ++i) {
      const size_t idx = (t - count + i + Capacity) & (Capacity - 1);
      out[i] = buffer_[idx];
    }
    return count;
  }

 private:
  alignas(64) std::atomic<size_t> head_;
  alignas(64) T buffer_[Capacity];
  alignas(64) std::atomic<size_t> tail_;
};

}  // namespace BTQuant