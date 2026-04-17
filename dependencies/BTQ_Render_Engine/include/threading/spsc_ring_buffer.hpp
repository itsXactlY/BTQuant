#ifndef BTQ_SPSC_RING_BUFFER_HPP
#define BTQ_SPSC_RING_BUFFER_HPP

#include <atomic>
#include <cstddef>
#include <new>
#include <optional>

namespace btq::threading {

template <typename T, size_t Capacity>
class SpscRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be power of 2");

public:
    SpscRingBuffer() noexcept = default;
    ~SpscRingBuffer() = default;

    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

    bool push(const T& item) noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = head + 1;

        // Check if full: writer is only one advancing head_
        // Full when next would catch up to tail_ on the ring
        if (next - tail_.load(std::memory_order_acquire) > Capacity) {
            return false;
        }

        data_[head & (Capacity - 1)] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() noexcept {
        const size_t tail = tail_.load(std::memory_order_relaxed);

        // Check if empty
        if (head_.load(std::memory_order_acquire) == tail) {
            return std::nullopt;
        }

        T item = std::move(data_[tail & (Capacity - 1)]);
        tail_.store(tail + 1, std::memory_order_release);
        return item;
    }

    size_t peek(size_t n, T* out) const noexcept {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);

        const size_t count = head - tail;
        const size_t to_copy = (n < count) ? n : count;

        for (size_t i = 0; i < to_copy; ++i) {
            size_t idx = (head - to_copy + i) & (Capacity - 1);
            out[i] = data_[idx];
        }

        return to_copy;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    size_t size() const noexcept {
        return head_.load(std::memory_order_acquire) -
               tail_.load(std::memory_order_acquire);
    }

private:
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    alignas(64) T data_[Capacity];
};

} // namespace btq::threading

#endif // BTQ_SPSC_RING_BUFFER_HPP
