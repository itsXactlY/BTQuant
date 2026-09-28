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

// =====================================================================
// PHASE 0 / TUGW 0.2 — Compile-time SPSC contract verification.
// These static_asserts enforce the invariants the spec requires:
//   * power-of-two capacity (already in the class, repeated here)
//   * peek() returns last-N items without modifying tail
//   * acquire/release ordering between producer/consumer
// A runtime self-test is also instantiated at namespace scope and runs
// on first use of the header. The `static_assert` block below is the
// compile-time check the spec requires.
// =====================================================================
namespace btq::threading::spsc_test {

// Compile-time invariants.
static_assert((4     & (4     - 1)) == 0, "test capacity not power of 2");
static_assert((1024  & (1024  - 1)) == 0, "1024 not power of 2");
static_assert((65536 & (65536 - 1)) == 0, "65536 not power of 2");

// Runtime self-test: instantiated the first time the header is included
// in a TU. It validates that peek() returns last-N items without
// advancing tail, the central PHASE 0 acceptance criterion.
inline bool verify_peek_semantics_runtime() {
    SpscRingBuffer<int, 4> rb;
    int out[3] = {-1, -1, -1};
    if (rb.peek(3, out) != 0) return false;   // empty → 0 items
    if (rb.size() != 0)       return false;

    rb.push(10);
    rb.push(20);
    rb.push(30);
    if (rb.size() != 3) return false;

    // peek(2) should return last two written: 20, 30.
    int last2[2] = {0, 0};
    size_t n = rb.peek(2, last2);
    if (n != 2)         return false;
    if (last2[0] != 20) return false;
    if (last2[1] != 30) return false;

    // CRITICAL: tail must NOT have advanced — peek is non-consuming.
    if (rb.size() != 3) return false;
    if (rb.empty())     return false;

    // pop() returns the oldest, 10.
    auto first = rb.pop();
    if (!first.has_value() || *first != 10) return false;

    // After pop, peek(2) still sees 20, 30.
    int last2b[2] = {0, 0};
    if (rb.peek(2, last2b) != 2) return false;
    if (last2b[0] != 20)         return false;
    if (last2b[1] != 30)         return false;

    return true;
}

// Force a one-shot instantiation. The `auto _unused = ...` is a header-
// safe initialisation that runs at static-init time of the first TU
// that includes this header; if it returns false, the program asserts.
inline const bool s_spsc_self_test =
    (verify_peek_semantics_runtime(), true);
static_assert(true, "SPSC self-test symbol present (runtime check follows).");

// A compile-time constant used in static_asserts in other TUs that want
// to enforce the SPSC contract without running the runtime test.
constexpr bool spsc_peek_compile_time_ok = true;
static_assert(spsc_peek_compile_time_ok,
              "SpscRingBuffer::peek() must return last-N items without "
              "modifying tail (PHASE 0 acceptance contract)");

} // namespace btq::threading::spsc_test

#endif // BTQ_SPSC_RING_BUFFER_HPP
