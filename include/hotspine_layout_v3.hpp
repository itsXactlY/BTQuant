#ifndef HOTSPINE_LAYOUT_V3_HPP
#define HOTSPINE_LAYOUT_V3_HPP

#include <atomic>
#include <cstdint>
#include <cstddef>

namespace hotspine {

// Ring buffer constants
constexpr size_t RING_BUFFER_SIZE = 8192; // Power of 2 for efficient masking
constexpr size_t RING_BUFFER_MASK = RING_BUFFER_SIZE - 1; // For indexing: idx = counter & MASK

struct alignas(64) RingBufferHeader {
    std::uint32_t magic;  // 0x42545155 "BTQ3"
    std::uint32_t version; // Version identifier
    alignas(64) std::atomic<std::uint64_t> write_head{0};  // Index of next write slot
    alignas(64) std::atomic<std::uint64_t> read_tail{0};   // Index of next read slot
    std::atomic<std::uint64_t> dropped_count{0};  // Count of dropped events due to overflow
    std::uint8_t reserved[24];  // Padding to align to 64-byte boundary

    // Inline helper methods
    inline std::uint64_t get_next_write_slot() const {
        return write_head.load(std::memory_order_acquire) & RING_BUFFER_MASK;
    }

    inline void commit_write() {
        write_head.fetch_add(1, std::memory_order_release);
    }

    inline std::uint64_t get_available_count() const {
        std::uint64_t write_idx = write_head.load(std::memory_order_acquire);
        std::uint64_t read_idx = read_tail.load(std::memory_order_acquire);
        return write_idx - read_idx;
    }

    inline bool is_full() const {
        return get_available_count() >= RING_BUFFER_SIZE;
    }

    inline bool is_empty() const {
        return get_available_count() == 0;
    }
};

// Constants
constexpr size_t HEADER_SIZE = sizeof(RingBufferHeader);

} // namespace hotspine

#endif // HOTSPINE_LAYOUT_V3_HPP