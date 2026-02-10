#ifndef HOTSPINE_LAYOUT_V3_HPP
#define HOTSPINE_LAYOUT_V3_HPP

#include <atomic>
#include <cstdint>
#include <cstddef>

namespace HotSpine::V3 {

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

// Core data structure for shared memory
struct alignas(64) HotspineData {
    std::uint64_t timestamp;            // Nanosecond timestamp of the event
    std::uint32_t symbol_id;            // Symbol identifier
    std::uint32_t event_type;           // Type of event (trade, quote, etc.)
    double price;                       // Price value
    double volume;                      // Volume value
    std::uint8_t flags;                 // Flags: Bit 0: IS_WARMUP, Bit 1: IS_SNAPSHOT
    std::uint8_t reserved_flags[3];     // Reserved for future flags
    std::uint32_t sequence_number;      // Sequence number for ordering
    std::uint32_t payload_size;         // Size of additional payload data
    std::uint8_t padding[20];           // Explicit padding to reach 64 bytes total

    // Flag bit positions
    static constexpr std::uint8_t IS_WARMUP = 0x01;    // Bit 0: Warm-up event
    static constexpr std::uint8_t IS_SNAPSHOT = 0x02;  // Bit 1: Snapshot event
};

// Constants
constexpr size_t HEADER_SIZE = sizeof(RingBufferHeader);

// Static assertions to ensure proper alignment and size
static_assert(sizeof(HotspineData) == 64, "HotspineData must be exactly 64 bytes for cache alignment");
static_assert(alignof(HotspineData) == 64, "HotspineData must be 64-byte aligned");

} // namespace HotSpine::V3

#endif // HOTSPINE_LAYOUT_V3_HPP