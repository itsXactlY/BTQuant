#include "hotspine_layout_v3.hpp"

namespace HotSpine::V3 {

uint64_t RingBufferHeader::get_next_write_slot() const {
    return write_head.load(std::memory_order_acquire) & RING_BUFFER_MASK;
}

void RingBufferHeader::commit_write() {
    write_head.fetch_add(1, std::memory_order_release);
}

uint64_t RingBufferHeader::get_available_count() const {
    uint64_t write_idx = write_head.load(std::memory_order_acquire);
    uint64_t read_idx = read_tail.load(std::memory_order_acquire);
    return write_idx - read_idx;
}

bool RingBufferHeader::is_full() const {
    return get_available_count() >= RING_BUFFER_SIZE;
}

bool RingBufferHeader::is_empty() const {
    return get_available_count() == 0;
}

size_t calculateTradeSharedMemorySize() {
    // Using HotspineData instead of HotTrade for the new zero-copy ingestion
    return sizeof(RingBufferHeader) + (RING_BUFFER_SIZE * sizeof(HotspineData));
}

} // namespace HotSpine::V3