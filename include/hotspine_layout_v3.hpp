#ifndef HOTSPINE_LAYOUT_V3_HPP
#define HOTSPINE_LAYOUT_V3_HPP

#include <cstdint>

namespace hotspine {

struct RingBufferHeader {
    std::uint64_t write_index;
    std::uint64_t read_index;
    std::uint64_t capacity;
    std::uint64_t element_size;
    std::uint32_t flags;
    std::uint32_t version;
    
    // Padding to ensure consistent size across platforms
    std::uint8_t reserved[32];
};

} // namespace hotspine

#endif // HOTSPINE_LAYOUT_V3_HPP