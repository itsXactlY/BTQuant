#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "../../../dependencies/BTQ_Render_Engine/include/trading/HotspineData.h"
#include "../../../dependencies/BTQ_Render_Engine/include/hotspine_layout_v3.hpp"

namespace BTQuant {

// Forward declaration of the canonical event structure
using HotspineData = RenderEngine::HotspineData;

class HotSpineDataBridge {
public:
    explicit HotSpineDataBridge(const std::string& shm_path = "/btquant_hotspine");
    ~HotSpineDataBridge();

    // Main method for writing data directly to the ring buffer
    bool write_direct(const HotspineData& event);

private:
    std::string shm_path_;
    int shm_fd_{-1};
    void* shm_ptr_{nullptr};
    
    // Pointers to shared memory structures
    HotSpine::V3::RingBufferHeader* header_{nullptr};
    uint8_t* ring_buffer_data_{nullptr};
    
    // Constants for ring buffer operations
    static constexpr size_t RING_BUFFER_SIZE = 8192; // Power of 2 for efficient masking
    static constexpr size_t RING_BUFFER_MASK = RING_BUFFER_SIZE - 1; // For indexing: idx = counter & MASK
    
    bool connect_to_shared_memory();
    void disconnect_from_shared_memory();
};

} // namespace BTQuant