#pragma once

#include "hotspine_layout.hpp"
#include <string>
#include <vector>
#include <memory>

namespace HotSpine {

class HotSpineReader {
public:
    /**
     * Constructor - attaches to existing shared memory segment
     * @param shm_name Name of shared memory segment to attach to
     */
    explicit HotSpineReader(const std::string& shm_name = "/btquant_hotspine");
    
    /**
     * Destructor - detaches from shared memory
     */
    ~HotSpineReader();
    
    /**
     * Check if successfully attached to shared memory
     * @return true if attached, false otherwise
     */
    bool isAttached() const;
    
    /**
     * Poll for a single trade (non-blocking)
     * @param trade Output parameter for the trade data
     * @return true if trade was read, false if no trades available
     */
    bool pollTrade(HotTrade& trade);
    
    /**
     * Read all available trades at once (more efficient for batch processing)
     * @return Vector of available trades (empty if none available)
     */
    std::vector<HotTrade> readAllAvailableTrades();
    
    /**
     * Get the number of lost trades (overflow counter)
     * @return Number of trades lost due to buffer overflow
     */
    uint64_t getLostCount() const;
    
    /**
     * Get current buffer utilization information
     * @return Pair of (current_size, capacity)
     */
    std::pair<uint64_t, uint64_t> getBufferUtilization() const;
    
    // Delete copy constructor and assignment operator
    HotSpineReader(const HotSpineReader&) = delete;
    HotSpineReader& operator=(const HotSpineReader&) = delete;

private:
    std::string shm_name_;
    int shm_fd_{-1};
    void* shm_ptr_{nullptr};
    SharedMemoryHeader* header_{nullptr};
    HotTrade* trades_buffer_{nullptr};
    
    bool attachToSharedMemory();
    bool detachFromSharedMemory();
    
    // Calculate current buffer size
    uint64_t calculateCurrentSize() const;
};

// Smart pointer type for HotSpineReader
using HotSpineReaderPtr = std::shared_ptr<HotSpineReader>;

} // namespace HotSpine