#pragma once

#include "vulkan_base_types.hpp"
#include <vector>
#include <atomic>
#include <chrono>

namespace BTQuant {

// ============================================================================
// Timeline Semaphore - Fine-grained synchronization using timeline values
// ============================================================================

class TimelineSemaphore {
public:
    using TimelineValue = uint64_t;

    TimelineSemaphore(VkDevice device);
    ~TimelineSemaphore();

    // Get current timeline value (atomic)
    TimelineValue getCurrentValue() const;

    // Wait for a specific timeline value
    bool waitForValue(TimelineValue value, uint64_t timeoutNs = UINT64_MAX);

    // Signal a specific timeline value
    bool signalValue(TimelineValue value);

    // Get the Vulkan semaphore handle
    VkSemaphore getHandle() const { return semaphore_; }

private:
    VkDevice device_;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
    std::atomic<TimelineValue> currentValue_ = 0;
};

// ============================================================================
// Render Pass Synchronization State
// ============================================================================

struct RenderPassSyncState {
    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;
    TimelineSemaphore::TimelineValue lastCompletedValue = 0;
};

// ============================================================================
// Vulkan Synchronization Context - Manages render frame synchronization
// ============================================================================

class VulkanSyncContext {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;

    VulkanSyncContext(VkDevice device, VkCommandPool commandPool);
    ~VulkanSyncContext();

    // Prepare for next frame - wait for previous frame to complete
    void prepareFrame(uint32_t frameIndex, TimelineSemaphore& timelineSemaphore);

    // Acquire command buffer for current frame
    VkCommandBuffer acquireCommandBuffer();

    // Submit command buffer with timeline synchronization
    bool submitCommandBuffer(VkQueue queue, VkCommandBuffer cmdBuffer,
                           TimelineSemaphore& timelineSemaphore,
                           TimelineSemaphore::TimelineValue signalValue,
                           TimelineSemaphore::TimelineValue waitValue);

    // Get current frame's sync state
    RenderPassSyncState& getSyncState(uint32_t frameIndex) {
        return syncStates_[frameIndex];
    }

    // Wait for all operations to complete
    void waitIdle() const;

private:
    VkDevice device_;
    VkCommandPool commandPool_;
    std::vector<RenderPassSyncState> syncStates_;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::atomic<uint32_t> currentFrameIndex_ = 0;

    // Helper for creating synchronization objects
    VkSemaphore createSemaphore();
    VkFence createFence(bool signaled = true);
};

// ============================================================================
// Synchronization Debug Utilities
// ============================================================================

class SyncDebugUtils {
public:
    // Measure command buffer execution time
    static uint64_t measureExecutionTime(VkDevice device,
                                       VkCommandBuffer cmdBuffer,
                                       VkQueue queue,
                                       const char* name);

    // Log synchronization state
    static void logSyncState(const VulkanSyncContext& context);

    // Performance profiling
    struct PerformanceStats {
        uint64_t minExecutionTimeNs = UINT64_MAX;
        uint64_t maxExecutionTimeNs = 0;
        uint64_t totalExecutionTimeNs = 0;
        uint32_t frameCount = 0;

        double getAverageTimeMs() const {
            return static_cast<double>(totalExecutionTimeNs) / (frameCount * 1000000.0);
        }

        double getMinTimeMs() const {
            return static_cast<double>(minExecutionTimeNs) / 1000000.0;
        }

        double getMaxTimeMs() const {
            return static_cast<double>(maxExecutionTimeNs) / 1000000.0;
        }
    };

    static PerformanceStats performanceStats;
};

// ============================================================================
// Ring Buffer Management - Lock-free buffer acquisition/release
// ============================================================================

class RingBufferManager {
public:
    static constexpr size_t SLOT_SIZE = 1024 * 1024;  // 1MB slots
    static constexpr uint32_t NUM_SLOTS = 2;         // Dual slot ring buffer

    RingBufferManager(VkDevice device, VkPhysicalDevice physicalDevice);
    ~RingBufferManager();

    // Acquire a buffer slot - returns offset in buffer
    uint32_t acquireSlot();

    // Release a buffer slot
    void releaseSlot(uint32_t slotIndex);

    // Get buffer allocation
    const BufferAllocation& getBufferAllocation() const {
        return bufferAllocation_;
    }

    // Get available slot count
    uint32_t getAvailableSlots() const {
        return availableSlots_.load(std::memory_order_acquire);
    }

private:
    VkDevice device_;
    BufferAllocation bufferAllocation_;
    std::atomic<uint32_t> availableSlots_ = NUM_SLOTS;
    std::atomic<uint32_t> writeIndex_ = 0;
    std::atomic<uint32_t> readIndex_ = 0;
};

} // namespace BTQuant
