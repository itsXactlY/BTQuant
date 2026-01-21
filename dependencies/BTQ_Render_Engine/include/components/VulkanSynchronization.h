/**
 * @file VulkanSynchronization.h
 * @brief Vulkan Synchronization Strategy for Hotspine Data Writes
 * 
 * Implements zero-latency synchronization between Hotspine ring buffer writes
 * and GPU reads using Vulkan barriers, semaphores, and fences. Prevents tearing
 * without locking the main thread.
 * 
 * @author Market Microstructure Renderer Team
 * @version 1.0.0
 */

#ifndef VULKAN_SYNCHRONIZATION_H
#define VULKAN_SYNCHRONIZATION_H

#include <vulkan/vulkan.h>
#include <memory>
#include <vector>
#include <atomic>
#include <chrono>

namespace vk {

// ============================================
// Forward Declarations
// ============================================

class VulkanDevice;

// ============================================
// Types and Constants
// ============================================

using FrameIndex = uint32_t;
using TimelineValue = uint64_t;
using TimestampNs = uint64_t;

constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;

// ============================================
// Timeline Semaphore Wrapper
// ============================================

class TimelineSemaphore {
public:
    TimelineSemaphore() = default;
    
    /**
     * @brief Create timeline semaphore
     * @param device Vulkan device pointer
     */
    explicit TimelineSemaphore(VkDevice device);
    
    ~TimelineSemaphore();
    
    TimelineSemaphore(const TimelineSemaphore&) = delete;
    TimelineSemaphore& operator=(const TimelineSemaphore&) = delete;
    TimelineSemaphore(TimelineSemaphore&& other) noexcept;
    TimelineSemaphore& operator=(TimelineSemaphore&& other) noexcept;
    
    /**
     * @brief Get current timeline value
     * @return Current value of the timeline semaphore
     */
    TimelineValue getCurrentValue() const;
    
    /**
     * @brief Wait for timeline to reach specified value
     * @param value Value to wait for
     * @param timeoutNs Timeout in nanoseconds (default: 1 second)
     * @return true if value was reached, false if timed out
     */
    bool waitForValue(TimelineValue value, uint64_t timeoutNs = 1'000'000'000) const;
    
    /**
     * @brief Signal timeline to reach specified value
     * @param value Value to signal
     * @return true if successful
     */
    bool signalValue(TimelineValue value);
    
    /**
     * @brief Get Vulkan semaphore handle
     */
    VkSemaphore handle() const { return semaphore_; }
    
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

// ============================================
// Fence Manager for Command Buffer Tracking
// ============================================

class FenceManager {
public:
    /**
     * @brief Create fence manager
     * @param device Vulkan device
     * @param maxFrames Number of in-flight frames to track
     */
    explicit FenceManager(VkDevice device, uint32_t maxFrames);
    
    ~FenceManager();
    
    FenceManager(const FenceManager&) = delete;
    FenceManager& operator=(const FenceManager&) = delete;
    
    /**
     * @brief Acquire a fence for next command buffer
     * @return Index of acquired fence (or -1 if all are in use)
     */
    int32_t acquireFence();
    
    /**
     * @brief Get fence at specified index
     * @param index Fence index
     */
    VkFence getFence(uint32_t index) const { return fences_[index]; }
    
    /**
     * @brief Wait for all fences to complete
     */
    void waitAll() const;
    
    /**
     * @brief Reset all fences
     */
    void resetAll() const;
    
private:
    VkDevice device_;
    uint32_t maxFrames_;
    std::vector<VkFence> fences_;
    std::vector<bool> used_;
};

// ============================================
// Command Buffer Pool and Allocation
// ============================================

class CommandBufferPool {
public:
    explicit CommandBufferPool(VkDevice device, VkCommandPool pool);
    
    ~CommandBufferPool();
    
    CommandBufferPool(const CommandBufferPool&) = delete;
    CommandBufferPool& operator=(const CommandBufferPool&) = delete;
    
    /**
     * @brief Acquire a command buffer from the pool
     * @param isPrimary True for primary command buffer, false for secondary
     */
    VkCommandBuffer acquireCommandBuffer(bool isPrimary = true);
    
    /**
     * @brief Release a command buffer back to the pool
     * @param cmdBuffer Command buffer to release
     */
    void releaseCommandBuffer(VkCommandBuffer cmdBuffer);
    
private:
    VkDevice device_;
    VkCommandPool pool_;
    std::vector<VkCommandBuffer> freeBuffers_;
    std::vector<VkCommandBuffer> activeBuffers_;
};

// ============================================
// Render Pass Synchronization State
// ============================================

struct RenderPassSyncState {
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;
    TimelineValue lastCompletedValue;
    bool isFirstFrame = true;
    
    RenderPassSyncState()
        : imageAvailableSemaphore(VK_NULL_HANDLE)
        , renderFinishedSemaphore(VK_NULL_HANDLE)
        , inFlightFence(VK_NULL_HANDLE)
        , lastCompletedValue(0)
    {}
};

// ============================================
// Vulkan Synchronization Context
// ============================================

class VulkanSyncContext {
public:
    explicit VulkanSyncContext(VkDevice device, VkCommandPool cmdPool);
    
    ~VulkanSyncContext();
    
    VulkanSyncContext(const VulkanSyncContext&) = delete;
    VulkanSyncContext& operator=(const VulkanSyncContext&) = delete;
    
    /**
     * @brief Prepare for next frame
     * @param currentFrame Current frame index
     * @param timelineSemaphore Timeline semaphore for synchronization
     * @return true if ready to proceed
     */
    bool prepareFrame(uint32_t currentFrame, TimelineSemaphore& timelineSemaphore);
    
    /**
     * @brief Submit command buffer with synchronization
     * @param queue Queue to submit to
     * @param cmdBuffer Command buffer to submit
     * @param timelineSemaphore Timeline semaphore
     * @param signalValue Value to signal
     * @param waitValue Value to wait for
     */
    bool submitCommandBuffer(VkQueue queue, VkCommandBuffer cmdBuffer,
                           TimelineSemaphore& timelineSemaphore,
                           TimelineValue signalValue,
                           TimelineValue waitValue = 0);
    
    /**
     * @brief Wait for all command buffers to complete
     */
    void waitForCompletion() const;
    
    /**
     * @brief Reset context state
     */
    void reset() const;
    
    /**
     * @brief Acquire a command buffer
     */
    VkCommandBuffer acquireCommandBuffer() {
        return cmdPool_->acquireCommandBuffer(true);
    }
    
    /**
     * @brief Release a command buffer
     */
    void releaseCommandBuffer(VkCommandBuffer cmdBuffer) {
        cmdPool_->releaseCommandBuffer(cmdBuffer);
    }
    
private:
    VkDevice device_;
    std::unique_ptr<CommandBufferPool> cmdPool_;
    std::unique_ptr<FenceManager> fenceManager_;
    std::vector<RenderPassSyncState> frameStates_;
};

// ============================================
// Barrier Management for Hotspine Data Access
// ============================================

class HotspineBarrierManager {
public:
    explicit HotspineBarrierManager(VkDevice device);
    
    /**
     * @brief Create memory barrier for buffer to image transitions
     * @param srcStage Source pipeline stage
     * @param dstStage Destination pipeline stage
     */
    VkMemoryBarrier createBufferMemoryBarrier(VkPipelineStageFlags srcStage,
                                           VkPipelineStageFlags dstStage) const;
    
    /**
     * @brief Create buffer memory barrier for SSBO to compute shader transitions
     */
    VkBufferMemoryBarrier createSSBOBufferBarrier(VkBuffer buffer,
                                                 VkDeviceSize offset,
                                                 VkDeviceSize size) const;
    
    /**
     * @brief Create image memory barrier for heatmap texture transitions
     */
    VkImageMemoryBarrier createHeatmapImageBarrier(VkImage image,
                                                 VkImageLayout oldLayout,
                                                 VkImageLayout newLayout) const;
    
    /**
     * @brief Create pipeline barrier for Hotspine data update
     * @param cmdBuffer Command buffer to record into
     * @param buffer Buffer being updated
     * @param offset Offset of update in buffer
     * @param size Size of update in bytes
     */
    void recordHotspineUpdateBarrier(VkCommandBuffer cmdBuffer,
                                   VkBuffer buffer,
                                   VkDeviceSize offset,
                                   VkDeviceSize size) const;
    
private:
    VkDevice device_;
};

// ============================================
// Atomic Queue for Zero-Latency Data Transfer
// ============================================

template<typename T>
class AtomicTransferQueue {
public:
    explicit AtomicTransferQueue(uint32_t maxSize)
        : maxSize_(maxSize)
        , readIndex_(0)
        , writeIndex_(0)
        , data_(std::make_unique<T[]>(maxSize))
    {}
    
    /**
     * @brief Check if queue has data available for reading
     */
    bool hasData() const {
        return readIndex_.load(std::memory_order_acquire) != 
               writeIndex_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Check if queue has space for writing
     */
    bool hasSpace() const {
        const auto nextWrite = (writeIndex_.load(std::memory_order_acquire) + 1) % maxSize_;
        return nextWrite != readIndex_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Enqueue data (atomic, lock-free)
     */
    bool enqueue(const T& value) {
        const auto currentWrite = writeIndex_.load(std::memory_order_acquire);
        const auto nextWrite = (currentWrite + 1) % maxSize_;
        
        if (nextWrite == readIndex_.load(std::memory_order_acquire)) {
            return false; // Queue full
        }
        
        data_[currentWrite] = value;
        writeIndex_.store(nextWrite, std::memory_order_release);
        return true;
    }
    
    /**
     * @brief Dequeue data (atomic, lock-free)
     */
    bool dequeue(T& value) {
        const auto currentRead = readIndex_.load(std::memory_order_acquire);
        
        if (currentRead == writeIndex_.load(std::memory_order_acquire)) {
            return false; // Queue empty
        }
        
        value = data_[currentRead];
        readIndex_.store((currentRead + 1) % maxSize_, std::memory_order_release);
        return true;
    }
    
private:
    uint32_t maxSize_;
    std::atomic<uint32_t> readIndex_;
    std::atomic<uint32_t> writeIndex_;
    std::unique_ptr<T[]> data_;
};

// ============================================
// Ring Buffer Synchronization Manager
// ============================================

struct RingBufferSlot {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
    TimelineValue lastUsed;
    bool isAvailable;
};

class RingBufferSyncManager {
public:
    explicit RingBufferSyncManager(VkDevice device,
                                 uint32_t slotCount,
                                 VkDeviceSize slotSize);
    
    ~RingBufferSyncManager();
    
    /**
     * @brief Acquire a slot from the ring buffer
     * @return Available slot index or -1 if all slots are in use
     */
    int32_t acquireSlot();
    
    /**
     * @brief Release a slot back to the ring buffer
     * @param slotIndex Index of slot to release
     */
    void releaseSlot(uint32_t slotIndex);
    
    /**
     * @brief Get slot information
     * @param slotIndex Slot index
     */
    RingBufferSlot& getSlot(uint32_t slotIndex) {
        return slots_[slotIndex];
    }
    
    /**
     * @brief Wait for a slot to become available
     * @param timeoutNs Timeout in nanoseconds
     * @return Available slot index or -1 if timed out
     */
    int32_t waitForSlot(uint64_t timeoutNs = 500'000'000);
    
private:
    VkDevice device_;
    uint32_t slotCount_;
    VkDeviceSize slotSize_;
    std::vector<RingBufferSlot> slots_;
    uint32_t currentIndex_ = 0;
    TimelineValue currentTime_ = 0;
};

// ============================================
// Debug Utilities
// ============================================

class SyncDebugUtils {
public:
    /**
     * @brief Measure command buffer execution time
     * @param device Vulkan device
     * @param cmdBuffer Command buffer to instrument
     * @param queue Queue command buffer will be submitted to
     * @param name Name for debugging
     */
    static uint64_t measureExecutionTime(VkDevice device,
                                       VkCommandBuffer cmdBuffer,
                                       VkQueue queue,
                                       const char* name);
    
    /**
     * @brief Log synchronization state for debugging
     */
    static void logSyncState(const VulkanSyncContext& context);
};

} // namespace vk

#endif // VULKAN_SYNCHRONIZATION_H