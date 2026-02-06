#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

#include "../vulkan_base_types.hpp"

namespace BTQuant {

// ============================================================================
// Timeline Semaphore - Fine-grained synchronization using timeline values
// ============================================================================

class TimelineSemaphore {
 public:
  TimelineSemaphore(VkDevice device);
  ~TimelineSemaphore();

  TimelineSemaphore(TimelineSemaphore&& other) noexcept;
  TimelineSemaphore& operator=(TimelineSemaphore&& other) noexcept;

  // Get current timeline value (atomic)
  uint64_t getCurrentValue() const;

  // Wait for a specific timeline value
  bool waitForValue(uint64_t value, uint64_t timeoutNs = UINT64_MAX) const;

  // Signal a specific timeline value
  bool signalValue(uint64_t value);

  // Get the Vulkan semaphore handle
  VkSemaphore getHandle() const { return semaphore_; }
  VkSemaphore handle() const { return semaphore_; }  // Alias for compatibility

 private:
  VkDevice device_;
  VkSemaphore semaphore_ = VK_NULL_HANDLE;
  std::atomic<uint64_t> currentValue_ = 0;
};

// ============================================================================
// Fence Manager - Manages Vulkan fences for frame synchronization
// ============================================================================

class FenceManager {
 public:
  FenceManager(VkDevice device, uint32_t maxFrames);
  ~FenceManager();

  // Acquire a fence for use
  int32_t acquireFence();

  // Wait for all fences to complete
  void waitAll() const;

  // Reset all fences
  void resetAll() const;

  // Get fence by index
  VkFence getFence(uint32_t index) const {
    return index < fences_.size() ? fences_[index] : VK_NULL_HANDLE;
  }

  // Release a fence slot
  void releaseSlot([[maybe_unused]] uint32_t index) { /* placeholder */ }

 private:
  VkDevice device_;
  uint32_t maxFrames_;
  std::vector<VkFence> fences_;
  std::vector<bool> used_;
};

// ============================================================================
// Command Buffer Pool - Manages command buffer allocation
// ============================================================================

class CommandBufferPool {
 public:
  CommandBufferPool(VkDevice device, VkCommandPool pool);
  ~CommandBufferPool();

  // Acquire a command buffer
  VkCommandBuffer acquireCommandBuffer(bool isPrimary = true);

  // Release a command buffer
  void releaseCommandBuffer(VkCommandBuffer cmdBuffer);

 private:
  VkDevice device_;
  VkCommandPool pool_;
  std::vector<VkCommandBuffer> freeBuffers_;
  std::vector<VkCommandBuffer> activeBuffers_;
};

// ============================================================================
// Render Pass Synchronization State
// ============================================================================

struct RenderPassSyncState {
  VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
  VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
  VkFence inFlightFence = VK_NULL_HANDLE;
  uint64_t lastCompletedValue = 0;
  bool isFirstFrame = true;
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
  bool prepareFrame(uint32_t currentFrame, TimelineSemaphore& timelineSemaphore);

  // Acquire command buffer for current frame
  VkCommandBuffer acquireCommandBuffer();

  // Submit command buffer with timeline synchronization
  bool submitCommandBuffer(VkQueue queue, VkCommandBuffer cmdBuffer,
                           TimelineSemaphore& timelineSemaphore, uint64_t signalValue,
                           uint64_t waitValue);

  // Get current frame's sync state
  RenderPassSyncState& getSyncState(uint32_t frameIndex) { return frameStates_[frameIndex]; }

  // Wait for all operations to complete
  void waitForCompletion() const;

  // Reset all fences
  void reset() const;

 private:
  VkDevice device_;
  std::unique_ptr<CommandBufferPool> cmdPool_;
  std::unique_ptr<FenceManager> fenceManager_;
  std::vector<RenderPassSyncState> frameStates_;
};

// ============================================================================
// Synchronization Debug Utilities
// ============================================================================

class SyncDebugUtils {
 public:
  // Measure command buffer execution time
  static uint64_t measureExecutionTime(VkDevice device, VkCommandBuffer cmdBuffer, VkQueue queue,
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

    double getMinTimeMs() const { return static_cast<double>(minExecutionTimeNs) / 1000000.0; }

    double getMaxTimeMs() const { return static_cast<double>(maxExecutionTimeNs) / 1000000.0; }
  };

  static PerformanceStats performanceStats;
};


// ============================================================================
// Ring Buffer Sync Manager - Lock-free ring buffer slot management
// ============================================================================

class RingBufferSyncManager {
 public:
  RingBufferSyncManager(VkDevice device, uint32_t slotCount, VkDeviceSize slotSize);
  ~RingBufferSyncManager();

  // Acquire a slot in the ring buffer
  int32_t acquireSlot();

  // Release a slot
  void releaseSlot(uint32_t slotIndex);

  // Wait for a slot to become available
  int32_t waitForSlot(uint64_t timeoutNs);

 private:
  struct Slot {
    bool isAvailable = true;
    uint64_t lastUsed = 0;
  };

  VkDevice device_;
  uint32_t slotCount_;
  VkDeviceSize slotSize_;
  std::vector<Slot> slots_;
  std::atomic<uint32_t> currentIndex_ = 0;
  std::atomic<uint64_t> currentTime_ = 0;
};

// ============================================================================
// Ring Buffer Management - Lock-free buffer acquisition/release
// ============================================================================

class RingBufferManager {
 public:
  static constexpr size_t SLOT_SIZE = 1024 * 1024;  // 1MB slots
  static constexpr uint32_t NUM_SLOTS = 2;          // Dual slot ring buffer

  RingBufferManager(VkDevice device, VkPhysicalDevice physicalDevice);
  RingBufferManager(VkDevice device, uint32_t slotCount, VkDeviceSize slotSize);
  ~RingBufferManager();

  // Acquire a buffer slot - returns offset in buffer
  uint32_t acquireSlot();

  // Release a buffer slot
  void releaseSlot(uint32_t slotIndex);

  // Get buffer allocation
  const BufferAllocation& getBufferAllocation() const { return bufferAllocation_; }

  // Get available slot count
  uint32_t getAvailableSlots() const { return availableSlots_.load(std::memory_order_acquire); }

 private:
  VkDevice device_;
  BufferAllocation bufferAllocation_;
  std::atomic<uint32_t> availableSlots_ = NUM_SLOTS;
  std::atomic<uint32_t> writeIndex_ = 0;
  std::atomic<uint32_t> readIndex_ = 0;
};

}  // namespace BTQuant
