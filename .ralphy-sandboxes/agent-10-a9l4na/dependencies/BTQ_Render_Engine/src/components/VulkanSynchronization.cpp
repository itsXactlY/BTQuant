/**
 * @file VulkanSynchronization.cpp
 * @brief Vulkan Synchronization Strategy Implementation (C++23/26)
 *
 * Zero-latency rendering synchronization primitives with modern C++ features:
 * - Improved atomic operations with memory ordering semantics
 * - constexpr where applicable
 * - [[nodiscard]] and [[likely]]/[[unlikely]] attributes
 * - Designated initializers
 *
 * @author Market Microstructure Renderer Team
 * @version 2.0.0 (C++23/26)
 */

#include "../../include/components/VulkanSynchronization.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace BTQuant {

// ============================================
// C++23/26 Error Types
// ============================================

enum class SyncError {
  SemaphoreCreationFailed,
  FenceCreationFailed,
  WaitTimeout,
  SubmissionFailed,
  NoAvailableSlots
};

[[nodiscard]] constexpr auto to_string(SyncError error) -> std::string_view {
  switch (error) {
  case SyncError::SemaphoreCreationFailed:
    return "Semaphore creation failed";
  case SyncError::FenceCreationFailed:
    return "Fence creation failed";
  case SyncError::WaitTimeout:
    return "Wait timeout";
  case SyncError::SubmissionFailed:
    return "Command buffer submission failed";
  case SyncError::NoAvailableSlots:
    return "No available slots";
  }
  return "Unknown error";
}

// ============================================
// TimelineSemaphore Implementation
// ============================================

TimelineSemaphore::TimelineSemaphore(VkDevice device) : device_(device) {
  VkSemaphoreTypeCreateInfo timelineInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .pNext = nullptr,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0};

  VkSemaphoreCreateInfo createInfo{.sType =
                                       VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                   .pNext = &timelineInfo,
                                   .flags = 0};

  if (auto result =
          vkCreateSemaphore(device_, &createInfo, nullptr, &semaphore_);
      result != VK_SUCCESS) [[unlikely]] {
    throw std::runtime_error("Failed to create timeline semaphore: " +
                             std::to_string(static_cast<int>(result)));
  }

  if (semaphore_ == VK_NULL_HANDLE) [[unlikely]] {
    throw std::runtime_error("Created semaphore is VK_NULL_HANDLE");
  }
}

TimelineSemaphore::~TimelineSemaphore() {
  if (semaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(device_, semaphore_, nullptr);
  }
}

TimelineSemaphore::TimelineSemaphore(TimelineSemaphore &&other) noexcept
    : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      semaphore_(std::exchange(other.semaphore_, VK_NULL_HANDLE)) {}

TimelineSemaphore &
TimelineSemaphore::operator=(TimelineSemaphore &&other) noexcept {
  if (this != &other) {
    if (semaphore_ != VK_NULL_HANDLE) {
      vkDestroySemaphore(device_, semaphore_, nullptr);
    }
    device_ = std::exchange(other.device_, VK_NULL_HANDLE);
    semaphore_ = std::exchange(other.semaphore_, VK_NULL_HANDLE);
  }
  return *this;
}

[[nodiscard]] uint64_t TimelineSemaphore::getCurrentValue() const {
  uint64_t value = 0;
  if (vkGetSemaphoreCounterValue(device_, semaphore_, &value) != VK_SUCCESS)
      [[unlikely]] {
    return 0;
  }
  return value;
}

[[nodiscard]] bool TimelineSemaphore::waitForValue(uint64_t value,
                                                   uint64_t timeoutNs) const {
  VkSemaphoreWaitInfo waitInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                               .pNext = nullptr,
                               .flags = 0,
                               .semaphoreCount = 1,
                               .pSemaphores = &semaphore_,
                               .pValues = &value};

  return vkWaitSemaphores(device_, &waitInfo, timeoutNs) == VK_SUCCESS;
}

[[nodiscard]] bool TimelineSemaphore::signalValue(uint64_t value) {
  VkSemaphoreSignalInfo signalInfo{.sType =
                                       VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                                   .pNext = nullptr,
                                   .semaphore = semaphore_,
                                   .value = value};

  return vkSignalSemaphore(device_, &signalInfo) == VK_SUCCESS;
}

// ============================================
// FenceManager Implementation
// ============================================

FenceManager::FenceManager(VkDevice device, uint32_t maxFrames)
    : device_(device), maxFrames_(maxFrames), used_(maxFrames, false) {
  fences_.reserve(maxFrames);

  VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                              .pNext = nullptr,
                              .flags = VK_FENCE_CREATE_SIGNALED_BIT};

  for (uint32_t i = 0; i < maxFrames; ++i) {
    VkFence fence{};
    if (vkCreateFence(device_, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
        [[unlikely]] {
      // Cleanup already created fences
      for (auto &createdFence : fences_) {
        vkDestroyFence(device_, createdFence, nullptr);
      }
      throw std::runtime_error("Failed to create fence manager");
    }
    fences_.push_back(fence);
  }
}

FenceManager::~FenceManager() {
  waitAll();
  for (auto &fence : fences_) {
    vkDestroyFence(device_, fence, nullptr);
  }
}

[[nodiscard]] int32_t FenceManager::acquireFence() {
  for (uint32_t i = 0; i < maxFrames_; ++i) {
    if (!used_[i]) [[likely]] {
      used_[i] = true;
      vkResetFences(device_, 1, &fences_[i]);
      return static_cast<int32_t>(i);
    }

    // Check if fence has completed
    if (vkGetFenceStatus(device_, fences_[i]) == VK_SUCCESS) {
      used_[i] = true;
      vkResetFences(device_, 1, &fences_[i]);
      return static_cast<int32_t>(i);
    }
  }
  return -1; // All fences in use
}

void FenceManager::waitAll() const {
  vkWaitForFences(device_, static_cast<uint32_t>(fences_.size()),
                  fences_.data(), VK_TRUE, UINT64_MAX);
}

void FenceManager::resetAll() const {
  vkResetFences(device_, static_cast<uint32_t>(fences_.size()), fences_.data());
}

// ============================================
// CommandBufferPool Implementation
// ============================================

CommandBufferPool::CommandBufferPool(VkDevice device, VkCommandPool pool)
    : device_(device), pool_(pool) {}

CommandBufferPool::~CommandBufferPool() {
  for (auto cmdBuffer : freeBuffers_) {
    vkFreeCommandBuffers(device_, pool_, 1, &cmdBuffer);
  }
  for (auto cmdBuffer : activeBuffers_) {
    vkFreeCommandBuffers(device_, pool_, 1, &cmdBuffer);
  }
}

[[nodiscard]] VkCommandBuffer
CommandBufferPool::acquireCommandBuffer(bool isPrimary) {
  if (freeBuffers_.empty()) [[unlikely]] {
    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = pool_,
        .level = isPrimary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY
                           : VK_COMMAND_BUFFER_LEVEL_SECONDARY,
        .commandBufferCount = 1};

    VkCommandBuffer cmdBuffer{};
    if (vkAllocateCommandBuffers(device_, &allocInfo, &cmdBuffer) != VK_SUCCESS)
        [[unlikely]] {
      throw std::runtime_error("Failed to allocate command buffer");
    }
    freeBuffers_.push_back(cmdBuffer);
  }

  auto cmdBuffer = freeBuffers_.back();
  freeBuffers_.pop_back();
  activeBuffers_.push_back(cmdBuffer);
  return cmdBuffer;
}

void CommandBufferPool::releaseCommandBuffer(VkCommandBuffer cmdBuffer) {
  auto it = std::find(activeBuffers_.begin(), activeBuffers_.end(), cmdBuffer);
  if (it != activeBuffers_.end()) {
    activeBuffers_.erase(it);
    freeBuffers_.push_back(cmdBuffer);
  }
}

// ============================================
// VulkanSyncContext Implementation
// ============================================

VulkanSyncContext::VulkanSyncContext(VkDevice device, VkCommandPool cmdPool)
    : device_(device) {
  cmdPool_ = std::make_unique<CommandBufferPool>(device_, cmdPool);
  fenceManager_ = std::make_unique<FenceManager>(device_, MAX_FRAMES_IN_FLIGHT);
  frameStates_.resize(MAX_FRAMES_IN_FLIGHT);

  VkSemaphoreCreateInfo semaphoreInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0};

  VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                              .pNext = nullptr,
                              .flags = VK_FENCE_CREATE_SIGNALED_BIT};

  for (auto &state : frameStates_) {
    if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                          &state.imageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                          &state.renderFinishedSemaphore) != VK_SUCCESS)
        [[unlikely]] {
      throw std::runtime_error("Failed to create render pass semaphores");
    }

    if (vkCreateFence(device_, &fenceInfo, nullptr, &state.inFlightFence) !=
        VK_SUCCESS) [[unlikely]] {
      throw std::runtime_error("Failed to create in-flight fence");
    }
  }
}

VulkanSyncContext::~VulkanSyncContext() {
  for (const auto &state : frameStates_) {
    vkDestroySemaphore(device_, state.imageAvailableSemaphore, nullptr);
    vkDestroySemaphore(device_, state.renderFinishedSemaphore, nullptr);
    vkDestroyFence(device_, state.inFlightFence, nullptr);
  }
}

[[nodiscard]] bool VulkanSyncContext::prepareFrame(
    uint32_t currentFrame,
    [[maybe_unused]] TimelineSemaphore &timelineSemaphore) {
  auto &state = frameStates_[currentFrame % MAX_FRAMES_IN_FLIGHT];

  if (!state.isFirstFrame) [[likely]] {
    constexpr uint64_t FRAME_TIMEOUT_NS = 1'000'000'000; // 1 second
    if (auto waitResult = vkWaitForFences(device_, 1, &state.inFlightFence,
                                          VK_TRUE, FRAME_TIMEOUT_NS);
        waitResult != VK_SUCCESS) [[unlikely]] {
      std::cerr << "Frame preparation timeout\n";
      return false;
    }
    vkResetFences(device_, 1, &state.inFlightFence);
  }

  state.isFirstFrame = false;
  return true;
}

[[nodiscard]] bool VulkanSyncContext::submitCommandBuffer(
    VkQueue queue, VkCommandBuffer cmdBuffer,
    [[maybe_unused]] TimelineSemaphore &timelineSemaphore,
    [[maybe_unused]] uint64_t signalValue,
    [[maybe_unused]] uint64_t waitValue) {

  VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext = nullptr,
                          .waitSemaphoreCount = 0,
                          .pWaitSemaphores = nullptr,
                          .pWaitDstStageMask = nullptr,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &cmdBuffer,
                          .signalSemaphoreCount = 0,
                          .pSignalSemaphores = nullptr};

  auto acquireFence = fenceManager_->acquireFence();
  if (acquireFence == -1) [[unlikely]] {
    std::cerr << "No available fences for submission\n";
    return false;
  }

  auto fence = fenceManager_->getFence(static_cast<uint32_t>(acquireFence));
  if (auto result = vkQueueSubmit(queue, 1, &submitInfo, fence);
      result != VK_SUCCESS) [[unlikely]] {
    std::cerr << "Failed to submit command buffer: " << static_cast<int>(result)
              << "\n";
    fenceManager_->releaseSlot(static_cast<uint32_t>(acquireFence));
    return false;
  }

  return true;
}

void VulkanSyncContext::waitForCompletion() const { fenceManager_->waitAll(); }

void VulkanSyncContext::reset() const { fenceManager_->resetAll(); }

// ============================================
// HotspineBarrierManager Implementation
// ============================================

HotspineBarrierManager::HotspineBarrierManager(VkDevice device)
    : device_(device) {}

[[nodiscard]] VkMemoryBarrier HotspineBarrierManager::createBufferMemoryBarrier(
    [[maybe_unused]] VkPipelineStageFlags srcStage,
    [[maybe_unused]] VkPipelineStageFlags dstStage) const {
  return VkMemoryBarrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                         .pNext = nullptr,
                         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                                          VK_ACCESS_HOST_WRITE_BIT,
                         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
}

[[nodiscard]] VkBufferMemoryBarrier
HotspineBarrierManager::createSSBOBufferBarrier(VkBuffer buffer,
                                                VkDeviceSize offset,
                                                VkDeviceSize size) const {
  return VkBufferMemoryBarrier{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                               .pNext = nullptr,
                               .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                                                VK_ACCESS_HOST_WRITE_BIT,
                               .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .buffer = buffer,
                               .offset = offset,
                               .size = size};
}

[[nodiscard]] VkImageMemoryBarrier
HotspineBarrierManager::createHeatmapImageBarrier(
    VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) const {
  return VkImageMemoryBarrier{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                           ? VkAccessFlags{0}
                           : VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = oldLayout,
      .newLayout = newLayout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           .baseMipLevel = 0,
                           .levelCount = 1,
                           .baseArrayLayer = 0,
                           .layerCount = 1}};
}

void HotspineBarrierManager::recordHotspineUpdateBarrier(
    VkCommandBuffer cmdBuffer, VkBuffer buffer, VkDeviceSize offset,
    VkDeviceSize size) const {
  auto bufferBarrier = createSSBOBufferBarrier(buffer, offset, size);

  vkCmdPipelineBarrier(
      cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      0, 0, nullptr, 1, &bufferBarrier, 0, nullptr);
}

// ============================================
// RingBufferSyncManager Implementation
// ============================================

RingBufferSyncManager::RingBufferSyncManager(VkDevice device,
                                             uint32_t slotCount,
                                             VkDeviceSize slotSize)
    : device_(device), slotCount_(slotCount), slotSize_(slotSize),
      slots_(slotCount) {
  // Initialize all slots as available
  for (auto &slot : slots_) {
    slot.isAvailable = true;
    slot.lastUsed = 0;
  }
}

RingBufferSyncManager::~RingBufferSyncManager() { waitForSlot(UINT64_MAX); }

[[nodiscard]] int32_t RingBufferSyncManager::acquireSlot() {
  auto currentTime = currentTime_.fetch_add(1, std::memory_order_relaxed);

  for (uint32_t i = 0; i < slotCount_; ++i) {
    auto index =
        (currentIndex_.load(std::memory_order_relaxed) + i) % slotCount_;
    auto &slot = slots_[index];

    if (slot.isAvailable) [[likely]] {
      slot.isAvailable = false;
      slot.lastUsed = currentTime;
      currentIndex_.store(index, std::memory_order_relaxed);
      return static_cast<int32_t>(index);
    }
  }
  return -1;
}

void RingBufferSyncManager::releaseSlot(uint32_t slotIndex) {
  assert(slotIndex < slotCount_);
  slots_[slotIndex].isAvailable = true;
}

[[nodiscard]] int32_t RingBufferSyncManager::waitForSlot(uint64_t timeoutNs) {
  auto startTime = std::chrono::high_resolution_clock::now();

  while (true) {
    if (auto slotIndex = acquireSlot(); slotIndex != -1) [[likely]] {
      return slotIndex;
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::high_resolution_clock::now() - startTime)
                       .count();

    if (static_cast<uint64_t>(elapsed) >= timeoutNs) [[unlikely]] {
      return -1;
    }

    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
}

// ============================================
// SyncDebugUtils Implementation
// ============================================

[[nodiscard]] uint64_t
SyncDebugUtils::measureExecutionTime([[maybe_unused]] VkDevice device,
                                     [[maybe_unused]] VkCommandBuffer cmdBuffer,
                                     [[maybe_unused]] VkQueue queue,
                                     [[maybe_unused]] const char *name) {
  // Placeholder - real implementation would use timestamp queries
  return 0;
}

void SyncDebugUtils::logSyncState(
    [[maybe_unused]] const VulkanSyncContext &context) {
  std::cout << "Sync context state logged\n";
}

SyncDebugUtils::PerformanceStats SyncDebugUtils::performanceStats = {};

// ============================================
// RingBufferManager Implementation
// ============================================

RingBufferManager::RingBufferManager(
    VkDevice device, [[maybe_unused]] VkPhysicalDevice physicalDevice)
    : device_(device), bufferAllocation_{.buffer = VK_NULL_HANDLE,
                                         .memory = VK_NULL_HANDLE,
                                         .mapped_ptr = nullptr,
                                         .size = SLOT_SIZE * NUM_SLOTS,
                                         .offset = 0,
                                         .is_mapped = false,
                                         .pool_id = 0} {}

RingBufferManager::RingBufferManager(VkDevice device, uint32_t slotCount,
                                     VkDeviceSize slotSize)
    : device_(device), bufferAllocation_{.buffer = VK_NULL_HANDLE,
                                         .memory = VK_NULL_HANDLE,
                                         .mapped_ptr = nullptr,
                                         .size = slotSize * slotCount,
                                         .offset = 0,
                                         .is_mapped = false,
                                         .pool_id = 0} {
  availableSlots_.store(slotCount, std::memory_order_release);
}

RingBufferManager::~RingBufferManager() {
  if (bufferAllocation_.buffer != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, bufferAllocation_.buffer, nullptr);
  }
  if (bufferAllocation_.memory != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, bufferAllocation_.memory, nullptr);
  }
}

[[nodiscard]] uint32_t RingBufferManager::acquireSlot() {
  auto currentSlots = availableSlots_.load(std::memory_order_acquire);

  if (currentSlots == 0) [[unlikely]] {
    return UINT32_MAX;
  }

  // Lock-free slot acquisition with CAS
  while (currentSlots > 0) {
    if (availableSlots_.compare_exchange_weak(currentSlots, currentSlots - 1,
                                              std::memory_order_acq_rel))
        [[likely]] {
      return writeIndex_.fetch_add(1, std::memory_order_relaxed) % NUM_SLOTS;
    }
  }
  return UINT32_MAX;
}

void RingBufferManager::releaseSlot(uint32_t slotIndex) {
  if (slotIndex >= NUM_SLOTS) [[unlikely]] {
    return;
  }
  readIndex_.fetch_add(1, std::memory_order_relaxed);
  availableSlots_.fetch_add(1, std::memory_order_release);
}

} // namespace BTQuant