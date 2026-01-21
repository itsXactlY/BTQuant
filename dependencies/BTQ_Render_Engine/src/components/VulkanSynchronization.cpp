/**
 * @file VulkanSynchronization.cpp
 * @brief Vulkan Synchronization Strategy Implementation
 * 
 * Provides concrete implementation of the synchronization primitives for
 * zero-latency rendering with Hotspine data.
 * 
 * @author Market Microstructure Renderer Team
 * @version 1.0.0
 */

#include "../../include/components/VulkanSynchronization.h"
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <thread>

namespace BTQuant {

// ============================================
// TimelineSemaphore Implementation
// ============================================

TimelineSemaphore::TimelineSemaphore(VkDevice device)
    : device_(device) {
    
    VkSemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue = 0;
    
    VkSemaphoreCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = &timelineInfo;
    
    if (vkCreateSemaphore(device_, &createInfo, nullptr, &semaphore_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create timeline semaphore");
    }
}

TimelineSemaphore::~TimelineSemaphore() {
    if (semaphore_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_, semaphore_, nullptr);
    }
}

TimelineSemaphore::TimelineSemaphore(TimelineSemaphore&& other) noexcept
    : device_(other.device_)
    , semaphore_(other.semaphore_) {
    
    other.device_ = VK_NULL_HANDLE;
    other.semaphore_ = VK_NULL_HANDLE;
}

TimelineSemaphore& TimelineSemaphore::operator=(TimelineSemaphore&& other) noexcept {
    if (this != &other) {
        if (semaphore_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, semaphore_, nullptr);
        }
        
        device_ = other.device_;
        semaphore_ = other.semaphore_;
        
        other.device_ = VK_NULL_HANDLE;
        other.semaphore_ = VK_NULL_HANDLE;
    }
    return *this;
}

uint64_t TimelineSemaphore::getCurrentValue() const {
    uint64_t value = 0;
    if (vkGetSemaphoreCounterValue(device_, semaphore_, &value) != VK_SUCCESS) {
        return 0;
    }
    return value;
}

bool TimelineSemaphore::waitForValue(uint64_t value, uint64_t timeoutNs) const {
    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &semaphore_;
    waitInfo.pValues = &value;
    
    const VkResult result = vkWaitSemaphores(device_, &waitInfo, timeoutNs);
    return result == VK_SUCCESS;
}

bool TimelineSemaphore::signalValue(uint64_t value) {
    VkSemaphoreSignalInfo signalInfo{};
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signalInfo.semaphore = semaphore_;
    signalInfo.value = value;
    
    return vkSignalSemaphore(device_, &signalInfo) == VK_SUCCESS;
}

// ============================================
// FenceManager Implementation
// ============================================

FenceManager::FenceManager(VkDevice device, uint32_t maxFrames)
    : device_(device)
    , maxFrames_(maxFrames)
    , used_(maxFrames, false) {
    
    fences_.reserve(maxFrames);
    
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Initially signaled
    
    for (uint32_t i = 0; i < maxFrames; ++i) {
        VkFence fence;
        if (vkCreateFence(device_, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            for (auto& createdFence : fences_) {
                vkDestroyFence(device_, createdFence, nullptr);
            }
            throw std::runtime_error("Failed to create fence manager");
        }
        fences_.push_back(fence);
    }
}

FenceManager::~FenceManager() {
    waitAll();
    for (auto& fence : fences_) {
        vkDestroyFence(device_, fence, nullptr);
    }
}

int32_t FenceManager::acquireFence() {
    for (uint32_t i = 0; i < maxFrames_; ++i) {
        if (!used_[i]) {
            used_[i] = true;
            vkResetFences(device_, 1, &fences_[i]);
            return static_cast<int32_t>(i);
        }
        
        // Check if fence has completed
        VkResult result = vkGetFenceStatus(device_, fences_[i]);
        if (result == VK_SUCCESS) {
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
    : device_(device)
    , pool_(pool) {
}

CommandBufferPool::~CommandBufferPool() {
    for (auto cmdBuffer : freeBuffers_) {
        vkFreeCommandBuffers(device_, pool_, 1, &cmdBuffer);
    }
    for (auto cmdBuffer : activeBuffers_) {
        vkFreeCommandBuffers(device_, pool_, 1, &cmdBuffer);
    }
}

VkCommandBuffer CommandBufferPool::acquireCommandBuffer(bool isPrimary) {
    if (freeBuffers_.empty()) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = pool_;
        allocInfo.level = isPrimary ? VK_COMMAND_BUFFER_LEVEL_PRIMARY 
                                   : VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        allocInfo.commandBufferCount = 1;
        
        VkCommandBuffer cmdBuffer;
        if (vkAllocateCommandBuffers(device_, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
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
    
    // Create semaphores for each frame
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    for (auto& state : frameStates_) {
        if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                           &state.imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                           &state.renderFinishedSemaphore) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create render pass semaphores");
        }
        
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        
        if (vkCreateFence(device_, &fenceInfo, nullptr, &state.inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create in-flight fence");
        }
    }
}

VulkanSyncContext::~VulkanSyncContext() {
    for (const auto& state : frameStates_) {
        vkDestroySemaphore(device_, state.imageAvailableSemaphore, nullptr);
        vkDestroySemaphore(device_, state.renderFinishedSemaphore, nullptr);
        vkDestroyFence(device_, state.inFlightFence, nullptr);
    }
}

bool VulkanSyncContext::prepareFrame(uint32_t currentFrame, TimelineSemaphore& timelineSemaphore) {
    auto& state = frameStates_[currentFrame % MAX_FRAMES_IN_FLIGHT];
    
    // Wait for previous frame to complete
    if (!state.isFirstFrame) {
        const auto waitResult = vkWaitForFences(device_, 1, &state.inFlightFence,
                                              VK_TRUE, 1'000'000'000);
        
        if (waitResult != VK_SUCCESS) {
            std::cerr << "Frame preparation timeout" << std::endl;
            return false;
        }
        
        // Reset the fence for current frame
        vkResetFences(device_, 1, &state.inFlightFence);
    }
    
    state.isFirstFrame = false;
    return true;
}

bool VulkanSyncContext::submitCommandBuffer(VkQueue queue, VkCommandBuffer cmdBuffer,
                                           TimelineSemaphore& timelineSemaphore,
                                           uint64_t signalValue,
                                           uint64_t waitValue) {

    // Simple submission without timeline for now
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pCommandBuffers = &cmdBuffer;
    submitInfo.commandBufferCount = 1;

    const auto acquireFence = fenceManager_->acquireFence();
    if (acquireFence == -1) {
        std::cerr << "No available fences for submission" << std::endl;
        return false;
    }

    const VkResult result = vkQueueSubmit(queue, 1, &submitInfo,
                                          fenceManager_->getFence(static_cast<uint32_t>(acquireFence)));

    if (result != VK_SUCCESS) {
        std::cerr << "Failed to submit command buffer: " << result << std::endl;
        fenceManager_->releaseSlot(static_cast<uint32_t>(acquireFence));
        return false;
    }

    return true;
}

void VulkanSyncContext::waitForCompletion() const {
    fenceManager_->waitAll();
}

void VulkanSyncContext::reset() const {
    fenceManager_->resetAll();
}

// ============================================
// HotspineBarrierManager Implementation
// ============================================

HotspineBarrierManager::HotspineBarrierManager(VkDevice device)
    : device_(device) {
}

VkMemoryBarrier HotspineBarrierManager::createBufferMemoryBarrier(VkPipelineStageFlags srcStage,
                                                                VkPipelineStageFlags dstStage) const {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    return barrier;
}

VkBufferMemoryBarrier HotspineBarrierManager::createSSBOBufferBarrier(VkBuffer buffer,
                                                                     VkDeviceSize offset,
                                                                     VkDeviceSize size) const {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = offset;
    barrier.size = size;
    return barrier;
}

VkImageMemoryBarrier HotspineBarrierManager::createHeatmapImageBarrier(VkImage image,
                                                                      VkImageLayout oldLayout,
                                                                      VkImageLayout newLayout) const {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        barrier.srcAccessMask = 0;
    } else {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    }
    
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    return barrier;
}

void HotspineBarrierManager::recordHotspineUpdateBarrier(VkCommandBuffer cmdBuffer,
                                                       VkBuffer buffer,
                                                       VkDeviceSize offset,
                                                       VkDeviceSize size) const {
    const auto bufferBarrier = createSSBOBufferBarrier(buffer, offset, size);
    
    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        1, &bufferBarrier,
        0, nullptr
    );
}

// ============================================
// RingBufferSyncManager Implementation
// ============================================

RingBufferSyncManager::RingBufferSyncManager(VkDevice device,
                                           uint32_t slotCount,
                                           VkDeviceSize slotSize)
    : device_(device)
    , slotCount_(slotCount)
    , slotSize_(slotSize)
    , slots_(slotCount) {
    
    // Initialize slots as available
    for (uint32_t i = 0; i < slotCount; ++i) {
        slots_[i].isAvailable = true;
        slots_[i].lastUsed = 0;
    }
}

RingBufferSyncManager::~RingBufferSyncManager() {
    // Wait for all slots to complete
    waitForSlot(UINT64_MAX);
}

int32_t RingBufferSyncManager::acquireSlot() {
    const auto currentTime = currentTime_++;
    
    for (uint32_t i = 0; i < slotCount_; ++i) {
        const uint32_t index = (currentIndex_ + i) % slotCount_;
        auto& slot = slots_[index];
        
        if (slot.isAvailable) {
            slot.isAvailable = false;
            slot.lastUsed = currentTime;
            currentIndex_ = index;
            return static_cast<int32_t>(index);
        }
    }
    
    return -1;
}

void RingBufferSyncManager::releaseSlot(uint32_t slotIndex) {
    assert(slotIndex < slotCount_);
    slots_[slotIndex].isAvailable = true;
}

int32_t RingBufferSyncManager::waitForSlot(uint64_t timeoutNs) {
    const auto startTime = std::chrono::high_resolution_clock::now();
    
    while (true) {
        const auto slotIndex = acquireSlot();
        if (slotIndex != -1) {
            return slotIndex;
        }
        
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - startTime
        ).count();
        
        if (elapsed >= timeoutNs) {
            return -1;
        }
        
        // Sleep briefly to yield CPU time
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

// ============================================
// SyncDebugUtils Implementation
// ============================================

uint64_t SyncDebugUtils::measureExecutionTime(VkDevice device,
                                             VkCommandBuffer cmdBuffer,
                                             VkQueue queue,
                                             const char* name) {
    // This is a placeholder - real implementation would use timestamp queries
    // and a high-resolution timer
    return 0;
}

void SyncDebugUtils::logSyncState(const VulkanSyncContext& context) {
    // Placeholder for sync state logging
    std::cout << "Sync context state logged" << std::endl;
}

} // namespace BTQuant