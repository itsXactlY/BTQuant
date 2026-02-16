/**
 * @file ssbo_snapshot_updater.cpp
 * @brief Vulkan SSBO Buffer Updater for OrderBookSnapshot with std430 alignment
 * 
 * Implementation file for CPU-side structures that mirror the GPU SSBO layout
 * defined in lob_heatmap.comp. All structures enforce std430 alignment rules
 * to ensure perfect mapping to vec4 arrays in the compute shader.
 */

#include "ssbo_snapshot_updater.hpp"
#include <cstring>

// ============================================================================
// SSBOBuffer Implementation
// ============================================================================

SSBOBuffer::~SSBOBuffer() {
    cleanup();
}

bool SSBOBuffer::create(VkDevice device, size_t numLevels, VkBufferUsageFlags usage) {
    device_ = device;
    bufferSize_ = OrderBookSnapshot::calculateBufferSize(numLevels);
    numLevels_ = numLevels;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize_;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer_) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool SSBOBuffer::bindMemory(VkDeviceMemory deviceMemory, VkDeviceSize memoryOffset) {
    if (vkBindBufferMemory(device_, buffer_, deviceMemory, memoryOffset) != VK_SUCCESS) {
        return false;
    }
    memory_ = deviceMemory;
    return true;
}

bool SSBOBuffer::writeData(const OrderBookSnapshot& snapshot, const OrderBookLevel* levels) {
    void* mappedData = nullptr;
    if (vkMapMemory(device_, memory_, 0, bufferSize_, 0, &mappedData) != VK_SUCCESS) {
        return false;
    }

    // Copy header
    std::memcpy(mappedData, &snapshot, sizeof(OrderBookSnapshot));

    // Copy levels array
    if (levels && snapshot.priceLevelsCount > 0) {
        auto* destLevels = reinterpret_cast<OrderBookLevel*>(
            reinterpret_cast<uint8_t*>(mappedData) + sizeof(OrderBookSnapshot));
        std::memcpy(destLevels, levels, snapshot.priceLevelsCount * sizeof(OrderBookLevel));
    }

    vkUnmapMemory(device_, memory_);
    return true;
}

VkDescriptorBufferInfo SSBOBuffer::getDescriptorInfo() const {
    VkDescriptorBufferInfo info{};
    info.buffer = buffer_;
    info.offset = 0;
    info.range = bufferSize_;
    return info;
}

void SSBOBuffer::cleanup() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
    }
    // Note: Memory is typically freed by the caller who allocated it
    memory_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    bufferSize_ = 0;
    numLevels_ = 0;
}

// ============================================================================
// Helper Functions
// ============================================================================

bool createSSBOBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                      size_t numLevels, VkMemoryPropertyFlags memoryFlags,
                      SSBOBuffer& outBuffer, VkDeviceMemory& outMemory) {
    if (!outBuffer.create(device, numLevels)) {
        return false;
    }

    // Get memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, outBuffer.getBuffer(), &memRequirements);

    // Find suitable memory type
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    uint32_t memoryTypeIndex = 0;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & memoryFlags) == memoryFlags) {
            memoryTypeIndex = i;
            break;
        }
    }

    allocInfo.memoryTypeIndex = memoryTypeIndex;

    if (vkAllocateMemory(device, &allocInfo, nullptr, &outMemory) != VK_SUCCESS) {
        return false;
    }

    if (!outBuffer.bindMemory(outMemory)) {
        vkFreeMemory(device, outMemory, nullptr);
        return false;
    }

    return true;
}
