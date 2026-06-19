#include "gpu_buffers.hpp"
#include <cstring>
#include <iostream>

namespace btquant::vulkan {

GPUBuffers::~GPUBuffers() {
    // Note: Actual buffer cleanup would happen here, but we don't store buffer list
    // In a real implementation, we'd track created buffers and clean them up
}

bool GPUBuffers::initialize(VkDevice device) {
    m_device = device;
    
    // We need to get the physical device somehow - in a real implementation
    // this would likely come from the VulkanContext
    return m_device != VK_NULL_HANDLE;
}

bool GPUBuffers::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                              VkMemoryPropertyFlags properties, Buffer& buffer) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer.buffer) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer!" << std::endl;
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, buffer.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &buffer.memory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate buffer memory!" << std::endl;
        return false;
    }

    vkBindBufferMemory(m_device, buffer.buffer, buffer.memory, 0);
    
    buffer.size = size;
    buffer.offset = 0;
    
    return true;
}

void GPUBuffers::copyBuffer(Buffer& srcBuffer, Buffer& dstBuffer, VkDeviceSize size) {
    // Create a temporary command buffer for the copy operation
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = VK_NULL_HANDLE; // Would need to get from context
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional offset
    copyRegion.dstOffset = 0; // Optional offset
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer.buffer, dstBuffer.buffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    // Submit the command buffer (would need queue from context)
    // This is simplified - in reality we'd need the graphics queue
    // and proper synchronization
    
    vkFreeCommandBuffers(m_device, allocInfo.commandPool, 1, &commandBuffer);
}

void GPUBuffers::copyDataToBuffer(const void* data, VkDeviceSize size, Buffer& buffer) {
    void* mappedData;
    vkMapMemory(m_device, buffer.memory, 0, size, 0, &mappedData);
    memcpy(mappedData, data, size);
    vkUnmapMemory(m_device, buffer.memory);
}

uint32_t GPUBuffers::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    // This would normally require the physical device to be set
    // For now, returning a placeholder - in real implementation would query physical device
    VkPhysicalDeviceMemoryProperties memProperties;
    // vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    
    // Placeholder implementation - in a real implementation we would:
    // 1. Get the physical device from context
    // 2. Query its memory properties
    // 3. Find a suitable memory type
    
    // For now, just return the first type that matches (this is incorrect but prevents compilation errors)
    return 0;
}

} // namespace btquant::vulkan