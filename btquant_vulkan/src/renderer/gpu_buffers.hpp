#ifndef BTQUANT_GPU_BUFFERS_HPP
#define BTQUANT_GPU_BUFFERS_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

namespace btquant::vulkan {

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize offset = 0;
};

class GPUBuffers {
public:
    GPUBuffers() = default;
    ~GPUBuffers();
    
    // Initialize buffer management
    bool initialize(VkDevice device);
    
    // Create a buffer
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                      VkMemoryPropertyFlags properties, Buffer& buffer);
    
    // Copy data to buffer
    void copyBuffer(Buffer& srcBuffer, Buffer& dstBuffer, VkDeviceSize size);
    
    // Copy data from host to device
    void copyDataToBuffer(const void* data, VkDeviceSize size, Buffer& buffer);
    
    // Getters
    VkDevice getDevice() const { return m_device; }
    
private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    
    // Helper function to find memory type
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

} // namespace btquant::vulkan

#endif