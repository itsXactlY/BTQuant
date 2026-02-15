#include <algorithm>
#include <iostream>
#include <mutex>  // For std::lock_guard
#include <stdexcept>
#include <vector>
#include <cstring> // For memcpy

#include "../../include/vulkan_base_types.hpp"

namespace BTQuant {

// Helper function to find memory type (could be a static method of MemoryPool
// or global)
uint32_t MemoryPool::find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter,
                                      VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

MemoryPool::MemoryPool(VkDevice device, VkPhysicalDevice physical_device, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags properties, VkDeviceSize pool_size)
    : device_(device),
      physical_device_(physical_device),
      usage_(usage),
      properties_(properties),
      pool_size_(pool_size),
      used_size_(0) {
  if (device == VK_NULL_HANDLE) {
    throw std::runtime_error("Cannot create MemoryPool with VK_NULL_HANDLE device");
  }

  // Create Pool Buffer
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = pool_size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &pool_buffer_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pool buffer!");
  }

  // Allocate Memory
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, pool_buffer_, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      find_memory_type(physical_device, memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &pool_memory_) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate pool memory!");
  }

  vkBindBufferMemory(device, pool_buffer_, pool_memory_, 0);

  if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    if (vkMapMemory(device, pool_memory_, 0, pool_size, 0, &mapped_ptr_) != VK_SUCCESS) {
      throw std::runtime_error("failed to map pool memory!");
    }
  } else {
    mapped_ptr_ = nullptr;
  }

  free_blocks_.push_back({0, pool_size});
}

MemoryPool::~MemoryPool() {
  // Wait for device to be idle before destroying resources
  vkDeviceWaitIdle(device_);
  
  if (mapped_ptr_) {
    vkUnmapMemory(device_, pool_memory_);
    mapped_ptr_ = nullptr;
  }
  
  // Clean up any additional buffers that were created when the pool was exhausted
  for (auto buffer : cleanup_buffers_) {
    if (buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, buffer, nullptr);
    }
  }
  cleanup_buffers_.clear();
  
  for (auto memory : cleanup_memories_) {
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(device_, memory, nullptr);
    }
  }
  cleanup_memories_.clear();
  
  // Clean up the main pool buffer and memory
  if (pool_buffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, pool_buffer_, nullptr);
    pool_buffer_ = VK_NULL_HANDLE;
  }
  if (pool_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, pool_memory_, nullptr);
    pool_memory_ = VK_NULL_HANDLE;
  }
}

BufferAllocation MemoryPool::allocate(VkDeviceSize size, VkDeviceSize alignment) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);

  for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
    VkDeviceSize aligned_offset = (it->offset + alignment - 1) & ~(alignment - 1);
    VkDeviceSize padding = aligned_offset - it->offset;

    if (it->size >= size + padding) {
      BufferAllocation alloc{};
      alloc.buffer = pool_buffer_;
      alloc.memory = pool_memory_;
      alloc.offset = aligned_offset;
      alloc.size = size;
      alloc.mapped_ptr = mapped_ptr_ ? static_cast<char*>(mapped_ptr_) + aligned_offset : nullptr;

      // Update free blocks
      VkDeviceSize remaining_size_after_alloc = it->size - (size + padding);
      if (remaining_size_after_alloc > 0) {
        it->offset = aligned_offset + size;
        it->size = remaining_size_after_alloc;
      } else {
        free_blocks_.erase(it);
      }

      // If there's padding, create a new free block for it
      if (padding > 0) {
        free_blocks_.push_back({it->offset, padding});
      }

      used_size_ += size;
      return alloc;
    }
  }

  // If no suitable block found, try to allocate a new buffer
  // This is a fallback mechanism for when the pool is exhausted
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage_;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer newBuffer;
  if (vkCreateBuffer(device_, &bufferInfo, nullptr, &newBuffer) != VK_SUCCESS) {
    return BufferAllocation{};  // Failed to create buffer
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device_, newBuffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      find_memory_type(physical_device_, memRequirements.memoryTypeBits, properties_);

  VkDeviceMemory newMemory;
  if (vkAllocateMemory(device_, &allocInfo, nullptr, &newMemory) != VK_SUCCESS) {
    vkDestroyBuffer(device_, newBuffer, nullptr);
    return BufferAllocation{};  // Failed to allocate memory
  }

  if (vkBindBufferMemory(device_, newBuffer, newMemory, 0) != VK_SUCCESS) {
    vkDestroyBuffer(device_, newBuffer, nullptr);
    vkFreeMemory(device_, newMemory, nullptr);
    return BufferAllocation{};  // Failed to bind memory
  }

  // Create a new allocation for the new buffer
  BufferAllocation alloc{};
  alloc.buffer = newBuffer;
  alloc.memory = newMemory;
  alloc.offset = 0;
  alloc.size = size;
  alloc.mapped_ptr = nullptr;  // Not mapped by default

  // Add to cleanup list
  cleanup_buffers_.push_back(newBuffer);
  cleanup_memories_.push_back(newMemory);

  used_size_ += size;
  return alloc;
}

void MemoryPool::deallocate(const BufferAllocation& allocation) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  // Simple deallocation (append to free blocks, no merging for now)
  // A more robust allocator would merge adjacent free blocks.
  free_blocks_.push_back({allocation.offset, allocation.size});
  used_size_ -= allocation.size;
}

GPUMemoryManager::GPUMemoryManager(VkDevice device, VkPhysicalDevice physical_device,
                                   const VulkanDashboardConfig& config)
    : device_(device), physical_device_(physical_device) {
  if (device == VK_NULL_HANDLE) {
    throw std::runtime_error("GPUMemoryManager initialized with VK_NULL_HANDLE device");
  }
  // Initialize memory pools
  vertex_pool_ = std::make_unique<MemoryPool>(
      device, physical_device,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, config.vertex_pool_size);

  uniform_pool_ = std::make_unique<MemoryPool>(
      device, physical_device,
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      config.uniform_pool_size);

  storage_pool_ = std::make_unique<MemoryPool>(
      device, physical_device,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, config.storage_pool_size);
}

GPUMemoryManager::~GPUMemoryManager() {
  // Smart pointers will handle cleanup of pools
  // The destructors of the unique_ptr objects will be called automatically
  // which will clean up the individual memory pools
}

BufferAllocation GPUMemoryManager::allocate_vertex_buffer(VkDeviceSize size) {
  return vertex_pool_->allocate(size, 256);  // Assuming a common alignment for vertex buffers
}

BufferAllocation GPUMemoryManager::allocate_index_buffer(VkDeviceSize size) {
  return vertex_pool_->allocate(size, 256);  // Index buffers can often share the vertex pool
}

BufferAllocation GPUMemoryManager::allocate_uniform_buffer(VkDeviceSize size) {
  // Uniform buffers have specific alignment requirements, typically 256 bytes
  return uniform_pool_->allocate(size, 256);
}

BufferAllocation GPUMemoryManager::allocate_storage_buffer(VkDeviceSize size) {
  return storage_pool_->allocate(size, 256);  // Assuming a common alignment for storage buffers
}

BufferAllocation GPUMemoryManager::allocate_staging_buffer(VkDeviceSize size) {
  // Staging buffers are typically host visible and coherent, uniform pool is
  // suitable
  return uniform_pool_->allocate(size, 256);
}

void GPUMemoryManager::deallocate_buffer(const BufferAllocation& allocation) {
  // Determine which pool the allocation came from and deallocate
  // This requires tracking the origin of the allocation, or checking buffer
  // handles. For simplicity, we'll assume we know which pool it belongs to
  // based on its type. A more robust system would store a pool ID in
  // BufferAllocation.
  if (allocation.buffer == vertex_pool_->get_pool_buffer()) {
    vertex_pool_->deallocate(allocation);
  } else if (allocation.buffer == uniform_pool_->get_pool_buffer()) {
    uniform_pool_->deallocate(allocation);
  } else if (allocation.buffer == storage_pool_->get_pool_buffer()) {
    storage_pool_->deallocate(allocation);
  } else {
    // This should not happen if all allocations come from these pools
    std::cerr << "Warning: Deallocating buffer not recognized by any pool." << std::endl;
  }
}

GPUMemoryManager::MemoryStats GPUMemoryManager::get_memory_stats() const {
  MemoryStats stats{};
  stats.vertex_pool_used = vertex_pool_->get_used_size();
  stats.uniform_pool_used = uniform_pool_->get_used_size();
  stats.storage_pool_used = storage_pool_->get_used_size();
  stats.vertex_pool_usage = vertex_pool_->get_usage_percentage();
  stats.uniform_pool_usage = uniform_pool_->get_usage_percentage();
  stats.storage_pool_usage = storage_pool_->get_usage_percentage();
  return stats;
}

ImageAllocation GPUMemoryManager::allocate_image(uint32_t width, uint32_t height, VkFormat format, 
                                               VkImageTiling tiling, VkImageUsageFlags usage, 
                                               VkMemoryPropertyFlags properties, uint32_t mip_levels) {
  ImageAllocation allocation{};
  
  // Create the image
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = mip_levels;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.flags = 0;

  if (vkCreateImage(device_, &imageInfo, nullptr, &allocation.image) != VK_SUCCESS) {
    throw std::runtime_error("failed to create image!");
  }

  // Get memory requirements
  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device_, allocation.image, &memRequirements);

  // Allocate memory
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device_, &allocInfo, nullptr, &allocation.memory) != VK_SUCCESS) {
    vkDestroyImage(device_, allocation.image, nullptr);
    throw std::runtime_error("failed to allocate image memory!");
  }

  // Bind memory to image
  vkBindImageMemory(device_, allocation.image, allocation.memory, 0);

  // Create image view
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = allocation.image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = mip_levels;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(device_, &viewInfo, nullptr, &allocation.view) != VK_SUCCESS) {
    vkDestroyImage(device_, allocation.image, nullptr);
    vkFreeMemory(device_, allocation.memory, nullptr);
    throw std::runtime_error("failed to create image view!");
  }

  // Store additional information
  allocation.size = memRequirements.size;
  allocation.format = format;
  allocation.width = width;
  allocation.height = height;
  allocation.mip_levels = mip_levels;
  allocation.usage = usage;

  return allocation;
}

void GPUMemoryManager::deallocate_image(const ImageAllocation& allocation) {
  if (allocation.view != VK_NULL_HANDLE) {
    vkDestroyImageView(device_, allocation.view, nullptr);
  }
  if (allocation.image != VK_NULL_HANDLE) {
    vkDestroyImage(device_, allocation.image, nullptr);
  }
  if (allocation.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device_, allocation.memory, nullptr);
  }
}

uint32_t GPUMemoryManager::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

// Note: The actual implementation of texture transfers would require access to command buffers
// and is typically handled by the rendering system (VulkanCore). The GPUMemoryManager focuses
// on allocation/deallocation. For a complete implementation, these methods would need to be
// called from the rendering system with proper command buffer access.

ImageAllocation GPUMemoryManager::create_texture_atlas(const std::vector<std::vector<uint8_t>>& icon_data, 
                                                       uint32_t icon_width, uint32_t icon_height, 
                                                       uint32_t cols, uint32_t rows) {
  uint32_t atlas_width = icon_width * cols;
  uint32_t atlas_height = icon_height * rows;
  
  // Calculate total size needed for the atlas (assuming RGBA format)
  VkDeviceSize image_size = atlas_width * atlas_height * 4; // 4 bytes per pixel (RGBA)

  // Create staging buffer to transfer image data
  BufferAllocation staging_buffer = allocate_staging_buffer(image_size);

  // Copy icon data to staging buffer in atlas layout
  uint8_t* data_ptr = static_cast<uint8_t*>(staging_buffer.mapped_ptr);
  
  // Initialize the entire atlas to transparent black
  memset(data_ptr, 0, image_size);
  
  // Place each icon in its grid position
  for (size_t i = 0; i < icon_data.size(); ++i) {
    if (i >= cols * rows) break; // Don't exceed atlas capacity
    
    uint32_t col = i % cols;
    uint32_t row = i / cols;
    
    uint32_t dest_x = col * icon_width;
    uint32_t dest_y = row * icon_height;
    
    // Copy each row of the icon to the appropriate position in the atlas
    for (uint32_t y = 0; y < icon_height; ++y) {
      uint32_t src_offset = y * icon_width * 4; // 4 bytes per pixel
      uint32_t dst_row_start = ((dest_y + y) * atlas_width + dest_x) * 4;
      
      if (src_offset + (icon_width * 4) <= icon_data[i].size()) {
        memcpy(&data_ptr[dst_row_start], &icon_data[i][src_offset], icon_width * 4);
      }
    }
  }

  // Create the final image in GPU memory
  ImageAllocation atlas_allocation = allocate_image(
      atlas_width, atlas_height,
      VK_FORMAT_R8G8B8A8_UNORM,  // Standard RGBA format
      VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,  // Can receive transfers and be sampled
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  );

  // The actual image transfer would happen in the rendering system using command buffers
  // GPUMemoryManager handles allocation, the rendering system handles the transfer

  // Clean up staging buffer
  deallocate_buffer(staging_buffer);

  return atlas_allocation;
}

void GPUMemoryManager::update_texture_atlas(const ImageAllocation& atlas, 
                                           const std::vector<std::vector<uint8_t>>& icon_data,
                                           uint32_t x_offset, uint32_t y_offset,
                                           uint32_t width, uint32_t height) {
  // This method would be called from the rendering system with proper command buffer access
  // For now, it serves as a placeholder for the intended functionality
}

VkResult GPUMemoryManager::copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
  // This method would be called from the rendering system with proper command buffer access
  // For now, it serves as a placeholder for the intended functionality
  return VK_SUCCESS;
}

}  // namespace BTQuant
