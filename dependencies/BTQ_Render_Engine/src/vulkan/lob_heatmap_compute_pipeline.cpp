#include "../../include/vulkan/lob_heatmap_compute_pipeline.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

// We need the SPIR-V data for the compute shader
#include "../../include/shader_spirv.hpp"

namespace BTQuant {

// Helper: find memory type (using MemoryPool's static helper)
static uint32_t find_mem_type(VkPhysicalDevice phys_dev, uint32_t type_filter,
                              VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type!");
}

void LobHeatmapComputePipeline::initialize(VkDevice device, VkPhysicalDevice physical_device,
                                           VkDescriptorPool pool, GPUMemoryManager& mem) {
  device_ = device;

  // ---- 1. Create compute shader module ----
  VkShaderModuleCreateInfo shader_ci{};
  shader_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_ci.codeSize = sizeof(LOB_HEATMAP_COMPUTE_SPIRV);
  shader_ci.pCode = LOB_HEATMAP_COMPUTE_SPIRV;

  VkShaderModule shader_module;
  if (vkCreateShaderModule(device, &shader_ci, nullptr, &shader_module) != VK_SUCCESS) {
    std::cerr << "[LobHeatmapComputePipeline] Failed to create shader module" << std::endl;
    return;
  }

  // ---- 2. Create descriptor set layout ----
  VkDescriptorSetLayoutBinding bindings[2]{};

  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layout_ci{};
  layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_ci.bindingCount = 2;
  layout_ci.pBindings = bindings;

  if (vkCreateDescriptorSetLayout(device, &layout_ci, nullptr, &descriptor_set_layout_) !=
      VK_SUCCESS) {
    std::cerr << "[LobHeatmapComputePipeline] Failed to create descriptor set layout" << std::endl;
    vkDestroyShaderModule(device, shader_module, nullptr);
    return;
  }

  // ---- 3. Create pipeline layout with push constants ----
  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = sizeof(HeatmapPushConstants);

  VkPipelineLayoutCreateInfo pipeline_layout_ci{};
  pipeline_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_ci.setLayoutCount = 1;
  pipeline_layout_ci.pSetLayouts = &descriptor_set_layout_;
  pipeline_layout_ci.pushConstantRangeCount = 1;
  pipeline_layout_ci.pPushConstantRanges = &push_range;

  if (vkCreatePipelineLayout(device, &pipeline_layout_ci, nullptr, &pipeline_layout_) !=
      VK_SUCCESS) {
    std::cerr << "[LobHeatmapComputePipeline] Failed to create pipeline layout" << std::endl;
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout_, nullptr);
    vkDestroyShaderModule(device, shader_module, nullptr);
    return;
  }

  // ---- 4. Create compute pipeline ----
  VkComputePipelineCreateInfo compute_ci{};
  compute_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  compute_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  compute_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  compute_ci.stage.module = shader_module;
  compute_ci.stage.pName = "main";
  compute_ci.layout = pipeline_layout_;

  if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_ci, nullptr, &pipeline_) !=
      VK_SUCCESS) {
    std::cerr << "[LobHeatmapComputePipeline] Failed to create compute pipeline" << std::endl;
    vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout_, nullptr);
    vkDestroyShaderModule(device, shader_module, nullptr);
    return;
  }

  // Shader module can be destroyed after pipeline creation
  vkDestroyShaderModule(device, shader_module, nullptr);

  // ---- 5. Create output image (1024x256 rgba16f) + view + sampler ----
  create_output_image(device, physical_device);

  // ---- 6. Register texture with GPUMemoryManager for ImGui access ----
  if (output_image_view_ != VK_NULL_HANDLE && output_sampler_ != VK_NULL_HANDLE) {
    imgui_texture_ =
        mem.add_texture(output_image_view_, output_sampler_, VK_IMAGE_LAYOUT_GENERAL);
  }

  // ---- 7. Allocate descriptor set ----
  VkDescriptorSetAllocateInfo desc_alloc{};
  desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  desc_alloc.descriptorPool = pool;
  desc_alloc.descriptorSetCount = 1;
  desc_alloc.pSetLayouts = &descriptor_set_layout_;

  if (vkAllocateDescriptorSets(device, &desc_alloc, &descriptor_set_) != VK_SUCCESS) {
    std::cerr << "[LobHeatmapComputePipeline] Failed to allocate descriptor set" << std::endl;
    return;
  }

  // Update image binding (binding 1) immediately since we have the image
  if (output_image_view_ != VK_NULL_HANDLE) {
    VkDescriptorImageInfo image_info{};
    image_info.imageView = output_image_view_;
    image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptor_set_;
    write.dstBinding = 1;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo = &image_info;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  }

  std::cout << "[LobHeatmapComputePipeline] Initialized successfully." << std::endl;
}

void LobHeatmapComputePipeline::create_output_image(VkDevice device,
                                                     VkPhysicalDevice physical_device) {
  // Create 1024x256 rgba16f storage image
  VkImageCreateInfo image_ci{};
  image_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_ci.imageType = VK_IMAGE_TYPE_2D;
  image_ci.extent.width = 1024;
  image_ci.extent.height = 256;
  image_ci.extent.depth = 1;
  image_ci.mipLevels = 1;
  image_ci.arrayLayers = 1;
  image_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  image_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_ci.samples = VK_SAMPLE_COUNT_1_BIT;

  if (vkCreateImage(device, &image_ci, nullptr, &output_image_) != VK_SUCCESS) {
    std::cerr << "[LobHeatmapComputePipeline] Failed to create output image" << std::endl;
    return;
  }

  VkMemoryRequirements mem_reqs;
  vkGetImageMemoryRequirements(device, output_image_, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = find_mem_type(physical_device, mem_reqs.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (vkAllocateMemory(device, &alloc_info, nullptr, &output_image_memory_) != VK_SUCCESS) {
    std::cerr << "[LobHeatmapComputePipeline] Failed to allocate output image memory" << std::endl;
    vkDestroyImage(device, output_image_, nullptr);
    output_image_ = VK_NULL_HANDLE;
    return;
  }

  vkBindImageMemory(device, output_image_, output_image_memory_, 0);

  // Create image view
  VkImageViewCreateInfo view_ci{};
  view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_ci.image = output_image_;
  view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_ci.subresourceRange.baseMipLevel = 0;
  view_ci.subresourceRange.levelCount = 1;
  view_ci.subresourceRange.baseArrayLayer = 0;
  view_ci.subresourceRange.layerCount = 1;

  if (vkCreateImageView(device, &view_ci, nullptr, &output_image_view_) != VK_SUCCESS) {
    std::cerr << "[LobHeatmapComputePipeline] Failed to create output image view" << std::endl;
    return;
  }

  // Create sampler
  VkSamplerCreateInfo sampler_ci{};
  sampler_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_ci.magFilter = VK_FILTER_LINEAR;
  sampler_ci.minFilter = VK_FILTER_LINEAR;
  sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_ci.anisotropyEnable = VK_FALSE;
  sampler_ci.maxAnisotropy = 1.0f;
  sampler_ci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_ci.unnormalizedCoordinates = VK_FALSE;
  sampler_ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

  if (vkCreateSampler(device, &sampler_ci, nullptr, &output_sampler_) != VK_SUCCESS) {
    std::cerr << "[LobHeatmapComputePipeline] Failed to create output sampler" << std::endl;
    return;
  }

  std::cout << "[LobHeatmapComputePipeline] Output image created (1024x256, rgba16f)." << std::endl;
}

void LobHeatmapComputePipeline::bind_ssbo(const BufferAllocation& ssbo_alloc) {
  ssbo_allocation_ = ssbo_alloc;

  if (descriptor_set_ == VK_NULL_HANDLE) return;

  // Update descriptor set with SSBO binding
  VkDescriptorBufferInfo buffer_info{};
  buffer_info.buffer = ssbo_alloc.buffer;
  buffer_info.offset = ssbo_alloc.offset;
  buffer_info.range = ssbo_alloc.size;

  VkWriteDescriptorSet writes[2]{};

  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = descriptor_set_;
  writes[0].dstBinding = 0;
  writes[0].dstArrayElement = 0;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].descriptorCount = 1;
  writes[0].pBufferInfo = &buffer_info;

  // Output image binding
  VkDescriptorImageInfo image_info{};
  image_info.imageView = output_image_view_;
  image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = descriptor_set_;
  writes[1].dstBinding = 1;
  writes[1].dstArrayElement = 0;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[1].descriptorCount = 1;
  writes[1].pImageInfo = &image_info;

  vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
}

void LobHeatmapComputePipeline::set_push_constants(float max_volume, float alpha) {
  push_constants_.max_volume = max_volume;
  push_constants_.alpha = alpha;
}

void LobHeatmapComputePipeline::transition_to_general(VkCommandBuffer cmd) {
  if (output_image_ == VK_NULL_HANDLE) return;

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = output_image_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void LobHeatmapComputePipeline::transition_to_shader_read(VkCommandBuffer cmd) {
  if (output_image_ == VK_NULL_HANDLE) return;

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = output_image_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void LobHeatmapComputePipeline::dispatch(VkCommandBuffer cmd, uint32_t columns, uint32_t rows) {
  if (pipeline_ == VK_NULL_HANDLE || descriptor_set_ == VK_NULL_HANDLE) return;

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                          0, 1, &descriptor_set_, 0, nullptr);
  vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(HeatmapPushConstants), &push_constants_);

  // Workgroup size is 16x16, so dispatch = (columns/16, rows/16, 1)
  uint32_t gx = (columns + 15) / 16;
  uint32_t gy = (rows + 15) / 16;
  vkCmdDispatch(cmd, gx, gy, 1);
}

void LobHeatmapComputePipeline::destroy(VkDevice device) {
  if (output_sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(device, output_sampler_, nullptr);
    output_sampler_ = VK_NULL_HANDLE;
  }
  if (output_image_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device, output_image_view_, nullptr);
    output_image_view_ = VK_NULL_HANDLE;
  }
  if (output_image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device, output_image_, nullptr);
    output_image_ = VK_NULL_HANDLE;
  }
  if (output_image_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device, output_image_memory_, nullptr);
    output_image_memory_ = VK_NULL_HANDLE;
  }
  if (pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(device, pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
  }
  if (pipeline_layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
    pipeline_layout_ = VK_NULL_HANDLE;
  }
  if (descriptor_set_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout_, nullptr);
    descriptor_set_layout_ = VK_NULL_HANDLE;
  }
}

}  // namespace BTQuant
