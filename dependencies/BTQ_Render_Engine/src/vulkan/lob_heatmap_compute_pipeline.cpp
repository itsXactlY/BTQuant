/// @file lob_heatmap_compute_pipeline.cpp
/// @brief Implementation of the LOB Heatmap Vulkan compute pipeline.

#include "vulkan/lob_heatmap_compute_pipeline.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace BTQuant {

// ============================================================================
// Helper: find_memory_type
// ============================================================================
uint32_t LobHeatmapComputePipeline::find_memory_type(VkPhysicalDevice physical_device,
                                                     uint32_t type_filter,
                                                     VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  return UINT32_MAX;
}

// ============================================================================
// initialize
// ============================================================================
bool LobHeatmapComputePipeline::initialize(VkDevice device, VkPhysicalDevice physical_device,
                                           VkDescriptorPool descriptor_pool) {
  if (!create_output_image(device, physical_device)) return false;
  if (!create_sampler(device)) return false;
  if (!create_descriptor_resources(device, descriptor_pool)) return false;
  if (!create_pipeline(device)) return false;
  return true;
}

// ============================================================================
// Output Image: VK_FORMAT_R16G16B16A16_SFLOAT, 1024×256
// ============================================================================
bool LobHeatmapComputePipeline::create_output_image(VkDevice device,
                                                    VkPhysicalDevice physical_device) {
  // Verify format supports required features for storage image and sampled image
  VkFormatProperties format_props{};
  vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16B16A16_SFLOAT, &format_props);
  
  const VkFormatFeatureFlags required_features = 
      VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
  
  if ((format_props.optimalTilingFeatures & required_features) != required_features) {
    std::cerr << "[LobHeatmapComputePipeline] VK_FORMAT_R16G16B16A16_SFLOAT does not support "
              << "required storage/sampled image features" << std::endl;
    return false;
  }

  VkImageCreateInfo img_ci{};
  img_ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  img_ci.imageType = VK_IMAGE_TYPE_2D;
  img_ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  img_ci.extent = {HEATMAP_WIDTH, HEATMAP_HEIGHT, 1};
  img_ci.mipLevels = 1;
  img_ci.arrayLayers = 1;
  img_ci.samples = VK_SAMPLE_COUNT_1_BIT;
  img_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  img_ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  img_ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  img_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  if (vkCreateImage(device, &img_ci, nullptr, &output_image_) != VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements mem_reqs{};
  vkGetImageMemoryRequirements(device, output_image_, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = find_memory_type(physical_device, mem_reqs.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (alloc_info.memoryTypeIndex == UINT32_MAX) return false;
  if (vkAllocateMemory(device, &alloc_info, nullptr, &output_memory_) != VK_SUCCESS) {
    return false;
  }
  if (vkBindImageMemory(device, output_image_, output_memory_, 0) != VK_SUCCESS) {
    return false;
  }

  // Image view
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

  return vkCreateImageView(device, &view_ci, nullptr, &output_view_) == VK_SUCCESS;
}

// ============================================================================
// Sampler: Nearest-neighbor for pixel-perfect heatmap
// ============================================================================
bool LobHeatmapComputePipeline::create_sampler(VkDevice device) {
  VkSamplerCreateInfo sampler_ci{};
  sampler_ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_ci.magFilter = VK_FILTER_NEAREST;
  sampler_ci.minFilter = VK_FILTER_NEAREST;
  sampler_ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_ci.maxLod = 0.0f;

  return vkCreateSampler(device, &sampler_ci, nullptr, &sampler_) == VK_SUCCESS;
}

// ============================================================================
// Descriptor Set Layout + Descriptor Set
// ============================================================================
bool LobHeatmapComputePipeline::create_descriptor_resources(VkDevice device,
                                                            VkDescriptorPool pool) {
  // Binding 0: SSBO (storage buffer) — VolumeNode array
  // Binding 1: Output image (storage image) — heatmap
  VkDescriptorSetLayoutBinding bindings[2]{};

  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo ds_layout_ci{};
  ds_layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ds_layout_ci.bindingCount = 2;
  ds_layout_ci.pBindings = bindings;

  if (vkCreateDescriptorSetLayout(device, &ds_layout_ci, nullptr, &ds_layout_) != VK_SUCCESS) {
    return false;
  }

  VkDescriptorSetAllocateInfo ds_alloc{};
  ds_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ds_alloc.descriptorPool = pool;
  ds_alloc.descriptorSetCount = 1;
  ds_alloc.pSetLayouts = &ds_layout_;

  if (vkAllocateDescriptorSets(device, &ds_alloc, &descriptor_set_) != VK_SUCCESS) {
    return false;
  }

  // Write the output image descriptor immediately (it's constant)
  VkDescriptorImageInfo image_info{};
  image_info.imageView = output_view_;
  image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  VkWriteDescriptorSet image_write{};
  image_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  image_write.dstSet = descriptor_set_;
  image_write.dstBinding = 1;
  image_write.dstArrayElement = 0;
  image_write.descriptorCount = 1;
  image_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  image_write.pImageInfo = &image_info;

  vkUpdateDescriptorSets(device, 1, &image_write, 0, nullptr);
  return true;
}

// ============================================================================
// Compute Pipeline
// ============================================================================
bool LobHeatmapComputePipeline::load_shader_module(VkDevice device, VkShaderModule& module) {
  // Load compiled SPIR-V from the spirv directory
  std::ifstream file("shaders/spirv/lob_heatmap.comp.spv", std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    // Try alternative path
    file.open("../shaders/spirv/lob_heatmap.comp.spv", std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
  }

  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> spirv(file_size / sizeof(uint32_t));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(spirv.data()), file_size);
  file.close();

  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = file_size;
  ci.pCode = spirv.data();

  return vkCreateShaderModule(device, &ci, nullptr, &module) == VK_SUCCESS;
}

bool LobHeatmapComputePipeline::create_pipeline(VkDevice device) {
  VkShaderModule shader_module = VK_NULL_HANDLE;
  if (!load_shader_module(device, shader_module)) return false;

  // Push constant range: 2 floats = 8 bytes
  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = sizeof(HeatmapPushConstants);

  VkPipelineLayoutCreateInfo layout_ci{};
  layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_ci.setLayoutCount = 1;
  layout_ci.pSetLayouts = &ds_layout_;
  layout_ci.pushConstantRangeCount = 1;
  layout_ci.pPushConstantRanges = &push_range;

  if (vkCreatePipelineLayout(device, &layout_ci, nullptr, &layout_) != VK_SUCCESS) {
    vkDestroyShaderModule(device, shader_module, nullptr);
    return false;
  }

  VkComputePipelineCreateInfo pipeline_ci{};
  pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_ci.stage.module = shader_module;
  pipeline_ci.stage.pName = "main";
  pipeline_ci.layout = layout_;

  VkResult result =
      vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr, &pipeline_);

  vkDestroyShaderModule(device, shader_module, nullptr);
  return result == VK_SUCCESS;
}

// ============================================================================
// update_descriptor — bind SSBO
// ============================================================================
void LobHeatmapComputePipeline::update_descriptor(VkDevice device, VkBuffer ssbo_buffer,
                                                  VkDeviceSize ssbo_size) {
  VkDescriptorBufferInfo buffer_info{};
  buffer_info.buffer = ssbo_buffer;
  buffer_info.offset = 0;
  buffer_info.range = ssbo_size;

  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = descriptor_set_;
  write.dstBinding = 0;
  write.dstArrayElement = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &buffer_info;

  vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

// ============================================================================
// update_descriptor_once — initialize SSBO binding (call once at startup)
// ============================================================================
void LobHeatmapComputePipeline::update_descriptor_once(VkDevice device, VkBuffer ssbo_buffer,
                                                       VkDeviceSize ssbo_size) {
  // Initial descriptor update for SSBO binding 0
  update_descriptor(device, ssbo_buffer, ssbo_size);
}

// ============================================================================
// dispatch
// ============================================================================
void LobHeatmapComputePipeline::dispatch(VkCommandBuffer cmd, const HeatmapPushConstants& pc) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &descriptor_set_, 0,
                          nullptr);
  vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HeatmapPushConstants),
                     &pc);

  // Dispatch: 1024/16 = 64 groups in X, 256/16 = 16 groups in Y
  vkCmdDispatch(cmd, HEATMAP_WIDTH / 16, HEATMAP_HEIGHT / 16, 1);
}

// ============================================================================
// Image Transitions
// ============================================================================
void LobHeatmapComputePipeline::transition_to_read(VkCommandBuffer cmd) {
  // Transition image from GENERAL (compute write) to SHADER_READ_ONLY_OPTIMAL (fragment read)
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
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

  // Ensure compute shader writes complete before any graphics stage reads the image
  // Using ALL_GRAPHICS_BIT to cover fragment shader reads inside render pass
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                       0,
                       0, nullptr, 0, nullptr, 1, &barrier);
}

void LobHeatmapComputePipeline::transition_to_general(VkCommandBuffer cmd) {
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = output_image_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  if (!initial_layout_done_) {
    // First time: transition from UNDEFINED to GENERAL
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    initial_layout_done_ = true;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr, 0, nullptr, 1, &barrier);
  } else {
    // Subsequent frames: transition from SHADER_READ_ONLY_OPTIMAL (previous frame's fragment read)
    // back to GENERAL for compute shader write
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr, 0, nullptr, 1, &barrier);
  }
}

// ============================================================================
// destroy
// ============================================================================
void LobHeatmapComputePipeline::destroy(VkDevice device) {
  if (pipeline_) {
    vkDestroyPipeline(device, pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
  }
  if (layout_) {
    vkDestroyPipelineLayout(device, layout_, nullptr);
    layout_ = VK_NULL_HANDLE;
  }
  if (ds_layout_) {
    vkDestroyDescriptorSetLayout(device, ds_layout_, nullptr);
    ds_layout_ = VK_NULL_HANDLE;
  }
  if (sampler_) {
    vkDestroySampler(device, sampler_, nullptr);
    sampler_ = VK_NULL_HANDLE;
  }
  if (output_view_) {
    vkDestroyImageView(device, output_view_, nullptr);
    output_view_ = VK_NULL_HANDLE;
  }
  if (output_image_) {
    vkDestroyImage(device, output_image_, nullptr);
    output_image_ = VK_NULL_HANDLE;
  }
  if (output_memory_) {
    vkFreeMemory(device, output_memory_, nullptr);
    output_memory_ = VK_NULL_HANDLE;
  }
  // descriptor_set_ is freed when the pool is destroyed/reset
  descriptor_set_ = VK_NULL_HANDLE;
  initial_layout_done_ = false;
}

}  // namespace BTQuant
