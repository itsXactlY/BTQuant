#pragma once

#include <cstdint>

#include "../vulkan_base_types.hpp"
#include "../hotspine_layout_v3.hpp"
#include "ssbo_snapshot_updater.hpp"

namespace BTQuant {

// Push constants matching the compute shader
struct HeatmapPushConstants {
  float max_volume;  // Maximum volume for normalization
  float alpha;       // Output alpha
};

// LOB Heatmap Compute Pipeline
// Creates a Vulkan compute pipeline from the lob_heatmap.comp shader,
// manages a 1024x256 rgba16f output image, and provides dispatch + barrier methods.
class LobHeatmapComputePipeline {
 public:
  LobHeatmapComputePipeline() = default;
  ~LobHeatmapComputePipeline() = default;

  // Initialize compute pipeline, descriptor sets, output image
  void initialize(VkDevice device, VkPhysicalDevice physical_device, VkDescriptorPool pool,
                  GPUMemoryManager& mem);

  // Record compute dispatch into command buffer
  // columns: number of history columns to process (max 1024)
  // rows:    number of price rows (max 256)
  void dispatch(VkCommandBuffer cmd, uint32_t columns, uint32_t rows);

  // Record image layout transition barriers:
  // - Call before dispatch: SHADER_READ_ONLY -> GENERAL
  // - Call after dispatch:  GENERAL -> SHADER_READ_ONLY
  void transition_to_general(VkCommandBuffer cmd);
  void transition_to_shader_read(VkCommandBuffer cmd);

  // Update push constants
  void set_push_constants(float max_volume, float alpha);

  // Get output image for external use
  VkImageView get_output_image_view() const { return output_image_view_; }
  VkImage get_output_image() const { return output_image_; }

  // Get cached texture for ImGui rendering
  CachedTexture get_imgui_texture() const { return imgui_texture_; }

  // Bind SSBO for the compute dispatch
  void bind_ssbo(const BufferAllocation& ssbo_alloc);

  // Cleanup all GPU resources
  void destroy(VkDevice device);

 private:
  VkDevice device_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

  // Output image resources
  VkImage output_image_ = VK_NULL_HANDLE;
  VkDeviceMemory output_image_memory_ = VK_NULL_HANDLE;
  VkImageView output_image_view_ = VK_NULL_HANDLE;
  VkSampler output_sampler_ = VK_NULL_HANDLE;

  // ImGui texture handle
  CachedTexture imgui_texture_;

  // Push constants
  HeatmapPushConstants push_constants_{1000.0f, 1.0f};

  // SSBO allocation reference
  BufferAllocation ssbo_allocation_{};

  // Helper: create output image + view + sampler
  void create_output_image(VkDevice device, VkPhysicalDevice physical_device);
  friend class VulkanDashboard;  // Allow access for initialization
};

}  // namespace BTQuant
