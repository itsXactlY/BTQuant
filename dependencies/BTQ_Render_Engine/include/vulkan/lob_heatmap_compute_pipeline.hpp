#pragma once

/// @file lob_heatmap_compute_pipeline.hpp
/// @brief Vulkan compute pipeline for LOB heatmap generation from VolumeNode SSBO.
/// Owns all Vulkan resources: descriptor set, pipeline, output image, sampler.

#include <vulkan/vulkan.h>

#include <cstdint>

namespace BTQuant {

class GPUMemoryManager;

/// Push constants mirroring the GLSL layout.
struct HeatmapPushConstants {
  float max_volume = 1.0f;  ///< Running max for normalization
  float alpha = 0.3f;       ///< Overall heatmap opacity
};
static_assert(sizeof(HeatmapPushConstants) == 8);

class LobHeatmapComputePipeline {
 public:
  // Heatmap dimensions — wired to HotSpine data:
  // 1024 ClusterColumns × 256 VolumeNode rows
  static constexpr uint32_t HEATMAP_WIDTH = 1024;
  static constexpr uint32_t HEATMAP_HEIGHT = 256;

  LobHeatmapComputePipeline() = default;
  ~LobHeatmapComputePipeline() = default;

  /// Create all Vulkan resources (pipeline, descriptors, output image).
  /// @param device Vulkan logical device
  /// @param physical_device Vulkan physical device (for memory type queries)
  /// @param descriptor_pool Pool to allocate descriptor set from
  /// @return true on success
  bool initialize(VkDevice device, VkPhysicalDevice physical_device,
                  VkDescriptorPool descriptor_pool);

  /// Update the descriptor set to point at a new SSBO buffer.
  void update_descriptor(VkDevice device, VkBuffer ssbo_buffer, VkDeviceSize ssbo_size);

  /// Initialize the SSBO descriptor binding (call once at startup).
  void update_descriptor_once(VkDevice device, VkBuffer ssbo_buffer, VkDeviceSize ssbo_size);

  /// Record compute dispatch into the command buffer.
  /// Issues push constants + vkCmdDispatch(64, 16, 1).
  void dispatch(VkCommandBuffer cmd, const HeatmapPushConstants& pc);

  /// Transition output image: keeps GENERAL layout for both compute write and fragment read.
  /// @param cmd Command buffer to record the barrier into
  /// @param compute_queue_family Queue family index for compute operations
  /// @param graphics_queue_family Queue family index for graphics operations
  void transition_to_read(VkCommandBuffer cmd, uint32_t compute_queue_family,
                          uint32_t graphics_queue_family);

  /// Transition output image: SHADER_READ_ONLY_OPTIMAL → GENERAL (before next compute dispatch).
  /// @param cmd Command buffer to record the barrier into
  /// @param compute_queue_family Queue family index for compute operations
  /// @param graphics_queue_family Queue family index for graphics operations
  void transition_to_general(VkCommandBuffer cmd, uint32_t compute_queue_family,
                             uint32_t graphics_queue_family);

  /// Clean up all Vulkan resources.
  void destroy(VkDevice device);

  // Accessors
  VkImageView get_output_image_view() const { return output_view_; }
  VkImage get_output_image() const { return output_image_; }
  VkSampler get_sampler() const { return sampler_; }
  bool is_initialized() const { return pipeline_ != VK_NULL_HANDLE; }

 private:
  bool create_output_image(VkDevice device, VkPhysicalDevice physical_device);
  bool create_sampler(VkDevice device);
  bool create_descriptor_resources(VkDevice device, VkDescriptorPool pool);
  bool create_pipeline(VkDevice device);
  bool load_shader_module(VkDevice device, VkShaderModule& module);

  uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter,
                            VkMemoryPropertyFlags properties);

  // Pipeline objects
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout ds_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

  // Output image
  VkImage output_image_ = VK_NULL_HANDLE;
  VkImageView output_view_ = VK_NULL_HANDLE;
  VkDeviceMemory output_memory_ = VK_NULL_HANDLE;
  VkSampler sampler_ = VK_NULL_HANDLE;

  // State tracking
  bool initial_layout_done_ = false;
  VkImageLayout current_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
};

}  // namespace BTQuant
