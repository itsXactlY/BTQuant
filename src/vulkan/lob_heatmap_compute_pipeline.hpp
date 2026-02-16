// LOB Heatmap Compute Pipeline Header
// Vulkan 1.3 - Market Microstructure Renderer
// Declares compute pipeline configuration and image transition functions

#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

// Workgroup configuration for LOB heatmap compute shader
// Matches the shader's local_size_x and local_size_y declarations
#define WORKGROUP_SIZE_X 16
#define WORKGROUP_SIZE_Y 16
#define WORKGROUP_SIZE_Z 1

// Pipeline configuration structure
struct LobHeatmapPipelineConfig {
    uint32_t local_size_x = WORKGROUP_SIZE_X;
    uint32_t local_size_y = WORKGROUP_SIZE_Y;
    uint32_t local_size_z = WORKGROUP_SIZE_Z;
};

/**
 * @brief Records a pipeline barrier to transition the heatmap image from
 * VK_IMAGE_LAYOUT_GENERAL (compute shader write) to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
 * (ImGui sampling).
 *
 * The compute shader uses imageStore() which requires VK_IMAGE_LAYOUT_GENERAL.
 * Before the heatmap can be sampled by ImGui's fragment shader, it must be
 * transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
 *
 * @param commandBuffer Vulkan command buffer to record the barrier into
 * @param heatmapImage The heatmap image to transition
 * @param width Width of the heatmap image (for subresource range)
 * @param height Height of the heatmap image (for subresource range)
 */
void recordHeatmapImageTransition(VkCommandBuffer commandBuffer,
                                  VkImage heatmapImage,
                                  uint32_t width,
                                  uint32_t height);

/**
 * @brief Records a pipeline barrier to transition the heatmap image back to
 * VK_IMAGE_LAYOUT_GENERAL for the next compute shader dispatch.
 *
 * This is used when the heatmap needs to be updated by the compute shader again.
 *
 * @param commandBuffer Vulkan command buffer to record the barrier into
 * @param heatmapImage The heatmap image to transition
 */
void recordHeatmapImageTransitionToGeneral(VkCommandBuffer commandBuffer,
                                           VkImage heatmapImage);
