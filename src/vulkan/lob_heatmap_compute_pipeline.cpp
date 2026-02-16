// LOB Heatmap Compute Pipeline
// Vulkan 1.3 - Market Microstructure Renderer
// Configures compute shader dispatch parameters for order book heatmap generation

#include "lob_heatmap_compute_pipeline.hpp"
#include <vulkan/vulkan.h>

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
                                  uint32_t height) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    // Source access: Compute shader write (imageStore)
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    // Destination access: Fragment shader read (sampling)
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    // Transition from GENERAL (compute write) to SHADER_READ_ONLY_OPTIMAL (ImGui sample)
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = heatmapImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    // Pipeline stages:
    // - Source: COMPUTE_SHADER_BIT (where the image was written)
    // - Destination: FRAGMENT_SHADER_BIT (where ImGui will sample)
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &barrier);
}

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
                                           VkImage heatmapImage) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    // Source access: Fragment shader read (previous frame's sampling)
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    // Destination access: Compute shader write (next dispatch)
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    // Transition from SHADER_READ_ONLY_OPTIMAL back to GENERAL
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = heatmapImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    // Pipeline stages:
    // - Source: FRAGMENT_SHADER_BIT (where the image was last read)
    // - Destination: COMPUTE_SHADER_BIT (where it will be written)
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &barrier);
}
