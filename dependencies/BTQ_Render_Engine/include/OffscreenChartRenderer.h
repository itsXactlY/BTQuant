#pragma once

#include "vulkan_base_types.hpp"
#include <cstdint>

namespace BTQuant {

class OffscreenChartRenderer {
public:
  OffscreenChartRenderer(VulkanCore *core);
  ~OffscreenChartRenderer();

  void resize(uint32_t width, uint32_t height);
  void create_resources(uint32_t width, uint32_t height);
  void begin_render(VkCommandBuffer cmd);
  void end_render(VkCommandBuffer cmd);

  VkDescriptorSet GetDescriptor() const { return descriptor_set_; }
  VkRenderPass GetRenderPass() const { return render_pass_; }
  VkExtent2D GetExtent() const { return {width_, height_}; }
  VkImage GetImage() const { return image_; }
  VkImageView GetImageView() const { return view_; }

private:
  void cleanup();

  VulkanCore *core_ = nullptr;
  uint32_t width_ = 0;
  uint32_t height_ = 0;

  VkImage image_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  VkImageView view_ = VK_NULL_HANDLE;
  VkFramebuffer framebuffer_ = VK_NULL_HANDLE;
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
};

} // namespace BTQuant
