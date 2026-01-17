#include "vulkan_dashboard_advanced.hpp"
#include <cstring>

namespace BTQuant {

OffscreenChartRenderer::OffscreenChartRenderer(VulkanCore *core)
    : core_(core) {}

OffscreenChartRenderer::~OffscreenChartRenderer() { cleanup(); }

void OffscreenChartRenderer::cleanup() {
  if (core_) {
    VkDevice device = core_->get_device();
    if (framebuffer_ != VK_NULL_HANDLE)
      vkDestroyFramebuffer(device, framebuffer_, nullptr);
    if (view_ != VK_NULL_HANDLE)
      vkDestroyImageView(device, view_, nullptr);
    if (image_ != VK_NULL_HANDLE)
      vkDestroyImage(device, image_, nullptr);
    if (memory_ != VK_NULL_HANDLE)
      vkFreeMemory(device, memory_, nullptr);
    if (render_pass_ != VK_NULL_HANDLE)
      vkDestroyRenderPass(device, render_pass_, nullptr);

    framebuffer_ = VK_NULL_HANDLE;
    view_ = VK_NULL_HANDLE;
    image_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    render_pass_ = VK_NULL_HANDLE;
    descriptor_set_ = VK_NULL_HANDLE;
  }
}

void OffscreenChartRenderer::resize(uint32_t width, uint32_t height) {
  if (width == width_ && height == height_)
    return;
  if (width == 0 || height == 0)
    return;

  vkDeviceWaitIdle(core_->get_device());
  cleanup();
  create_resources(width, height);
}

void OffscreenChartRenderer::create_resources(uint32_t width, uint32_t height) {
  width_ = width;
  height_ = height;
  VkDevice device = core_->get_device();

  // 1. Create Render Pass
  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = VK_FORMAT_R8G8B8A8_UNORM;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;

  // Explicit dependency for ImGui sampling
  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkSubpassDependency samplingDependency{};
  samplingDependency.srcSubpass = 0;
  samplingDependency.dstSubpass = VK_SUBPASS_EXTERNAL;
  samplingDependency.srcStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  samplingDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  samplingDependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  samplingDependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  VkSubpassDependency dependencies[] = {dependency, samplingDependency};

  VkRenderPassCreateInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 2;
  renderPassInfo.pDependencies = dependencies;

  if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &render_pass_) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create offscreen render pass");
  }

  // 2. Create Image
  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

  if (vkCreateImage(device, &imageInfo, nullptr, &image_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create offscreen image");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, image_, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = core_->find_memory_type(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &memory_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate offscreen image memory");
  }

  vkBindImageMemory(device, image_, memory_, 0);

  // 3. Create Image View
  VkImageViewCreateInfo viewInfo = {};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image_;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(device, &viewInfo, nullptr, &view_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create offscreen image view");
  }

  // 4. Create Framebuffer
  VkFramebufferCreateInfo framebufferInfo = {};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = render_pass_;
  framebufferInfo.attachmentCount = 1;
  framebufferInfo.pAttachments = &view_;
  framebufferInfo.width = width;
  framebufferInfo.height = height;
  framebufferInfo.layers = 1;

  if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer_) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create offscreen framebuffer");
  }

  // 5. Build ImGui Descriptor Set
  descriptor_set_ = core_->create_texture_descriptor(view_);
}

void OffscreenChartRenderer::begin_render(VkCommandBuffer cmd) {
  VkRenderPassBeginInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = render_pass_;
  renderPassInfo.framebuffer = framebuffer_;
  renderPassInfo.renderArea.extent = {width_, height_};

  VkClearValue clearColor = {{{0.04f, 0.04f, 0.04f, 1.0f}}};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearColor;

  vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport{0.0f, 0.0f, (float)width_, (float)height_, 0.0f, 1.0f};
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor{{0, 0}, {width_, height_}};
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void OffscreenChartRenderer::end_render(VkCommandBuffer cmd) {
  vkCmdEndRenderPass(cmd);
}

} // namespace BTQuant
