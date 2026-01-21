// #define VK_USE_PLATFORM_XLIB_KHR
#include "../../include/vulkan_base_types.hpp"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

namespace BTQuant {

VulkanCore::VulkanCore(const VulkanDashboardConfig &config) : config_(config) {}

VulkanCore::~VulkanCore() { cleanup(); }

void VulkanCore::initialize(GLFWwindow *window, uint32_t width,
                            uint32_t height) {
  // Basic Vulkan initialization logic
  create_instance();
  create_surface(window);
  select_physical_device();
  create_logical_device();

  memory_manager_ =
      std::make_unique<GPUMemoryManager>(device_, physical_device_, config_);

  create_swapchain(width, height);
  create_image_views();
  create_render_pass();
  create_msaa_resources();
  create_depth_resources();
  create_framebuffers();
  create_command_pool();
  create_command_buffers();
  create_descriptor_pool();
  create_sync_objects();
  init_imgui();
  std::cout << "[VulkanCore] init_imgui done." << std::endl;
  create_default_sampler();
  std::cout << "[VulkanCore] create_default_sampler done." << std::endl;
}

void VulkanCore::cleanup() {
  cleanup_swapchain();
  cleanup_imgui();

  vkDestroySampler(device_, default_sampler_, nullptr);
  vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
  vkDestroyDescriptorPool(device_, imgui_descriptor_pool_, nullptr);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(device_, render_finished_semaphores_[i], nullptr);
    vkDestroySemaphore(device_, image_available_semaphores_[i], nullptr);
    vkDestroyFence(device_, in_flight_fences_[i], nullptr);
  }

  vkDestroyCommandPool(device_, command_pool_, nullptr);

  vkDestroyDevice(device_, nullptr);
  vkDestroySurfaceKHR(instance_, surface_, nullptr);
  vkDestroyInstance(instance_, nullptr);
}

void VulkanCore::RecreateSwapchain() {
  recreate_swapchain(swapchain_extent_.width, swapchain_extent_.height);
}

VkResult VulkanCore::PrepareFrame(uint32_t &imageIndex) {
  // Measure frame time
  auto current_time = std::chrono::high_resolution_clock::now();
  if (last_frame_time_.time_since_epoch().count() > 0) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        current_time - last_frame_time_);
    frame_time_ms_ = duration.count() / 1000.0f;
    fps_ = 1000.0f / frame_time_ms_;

    // Update frame time history
    frame_time_history_.push_back(frame_time_ms_);
    if (frame_time_history_.size() > FRAME_TIME_HISTORY_SIZE) {
      frame_time_history_.erase(frame_time_history_.begin());
    }
  }
  last_frame_time_ = current_time;

  vkWaitForFences(device_, 1, &in_flight_fences_[current_frame_], VK_TRUE,
                  UINT64_MAX);

  VkResult result = vkAcquireNextImageKHR(
      device_, swapchain_,
      UINT64_MAX, // Unbegrenzter Timeout für bessere Stabilität
      image_available_semaphores_[current_frame_], VK_NULL_HANDLE, &imageIndex);

  if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
    vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);
    current_command_buffer_ = command_buffers_[current_frame_];
    vkResetCommandBuffer(current_command_buffer_, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = config_.enable_command_buffer_recycling
                          ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
                          : 0;
    vkBeginCommandBuffer(current_command_buffer_, &beginInfo);
  }

  return result;
}

void VulkanCore::RecordCommandBuffer(uint32_t imageIndex,
                                     ImDrawData *drawData) {
  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = render_pass_;
  renderPassInfo.framebuffer = framebuffers_[imageIndex];
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = swapchain_extent_;

  std::array<VkClearValue, 2> clearValues{};
  clearValues[0].color = {{0.01f, 0.01f, 0.01f, 1.0f}};
  clearValues[1].depthStencil = {1.0f, 0};

  renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
  renderPassInfo.pClearValues = clearValues.data();

  vkCmdBeginRenderPass(current_command_buffer_, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);

  if (drawData) {
    ImGui_ImplVulkan_RenderDrawData(drawData, current_command_buffer_);
  }

  vkCmdEndRenderPass(current_command_buffer_);

  if (vkEndCommandBuffer(current_command_buffer_) != VK_SUCCESS) {
    throw std::runtime_error("failed to record command buffer!");
  }
}

VkResult VulkanCore::PresentFrame(uint32_t imageIndex) {
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore waitSemaphores[] = {image_available_semaphores_[current_frame_]};
  VkPipelineStageFlags waitStages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &current_command_buffer_;

  VkSemaphore signalSemaphores[] = {
      render_finished_semaphores_[current_frame_]};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  if (vkQueueSubmit(graphics_queue_, 1, &submitInfo,
                    in_flight_fences_[current_frame_]) != VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = signalSemaphores;

  VkSwapchainKHR swapChains[] = {swapchain_};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;
  presentInfo.pImageIndices = &imageIndex;

  VkResult result = vkQueuePresentKHR(present_queue_, &presentInfo);

  current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;

  return result;
}

void VulkanCore::begin_main_render_pass() {}
void VulkanCore::end_frame() {}

void VulkanCore::set_low_latency_mode(bool enabled) {
  config_.enable_low_latency_mode = enabled;
  std::cout << "[VulkanCore] Low latency mode "
            << (enabled ? "enabled" : "disabled") << std::endl;
}

VkCommandBuffer VulkanCore::acquire_command_buffer() {
  // Simple implementation - in production, this would manage a pool of command
  // buffers
  return command_buffers_[current_frame_];
}

void VulkanCore::release_command_buffer(VkCommandBuffer cmd_buf) {
  // For now, just reset the command buffer for reuse
  if (config_.enable_command_buffer_recycling) {
    vkResetCommandBuffer(cmd_buf,
                         VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
  }
}

void VulkanCore::recreate_swapchain(uint32_t width, uint32_t height) {
  vkDeviceWaitIdle(device_);

  cleanup_swapchain();

  create_swapchain(width, height);
  create_image_views();
  create_render_pass();
  create_msaa_resources();
  create_depth_resources();
  create_framebuffers();
}

uint32_t VulkanCore::find_memory_type(uint32_t type_filter,
                                      VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

VkCommandBuffer VulkanCore::begin_single_time_commands() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = command_pool_;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

void VulkanCore::end_single_time_commands(VkCommandBuffer commandBuffer) {
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(graphics_queue_, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue_);

  vkFreeCommandBuffers(device_, command_pool_, 1, &commandBuffer);
}
// Private implementation stubs
void VulkanCore::create_instance() {
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "BTQuant Unified Dashboard";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "BTQ Render Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  auto extensions = get_required_extensions();
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};
  // Temporarily disable validation layers if they might be missing
  if (config_.enable_validation_layers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
  if (result != VK_SUCCESS) {
    std::cerr << "[VulkanCore] vkCreateInstance failed with error: " << result
              << std::endl;
    throw std::runtime_error("failed to create instance!");
  }
}

void VulkanCore::select_physical_device() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

  for (const auto &device : devices) {
    if (is_device_suitable(device)) {
      physical_device_ = device;
      break;
    }
  }

  if (physical_device_ == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void VulkanCore::create_logical_device() {
  // Find queue families
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount,
                                           queueFamilies.data());

  int graphicsFamily = -1;
  int computeFamily = -1;
  int presentFamily = -1;

  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, i, surface_,
                                         &presentSupport);

    if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
        presentSupport) {
      graphicsFamily = i;
      presentFamily = i;
    }

    if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        computeFamily == -1) {
      computeFamily = i;
    }

    // If we found all queue families, break early
    if (graphicsFamily != -1 && computeFamily != -1 && presentFamily != -1) {
      break;
    }
  }

  // Fallback: if no separate compute queue, use graphics queue
  if (computeFamily == -1) {
    computeFamily = graphicsFamily;
  }

  if (graphicsFamily == -1) {
    throw std::runtime_error("No graphics queue family found!");
  }

  if (presentFamily == -1) {
    throw std::runtime_error("No present queue family found!");
  }

  // Create queue create info structures
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::vector<float> queuePriorities = {1.0f};

  if (graphicsFamily == computeFamily && graphicsFamily == presentFamily) {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = graphicsFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = queuePriorities.data();
    queueCreateInfos.push_back(queueCreateInfo);
  } else {
    // Graphics queue
    VkDeviceQueueCreateInfo graphicsQueueInfo{};
    graphicsQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    graphicsQueueInfo.queueFamilyIndex = graphicsFamily;
    graphicsQueueInfo.queueCount = 1;
    graphicsQueueInfo.pQueuePriorities = queuePriorities.data();
    queueCreateInfos.push_back(graphicsQueueInfo);

    // Compute queue
    if (computeFamily != graphicsFamily) {
      VkDeviceQueueCreateInfo computeQueueInfo{};
      computeQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      computeQueueInfo.queueFamilyIndex = computeFamily;
      computeQueueInfo.queueCount = 1;
      computeQueueInfo.pQueuePriorities = queuePriorities.data();
      queueCreateInfos.push_back(computeQueueInfo);
    }

    // Present queue
    if (presentFamily != graphicsFamily && presentFamily != computeFamily) {
      VkDeviceQueueCreateInfo presentQueueInfo{};
      presentQueueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      presentQueueInfo.queueFamilyIndex = presentFamily;
      presentQueueInfo.queueCount = 1;
      presentQueueInfo.pQueuePriorities = queuePriorities.data();
      queueCreateInfos.push_back(presentQueueInfo);
    }
  }

  VkPhysicalDeviceFeatures deviceFeatures{};

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pEnabledFeatures = &deviceFeatures;

  std::vector<const char *> deviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  VkResult result =
      vkCreateDevice(physical_device_, &createInfo, nullptr, &device_);
  if (result != VK_SUCCESS) {
    std::cerr
        << "[VulkanCore] CRITICAL: vkCreateDevice failed with error code: "
        << result << std::endl;
    throw std::runtime_error("failed to create logical device!");
  }

  vkGetDeviceQueue(device_, graphicsFamily, 0, &graphics_queue_);
  vkGetDeviceQueue(device_, presentFamily, 0, &present_queue_);
  vkGetDeviceQueue(device_, computeFamily, 0, &compute_queue_);

  graphics_queue_family_ = graphicsFamily;
  present_queue_family_ = presentFamily;
  compute_queue_family_ = computeFamily;

  std::cout << "[VulkanCore] Queue families - Graphics: "
            << graphics_queue_family_ << ", Compute: " << compute_queue_family_
            << ", Present: " << present_queue_family_ << std::endl;
}

void VulkanCore::create_surface(GLFWwindow *window) {
  if (glfwCreateWindowSurface(instance_, window, nullptr, &surface_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }
}
void VulkanCore::create_swapchain(uint32_t width, uint32_t height) {
  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_,
                                            &capabilities);

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &formatCount,
                                       nullptr);
  std::vector<VkSurfaceFormatKHR> formats(formatCount);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &formatCount,
                                       formats.data());

  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_,
                                            &presentModeCount, nullptr);
  std::vector<VkPresentModeKHR> presentModes(presentModeCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      physical_device_, surface_, &presentModeCount, presentModes.data());

  VkSurfaceFormatKHR surfaceFormat = formats[0];
  for (const auto &availableFormat : formats) {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
        availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      surfaceFormat = availableFormat;
      break;
    }
  }

  VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
  for (const auto &availablePresentMode : presentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      presentMode = availablePresentMode;
      break;
    }
  }

  if (capabilities.currentExtent.width != UINT32_MAX) {
    swapchain_extent_ = capabilities.currentExtent;
  } else {
    VkExtent2D actualExtent = {width, height};
    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);
    swapchain_extent_ = actualExtent;
  }

  std::cout << "[VulkanCore] Surface Capabilities: minExtent="
            << capabilities.minImageExtent.width << "x"
            << capabilities.minImageExtent.height
            << " maxExtent=" << capabilities.maxImageExtent.width << "x"
            << capabilities.maxImageExtent.height
            << " currentExtent=" << capabilities.currentExtent.width << "x"
            << capabilities.currentExtent.height << std::endl;
  std::cout << "[VulkanCore] Chosen Swapchain Extent: "
            << swapchain_extent_.width << "x" << swapchain_extent_.height
            << std::endl;

  swapchain_image_format_ = surfaceFormat.format;

  uint32_t imageCount = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 &&
      imageCount > capabilities.maxImageCount) {
    imageCount = capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface_;
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = swapchain_extent_;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  if (graphics_queue_family_ != present_queue_family_) {
    uint32_t queueFamilyIndices[] = {graphics_queue_family_,
                                     present_queue_family_};
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;

  if (vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create swapchain!");
  }

  vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
  swapchain_images_.resize(imageCount);
  vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount,
                          swapchain_images_.data());
}

void VulkanCore::create_image_views() {
  swapchain_image_views_.resize(swapchain_images_.size());

  for (size_t i = 0; i < swapchain_images_.size(); i++) {
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = swapchain_images_[i];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = swapchain_image_format_;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device_, &createInfo, nullptr,
                          &swapchain_image_views_[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create image views!");
    }
  }
}

void VulkanCore::create_render_pass() {
  std::vector<VkAttachmentDescription> attachments;

  if (config_.enable_msaa) {
    // MSAA color attachment
    VkAttachmentDescription msaaAttachment{};
    msaaAttachment.format = swapchain_image_format_;
    msaaAttachment.samples = get_max_usable_sample_count();
    msaaAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    msaaAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    msaaAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    msaaAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    msaaAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    msaaAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments.push_back(msaaAttachment);

    // Resolve attachment for MSAA
    VkAttachmentDescription resolveAttachment{};
    resolveAttachment.format = swapchain_image_format_;
    resolveAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    resolveAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    resolveAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolveAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    resolveAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    resolveAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachments.push_back(resolveAttachment);
  } else {
    // Direct color attachment (no MSAA)
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapchain_image_format_;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    attachments.push_back(colorAttachment);
  }

  // Depth attachment
  VkAttachmentDescription depthAttachment{};
  depthAttachment.format = find_depth_format();
  depthAttachment.samples = config_.enable_msaa ? get_max_usable_sample_count()
                                                : VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachments.push_back(depthAttachment);

  // Subpass setup
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

  std::vector<VkAttachmentReference> colorAttachmentRefs;
  std::vector<VkAttachmentReference> resolveAttachmentRefs;
  if (config_.enable_msaa) {
    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachmentRefs.push_back(colorRef);

    VkAttachmentReference resolveRef{};
    resolveRef.attachment = 1;
    resolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    resolveAttachmentRefs.push_back(resolveRef);
    subpass.pResolveAttachments = resolveAttachmentRefs.data();
  } else {
    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachmentRefs.push_back(colorRef);
  }

  subpass.colorAttachmentCount =
      static_cast<uint32_t>(colorAttachmentRefs.size());
  subpass.pColorAttachments = colorAttachmentRefs.data();

  VkAttachmentReference depthAttachmentRef{};
  depthAttachmentRef.attachment = config_.enable_msaa ? 2 : 1;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  subpass.pDepthStencilAttachment = &depthAttachmentRef;

  // Subpass dependencies
  std::vector<VkSubpassDependency> dependencies;
  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  dependencies.push_back(dependency);

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
  renderPassInfo.pDependencies = dependencies.data();

  if (vkCreateRenderPass(device_, &renderPassInfo, nullptr, &render_pass_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }

  std::cout << "[VulkanCore] Render pass created with "
            << (config_.enable_msaa ? "MSAA" : "no MSAA") << " and depth buffer"
            << std::endl;
}
void VulkanCore::create_framebuffers() {
  framebuffers_.resize(swapchain_image_views_.size());

  for (size_t i = 0; i < swapchain_image_views_.size(); i++) {
    std::vector<VkImageView> attachments;

    if (config_.enable_msaa) {
      attachments.push_back(msaa_color_image_view_);
      attachments.push_back(swapchain_image_views_[i]);
    } else {
      attachments.push_back(swapchain_image_views_[i]);
    }

    attachments.push_back(depth_image_view_);

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = render_pass_;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebufferInfo.pAttachments = attachments.data();
    framebufferInfo.width = swapchain_extent_.width;
    framebufferInfo.height = swapchain_extent_.height;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device_, &framebufferInfo, nullptr,
                            &framebuffers_[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }

  std::cout << "[VulkanCore] Framebuffers created with "
            << (config_.enable_msaa ? "MSAA" : "no MSAA") << " and depth buffer"
            << std::endl;
}
void VulkanCore::create_command_pool() {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = graphics_queue_family_;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(device_, &poolInfo, nullptr, &command_pool_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }
}

void VulkanCore::create_command_buffers() {
  command_buffers_.resize(MAX_FRAMES_IN_FLIGHT);

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = command_pool_;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = (uint32_t)command_buffers_.size();

  if (vkAllocateCommandBuffers(device_, &allocInfo, command_buffers_.data()) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate command buffers!");
  }
}

void VulkanCore::create_descriptor_pool() {
  std::vector<VkDescriptorPoolSize> poolSizes = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = 1000 * static_cast<uint32_t>(poolSizes.size());
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptor_pool_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}
void VulkanCore::create_sync_objects() {
  image_available_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
  render_finished_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
  in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                          &image_available_semaphores_[i]) != VK_SUCCESS ||
        vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                          &render_finished_semaphores_[i]) != VK_SUCCESS ||
        vkCreateFence(device_, &fenceInfo, nullptr, &in_flight_fences_[i]) !=
            VK_SUCCESS) {
      throw std::runtime_error("failed to create synchronization objects!");
    }
  }
}
void VulkanCore::cleanup_swapchain() {
  for (auto framebuffer : framebuffers_) {
    vkDestroyFramebuffer(device_, framebuffer, nullptr);
  }
  framebuffers_.clear();

  for (auto imageView : swapchain_image_views_) {
    vkDestroyImageView(device_, imageView, nullptr);
  }
  swapchain_image_views_.clear();

  if (msaa_color_image_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_, msaa_color_image_view_, nullptr);
    msaa_color_image_view_ = VK_NULL_HANDLE;
  }
  if (msaa_color_image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device_, msaa_color_image_, nullptr);
    msaa_color_image_ = VK_NULL_HANDLE;
  }
  if (msaa_color_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, msaa_color_memory_, nullptr);
    msaa_color_memory_ = VK_NULL_HANDLE;
  }

  if (depth_image_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_, depth_image_view_, nullptr);
    depth_image_view_ = VK_NULL_HANDLE;
  }
  if (depth_image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device_, depth_image_, nullptr);
    depth_image_ = VK_NULL_HANDLE;
  }
  if (depth_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, depth_memory_, nullptr);
    depth_memory_ = VK_NULL_HANDLE;
  }

  if (swapchain_ != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
  }

  if (render_pass_ != VK_NULL_HANDLE) {
    vkDestroyRenderPass(device_, render_pass_, nullptr);
    render_pass_ = VK_NULL_HANDLE;
  }

  std::cout << "[VulkanCore] Swapchain resources cleaned up" << std::endl;
}
void VulkanCore::init_imgui() {
  std::cout << "[VulkanCore] init_imgui: Instance=" << instance_
            << " Device=" << device_ << " PhysDevice=" << physical_device_
            << std::endl;
  std::cout << "[VulkanCore] init_imgui: DescPool=" << descriptor_pool_
            << " RenderPass=" << render_pass_ << std::endl;

  // Create a separate descriptor pool for ImGui
  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
  pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;

  if (vkCreateDescriptorPool(device_, &pool_info, nullptr,
                             &imgui_descriptor_pool_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create ImGui descriptor pool!");
  }

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.ApiVersion = VK_API_VERSION_1_2;
  init_info.Instance = instance_;
  init_info.PhysicalDevice = physical_device_;
  init_info.Device = device_;
  init_info.QueueFamily = graphics_queue_family_;
  init_info.Queue = graphics_queue_;
  init_info.DescriptorPool = imgui_descriptor_pool_; // Use the dedicated pool
  init_info.PipelineInfoMain.RenderPass = render_pass_;
  init_info.PipelineInfoMain.MSAASamples = config_.enable_msaa
                                               ? get_max_usable_sample_count()
                                               : VK_SAMPLE_COUNT_1_BIT;
  init_info.MinImageCount = 2;
  init_info.ImageCount = static_cast<uint32_t>(swapchain_images_.size());
  init_info.CheckVkResultFn = [](VkResult err) {
    if (err == 0)
      return;
    std::cerr << "[ImGui][Vulkan] Error: " << err << std::endl;
  };

  // Initialize Glfw Backend
  // Note: We need access to the window pointer here ideally, but
  // ImGui_ImplGlfw_InitForVulkan should have been called in VulkanDashboard or
  // we need to pass it here. Actually, standard practice is to Init GLFW
  // backend in Dashboard, and Vulkan backend here. But let's assume valid GLFW
  // context is set current or passed. Wait, I need the window for
  // ImGui_ImplGlfw_InitForVulkan. I will refactor init_imgui to take
  // GLFWwindow* or assert it's initialized before. For now, let's just
  // initialize the Vulkan part here, and ensuring GLFW part is done in
  // Dashboard.

  std::cout << "[VulkanCore] init_imgui: Calling ImGui_ImplVulkan_Init..."
            << std::endl;
  if (!ImGui_ImplVulkan_Init(&init_info)) {
    throw std::runtime_error("failed to initialize ImGui Vulkan backend!");
  }
  std::cout << "[VulkanCore] init_imgui: ImGui_ImplVulkan_Init success."
            << std::endl;

  // Fonts are uploaded automatically by ImGui_ImplVulkan_NewFrame() the first
  // time.
}

void VulkanCore::cleanup_imgui() {
  // Do not call ImGui_ImplVulkan_Shutdown() here - it's called from
  // VulkanDashboard::shutdown() to ensure correct shutdown order
}

void VulkanCore::create_default_sampler() {
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.anisotropyEnable = VK_FALSE;
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(device_, &samplerInfo, nullptr, &default_sampler_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
}
std::vector<const char *> VulkanCore::get_required_extensions() {
  std::vector<const char *> extensions;
  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);

  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  for (uint32_t i = 0; i < glfwExtensionCount; i++) {
    extensions.push_back(glfwExtensions[i]);
  }

  // Duplicate check might be needed but usually GLFW returns the correct
  // surface extensions. references: VK_KHR_XLIB_SURFACE_EXTENSION_NAME removed.
  if (config_.enable_validation_layers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  return extensions;
}

bool VulkanCore::check_validation_layer_support() { return true; }
bool VulkanCore::is_device_suitable(VkPhysicalDevice device) {
  // Check if device supports required extensions and features
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

  // Check if device supports geometry shaders, tessellation, etc. (if needed)
  // For now, check if device supports depth testing and MSAA
  if (!deviceFeatures.depthClamp) {
    std::cout << "[VulkanCore] Device does not support required depth features"
              << std::endl;
    return false;
  }

  // Check MSAA support
  VkPhysicalDeviceProperties physicalDeviceProperties;
  vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);

  VkSampleCountFlags counts =
      physicalDeviceProperties.limits.framebufferColorSampleCounts &
      physicalDeviceProperties.limits.framebufferDepthSampleCounts;

  bool msaaSupported =
      counts & VK_SAMPLE_COUNT_4_BIT; // Check for at least 4x MSAA
  if (config_.enable_msaa && !msaaSupported) {
    std::cout << "[VulkanCore] Device does not support MSAA (4x samples)"
              << std::endl;
    return false;
  }

  // Check extension support
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       availableExtensions.data());

  bool swapchainSupported = false;
  for (const auto &extension : availableExtensions) {
    if (std::string(VK_KHR_SWAPCHAIN_EXTENSION_NAME) ==
        extension.extensionName) {
      swapchainSupported = true;
      break;
    }
  }

  if (!swapchainSupported) {
    std::cout << "[VulkanCore] Device does not support swapchain extension"
              << std::endl;
    return false;
  }

  return true;
}
VkSampleCountFlagBits VulkanCore::get_max_usable_sample_count() {
  VkPhysicalDeviceProperties physicalDeviceProperties;
  vkGetPhysicalDeviceProperties(physical_device_, &physicalDeviceProperties);

  VkSampleCountFlags counts =
      physicalDeviceProperties.limits.framebufferColorSampleCounts &
      physicalDeviceProperties.limits.framebufferDepthSampleCounts;

  if (counts & VK_SAMPLE_COUNT_8_BIT)
    return VK_SAMPLE_COUNT_8_BIT;
  if (counts & VK_SAMPLE_COUNT_4_BIT)
    return VK_SAMPLE_COUNT_4_BIT;
  if (counts & VK_SAMPLE_COUNT_2_BIT)
    return VK_SAMPLE_COUNT_2_BIT;

  return VK_SAMPLE_COUNT_1_BIT;
}

VkFormat
VulkanCore::find_supported_format(const std::vector<VkFormat> &candidates,
                                  VkImageTiling tiling,
                                  VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(physical_device_, format, &props);

    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  throw std::runtime_error("failed to find supported format!");
}

VkFormat VulkanCore::find_depth_format() {
  return find_supported_format(
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void VulkanCore::create_msaa_resources() {
  if (!config_.enable_msaa) {
    std::cout << "[VulkanCore] MSAA disabled" << std::endl;
    return;
  }

  VkSampleCountFlagBits msaaSamples = get_max_usable_sample_count();
  if (msaaSamples == VK_SAMPLE_COUNT_1_BIT) {
    std::cout << "[VulkanCore] MSAA not supported by device" << std::endl;
    config_.enable_msaa = false;
    return;
  }

  std::cout << "[VulkanCore] Creating MSAA resources with " << msaaSamples
            << " samples" << std::endl;

  // Create MSAA color image
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = swapchain_extent_.width;
  imageInfo.extent.height = swapchain_extent_.height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = swapchain_image_format_;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = msaaSamples;
  imageInfo.flags = 0;

  if (vkCreateImage(device_, &imageInfo, nullptr, &msaa_color_image_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create MSAA color image!");
  }

  // Allocate memory for MSAA image
  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device_, msaa_color_image_, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = find_memory_type(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (vkAllocateMemory(device_, &allocInfo, nullptr, &msaa_color_memory_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate MSAA color image memory!");
  }

  vkBindImageMemory(device_, msaa_color_image_, msaa_color_memory_, 0);

  // Create MSAA image view
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = msaa_color_image_;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = swapchain_image_format_;
  viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(device_, &viewInfo, nullptr, &msaa_color_image_view_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create MSAA color image view!");
  }
}

void VulkanCore::create_depth_resources() {
  VkFormat depthFormat = find_depth_format();
  std::cout << "[VulkanCore] Creating depth buffer with format: " << depthFormat
            << std::endl;

  // Create depth image
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = swapchain_extent_.width;
  imageInfo.extent.height = swapchain_extent_.height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = depthFormat;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = config_.enable_msaa ? get_max_usable_sample_count()
                                          : VK_SAMPLE_COUNT_1_BIT;
  imageInfo.flags = 0;

  if (vkCreateImage(device_, &imageInfo, nullptr, &depth_image_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create depth image!");
  }

  // Allocate memory for depth image
  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device_, depth_image_, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = find_memory_type(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (vkAllocateMemory(device_, &allocInfo, nullptr, &depth_memory_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate depth image memory!");
  }

  vkBindImageMemory(device_, depth_image_, depth_memory_, 0);

  // Create depth image view
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = depth_image_;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = depthFormat;
  viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  if (depthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT ||
      depthFormat == VK_FORMAT_D24_UNORM_S8_UINT) {
    viewInfo.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
  }
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(device_, &viewInfo, nullptr, &depth_image_view_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create depth image view!");
  }
}

} // namespace BTQuant
