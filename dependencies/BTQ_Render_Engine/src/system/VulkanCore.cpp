// #define VK_USE_PLATFORM_XLIB_KHR
#include "../../include/vulkan_base_types.hpp"
#include "backends/imgui_impl_glfw.h"
#include "imgui.h"
#include "imgui_impl_vulkan.h"
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
  create_default_sampler();
}

void VulkanCore::cleanup() {
  cleanup_swapchain();
  cleanup_imgui();

  vkDestroySampler(device_, default_sampler_, nullptr);
  vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);

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
  vkWaitForFences(device_, 1, &in_flight_fences_[current_frame_], VK_TRUE,
                  UINT64_MAX);

  VkResult result = vkAcquireNextImageKHR(
      device_, swapchain_, UINT64_MAX,
      image_available_semaphores_[current_frame_], VK_NULL_HANDLE, &imageIndex);

  if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
    vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);
    current_command_buffer_ = command_buffers_[current_frame_];
    vkResetCommandBuffer(current_command_buffer_, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
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

static std::string resolve_shader_path(const std::string &path) {
  // 1. Try directly (as relative or absolute)
  {
    std::ifstream f(path);
    if (f.good())
      return path;
  }

  // 2. Try prefixing with project root shaders dir
  std::string absolute_fallback =
      "/home/alca/projects/PubBTQuant/dependencies/BTQ_Render_Engine/" + path;
  {
    std::ifstream f(absolute_fallback);
    if (f.good())
      return absolute_fallback;
  }

  // 3. Try ../ prefix (common if running from build/bin)
  std::string parent_prefix = "../" + path;
  {
    std::ifstream f(parent_prefix);
    if (f.good())
      return parent_prefix;
  }

  return path; // Return original if not found, will fail in read_file with
               // error
}

static std::vector<char> read_file(const std::string &filename) {
  std::string resolved_path = resolve_shader_path(filename);
  std::ifstream file(resolved_path, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file: " + filename +
                             " (resolved to: " + resolved_path + ")");
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);

  file.close();

  return buffer;
}

VkPipeline VulkanCore::create_graphics_pipeline(
    const std::string &vert_path, const std::string &frag_path,
    const std::vector<VkVertexInputBindingDescription> &bindings,
    const std::vector<VkVertexInputAttributeDescription> &attributes,
    VkPipelineLayout layout, VkRenderPass render_pass) {

  auto vertShaderCode = read_file(vert_path);
  auto fragShaderCode = read_file(frag_path);

  VkShaderModule vertShaderModule = create_shader_module(vertShaderCode);
  VkShaderModule fragShaderModule = create_shader_module(fragShaderCode);

  VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
  vertShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
  fragShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                    fragShaderStageInfo};

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = bindings.size();
  vertexInputInfo.pVertexBindingDescriptions = bindings.data();
  vertexInputInfo.vertexAttributeDescriptionCount = attributes.size();
  vertexInputInfo.pVertexAttributeDescriptions = attributes.data();

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)swapchain_extent_.width;
  viewport.height = (float)swapchain_extent_.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = swapchain_extent_;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_TRUE;
  colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  colorBlendAttachment.dstColorBlendFactor =
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
  colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.layout = layout;
  pipelineInfo.renderPass = render_pass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  VkPipeline graphicsPipeline;
  if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo,
                                nullptr, &graphicsPipeline) != VK_SUCCESS) {
    throw std::runtime_error("failed to create graphics pipeline!");
  }

  vkDestroyShaderModule(device_, fragShaderModule, nullptr);
  vkDestroyShaderModule(device_, vertShaderModule, nullptr);

  return graphicsPipeline;
}

VkPipeline VulkanCore::create_compute_pipeline(const std::string &shader_path,
                                               VkPipelineLayout layout) {
  (void)shader_path;
  (void)layout;
  return VK_NULL_HANDLE;
}

VkShaderModule VulkanCore::create_shader_module(const std::vector<char> &code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }
  return shaderModule;
}

VkPipelineLayout VulkanCore::create_pipeline_layout(
    const std::vector<VkDescriptorSetLayout> &layouts,
    const std::vector<VkPushConstantRange> &push_constants) {

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = layouts.size();
  pipelineLayoutInfo.pSetLayouts = layouts.data();
  pipelineLayoutInfo.pushConstantRangeCount = push_constants.size();
  pipelineLayoutInfo.pPushConstantRanges = push_constants.data();

  VkPipelineLayout pipelineLayout;
  if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr,
                             &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }
  return pipelineLayout;
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

VkDescriptorSet VulkanCore::create_texture_descriptor(VkImageView view) {
  if (default_sampler_ == VK_NULL_HANDLE) {
    create_default_sampler();
  }
  return ImGui_ImplVulkan_AddTexture(default_sampler_, view,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
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

void VulkanCore::copy_buffer(VkBuffer srcBuffer, VkBuffer dstBuffer,
                             VkDeviceSize size) {
  (void)srcBuffer;
  (void)dstBuffer;
  (void)size;
}

void VulkanCore::create_placeholder_texture(VkImage &image,
                                            VkDeviceMemory &memory,
                                            VkImageView &view,
                                            VkSampler &sampler) {
  (void)image;
  (void)memory;
  (void)view;
  (void)sampler;
  // Code to create a 1x1 placeholder texture
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
  // Simple graphics queue request
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queueFamilyCount,
                                           queueFamilies.data());

  int graphicsFamily = -1;
  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_, i, surface_,
                                         &presentSupport);

    if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
        presentSupport) {
      graphicsFamily = i;
      break;
    }
  }

  if (graphicsFamily == -1) {
    // Fallback: search separately if needed, but for most GPUs graphics ==
    // present
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
      if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        graphicsFamily = i;
        break;
      }
    }
  }

  if (graphicsFamily == -1) {
    throw std::runtime_error("No graphics queue family found!");
  }

  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = graphicsFamily;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures{};

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.queueCreateInfoCount = 1;
  createInfo.pEnabledFeatures = &deviceFeatures;

  std::vector<const char *> deviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  if (vkCreateDevice(physical_device_, &createInfo, nullptr, &device_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  vkGetDeviceQueue(device_, graphicsFamily, 0, &graphics_queue_);
  vkGetDeviceQueue(device_, graphicsFamily, 0, &present_queue_);
  graphics_queue_family_ = graphicsFamily;
  present_queue_family_ = graphicsFamily;
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
  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = swapchain_image_format_;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;

  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  if (vkCreateRenderPass(device_, &renderPassInfo, nullptr, &render_pass_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create render pass!");
  }
}
void VulkanCore::create_msaa_resources() {}
void VulkanCore::create_depth_resources() {}
void VulkanCore::create_framebuffers() {
  framebuffers_.resize(swapchain_image_views_.size());

  for (size_t i = 0; i < swapchain_image_views_.size(); i++) {
    VkImageView attachments[] = {swapchain_image_views_[i]};

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = render_pass_;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = swapchain_extent_.width;
    framebufferInfo.height = swapchain_extent_.height;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(device_, &framebufferInfo, nullptr,
                            &framebuffers_[i]) != VK_SUCCESS) {
      throw std::runtime_error("failed to create framebuffer!");
    }
  }
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

  if (swapchain_ != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
  }

  if (render_pass_ != VK_NULL_HANDLE) {
    vkDestroyRenderPass(device_, render_pass_, nullptr);
    render_pass_ = VK_NULL_HANDLE;
  }
}
void VulkanCore::init_imgui() {
  std::cout << "[VulkanCore] init_imgui: Instance=" << instance_
            << " Device=" << device_ << " PhysDevice=" << physical_device_
            << std::endl;
  std::cout << "[VulkanCore] init_imgui: DescPool=" << descriptor_pool_
            << " RenderPass=" << render_pass_ << std::endl;

  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.ApiVersion = VK_API_VERSION_1_2;
  init_info.Instance = instance_;
  init_info.PhysicalDevice = physical_device_;
  init_info.Device = device_;
  init_info.QueueFamily = graphics_queue_family_;
  init_info.Queue = graphics_queue_;
  init_info.DescriptorPool = descriptor_pool_;
  init_info.PipelineInfoMain.RenderPass = render_pass_;
  init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
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

  if (!ImGui_ImplVulkan_Init(&init_info)) {
    throw std::runtime_error("failed to initialize ImGui Vulkan backend!");
  }

  // Fonts are uploaded automatically by ImGui_ImplVulkan_NewFrame() the first
  // time.
}

void VulkanCore::cleanup_imgui() { ImGui_ImplVulkan_Shutdown(); }

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
  (void)device;
  return true;
}
VkSampleCountFlagBits VulkanCore::get_max_usable_sample_count() {
  return VK_SAMPLE_COUNT_1_BIT;
}
VkFormat VulkanCore::find_supported_format(
    const std::vector<VkFormat> &candidates,
    [[maybe_unused]] VkImageTiling tiling,
    [[maybe_unused]] VkFormatFeatureFlags features) {
  return candidates[0];
}
VkFormat VulkanCore::find_depth_format() { return VK_FORMAT_D32_SFLOAT; }

} // namespace BTQuant
