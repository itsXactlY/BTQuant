#define VK_USE_PLATFORM_XLIB_KHR
#include "../../include/vulkan_base_types.hpp"
#include "imgui.h"
#include <iostream>
#include <vector>

namespace BTQuant {

VulkanCore::VulkanCore(const VulkanDashboardConfig &config) : config_(config) {}

VulkanCore::~VulkanCore() { cleanup(); }

void VulkanCore::initialize(Display *display, Window window, uint32_t width,
                            uint32_t height) {
  // Basic Vulkan initialization logic
  create_instance();
  create_surface(display, window);
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

void VulkanCore::init_vulkan_components() {
  // Additional initialization if needed
}

void VulkanCore::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
    // cleanup_imgui();
    // cleanup_swapchain();

    memory_manager_.reset();

    if (descriptor_pool_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
      descriptor_pool_ = VK_NULL_HANDLE;
    }

    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, command_pool_, nullptr);
      command_pool_ = VK_NULL_HANDLE;
    }

    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }

  if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
  }

  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }
}

bool VulkanCore::begin_frame() {
  current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
  // In a real implementation we would wait for fences and acquire image here.
  // For now, we just ensure a command buffer is available for components to
  // use.
  if (command_buffers_.empty()) {
    create_command_buffers();
  }
  current_command_buffer_ = command_buffers_[current_frame_];
  return true;
}

void VulkanCore::begin_command_buffer() {
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(current_command_buffer_, &beginInfo);
}

void VulkanCore::begin_main_render_pass() {
  // Logic to begin main render pass on the swapchain
}

void VulkanCore::end_frame() {
  if (current_command_buffer_ != VK_NULL_HANDLE) {
    vkEndCommandBuffer(current_command_buffer_);
    // In a real implementation, we would submit to the queue and present.
  }
}

VkPipeline VulkanCore::create_graphics_pipeline(
    const std::string &vert_path, const std::string &frag_path,
    const std::vector<VkVertexInputBindingDescription> &bindings,
    const std::vector<VkVertexInputAttributeDescription> &attributes,
    VkPipelineLayout layout, VkRenderPass render_pass) {
  (void)vert_path;
  (void)frag_path;
  (void)bindings;
  (void)attributes;
  (void)layout;
  (void)render_pass;
  // Minimal pipeline creation logic
  return VK_NULL_HANDLE;
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
  cleanup_swapchain();
  create_swapchain(width, height);
  create_image_views();
  create_depth_resources();
  create_msaa_resources();
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
  (void)view;
  return VK_NULL_HANDLE;
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
    if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      graphicsFamily = i;
      break;
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
  graphics_queue_family_ = graphicsFamily;
}

void VulkanCore::create_surface(Display *display, Window window) {
  VkXlibSurfaceCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
  createInfo.dpy = display;
  createInfo.window = window;

  if (vkCreateXlibSurfaceKHR(instance_, &createInfo, nullptr, &surface_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface!");
  }
}
void VulkanCore::create_swapchain(uint32_t width, uint32_t height) {
  (void)width;
  (void)height;
}
void VulkanCore::create_image_views() {}
void VulkanCore::create_render_pass() {}
void VulkanCore::create_msaa_resources() {}
void VulkanCore::create_depth_resources() {}
void VulkanCore::create_framebuffers() {}
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

  if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptor_pool_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}
void VulkanCore::create_sync_objects() {}
void VulkanCore::cleanup_swapchain() {}
void VulkanCore::init_imgui() {
  // Normally we would call ImGui_ImplX11_Init and ImGui_ImplVulkan_Init here.
  // For the fix, we ensure the IO structure is at least minimally populated
  // if NewFrame is going to be called.
  ImGuiIO &io = ImGui::GetIO();
  io.DisplaySize =
      ImVec2((float)swapchain_extent_.width, (float)swapchain_extent_.height);
  if (io.DisplaySize.x <= 0.0f)
    io.DisplaySize.x = 800.0f;
  if (io.DisplaySize.y <= 0.0f)
    io.DisplaySize.y = 600.0f;

  io.DeltaTime = 1.0f / 60.0f;

  unsigned char *pixels;
  int width, height;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

  // Set backend names to avoid null derefs if ImGui checks them
  io.BackendPlatformName = "imgui_impl_btquant_stub";
  io.BackendRendererName = "imgui_impl_btquant_vulkan_stub";
}

void VulkanCore::cleanup_imgui() {
  // Normally ImGui_ImplVulkan_Shutdown(); ImGui_ImplX11_Shutdown();
}
std::vector<const char *> VulkanCore::get_required_extensions() {
  std::vector<const char *> extensions;
  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
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

void VulkanCore::create_default_sampler() {}

} // namespace BTQuant
