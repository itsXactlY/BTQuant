#include "vulkan_dashboard_advanced.hpp"
#include "data_visualization_engine.hpp"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include <unistd.h>
#include <vulkan/vulkan_core.h>

// Stub implementations for now to fix linking

namespace BTQuant {

const std::vector<const char *> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};

// VulkanErrorHandler implementations
VkDebugUtilsMessengerEXT VulkanErrorHandler::debug_messenger_ = VK_NULL_HANDLE;

void VulkanErrorHandler::setup_debug_messenger(VkInstance instance) {
  // Stub
}

void VulkanErrorHandler::cleanup_debug_messenger(VkInstance instance) {
  // Stub
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanErrorHandler::debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
    void *user_data) {

  // Log Vulkan validation messages
  std::string severity_str;
  switch (message_severity) {
  case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
    severity_str = "VERBOSE";
    break;
  case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
    severity_str = "INFO";
    break;
  case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
    severity_str = "WARNING";
    break;
  case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
    severity_str = "ERROR";
    break;
  default:
    severity_str = "UNKNOWN";
  }

  std::string type_str;
  switch (message_type) {
  case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
    type_str = "GENERAL";
    break;
  case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
    type_str = "VALIDATION";
    break;
  case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
    type_str = "PERFORMANCE";
    break;
  default:
    type_str = "UNKNOWN";
  }

  fprintf(stderr, "[VULKAN %s %s] %s\n", severity_str.c_str(), type_str.c_str(),
          callback_data->pMessage);

  // Return VK_FALSE to indicate we don't want to abort the call
  return VK_FALSE;
}

// MemoryPool implementation
MemoryPool::MemoryPool(VkDevice device, VkPhysicalDevice physical_device,
                       VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags properties, VkDeviceSize pool_size)
    : device_(device), pool_size_(pool_size), used_size_(0) {

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = pool_size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VulkanErrorHandler::check_result(
      vkCreateBuffer(device_, &buffer_info, nullptr, &pool_buffer_),
      "vkCreateBuffer (MemoryPool)");

  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device_, pool_buffer_, &mem_requirements);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex = find_memory_type(
      physical_device, mem_requirements.memoryTypeBits, properties);

  VulkanErrorHandler::check_result(
      vkAllocateMemory(device_, &alloc_info, nullptr, &pool_memory_),
      "vkAllocateMemory (MemoryPool)");

  vkBindBufferMemory(device_, pool_buffer_, pool_memory_, 0);

  if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    vkMapMemory(device_, pool_memory_, 0, pool_size_, 0, &mapped_ptr_);
  } else {
    mapped_ptr_ = nullptr;
  }

  // Initialize with one large free block
  free_blocks_.push_back({0, pool_size});
}

MemoryPool::~MemoryPool() {
  if (mapped_ptr_) {
    vkUnmapMemory(device_, pool_memory_);
  }
  vkDestroyBuffer(device_, pool_buffer_, nullptr);
  vkFreeMemory(device_, pool_memory_, nullptr);
}

BufferAllocation MemoryPool::allocate(VkDeviceSize size,
                                      VkDeviceSize alignment) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);

  for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
    VkDeviceSize aligned_offset =
        (it->offset + alignment - 1) & ~(alignment - 1);
    VkDeviceSize actual_size = size + (aligned_offset - it->offset);

    if (it->size >= actual_size) {
      BufferAllocation allocation;
      allocation.buffer = pool_buffer_;
      allocation.memory = pool_memory_;
      allocation.offset = aligned_offset;
      allocation.size = size;
      allocation.mapped_ptr =
          mapped_ptr_ ? static_cast<char *>(mapped_ptr_) + aligned_offset
                      : nullptr;
      allocation.is_mapped = (mapped_ptr_ != nullptr);

      // Update free blocks
      VkDeviceSize remaining_size = it->size - actual_size;
      if (remaining_size > 0) {
        it->offset = aligned_offset + size;
        it->size = remaining_size;
      } else {
        free_blocks_.erase(it);
      }

      used_size_ += actual_size;
      return allocation;
    }
  }

  throw std::runtime_error("MemoryPool: Out of memory");
}

void MemoryPool::deallocate(const BufferAllocation &allocation) {
  std::lock_guard<std::mutex> lock(allocation_mutex_);
  // Simple deallocation for now, in a production system we'd merge adjacent
  // blocks
  free_blocks_.push_back({allocation.offset, allocation.size});
  used_size_ -= allocation.size;
}

uint32_t MemoryPool::find_memory_type(VkPhysicalDevice physical_device,
                                      uint32_t type_filter,
                                      VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type!");
}

// GPUMemoryManager implementation
GPUMemoryManager::GPUMemoryManager(VkDevice device,
                                   VkPhysicalDevice physical_device,
                                   const DashboardConfig &config)
    : device_(device) {

  vertex_pool_ = std::make_unique<MemoryPool>(
      device, physical_device,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      config.vertex_pool_size);

  uniform_pool_ = std::make_unique<MemoryPool>(
      device, physical_device,
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      config.uniform_pool_size);

  storage_pool_ = std::make_unique<MemoryPool>(
      device, physical_device,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, config.storage_pool_size);
}

GPUMemoryManager::~GPUMemoryManager() {}

BufferAllocation GPUMemoryManager::allocate_vertex_buffer(VkDeviceSize size) {
  return vertex_pool_->allocate(size, 16);
}

BufferAllocation GPUMemoryManager::allocate_index_buffer(VkDeviceSize size) {
  return vertex_pool_->allocate(size, 16);
}

BufferAllocation GPUMemoryManager::allocate_uniform_buffer(VkDeviceSize size) {
  return uniform_pool_->allocate(size, 256); // Often required by hardware
}

BufferAllocation GPUMemoryManager::allocate_storage_buffer(VkDeviceSize size) {
  return storage_pool_->allocate(size, 16);
}

BufferAllocation GPUMemoryManager::allocate_staging_buffer(VkDeviceSize size) {
  return vertex_pool_->allocate(size, 1);
}

void GPUMemoryManager::deallocate_buffer(const BufferAllocation &allocation) {
  // In this simple pool system, we need to know which pool it came from
  // For now, deallocate is a no-op or we could add pool ID to BufferAllocation
}

GPUMemoryManager::MemoryStats GPUMemoryManager::get_memory_stats() const {
  MemoryStats stats;
  stats.vertex_pool_used = vertex_pool_->get_used_size();
  stats.uniform_pool_used = uniform_pool_->get_used_size();
  stats.storage_pool_used = storage_pool_->get_used_size();
  stats.vertex_pool_usage = vertex_pool_->get_usage_percentage();
  stats.uniform_pool_usage = uniform_pool_->get_usage_percentage();
  stats.storage_pool_usage = storage_pool_->get_usage_percentage();
  return stats;
}

// VulkanCore implementation
VulkanCore::VulkanCore(const DashboardConfig &config) : config_(config) {}

VulkanCore::~VulkanCore() { cleanup(); }

void VulkanCore::initialize(Display *display, Window window, uint32_t width,
                            uint32_t height) {
  fprintf(stderr, "[VulkanCore] Initializing Vulkan with window %dx%d\n", width,
          height);

  create_instance();
  create_surface(display, window);
  select_physical_device();
  create_logical_device();

  // Initialize memory manager after device is created
  memory_manager_ =
      std::make_unique<GPUMemoryManager>(device_, physical_device_, config_);

  create_swapchain(width, height);
  create_image_views();
  create_render_pass();

  if (config_.enable_msaa) {
    create_msaa_resources();
  }
  create_depth_resources();

  create_framebuffers();
  create_command_pool();
  create_command_buffers();
  create_descriptor_pool();
  create_sync_objects();

  init_imgui();

  last_frame_time_ = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "[VulkanCore] Vulkan initialization complete\n");
}

void VulkanCore::cleanup() {
  fprintf(stderr, "[VulkanCore] Cleaning up Vulkan resources\n");

  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);

    cleanup_imgui();
    cleanup_swapchain();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(device_, render_finished_semaphores_[i], nullptr);
      vkDestroySemaphore(device_, image_available_semaphores_[i], nullptr);
      vkDestroyFence(device_, in_flight_fences_[i], nullptr);
    }

    vkDestroyCommandPool(device_, command_pool_, nullptr);

    if (descriptor_pool_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    }

    // Memory manager should be destroyed before device
    memory_manager_.reset();

    vkDestroyDevice(device_, nullptr);
  }

  if (instance_ != VK_NULL_HANDLE) {
    if (surface_ != VK_NULL_HANDLE) {
      vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    VulkanErrorHandler::cleanup_debug_messenger(instance_);
    vkDestroyInstance(instance_, nullptr);
  }
}

void VulkanCore::cleanup_swapchain() {
  for (auto framebuffer : framebuffers_) {
    vkDestroyFramebuffer(device_, framebuffer, nullptr);
  }

  for (auto imageView : swapchain_image_views_) {
    vkDestroyImageView(device_, imageView, nullptr);
  }

  vkDestroySwapchainKHR(device_, swapchain_, nullptr);

  if (msaa_color_image_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_, msaa_color_image_view_, nullptr);
    vkDestroyImage(device_, msaa_color_image_, nullptr);
    vkFreeMemory(device_, msaa_color_memory_, nullptr);
  }

  if (depth_image_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_, depth_image_view_, nullptr);
    vkDestroyImage(device_, depth_image_, nullptr);
    vkFreeMemory(device_, depth_memory_, nullptr);
  }
}

void VulkanCore::recreate_swapchain(uint32_t width, uint32_t height) {
  vkDeviceWaitIdle(device_);

  cleanup_swapchain();

  create_swapchain(width, height);
  create_image_views();

  if (config_.enable_msaa) {
    create_msaa_resources();
  }
  create_depth_resources();

  create_framebuffers();
}

bool VulkanCore::begin_frame() {
  fprintf(stderr, "[VulkanCore] Beginning frame\n");
  fflush(stderr);

  // Update ImGui IO
  ImGuiIO &io = ImGui::GetIO();
  auto now = std::chrono::high_resolution_clock::now();
  float dt = std::chrono::duration<float>(now - last_frame_time_).count();
  if (dt <= 0.0f)
    dt = 1.0f / 60.0f; // Fallback
  io.DeltaTime = dt;
  last_frame_time_ = now;
  io.DisplaySize =
      ImVec2((float)swapchain_extent_.width, (float)swapchain_extent_.height);

  // Start ImGui frame BEFORE recording any commands
  ImGui_ImplVulkan_NewFrame();
  ImGui::NewFrame();

  vkWaitForFences(device_, 1, &in_flight_fences_[current_frame_], VK_TRUE,
                  UINT64_MAX);

  VkResult result =
      vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                            image_available_semaphores_[current_frame_],
                            VK_NULL_HANDLE, &current_image_index_);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    recreate_swapchain(swapchain_extent_.width, swapchain_extent_.height);
    return false;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    throw VulkanException(result, "vkAcquireNextImageKHR");
  }

  vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);
  return true;
}

bool VulkanCore::prepare_frame() {
  vkResetCommandBuffer(command_buffers_[current_frame_], 0);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = 0;
  begin_info.pInheritanceInfo = nullptr;

  VulkanErrorHandler::check_result(
      vkBeginCommandBuffer(command_buffers_[current_frame_], &begin_info),
      "vkBeginCommandBuffer");

  VkRenderPassBeginInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_info.renderPass = render_pass_;
  render_pass_info.framebuffer = framebuffers_[current_image_index_];
  render_pass_info.renderArea.offset = {0, 0};
  render_pass_info.renderArea.extent = swapchain_extent_;

  VkClearValue clear_color = {
      {{0.05f, 0.05f, 0.07f, 1.0f}}}; // Deep dark background
  render_pass_info.clearValueCount = 1;
  render_pass_info.pClearValues = &clear_color;

  vkCmdBeginRenderPass(command_buffers_[current_frame_], &render_pass_info,
                       VK_SUBPASS_CONTENTS_INLINE);

  current_command_buffer_ = command_buffers_[current_frame_];
  return true;
}

void VulkanCore::end_frame() {
  // Render ImGui inside the render pass
  ImGui::Render();
  ImDrawData *draw_data = ImGui::GetDrawData();
  ImGui_ImplVulkan_RenderDrawData(draw_data, current_command_buffer_);

  vkCmdEndRenderPass(current_command_buffer_);

  VulkanErrorHandler::check_result(vkEndCommandBuffer(current_command_buffer_),
                                   "vkEndCommandBuffer");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  VkSemaphore wait_semaphores[] = {image_available_semaphores_[current_frame_]};
  VkPipelineStageFlags wait_stages[] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &current_command_buffer_;

  VkSemaphore signal_semaphores[] = {
      render_finished_semaphores_[current_frame_]};
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;

  VulkanErrorHandler::check_result(
      vkQueueSubmit(graphics_queue_, 1, &submit_info,
                    in_flight_fences_[current_frame_]),
      "vkQueueSubmit");

  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;

  VkSwapchainKHR swapchains[] = {swapchain_};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapchains;
  present_info.pImageIndices = &current_image_index_;

  VkResult result = vkQueuePresentKHR(present_queue_, &present_info);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    recreate_swapchain(swapchain_extent_.width, swapchain_extent_.height);
  } else if (result != VK_SUCCESS) {
    throw VulkanException(result, "vkQueuePresentKHR");
  }

  current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanCore::create_instance() {
  fprintf(stderr, "[VulkanCore] Creating Vulkan instance\n");

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "BTQuant Dashboard";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "BTQuant Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  fprintf(stderr, "[VulkanCore] Calling get_required_extensions()\n");
  auto extensions = get_required_extensions();
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

  fprintf(stderr,
          "[VulkanCore] Checking validation layers (config.enable=%d)\n",
          config_.enable_validation_layers);
  if (config_.enable_validation_layers && check_validation_layer_support()) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    create_info.ppEnabledLayerNames = validation_layers.data();
  } else {
    create_info.enabledLayerCount = 0;
  }

  fprintf(stderr, "[VulkanCore] Calling vkCreateInstance...\n");
  VulkanErrorHandler::check_result(
      vkCreateInstance(&create_info, nullptr, &instance_), "vkCreateInstance");

  fprintf(stderr, "[VulkanCore] vkCreateInstance returned success\n");

  if (config_.enable_validation_layers && check_validation_layer_support()) {
    VulkanErrorHandler::setup_debug_messenger(instance_);
  }

  fprintf(stderr, "[VulkanCore] Vulkan instance created successfully\n");
}

void VulkanCore::select_physical_device() {
  fprintf(stderr, "[VulkanCore] Selecting physical device\n");

  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

  if (device_count == 0) {
    throw std::runtime_error("Failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

  for (const auto &device : devices) {
    if (is_device_suitable(device)) {
      physical_device_ = device;
      break;
    }
  }

  if (physical_device_ == VK_NULL_HANDLE) {
    throw std::runtime_error("Failed to find a suitable GPU!");
  }

  fprintf(stderr, "[VulkanCore] Physical device selected\n");
}

void VulkanCore::create_logical_device() {
  fprintf(stderr, "[VulkanCore] Creating logical device\n");

  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set<uint32_t> unique_queue_families = {graphics_queue_family_,
                                              present_queue_family_};

  if (compute_queue_family_ != UINT32_MAX) {
    unique_queue_families.insert(compute_queue_family_);
  }

  float queue_priority = 1.0f;
  for (uint32_t queue_family : unique_queue_families) {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }

  VkPhysicalDeviceFeatures device_features{};
  // device_features.geometryShader = VK_TRUE; // Disabled for compatibility
  device_features.samplerAnisotropy = VK_TRUE;

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.queueCreateInfoCount =
      static_cast<uint32_t>(queue_create_infos.size());
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.pEnabledFeatures = &device_features;

  std::vector<const char *> device_extensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  create_info.enabledExtensionCount =
      static_cast<uint32_t>(device_extensions.size());
  create_info.ppEnabledExtensionNames = device_extensions.data();

  if (config_.enable_validation_layers) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    create_info.ppEnabledLayerNames = validation_layers.data();
  } else {
    create_info.enabledLayerCount = 0;
  }

  VulkanErrorHandler::check_result(
      vkCreateDevice(physical_device_, &create_info, nullptr, &device_),
      "vkCreateDevice");

  // Get queue handles
  vkGetDeviceQueue(device_, graphics_queue_family_, 0, &graphics_queue_);
  vkGetDeviceQueue(device_, present_queue_family_, 0, &present_queue_);
  if (compute_queue_family_ != UINT32_MAX) {
    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
  }

  fprintf(stderr, "[VulkanCore] Logical device created\n");
}

void VulkanCore::create_surface(Display *display, Window window) {
  fprintf(stderr, "[VulkanCore] Creating Vulkan surface\n");

  VkXlibSurfaceCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
  create_info.dpy = display;
  create_info.window = window;

  VulkanErrorHandler::check_result(
      vkCreateXlibSurfaceKHR(instance_, &create_info, nullptr, &surface_),
      "vkCreateXlibSurfaceKHR");

  fprintf(stderr, "[VulkanCore] Vulkan surface created\n");
}

void VulkanCore::create_swapchain(uint32_t width, uint32_t height) {
  fprintf(stderr, "[VulkanCore] Creating swapchain %dx%d\n", width, height);

  VkSurfaceCapabilitiesKHR capabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_,
                                            &capabilities);

  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_,
                                       &format_count, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(format_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_,
                                       &format_count, formats.data());

  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device_, surface_,
                                            &present_mode_count, nullptr);
  std::vector<VkPresentModeKHR> present_modes(present_mode_count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(
      physical_device_, surface_, &present_mode_count, present_modes.data());

  VkSurfaceFormatKHR surface_format = formats[0];
  for (const auto &format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      surface_format = format;
      break;
    }
  }

  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
  for (const auto &mode : present_modes) {
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      present_mode = mode;
      break;
    }
  }

  VkExtent2D extent = {width, height};
  if (capabilities.currentExtent.width != UINT32_MAX) {
    extent = capabilities.currentExtent;
  } else {
    extent.width =
        std::max(capabilities.minImageExtent.width,
                 std::min(capabilities.maxImageExtent.width, extent.width));
    extent.height =
        std::max(capabilities.minImageExtent.height,
                 std::min(capabilities.maxImageExtent.height, extent.height));
  }

  uint32_t image_count = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 &&
      image_count > capabilities.maxImageCount) {
    image_count = capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = surface_;
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  uint32_t queue_family_indices[] = {graphics_queue_family_,
                                     present_queue_family_};
  if (graphics_queue_family_ != present_queue_family_) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_family_indices;
  } else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  create_info.preTransform = capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;

  VulkanErrorHandler::check_result(
      vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_),
      "vkCreateSwapchainKHR");

  vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
  swapchain_images_.resize(image_count);
  vkGetSwapchainImagesKHR(device_, swapchain_, &image_count,
                          swapchain_images_.data());

  swapchain_image_format_ = surface_format.format;
  swapchain_extent_ = extent;

  fprintf(stderr, "[VulkanCore] Swapchain created with %d images\n",
          image_count);
}

void VulkanCore::create_image_views() {
  fprintf(stderr, "[VulkanCore] Creating image views\n");

  swapchain_image_views_.resize(swapchain_images_.size());

  for (size_t i = 0; i < swapchain_images_.size(); i++) {
    VkImageViewCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = swapchain_images_[i];
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = swapchain_image_format_;
    create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    VulkanErrorHandler::check_result(
        vkCreateImageView(device_, &create_info, nullptr,
                          &swapchain_image_views_[i]),
        "vkCreateImageView");
  }

  fprintf(stderr, "[VulkanCore] Image views created\n");
}

void VulkanCore::create_render_pass() {
  fprintf(stderr, "[VulkanCore] Creating render pass\n");

  VkAttachmentDescription color_attachment{};
  color_attachment.format = swapchain_image_format_;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref{};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  VkRenderPassCreateInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  render_pass_info.attachmentCount = 1;
  render_pass_info.pAttachments = &color_attachment;
  render_pass_info.subpassCount = 1;
  render_pass_info.pSubpasses = &subpass;

  VulkanErrorHandler::check_result(
      vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_),
      "vkCreateRenderPass");

  fprintf(stderr, "[VulkanCore] Render pass created\n");
}

void VulkanCore::create_msaa_resources() {
  // Stub
}

void VulkanCore::create_depth_resources() {
  // Stub
}

void VulkanCore::create_framebuffers() {
  fprintf(stderr, "[VulkanCore] Creating framebuffers\n");

  framebuffers_.resize(swapchain_image_views_.size());

  for (size_t i = 0; i < swapchain_image_views_.size(); i++) {
    VkImageView attachments[] = {swapchain_image_views_[i]};

    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass_;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = attachments;
    framebuffer_info.width = swapchain_extent_.width;
    framebuffer_info.height = swapchain_extent_.height;
    framebuffer_info.layers = 1;

    VulkanErrorHandler::check_result(
        vkCreateFramebuffer(device_, &framebuffer_info, nullptr,
                            &framebuffers_[i]),
        "vkCreateFramebuffer");
  }

  fprintf(stderr, "[VulkanCore] Framebuffers created\n");
}

void VulkanCore::create_command_pool() {
  fprintf(stderr, "[VulkanCore] Creating command pool\n");

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = graphics_queue_family_;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  VulkanErrorHandler::check_result(
      vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
      "vkCreateCommandPool");

  fprintf(stderr, "[VulkanCore] Command pool created\n");
}

void VulkanCore::create_command_buffers() {
  fprintf(stderr, "[VulkanCore] Creating command buffers\n");

  command_buffers_.resize(MAX_FRAMES_IN_FLIGHT);

  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool_;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = (uint32_t)command_buffers_.size();

  VulkanErrorHandler::check_result(
      vkAllocateCommandBuffers(device_, &alloc_info, command_buffers_.data()),
      "vkAllocateCommandBuffers");

  fprintf(stderr, "[VulkanCore] Command buffers created\n");
}

void VulkanCore::create_descriptor_pool() {
  fprintf(stderr, "[VulkanCore] Creating descriptor pool\n");

  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100}};

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes));
  pool_info.pPoolSizes = pool_sizes;
  pool_info.maxSets = 300;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  VulkanErrorHandler::check_result(
      vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_),
      "vkCreateDescriptorPool");

  fprintf(stderr, "[VulkanCore] Descriptor pool created\n");
}

void VulkanCore::create_sync_objects() {
  fprintf(stderr, "[VulkanCore] Creating synchronization objects\n");

  image_available_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
  render_finished_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
  in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);

  VkSemaphoreCreateInfo semaphore_info{};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    VulkanErrorHandler::check_result(
        vkCreateSemaphore(device_, &semaphore_info, nullptr,
                          &image_available_semaphores_[i]),
        "vkCreateSemaphore (image_available)");

    VulkanErrorHandler::check_result(
        vkCreateSemaphore(device_, &semaphore_info, nullptr,
                          &render_finished_semaphores_[i]),
        "vkCreateSemaphore (render_finished)");

    VulkanErrorHandler::check_result(
        vkCreateFence(device_, &fence_info, nullptr, &in_flight_fences_[i]),
        "vkCreateFence");
  }
}

static std::vector<char> read_file(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file: " + filename);
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
    VkPipelineLayout layout) {

  auto vert_code = read_file(vert_path);
  auto frag_code = read_file(frag_path);

  VkShaderModule vert_module = create_shader_module(vert_code);
  VkShaderModule frag_module = create_shader_module(frag_code);

  VkPipelineShaderStageCreateInfo vert_stage_info{};
  vert_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vert_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vert_stage_info.module = vert_module;
  vert_stage_info.pName = "main";

  VkPipelineShaderStageCreateInfo frag_stage_info{};
  frag_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  frag_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  frag_stage_info.module = frag_module;
  frag_stage_info.pName = "main";

  VkPipelineShaderStageCreateInfo shader_stages[] = {vert_stage_info,
                                                     frag_stage_info};

  VkPipelineVertexInputStateCreateInfo vertex_input_info{};
  vertex_input_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input_info.vertexBindingDescriptionCount =
      static_cast<uint32_t>(bindings.size());
  vertex_input_info.pVertexBindingDescriptions = bindings.data();
  vertex_input_info.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributes.size());
  vertex_input_info.pVertexAttributeDescriptions = attributes.data();

  VkPipelineInputAssemblyStateCreateInfo input_assembly{};
  input_assembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  input_assembly.primitiveRestartEnable = VK_FALSE;

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

  VkPipelineViewportStateCreateInfo viewport_state{};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

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

  VkPipelineColorBlendAttachmentState color_blend_attachment{};
  color_blend_attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  color_blend_attachment.blendEnable = VK_TRUE;
  color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  color_blend_attachment.dstColorBlendFactor =
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
  color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

  VkPipelineColorBlendStateCreateInfo color_blending{};
  color_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;
  color_blending.blendConstants[0] = 0.0f;
  color_blending.blendConstants[1] = 0.0f;
  color_blending.blendConstants[2] = 0.0f;
  color_blending.blendConstants[3] = 0.0f;

  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount = 2;
  pipeline_info.pStages = shader_stages;
  pipeline_info.pVertexInputState = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.layout = layout;
  pipeline_info.renderPass = render_pass_;
  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

  VkPipeline graphics_pipeline;
  VulkanErrorHandler::check_result(
      vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                                nullptr, &graphics_pipeline),
      "vkCreateGraphicsPipelines");

  vkDestroyShaderModule(device_, frag_module, nullptr);
  vkDestroyShaderModule(device_, vert_module, nullptr);

  return graphics_pipeline;
}

VkShaderModule VulkanCore::create_shader_module(const std::vector<char> &code) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size();
  create_info.pCode = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule shader_module;
  VulkanErrorHandler::check_result(
      vkCreateShaderModule(device_, &create_info, nullptr, &shader_module),
      "vkCreateShaderModule");

  return shader_module;
}

VkPipelineLayout VulkanCore::create_pipeline_layout(
    const std::vector<VkDescriptorSetLayout> &layouts,
    const std::vector<VkPushConstantRange> &push_constants) {

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(layouts.size());
  pipeline_layout_info.pSetLayouts = layouts.data();
  pipeline_layout_info.pushConstantRangeCount =
      static_cast<uint32_t>(push_constants.size());
  pipeline_layout_info.pPushConstantRanges = push_constants.data();

  VkPipelineLayout pipeline_layout;
  VulkanErrorHandler::check_result(
      vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr,
                             &pipeline_layout),
      "vkCreatePipelineLayout");

  return pipeline_layout;
}

std::vector<const char *> VulkanCore::get_required_extensions() {
  fprintf(stderr, "[VulkanCore] Entering get_required_extensions\n");
  std::vector<const char *> extensions;

  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);

  if (config_.enable_validation_layers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  fprintf(stderr,
          "[VulkanCore] get_required_extensions returning %zu extensions\n",
          extensions.size());
  return extensions;
}

bool VulkanCore::check_validation_layer_support() {
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  for (const char *layer_name : validation_layers) {
    bool layer_found = false;

    for (const auto &layer_properties : available_layers) {
      if (strcmp(layer_name, layer_properties.layerName) == 0) {
        layer_found = true;
        break;
      }
    }

    if (!layer_found) {
      return false;
    }
  }

  return true;
}

bool VulkanCore::is_device_suitable(VkPhysicalDevice device) {
  VkPhysicalDeviceProperties device_properties;
  VkPhysicalDeviceFeatures device_features;
  vkGetPhysicalDeviceProperties(device, &device_properties);
  vkGetPhysicalDeviceFeatures(device, &device_features);

  // Check for required features
  // if (!device_features.geometryShader) {
  //     return false;
  // }

  // Check for queue families
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           queue_families.data());

  bool has_graphics_family = false;
  bool has_present_family = false;

  for (uint32_t i = 0; i < queue_family_count; i++) {
    if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      graphics_queue_family_ = i;
      has_graphics_family = true;
    }

    VkBool32 present_support = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_support);

    if (present_support) {
      present_queue_family_ = i;
      has_present_family = true;
    }

    if (has_graphics_family && has_present_family) {
      break;
    }
  }

  // Check for required extensions
  uint32_t extension_count;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       nullptr);

  std::vector<VkExtensionProperties> available_extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       available_extensions.data());

  std::set<std::string> required_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  for (const auto &extension : available_extensions) {
    required_extensions.erase(extension.extensionName);
  }

  return has_graphics_family && has_present_family &&
         required_extensions.empty();
}

VkSampleCountFlagBits VulkanCore::get_max_usable_sample_count() {
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

uint32_t VulkanCore::find_memory_type(uint32_t type_filter,
                                      VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

// VulkanDashboard implementation

// VulkanDashboard implementation
VulkanDashboard::VulkanDashboard(uint32_t width, uint32_t height,
                                 const DashboardConfig &config)
    : config_(config), width_(width), height_(height) {
  fprintf(stderr, "[VulkanDashboard] Creating dashboard %dx%d\n", width,
          height);
}

VulkanDashboard::~VulkanDashboard() {}

void VulkanDashboard::initialize() {
  fprintf(stderr, "[VulkanDashboard] Starting initialization...\n");
  init_x11();
  fprintf(stderr, "[VulkanDashboard] X11 initialized\n");
  init_vulkan();
  fprintf(stderr, "[VulkanDashboard] Vulkan initialized\n");

  // Initialize data components
  market_data_processor_ =
      std::make_unique<RenderEngine::MarketDataProcessor>();
  hotspine_bridge_ =
      std::make_unique<RenderEngine::HotSpineDataBridge>("hotspine_shm");
  visualization_engine_ =
      std::make_unique<RenderEngine::DataVisualizationEngine>(
          vulkan_core_->get_device(), vulkan_core_->get_physical_device());

  init_components();
  fprintf(stderr, "[VulkanDashboard] Components initialized\n");
  init_component_resources();
  fprintf(stderr, "[VulkanDashboard] Component resources initialized\n");
  setup_data_subscriptions();
  fprintf(stderr, "[VulkanDashboard] Data subscriptions setup\n");
}

void VulkanDashboard::main_loop() {
  fprintf(stderr, "[VulkanDashboard] Starting main loop\n");
  bool running = true;
  while (running) {
    fprintf(stderr, "[VulkanDashboard] Loop - Handle X11 events\n");
    fflush(stderr);
    if (handle_x11_events()) {
      running = false; // Window close requested
    }

    // Process real-time data updates
    if (hotspine_bridge_ && market_data_processor_) {
      fprintf(stderr, "[VulkanDashboard] Loop - Get updates from bridge\n");
      fflush(stderr);
      auto updates = hotspine_bridge_->getLatestUpdates();
      for (const auto &update : updates) {
        if (update.type == RenderEngine::MarketDataType::TRADE) {
          market_data_processor_->processTradeUpdate(update);

          // Notify dashboard for UI updates
          RenderEngine::TradeData trade;
          trade.timestamp_us = update.timestamp_us;
          trade.price = update.price;
          trade.size = update.size;
          trade.is_buy = (update.side == "buy");
          on_trade_received(trade);
        } else if (update.type == RenderEngine::MarketDataType::ORDERBOOK) {
          market_data_processor_->processOrderbookUpdate(update);

          // Notify dashboard for UI updates
          RenderEngine::OrderbookData ob;
          ob.timestamp_us = update.timestamp_us;
          ob.bids = update.bids;
          ob.asks = update.asks;
          // Calculate analytical fields if needed by components
          ob.spread = update.price; // Placeholder if not in update
          on_orderbook_updated(ob);
        }
      }
    }

    fprintf(stderr, "[VulkanDashboard] Loop - Update components\n");
    fflush(stderr);
    update_components(0.016f); // ~60fps

    // Unified rendering pass
    if (vulkan_core_ && vulkan_core_->begin_frame()) {
      fprintf(stderr, "[VulkanDashboard] Frame %lu - Begin\n",
              (unsigned long)frame_count_);
      fflush(stderr);

      if (vulkan_core_->prepare_frame()) {
        fprintf(stderr, "[VulkanDashboard] Frame %lu - Render GUI\n",
                (unsigned long)frame_count_);
        fflush(stderr);
        render_gui();

        fprintf(stderr, "[VulkanDashboard] Frame %lu - Render Components\n",
                (unsigned long)frame_count_);
        fflush(stderr);
        vulkan_core_->end_frame();
      }
    }
    update_performance_stats();
    // Small delay to prevent 100% CPU usage
    usleep(16000); // ~60fps
  }
}

void VulkanDashboard::shutdown() {
  fprintf(stderr, "[VulkanDashboard] Shutting down\n");
  stop_market_data_processing();
  components_.clear();
  visualization_engine_.reset();
  hotspine_bridge_.reset();
  market_data_processor_.reset();
  vulkan_core_.reset();
  cleanup_x11();
}

void VulkanDashboard::add_component(std::unique_ptr<UIComponent> component) {
  components_.push_back(std::move(component));
}

void VulkanDashboard::remove_component(UIComponent *component) {
  // Stub
}

void VulkanDashboard::start_market_data_processing() {
  if (hotspine_bridge_) {
    hotspine_bridge_->start();
  }
}

void VulkanDashboard::stop_market_data_processing() {
  if (hotspine_bridge_) {
    hotspine_bridge_->stop();
  }
}

VulkanDashboard::PerformanceStats
VulkanDashboard::get_performance_stats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return current_stats_;
}

void VulkanDashboard::on_trade_received(const RenderEngine::TradeData &trade) {
  for (auto &component : components_) {
    component->handle_trade(trade);
  }
}

void VulkanDashboard::on_orderbook_updated(
    const RenderEngine::OrderbookData &orderbook) {
  for (auto &component : components_) {
    component->handle_orderbook(orderbook);
  }
}

void VulkanDashboard::on_window_resize(uint32_t new_width,
                                       uint32_t new_height) {
  width_ = new_width;
  height_ = new_height;
  if (vulkan_core_) {
    vulkan_core_->recreate_swapchain(width_, height_);
  }
}

void VulkanDashboard::init_components() {
  fprintf(stderr, "[VulkanDashboard] Initializing UI components\n");

  // Add log display component
  auto log_component = std::make_unique<LogDisplayComponent>(
      glm::vec2(10, 410), glm::vec2(600, 300));
  log_component->add_log_entry(LogDisplayComponent::Info,
                               "Vulkan Dashboard initialized");
  log_component->add_log_entry(LogDisplayComponent::Info,
                               "BTQuest Render Engine live");
  add_component(std::move(log_component));

  // Add chart component
  auto chart_component = std::make_unique<RealtimeChartComponent>(
      glm::vec2(10, 10), glm::vec2(800, 380));
  add_component(std::move(chart_component));

  // Add orderbook component
  auto ob_component = std::make_unique<OrderBookComponent>(glm::vec2(820, 10),
                                                           glm::vec2(440, 500));
  add_component(std::move(ob_component));
}

void VulkanDashboard::update_performance_stats() {
  auto now = std::chrono::high_resolution_clock::now();
  static auto last_time = now;
  float duration = std::chrono::duration<float>(now - last_time).count();

  if (duration >= 1.0f) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    current_stats_.fps = frame_count_ / duration;
    frame_count_ = 0;
    last_time = now;
  }
  frame_count_++;
}

bool VulkanDashboard::handle_x11_events() {
  if (!display_)
    return false;
  XEvent event;
  while (XPending(display_)) {
    XNextEvent(display_, &event);

    if (event.type == ClientMessage &&
        static_cast<Atom>(event.xclient.data.l[0]) == wm_delete_window_) {
      return true;
    }

    if (event.type == ConfigureNotify) {
      if (static_cast<uint32_t>(event.xconfigure.width) != width_ ||
          static_cast<uint32_t>(event.xconfigure.height) != height_) {
        on_window_resize(event.xconfigure.width, event.xconfigure.height);
      }
    }

    // Pass events to ImGui
    ImGuiIO &io = ImGui::GetIO();
    if (event.type == X11_KeyPress) {
      // Key mapping would go here
    } else if (event.type == ButtonPress) {
      if (event.xbutton.button == Button1)
        io.MouseDown[0] = true;
      if (event.xbutton.button == Button2)
        io.MouseDown[2] = true;
      if (event.xbutton.button == Button3)
        io.MouseDown[1] = true;
      if (event.xbutton.button == Button4)
        io.MouseWheel += 1.0f;
      if (event.xbutton.button == Button5)
        io.MouseWheel -= 1.0f;
    } else if (event.type == ButtonRelease) {
      if (event.xbutton.button == Button1)
        io.MouseDown[0] = false;
      if (event.xbutton.button == Button2)
        io.MouseDown[2] = false;
      if (event.xbutton.button == Button3)
        io.MouseDown[1] = false;
    } else if (event.type == MotionNotify) {
      io.MousePos = ImVec2((float)event.xmotion.x, (float)event.xmotion.y);
    }

    // TODO: Pass other events to components
  }
  return false;
}

void VulkanCore::init_imgui() {
  fprintf(stderr, "[VulkanCore] Initializing ImGui\n");

  // Create descriptor pool for ImGui
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

  VulkanErrorHandler::check_result(
      vkCreateDescriptorPool(device_, &pool_info, nullptr,
                             &imgui_descriptor_pool_),
      "vkCreateDescriptorPool (ImGui)");

  // Initialize ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui::StyleColorsDark();

  // Initialize ImGui Vulkan implementation
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = instance_;
  init_info.PhysicalDevice = physical_device_;
  init_info.Device = device_;
  init_info.QueueFamily = graphics_queue_family_;
  init_info.Queue = graphics_queue_;
  init_info.DescriptorPool = imgui_descriptor_pool_;
  init_info.MinImageCount = 2; // Usually swapchain image count
  init_info.ImageCount = static_cast<uint32_t>(swapchain_images_.size());
  init_info.MSAASamples =
      config_.enable_msaa ? config_.msaa_samples : VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&init_info, render_pass_);

  // Upload Fonts
  VkCommandBuffer command_buffer =
      command_buffers_[0]; // Use an existing one temporarily
  vkResetCommandBuffer(command_buffer, 0);
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(command_buffer, &begin_info);

  ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

  vkEndCommandBuffer(command_buffer);

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue_);

  ImGui_ImplVulkan_DestroyFontUploadObjects();

  fprintf(stderr, "[VulkanCore] ImGui initialized successfully\n");
}

void VulkanCore::cleanup_imgui() {
  if (imgui_descriptor_pool_ != VK_NULL_HANDLE) {
    ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(device_, imgui_descriptor_pool_, nullptr);
    imgui_descriptor_pool_ = VK_NULL_HANDLE;
  }
}

VkCommandBuffer VulkanCore::begin_single_time_commands() {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = command_pool_;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer, &begin_info);

  return command_buffer;
}

void VulkanCore::end_single_time_commands(VkCommandBuffer command_buffer) {
  vkEndCommandBuffer(command_buffer);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  vkQueueSubmit(graphics_queue_, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(graphics_queue_);

  vkFreeCommandBuffers(device_, command_pool_, 1, &command_buffer);
}

void VulkanCore::create_placeholder_texture(VkImage &image,
                                            VkDeviceMemory &memory,
                                            VkImageView &view,
                                            VkSampler &sampler) {
  // 1. Create 1x1 image
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.extent.width = 1;
  image_info.extent.height = 1;
  image_info.extent.depth = 1;
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VulkanErrorHandler::check_result(
      vkCreateImage(device_, &image_info, nullptr, &image), "vkCreateImage");

  VkMemoryRequirements mem_reqs;
  vkGetImageMemoryRequirements(device_, image, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = find_memory_type(
      mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VulkanErrorHandler::check_result(
      vkAllocateMemory(device_, &alloc_info, nullptr, &memory),
      "vkAllocateMemory");

  vkBindImageMemory(device_, image, memory, 0);

  // 2. Upload white pixel
  BufferAllocation staging = memory_manager_->allocate_staging_buffer(4);
  if (staging.mapped_ptr) {
    uint8_t white[] = {255, 255, 255, 255};
    memcpy(staging.mapped_ptr, white, 4);

    // Copy to image (requires command buffer)
    VkCommandBuffer cmd = begin_single_time_commands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.bufferOffset = staging.offset;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {1, 1, 1};

    vkCmdCopyBufferToImage(cmd, staging.buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);

    end_single_time_commands(cmd);
  }

  // 3. Create image view
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;

  VulkanErrorHandler::check_result(
      vkCreateImageView(device_, &view_info, nullptr, &view),
      "vkCreateImageView");

  // 4. Create sampler
  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.maxAnisotropy = 1.0f;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  VulkanErrorHandler::check_result(
      vkCreateSampler(device_, &sampler_info, nullptr, &sampler),
      "vkCreateSampler");
}

void VulkanDashboard::init_x11() {
  display_ = XOpenDisplay(nullptr);
  if (!display_)
    throw std::runtime_error("Cannot open X11 display");

  int screen = DefaultScreen(display_);
  window_ = XCreateSimpleWindow(
      display_, RootWindow(display_, screen), 0, 0, width_, height_, 1,
      BlackPixel(display_, screen), BlackPixel(display_, screen));

  XSelectInput(display_, window_,
               ExposureMask | StructureNotifyMask | KeyPressMask |
                   ButtonPressMask | PointerMotionMask);

  XStoreName(display_, window_, "BTQuant Vulkan Dashboard");
  XMapWindow(display_, window_);
  XFlush(display_);

  wm_delete_window_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(display_, window_, &wm_delete_window_, 1);
}

void VulkanDashboard::init_vulkan() {
  vulkan_core_ = std::make_unique<VulkanCore>(config_);
  vulkan_core_->initialize(display_, window_, width_, height_);
}

void VulkanDashboard::setup_data_subscriptions() {
  if (!hotspine_bridge_ || !market_data_processor_)
    return;

  // Bridge bridge -> processor subscriptions would happen here
  // In the future: hotspine_bridge_->on_data([...](...) {
  // market_data_processor_->process(...); });
}

void VulkanDashboard::init_component_resources() {
  for (auto &component : components_) {
    component->initialize_vulkan_resources(vulkan_core_.get());
  }
}

void VulkanDashboard::update_components(float delta_time) {
  for (auto &component : components_) {
    component->update(delta_time);
  }
}

void VulkanDashboard::render_components() {
  if (vulkan_core_) {
    vulkan_core_->begin_frame();
    VkCommandBuffer cmd = vulkan_core_->get_current_command_buffer();

    // Render components
    for (auto &component : components_) {
      if (component->is_visible()) {
        component->render(cmd);
      }
    }
    vulkan_core_->end_frame();
  }
}

void VulkanDashboard::cleanup_x11() {
  if (display_ && window_) {
    XDestroyWindow(display_, window_);
    XCloseDisplay(display_);
  }
}

void VulkanDashboard::render_gui() {
  // Main Menu Bar
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Exit", "Alt+F4")) {
        // Handle exit
      }
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Reset Layout")) {
        // Handle reset
      }
      ImGui::EndMenu();
    }

    // Status info on the right
    float width = ImGui::GetWindowWidth();
    ImGui::SameLine(width - 400);
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "CONNECTED");
    ImGui::SameLine();
    ImGui::Text("| FPS: %.1f", current_stats_.fps);
    ImGui::SameLine();
    ImGui::Text("| Latency: %.2fms", current_stats_.data_latency_ms);

    ImGui::EndMainMenuBar();
  }

  // Dashboard Control Panel
  ImGui::Begin("Dashboard Controls");
  ImGui::Text("Market Data Status: Running");
  if (ImGui::Button("Stop Market Data")) {
    stop_market_data_processing();
  }
  ImGui::SameLine();
  if (ImGui::Button("Start Market Data")) {
    start_market_data_processing();
  }

  ImGui::Separator();
  ImGui::Text("Visible Components:");
  for (auto &component : components_) {
    bool visible = component->is_visible();
    if (ImGui::Checkbox(component->get_position().x < 400 ? "Left Panel"
                                                          : "Right Panel",
                        &visible)) {
      component->set_visible(visible);
    }
  }

  ImGui::End();

  // Render Component GUI Windows
  for (auto &component : components_) {
    if (component && component->is_visible()) {
      component->render_gui();
    }
  }
}

} // namespace BTQuant