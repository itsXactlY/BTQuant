#include "vulkan_dashboard_advanced.hpp"
#include "data_visualization_engine.hpp"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
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

  // Phase 7 Optimization: Ensure Host Coherent memory for mapped buffers to
  // avoid manual flushing
  if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    properties |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }

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
  std::lock_guard lock(allocation_mutex_);

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
  std::lock_guard lock(allocation_mutex_);
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
  auto alloc = vertex_pool_->allocate(size, 16);
  alloc.pool_id = 1;
  return alloc;
}

BufferAllocation GPUMemoryManager::allocate_index_buffer(VkDeviceSize size) {
  auto alloc = vertex_pool_->allocate(size, 16);
  alloc.pool_id = 1;
  return alloc;
}

BufferAllocation GPUMemoryManager::allocate_uniform_buffer(VkDeviceSize size) {
  auto alloc = uniform_pool_->allocate(size, 256); // Often required by hardware
  alloc.pool_id = 2;
  return alloc;
}

BufferAllocation GPUMemoryManager::allocate_storage_buffer(VkDeviceSize size) {
  auto alloc = storage_pool_->allocate(size, 16);
  alloc.pool_id = 3;
  return alloc;
}

BufferAllocation GPUMemoryManager::allocate_staging_buffer(VkDeviceSize size) {
  auto alloc = vertex_pool_->allocate(size, 1);
  alloc.pool_id = 4;
  return alloc;
}

void GPUMemoryManager::deallocate_buffer(const BufferAllocation &allocation) {
  switch (allocation.pool_id) {
  case 1:
  case 4:
    vertex_pool_->deallocate(allocation);
    break;
  case 2:
    uniform_pool_->deallocate(allocation);
    break;
  case 3:
    storage_pool_->deallocate(allocation);
    break;
  default:
    break;
  }
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

void VulkanCore::create_default_sampler() {
  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.maxAnisotropy = 1.0f;
  sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_info.mipLodBias = 0.0f;
  sampler_info.minLod = 0.0f;
  sampler_info.maxLod = 0.0f;

  VulkanErrorHandler::check_result(
      vkCreateSampler(device_, &sampler_info, nullptr, &default_sampler_),
      "vkCreateSampler (Default)");
}

VkDescriptorSet VulkanCore::create_texture_descriptor(VkImageView view) {
  if (imgui_descriptor_pool_ == VK_NULL_HANDLE)
    return VK_NULL_HANDLE;
  return ImGui_ImplVulkan_AddTexture(default_sampler_, view,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// OffscreenChartRenderer Implementation
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
  }
}

void OffscreenChartRenderer::resize(uint32_t width, uint32_t height) {
  if (width == width_ && height == height_)
    return;
  if (width == 0 || height == 0)
    return;
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
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

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

  VulkanErrorHandler::check_result(
      vkCreateRenderPass(device, &renderPassInfo, nullptr, &render_pass_),
      "vkCreateRenderPass (Offscreen)");

  // 2. Create Image
  VkImageCreateInfo imageInfo{};
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
  imageInfo.flags = 0;

  VulkanErrorHandler::check_result(
      vkCreateImage(device, &imageInfo, nullptr, &image_),
      "vkCreateImage (Offscreen)");

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, image_, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = core_->find_memory_type(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VulkanErrorHandler::check_result(
      vkAllocateMemory(device, &allocInfo, nullptr, &memory_),
      "vkAllocateMemory (Offscreen)");

  vkBindImageMemory(device, image_, memory_, 0);

  // 3. Create Image View
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image_;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  VulkanErrorHandler::check_result(
      vkCreateImageView(device, &viewInfo, nullptr, &view_),
      "vkCreateImageView (Offscreen)");

  // 4. Create Framebuffer
  VkFramebufferCreateInfo framebufferInfo{};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = render_pass_;
  framebufferInfo.attachmentCount = 1;
  framebufferInfo.pAttachments = &view_;
  framebufferInfo.width = width;
  framebufferInfo.height = height;
  framebufferInfo.layers = 1;

  VulkanErrorHandler::check_result(
      vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer_),
      "vkCreateFramebuffer (Offscreen)");

  // 5. Create Descriptor Set for ImGui
  descriptor_set_ = core_->create_texture_descriptor(view_);
}

void OffscreenChartRenderer::begin_render(VkCommandBuffer cmd) {
  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = render_pass_;
  renderPassInfo.framebuffer = framebuffer_;
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = {width_, height_};

  VkClearValue clearColor = {{{0.01f, 0.01f, 0.01f, 1.0f}}};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearColor;

  vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)width_;
  viewport.height = (float)height_;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {width_, height_};
  vkCmdSetScissor(cmd, 0, 1, &scissor);
}

void OffscreenChartRenderer::end_render(VkCommandBuffer cmd) {
  vkCmdEndRenderPass(cmd);
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

  create_default_sampler();
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

void VulkanCore::begin_command_buffer() {
  vkResetCommandBuffer(command_buffers_[current_frame_], 0);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = 0;
  begin_info.pInheritanceInfo = nullptr;

  VulkanErrorHandler::check_result(
      vkBeginCommandBuffer(command_buffers_[current_frame_], &begin_info),
      "vkBeginCommandBuffer");
  current_command_buffer_ = command_buffers_[current_frame_];
}

void VulkanCore::begin_main_render_pass() {
  VkRenderPassBeginInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_info.renderPass = render_pass_;
  render_pass_info.framebuffer = framebuffers_[current_image_index_];
  render_pass_info.renderArea.offset = {0, 0};
  render_pass_info.renderArea.extent = swapchain_extent_;

  std::array<VkClearValue, 2> clear_values{};
  clear_values[0].color = {{0.01f, 0.01f, 0.01f, 1.0f}};
  clear_values[1].depthStencil = {1.0f, 0};

  render_pass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
  render_pass_info.pClearValues = clear_values.data();

  vkCmdBeginRenderPass(current_command_buffer_, &render_pass_info,
                       VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)swapchain_extent_.width;
  viewport.height = (float)swapchain_extent_.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  vkCmdSetViewport(current_command_buffer_, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = swapchain_extent_;
  vkCmdSetScissor(current_command_buffer_, 0, 1, &scissor);
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
    VkPipelineLayout layout, VkRenderPass render_pass) {

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
  pipeline_info.renderPass =
      (render_pass != VK_NULL_HANDLE) ? render_pass : render_pass_;
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

VkPipeline VulkanCore::create_compute_pipeline(const std::string &shader_path,
                                               VkPipelineLayout layout) {
  auto compute_code = read_file(shader_path);
  VkShaderModule compute_module = create_shader_module(compute_code);

  VkPipelineShaderStageCreateInfo stage_info{};
  stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage_info.module = compute_module;
  stage_info.pName = "main";

  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.layout = layout;
  pipeline_info.stage = stage_info;

  VkPipeline compute_pipeline;
  VulkanErrorHandler::check_result(
      vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info,
                               nullptr, &compute_pipeline),
      "vkCreateComputePipelines");

  vkDestroyShaderModule(device_, compute_module, nullptr);
  return compute_pipeline;
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

  apply_theme(current_theme_);

  // Initialize data components
  market_data_processor_ =
      std::make_unique<RenderEngine::MarketDataProcessor>();
  hotspine_bridge_ =
      std::make_unique<RenderEngine::HotSpineDataBridge>("/btquant_hotspine");
  visualization_engine_ =
      std::make_unique<RenderEngine::DataVisualizationEngine>(
          vulkan_core_->get_device(), vulkan_core_->get_physical_device());
  if (!visualization_engine_->isValid()) {
    fprintf(stderr, "[VulkanDashboard] FATAL: DataVisualizationEngine "
                    "failed to initialize GPU buffers. Possible OOM.\n");
    throw VulkanException(VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "DataVisualizationEngine::initializeBuffers");
  }

  init_components();
  fprintf(stderr, "[VulkanDashboard] Components initialized\n");
  init_component_resources();
  fprintf(stderr, "[VulkanDashboard] Component resources initialized\n");
  setup_data_subscriptions();
  fprintf(stderr, "[VulkanDashboard] Data subscriptions setup\n");

  // Retrieve the loaded fonts from ImGui
  ImGuiIO &io = ImGui::GetIO();
  if (io.Fonts->Fonts.Size > 0) {
    theme_.sans_font = io.Fonts->Fonts[0];
  }
  if (io.Fonts->Fonts.Size > 1) {
    theme_.monospace_font = io.Fonts->Fonts[1];
  } else if (io.Fonts->Fonts.Size > 0) {
    theme_.monospace_font = io.Fonts->Fonts[0];
  }

  // Auto-load saved layout if it exists
  try {
    std::ifstream test("config/last_layout.json");
    if (test.good()) {
      test.close();
      load_layout("config/last_layout.json");
      fprintf(stderr, "[VulkanDashboard] Loaded saved layout\n");
    } else {
      fprintf(stderr,
              "[VulkanDashboard] No saved layout found, using defaults\n");
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "[VulkanDashboard] Failed to load layout: %s\n", e.what());
  }

  // Start real-time data data bridge
  hotspine_bridge_->start();
  fprintf(stderr, "[VulkanDashboard] HotSpine bridge started\n");

  setup_hotkeys();
  fprintf(stderr, "[VulkanDashboard] Hotkeys configured\n");
}

void VulkanDashboard::setup_hotkeys() {
  hotkey_manager_.clear();

  // Layout Controls (Ctrl)
  hotkey_manager_.add_binding(XK_s, ControlMask, HotkeyAction::SAVE_LAYOUT);
  hotkey_manager_.add_binding(XK_l, ControlMask, HotkeyAction::LOAD_LAYOUT);
  hotkey_manager_.add_binding(XK_l, ControlMask, HotkeyAction::LOAD_LAYOUT);
  hotkey_manager_.add_binding(XK_r, ControlMask, HotkeyAction::RESET_LAYOUT);
  hotkey_manager_.add_binding(XK_p, ControlMask,
                              HotkeyAction::OPEN_COMMAND_PALETTE);

  // View Controls
  hotkey_manager_.add_binding(XK_F11, 0, HotkeyAction::TOGGLE_FULLSCREEN);
  hotkey_manager_.add_binding(XK_F3, 0,
                              HotkeyAction::TOGGLE_PERFORMANCE_OVERLAY);
  hotkey_manager_.add_binding(XK_space, 0, HotkeyAction::TOGGLE_DATA_FEED);

  // Timeframe Controls
  hotkey_manager_.add_binding(XK_1, 0, HotkeyAction::TIMEFRAME_1M);
  hotkey_manager_.add_binding(XK_3, 0, HotkeyAction::TIMEFRAME_3M);
  hotkey_manager_.add_binding(XK_5, 0, HotkeyAction::TIMEFRAME_5M);
  hotkey_manager_.add_binding(XK_0, 0, HotkeyAction::TIMEFRAME_15M);

  // Tab Controls (Alt + 1-9)
  hotkey_manager_.add_binding(XK_1, Mod1Mask, HotkeyAction::SWITCH_TAB_1);
  hotkey_manager_.add_binding(XK_2, Mod1Mask, HotkeyAction::SWITCH_TAB_2);
  hotkey_manager_.add_binding(XK_3, Mod1Mask, HotkeyAction::SWITCH_TAB_3);
  hotkey_manager_.add_binding(XK_4, Mod1Mask, HotkeyAction::SWITCH_TAB_4);
  hotkey_manager_.add_binding(XK_5, Mod1Mask, HotkeyAction::SWITCH_TAB_5);
  hotkey_manager_.add_binding(XK_6, Mod1Mask, HotkeyAction::SWITCH_TAB_6);
  hotkey_manager_.add_binding(XK_7, Mod1Mask, HotkeyAction::SWITCH_TAB_7);
  hotkey_manager_.add_binding(XK_8, Mod1Mask, HotkeyAction::SWITCH_TAB_8);
  hotkey_manager_.add_binding(XK_9, Mod1Mask, HotkeyAction::SWITCH_TAB_9);
}

void VulkanDashboard::handle_hotkey(HotkeyAction action) {
  switch (action) {
  case HotkeyAction::SAVE_LAYOUT:
    save_layout("config/last_layout.json");
    fprintf(stderr, "[VulkanDashboard] Layout saved via hotkey\n");
    break;
  case HotkeyAction::LOAD_LAYOUT:
    load_layout("config/last_layout.json");
    fprintf(stderr, "[VulkanDashboard] Layout loaded\n");
    break;
  case HotkeyAction::RESET_LAYOUT:
    load_layout("config/default_layout.json");
    fprintf(stderr, "[VulkanDashboard] Reset to default layout\n");
    break;
  case HotkeyAction::TOGGLE_FULLSCREEN:
    fprintf(stderr, "[VulkanDashboard] Fullscreen toggle placeholder\n");
    break;
  case HotkeyAction::TOGGLE_PERFORMANCE_OVERLAY:
    show_performance_overlay_ = !show_performance_overlay_;
    break;
  case HotkeyAction::OPEN_COMMAND_PALETTE:
    show_command_palette_ = !show_command_palette_;
    memset(command_buffer_, 0, sizeof(command_buffer_));
    break;
  case HotkeyAction::TOGGLE_DATA_FEED:
    live_execution_active_ = !live_execution_active_;
    fprintf(stderr, "[VulkanDashboard] Data feed %s\n",
            live_execution_active_ ? "resumed" : "paused");
    break;
  case HotkeyAction::TIMEFRAME_1M:
    handle_menu_command("TIMEFRAME_1M");
    break;
  case HotkeyAction::TIMEFRAME_3M:
    handle_menu_command("TIMEFRAME_3M");
    break;
  case HotkeyAction::TIMEFRAME_5M:
    handle_menu_command("TIMEFRAME_5M");
    break;
  case HotkeyAction::TIMEFRAME_15M:
    handle_menu_command("TIMEFRAME_15M");
    break;
  case HotkeyAction::SWITCH_TAB_1:
    active_chart_index_ = 0;
    break;
  case HotkeyAction::SWITCH_TAB_2:
    active_chart_index_ = 1;
    break;
  case HotkeyAction::SWITCH_TAB_3:
    active_chart_index_ = 2;
    break;
  case HotkeyAction::SWITCH_TAB_4:
    active_chart_index_ = 3;
    break;
  case HotkeyAction::SWITCH_TAB_5:
    active_chart_index_ = 4;
    break;
  case HotkeyAction::SWITCH_TAB_6:
    active_chart_index_ = 5;
    break;
  case HotkeyAction::SWITCH_TAB_7:
    active_chart_index_ = 6;
    break;
  case HotkeyAction::SWITCH_TAB_8:
    active_chart_index_ = 7;
    break;
  case HotkeyAction::SWITCH_TAB_9:
    active_chart_index_ = 8;
    break;
  default:
    break;
  }
}

void VulkanDashboard::main_loop() {
  fprintf(stderr, "[VulkanDashboard] Starting main loop\n");
  bool running = true;
  while (running) {
    auto frame_start = std::chrono::high_resolution_clock::now();

    // 1. Event Processing
    auto event_start = std::chrono::high_resolution_clock::now();
    if (handle_x11_events()) {
      running = false;
    }
    auto event_end = std::chrono::high_resolution_clock::now();
    {
      std::lock_guard lock(stats_mutex_);
      current_stats_.event_processing_ms =
          std::chrono::duration<float, std::milli>(event_end - event_start)
              .count();
    }

    // 2. Data Updates
    auto data_start = std::chrono::high_resolution_clock::now();
    uint64_t trades_this_frame = 0;
    uint64_t obs_this_frame = 0;
    if (hotspine_bridge_ && market_data_processor_) {
      auto updates = hotspine_bridge_->getLatestUpdates();

      // LAG RECOVERY: If we have an extreme burst (e.g. >50k updates),
      // we are likely reading stale/partially-overwritten data or will stall
      // the UI.
      if (updates.size() > 50000) {
        fprintf(stderr,
                "[VulkanDashboard] WARN: Extreme updates lag detected (%zu "
                "updates). Flushing buffers.\n",
                updates.size());
        for (auto &chart : chart_components_) {
          if (chart)
            chart->clear_data();
        }
        for (auto &comp : components_) {
          if (comp)
            comp->clear_data();
        }
      }

      for (const auto &update : updates) {
        if (update.type == RenderEngine::MarketDataType::TRADE) {
          trades_this_frame++;
          market_data_processor_->processTradeUpdate(update);
          RenderEngine::TradeData trade;
          trade.symbol = update.symbol;
          trade.timestamp_us = update.timestamp_us;
          trade.price = update.price;
          trade.size = update.size;
          trade.is_buy = (update.side == "buy");
          on_trade_received(trade);
        } else if (update.type == RenderEngine::MarketDataType::ORDERBOOK) {
          obs_this_frame++;
          market_data_processor_->processOrderbookUpdate(update);
          RenderEngine::OrderbookData ob;
          ob.symbol = update.symbol;
          ob.timestamp_us = update.timestamp_us;
          ob.bids = update.bids;
          ob.asks = update.asks;
          ob.spread = update.price;
          on_orderbook_updated(ob);
        }
      }
    }
    synchronize_market_data();
    auto data_end = std::chrono::high_resolution_clock::now();
    {
      std::lock_guard lock(stats_mutex_);
      current_stats_.data_bridge_update_ms =
          std::chrono::duration<float, std::milli>(data_end - data_start)
              .count();
      current_stats_.trades_processed += trades_this_frame;
      current_stats_.orderbooks_processed += obs_this_frame;
    }

    // 3. UI and Logic Updates
    auto logic_start = std::chrono::high_resolution_clock::now();
    update_components(0.016f); // ~60fps
    auto logic_end = std::chrono::high_resolution_clock::now();
    {
      std::lock_guard lock(stats_mutex_);
      current_stats_.geometry_rebuild_ms =
          std::chrono::duration<float, std::milli>(logic_end - logic_start)
              .count();
    }

    // 4. Rendering
    auto render_start = std::chrono::high_resolution_clock::now();
    if (vulkan_core_ && vulkan_core_->begin_frame()) {
      vulkan_core_->begin_command_buffer();

      // 4a. Offscreen rendering (e.g. charts)
      render_offscreen_components();

      // 4b. Main render pass (UI and overlays)
      vulkan_core_->begin_main_render_pass();
      render_gui();
      render_components();

      vulkan_core_->end_frame();
    }
    auto render_end = std::chrono::high_resolution_clock::now();
    {
      std::lock_guard lock(stats_mutex_);
      current_stats_.render_dispatch_ms =
          std::chrono::duration<float, std::milli>(render_end - render_start)
              .count();
    }

    update_performance_stats();
  }
}

void VulkanDashboard::apply_theme(AppTheme theme) {
  current_theme_ = theme;
  ImGuiStyle &style = ImGui::GetStyle();
  ImVec4 *colors = style.Colors;

  style.WindowRounding = 0.0f;
  style.FrameRounding = 3.0f;
  style.PopupRounding = 3.0f;
  style.ScrollbarRounding = 9.0f;
  style.GrabRounding = 3.0f;
  style.TabRounding = 4.0f;
  style.WindowBorderSize = 1.0f;
  style.FrameBorderSize = 1.0f;
  style.PopupBorderSize = 1.0f;

  // Set colors based on chosen theme
  if (theme == AppTheme::BloombergTerminal) {
    ImGui::StyleColorsDark();
    theme_.background_primary = glm::vec4(0.02f, 0.02f, 0.05f, 1.00f);
    theme_.accent_primary = glm::vec4(1.00f, 0.65f, 0.00f, 1.00f); // Amber
    theme_.text_primary = glm::vec4(1.00f, 0.65f, 0.00f, 1.00f);
  } else if (theme == AppTheme::LightMode) {
    ImGui::StyleColorsLight();
    theme_.background_primary = glm::vec4(0.95f, 0.95f, 0.95f, 1.00f);
    theme_.text_primary = glm::vec4(0.1f, 0.1f, 0.1f, 1.00f);
  } else if (theme == AppTheme::TealStreet) {
    // Teal Street Inspired (Deep Black & Cyan/Teal)
    ImGui::StyleColorsDark();
    theme_.background_primary =
        glm::vec4(0.02f, 0.02f, 0.02f, 1.00f); // #050505
    theme_.background_secondary =
        glm::vec4(0.04f, 0.04f, 0.04f, 1.00f); // Slightly lighter
    theme_.background_panel =
        glm::vec4(0.06f, 0.06f, 0.08f, 0.90f); // Glassy navy-black
    theme_.border_color =
        glm::vec4(0.15f, 0.20f, 0.25f, 1.00f); // Cyan-grey border

    theme_.text_muted = glm::vec4(0.30f, 0.35f, 0.40f, 1.00f);

    theme_.accent_primary =
        glm::vec4(0.00f, 0.96f, 1.00f, 1.00f); // Vibrant Teal/Cyan
    theme_.accent_secondary =
        glm::vec4(0.00f, 0.70f, 0.75f, 1.00f); // Deeper Teal

    theme_.price_up = glm::vec4(0.00f, 1.00f, 0.60f, 1.00f);   // Neon Green
    theme_.price_down = glm::vec4(1.00f, 0.20f, 0.35f, 1.00f); // Neon Red/Pink

    // Tighten style for density
    style.WindowPadding = ImVec2(4.0f, 4.0f);
    style.FramePadding = ImVec2(4.0f, 2.0f);
    style.ItemSpacing = ImVec2(4.0f, 2.0f);
    style.ScrollbarSize = 10.0f;
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.WindowRounding = 0.0f;
    style.ChildRounding = 0.0f;
    style.FrameRounding = 2.0f;
    style.PopupRounding = 2.0f;
    style.ScrollbarRounding = 0.0f;
    style.GrabRounding = 2.0f;
    style.TabRounding = 2.0f;
  }

  // Map our theme tokens to ImGui colors
  auto to_imvec4 = [](const glm::vec4 &v) {
    return ImVec4(v.x, v.y, v.z, v.w);
  };

  colors[ImGuiCol_WindowBg] = to_imvec4(theme_.background_primary);
  colors[ImGuiCol_ChildBg] = to_imvec4(theme_.background_secondary);
  colors[ImGuiCol_PopupBg] = to_imvec4(theme_.background_panel);
  colors[ImGuiCol_Border] = to_imvec4(theme_.border_color);
  colors[ImGuiCol_FrameBg] = to_imvec4(theme_.background_secondary);
  colors[ImGuiCol_FrameBgHovered] = to_imvec4(theme_.background_panel);
  colors[ImGuiCol_FrameBgActive] = to_imvec4(theme_.accent_primary * 0.5f);

  colors[ImGuiCol_TitleBg] = to_imvec4(theme_.background_secondary);
  colors[ImGuiCol_TitleBgActive] = to_imvec4(theme_.background_panel);

  colors[ImGuiCol_MenuBarBg] = to_imvec4(theme_.background_secondary);

  colors[ImGuiCol_Header] = to_imvec4(theme_.background_panel);
  colors[ImGuiCol_HeaderHovered] = to_imvec4(theme_.accent_primary * 0.4f);
  colors[ImGuiCol_HeaderActive] = to_imvec4(theme_.accent_primary * 0.6f);

  colors[ImGuiCol_Button] = to_imvec4(theme_.background_panel);
  colors[ImGuiCol_ButtonHovered] = to_imvec4(theme_.accent_primary * 0.7f);
  colors[ImGuiCol_ButtonActive] = to_imvec4(theme_.accent_primary);

  colors[ImGuiCol_Text] = to_imvec4(theme_.text_primary);
  colors[ImGuiCol_TextDisabled] = to_imvec4(theme_.text_muted);

  colors[ImGuiCol_CheckMark] = to_imvec4(theme_.accent_primary);
  colors[ImGuiCol_SliderGrab] = to_imvec4(theme_.accent_primary * 0.8f);
  colors[ImGuiCol_SliderGrabActive] = to_imvec4(theme_.accent_primary);

  colors[ImGuiCol_Separator] = to_imvec4(theme_.border_color);
  colors[ImGuiCol_SeparatorHovered] = to_imvec4(theme_.accent_primary);
  colors[ImGuiCol_SeparatorActive] = to_imvec4(theme_.accent_primary);

  colors[ImGuiCol_Tab] = to_imvec4(theme_.background_secondary);
  colors[ImGuiCol_TabHovered] = to_imvec4(theme_.accent_primary * 0.8f);
  colors[ImGuiCol_TabActive] = to_imvec4(theme_.accent_primary * 0.6f);
  colors[ImGuiCol_TabUnfocused] = to_imvec4(theme_.background_secondary);
  colors[ImGuiCol_TabUnfocusedActive] = to_imvec4(theme_.background_panel);
}

void VulkanDashboard::shutdown() {
  fprintf(stderr, "[VulkanDashboard] Shutting down\n");

  // Auto-save layout before shutdown
  try {
    save_layout("config/last_layout.json");
  } catch (const std::exception &e) {
    fprintf(stderr, "[VulkanDashboard] Failed to save layout: %s\n", e.what());
  }

  stop_market_data_processing();
  components_.clear();
  visualization_engine_.reset();
  hotspine_bridge_.reset();
  market_data_processor_.reset();
  vulkan_core_.reset();
  cleanup_x11();
}

void VulkanDashboard::add_component(std::unique_ptr<UIComponent> component) {
  component->set_dashboard(this);
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
  std::lock_guard lock(stats_mutex_);
  return current_stats_;
}

void VulkanDashboard::on_trade_received(const RenderEngine::TradeData &trade) {
  // Check alerts
  alert_manager_.check_alerts(trade.symbol, trade.price, trade.size);

  if (watchlist_component_) {
    watchlist_component_->update_quote(trade.symbol, trade.price, 0.0,
                                       trade.size);
  }

  // Dispatch to all components
  for (auto &component : components_) {
    if (component && component->get_target_symbol() == trade.symbol) {
      component->handle_trade(trade);
    }
  }

  // Dispatch to chart components
  for (auto &chart : chart_components_) {
    if (chart && chart->get_target_symbol() == trade.symbol) {
      chart->handle_trade(trade);
    }
  }
}

void VulkanDashboard::on_orderbook_updated(
    const RenderEngine::OrderbookData &orderbook) {
  // Dispatch to all components
  for (auto &component : components_) {
    if (component && component->get_target_symbol() == orderbook.symbol) {
      component->handle_orderbook(orderbook);
    }
  }

  // Dispatch to chart components
  for (auto &chart : chart_components_) {
    if (chart && chart->get_target_symbol() == orderbook.symbol) {
      chart->handle_orderbook(orderbook);
    }
  }
}

void VulkanDashboard::synchronize_market_data() {
  if (!hotspine_bridge_)
    return;

  auto all_symbols = hotspine_bridge_->getAllSymbols();
  if (all_symbols.empty())
    return;

  // Find DataGrid and Heatmap components
  DataGridComponent *grid = nullptr;
  HeatmapComponent *heatmap = nullptr;

  for (auto &comp : components_) {
    if (comp->get_name() == "Market Data Grid") {
      grid = dynamic_cast<DataGridComponent *>(comp.get());
    } else if (comp->get_name() == "Momentum Heatmap") {
      heatmap = dynamic_cast<HeatmapComponent *>(comp.get());
    }
  }

  // Update Data Grid
  if (grid) {
    for (size_t i = 0; i < all_symbols.size(); ++i) {
      const auto &sym = all_symbols[i];
      std::vector<DataGridComponent::CellData> row;

      // Symbol
      row.push_back({sym.symbol, {1.0f, 1.0f, 1.0f, 1.0f}, 0.0f, false, false});

      // Price
      row.push_back({std::to_string(sym.last_price),
                     {1.0f, 1.0f, 1.0f, 1.0f},
                     (float)sym.last_price,
                     false,
                     true});

      // Change %
      glm::vec4 change_color = {1, 1, 1, 1};
      if (sym.price_change_percent > 0)
        change_color = {0.3f, 1.0f, 0.3f, 1.0f};
      else if (sym.price_change_percent < 0)
        change_color = {1.0f, 0.3f, 0.3f, 1.0f};

      char change_buf[32];
      snprintf(change_buf, sizeof(change_buf), "%.2f%%",
               (float)sym.price_change_percent);
      row.push_back({change_buf, change_color, (float)sym.price_change_percent,
                     false, true});

      // Bid/Ask
      row.push_back({std::to_string(sym.bid_price),
                     {0.3f, 1.0f, 0.3f, 1.0f},
                     (float)sym.bid_price,
                     false,
                     true});
      row.push_back({std::to_string(sym.ask_price),
                     {1.0f, 0.3f, 0.3f, 1.0f},
                     (float)sym.ask_price,
                     false,
                     true});

      // Spread
      row.push_back({std::to_string(sym.spread),
                     {1.0f, 0.9f, 0.2f, 1.0f},
                     (float)sym.spread,
                     false,
                     true});

      grid->set_row_data(i, row);
    }
  }

  // Update Heatmap
  if (heatmap) {
    // Arrange symbols in a grid
    size_t w = 10, h = 10;
    std::vector<std::vector<HeatmapComponent::HeatmapData>> heatmap_grid(
        h, std::vector<HeatmapComponent::HeatmapData>(w));

    for (size_t i = 0; i < all_symbols.size() && i < w * h; ++i) {
      const auto &sym = all_symbols[i];
      glm::vec4 color = {0.5f, 0.5f, 0.5f, 1.0f};
      if (sym.momentum > 0)
        color = {0.0f, (float)sym.momentum * 0.5f + 0.5f, 0.0f, 1.0f};
      else if (sym.momentum < 0)
        color = {std::abs((float)sym.momentum) * 0.5f + 0.5f, 0.0f, 0.0f, 1.0f};

      heatmap_grid[i / w][i % w] = {(float)sym.momentum, color, sym.symbol,
                                    sym.symbol_id};
    }
    heatmap->set_data(heatmap_grid);
  }
}

void VulkanDashboard::synchronize_crosshair(const CrosshairState &state) {
  shared_crosshair_ = state;
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
  fprintf(
      stderr,
      "[VulkanDashboard] Initializing Institutional TealStreet Workspace\n");

  const float grid = 4.0f;    // Finer grid for higher density
  const float padding = 2.0f; // Minimal padding
  float w = (float)width_;
  float h = (float)height_;

  // 1. Market Overview Top Bar (Pinned to top, fixed height)
  float top_bar_h = 24.0f;
  auto overview = std::make_unique<MarketOverviewPanel>(
      glm::vec2(0, 0), glm::vec2(w, top_bar_h));
  add_component(std::move(overview));

  float workspace_y = top_bar_h;
  float workspace_h = h - top_bar_h;

  // Workspace Division:
  // Left: Watchlist + Positions
  // Center: Charts (Main Column)
  // Right: Orderbook + Tape + Order Mgmt

  float left_w = std::round((w * 0.16f) / grid) * grid;
  float right_w = std::round((w * 0.28f) / grid) * grid;
  float center_w = w - left_w - right_w;

  // --- LEFT COLUMN ---
  float wl_h = std::round((workspace_h * 0.45f) / grid) * grid;
  auto wl = std::make_unique<WatchlistComponent>(glm::vec2(0, workspace_y),
                                                 glm::vec2(left_w, wl_h));
  watchlist_component_ = wl.get();
  add_component(std::move(wl));

  auto pos_panel = std::make_unique<PositionPanelComponent>(
      glm::vec2(0, workspace_y + wl_h), glm::vec2(left_w, workspace_h - wl_h));
  add_component(std::move(pos_panel));

  // --- CENTER COLUMN ---
  auto main_chart = std::make_unique<RealtimeChartComponent>(
      glm::vec2(left_w, workspace_y), glm::vec2(center_w, workspace_h));
  main_chart->enable_candlestick_mode(true);
  add_component(std::move(main_chart));

  // --- RIGHT COLUMN ---
  float ob_w = std::round((right_w * 0.55f) / grid) * grid;
  float tape_w = right_w - ob_w;
  float ob_h = std::round((workspace_h * 0.65f) / grid) * grid;

  auto ob = std::make_unique<OrderBookComponent>(
      glm::vec2(left_w + center_w, workspace_y), glm::vec2(ob_w, ob_h));
  add_component(std::move(ob));

  auto tape = std::make_unique<TapeComponent>(
      glm::vec2(left_w + center_w + ob_w, workspace_y),
      glm::vec2(tape_w, ob_h));
  add_component(std::move(tape));

  // Bottom-Right: Order Management & Alerts (Split)
  float bottom_right_y = workspace_y + ob_h;
  float bottom_right_h = workspace_h - ob_h;
  float order_mgmt_h = std::round((bottom_right_h * 0.60f) / grid) * grid;

  auto order_mgmt = std::make_unique<OrderManagementComponent>(
      glm::vec2(left_w + center_w, bottom_right_y),
      glm::vec2(right_w, order_mgmt_h));
  add_component(std::move(order_mgmt));

  auto alerts = std::make_unique<AlertComponent>(
      glm::vec2(left_w + center_w, bottom_right_y + order_mgmt_h),
      glm::vec2(right_w, bottom_right_h - order_mgmt_h), alert_manager_);
  add_component(std::move(alerts));

  // Additional Floating/Optional components in bottom bar if space allowed...
}

void VulkanDashboard::update_performance_stats() {
  auto now = std::chrono::high_resolution_clock::now();
  static auto last_time = now;
  float duration = std::chrono::duration<float>(now - last_time).count();

  frame_count_++;

  // Update history buffers every frame for smooth graphing
  {
    std::lock_guard lock(stats_mutex_);
    event_times_.push_back(current_stats_.event_processing_ms);
    data_times_.push_back(current_stats_.data_bridge_update_ms);
    render_times_.push_back(current_stats_.render_dispatch_ms);

    // Calculate total frame time (approximate delta from last frame)
    static auto last_frame = now;
    float delta =
        std::chrono::duration<float, std::milli>(now - last_frame).count();
    frame_times_.push_back(delta);
    last_frame = now;

    if (event_times_.size() > 100)
      event_times_.pop_front();
    if (data_times_.size() > 100)
      data_times_.pop_front();
    if (render_times_.size() > 100)
      render_times_.pop_front();
    if (frame_times_.size() > 100)
      frame_times_.pop_front();
  }

  if (duration >= 1.0f) {
    std::lock_guard lock(stats_mutex_);
    current_stats_.fps = frame_count_ / duration;
    current_stats_.frame_time_ms = (duration / frame_count_) * 1000.0f;

    frame_count_ = 0;
    last_time = now;
  }
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

    ImGuiIO &io = ImGui::GetIO();
    InputEvent input_ev;
    input_ev.position = glm::vec2(io.MousePos.x, io.MousePos.y);
    input_ev.timestamp = std::chrono::high_resolution_clock::now();

    if (event.type == MotionNotify) {
      input_ev.type = InputEventType::MouseMove;
      for (auto &comp : components_)
        comp->handle_input(input_ev);
    } else if (event.type == ButtonPress) {
      if (event.xbutton.button == Button1)
        io.MouseDown[0] = true;
      if (event.xbutton.button == Button2)
        io.MouseDown[2] = true;
      if (event.xbutton.button == Button3)
        io.MouseDown[1] = true;

      input_ev.type = InputEventType::MouseButton;
      input_ev.pressed = true;
      if (event.xbutton.button == Button4 || event.xbutton.button == Button5) {
        input_ev.type = InputEventType::Scroll;
        input_ev.scroll_delta = {0, event.xbutton.button == Button4 ? 1.0f
                                                                    : -1.0f};
        io.MouseWheel += input_ev.scroll_delta.y;
      } else {
        if (event.xbutton.button == Button1)
          input_ev.mouse_button = MouseButton::Left;
        if (event.xbutton.button == Button2)
          input_ev.mouse_button = MouseButton::Middle;
        if (event.xbutton.button == Button3)
          input_ev.mouse_button = MouseButton::Right;
      }
      for (auto &comp : components_)
        comp->handle_input(input_ev);
    } else if (event.type == ButtonRelease) {
      if (event.xbutton.button == Button1)
        io.MouseDown[0] = false;
      if (event.xbutton.button == Button2)
        io.MouseDown[2] = false;
      if (event.xbutton.button == Button3)
        io.MouseDown[1] = false;

      input_ev.type = InputEventType::MouseButton;
      input_ev.pressed = false;
      if (event.xbutton.button == Button1)
        input_ev.mouse_button = MouseButton::Left;
      if (event.xbutton.button == Button2)
        input_ev.mouse_button = MouseButton::Middle;
      if (event.xbutton.button == Button3)
        input_ev.mouse_button = MouseButton::Right;
      for (auto &comp : components_)
        comp->handle_input(input_ev);
    } else if (event.type == X11_KeyPress) {
      KeySym keysym = XLookupKeysym(&event.xkey, 0);
      uint32_t modifiers = 0;
      if (event.xkey.state & ControlMask)
        modifiers |= ControlMask;
      if (event.xkey.state & ShiftMask)
        modifiers |= ShiftMask;
      if (event.xkey.state & Mod1Mask)
        modifiers |= Mod1Mask;

      HotkeyAction action = hotkey_manager_.get_action(keysym, modifiers);
      if (action != HotkeyAction::NONE) {
        handle_hotkey(action);
      }
    }
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

  // Theme will be applied by VulkanDashboard after initialization

  // Load specialized fonts
  const char *sans_font_paths[] = {
      "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"};
  const char *mono_font_paths[] = {
      "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
      "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"};

  bool sans_loaded = false;
  for (const char *path : sans_font_paths) {
    if (access(path, F_OK) != -1) {
      io.Fonts->AddFontFromFileTTF(path, 14.0f);
      sans_loaded = true;
      break;
    }
  }
  if (!sans_loaded) {
    io.Fonts->AddFontDefault();
  }

  bool mono_loaded = false;
  for (const char *path : mono_font_paths) {
    if (access(path, F_OK) != -1) {
      io.Fonts->AddFontFromFileTTF(path, 13.0f);
      mono_loaded = true;
      break;
    }
  }

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
                   KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
                   PointerMotionMask);

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
  fprintf(stderr,
          "[VulkanDashboard] Initializing component Vulkan resources\n");

  for (auto &component : components_) {
    if (component) {
      component->initialize_vulkan_resources(vulkan_core_.get());
    }
  }

  for (auto &chart : chart_components_) {
    if (chart) {
      chart->initialize_vulkan_resources(vulkan_core_.get());
    }
  }
}

void VulkanDashboard::update_components(float delta_time) {
  for (auto &component : components_) {
    component->update(delta_time);
  }
  for (auto &chart : chart_components_) {
    chart->update(delta_time);
  }
}

void VulkanDashboard::render_components() {
  if (vulkan_core_) {
    VkCommandBuffer cmd = vulkan_core_->get_current_command_buffer();

    // Render global components that use the main render pass
    for (auto &component : components_) {
      if (component && component->is_visible()) {
        // Skip components that manage their own offscreen passes
        if (component->get_name() != "Price Chart") {
          component->render(cmd);
        }
      }
    }
  }
}

void VulkanDashboard::render_offscreen_components() {
  if (vulkan_core_) {
    VkCommandBuffer cmd = vulkan_core_->get_current_command_buffer();

    // Render workspace charts (they manage their own offscreen render passes)
    for (auto &chart : chart_components_) {
      if (chart && chart->is_visible()) {
        chart->render(cmd);
      }
    }
  }
}

void VulkanDashboard::cleanup_x11() {
  if (display_ && window_) {
    XDestroyWindow(display_, window_);
    XCloseDisplay(display_);
  }
}

void VulkanDashboard::render_gui() {
  render_main_menu_bar();
  render_symbol_selector();

  // Render Global UI Components (Fixed positions/panels)
  for (auto &component : components_) {
    if (component->is_visible()) {
      component->render_gui();
    }
  }

  // Render Chart Workspace
  if (current_layout_mode_ == LayoutMode::Tabs) {
    render_chart_tabs();
  }

  for (size_t i = 0; i < chart_components_.size(); ++i) {
    auto &chart = chart_components_[i];
    if (current_layout_mode_ == LayoutMode::Tabs) {
      if (static_cast<int>(i) == active_chart_index_) {
        chart->set_visible(true);
        chart->render_gui();
      } else {
        chart->set_visible(false);
      }
    } else {
      // Grid or Single View (handled by layout logic)
      if (chart->is_visible()) {
        chart->render_gui();
      }
    }
  }

  // Status overlay and specialized dialogs
  if (show_performance_overlay_)
    render_performance_overlay();
  if (show_command_palette_)
    render_command_palette();
  if (show_backtest_dialog_)
    render_backtest_dialog();
  if (show_risk_manager_)
    render_risk_manager();

  render_status_bar();
}

void VulkanDashboard::set_active_symbol(const std::string &symbol) {
  if (active_symbol_ == symbol)
    return;
  active_symbol_ = symbol;

  // In multi-chart or tabbed mode, change the symbol for the active view
  if (current_layout_mode_ == LayoutMode::Tabs && !chart_components_.empty()) {
    if (active_chart_index_ >= 0 &&
        active_chart_index_ < static_cast<int>(chart_components_.size())) {
      chart_components_[active_chart_index_]->set_target_symbol(symbol);
      chart_components_[active_chart_index_]->clear_data();
    }
  }

  // Notify global components that might want to follow the active symbol
  for (auto &comp : components_) {
    // Components like OrderBook might want to follow the global search
    // if they don't have a fixed pin. For now, we update their target.
    if (comp->get_name() == "Order Book" ||
        comp->get_name() == "Time & Sales") {
      comp->set_target_symbol(symbol);
      comp->clear_data();
    }
  }
}

void VulkanDashboard::render_main_menu_bar() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("New Workspace", "Ctrl+N")) {
      }
      if (ImGui::MenuItem("Open Workspace...", "Ctrl+O")) {
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Save Workspace", "Ctrl+S")) {
        save_layout();
      }
      if (ImGui::MenuItem("Save Layout As...")) {
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Export Data...")) {
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Settings", "Ctrl+,")) {
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Exit", "Alt+F4")) {
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Markets")) {
      if (ImGui::MenuItem("Search Market...", "Ctrl+M")) {
        is_symbol_selector_open_ = true;
      }
      ImGui::Separator();
      if (ImGui::BeginMenu("Exchanges")) {
        if (ImGui::MenuItem("Binance Connectivity")) {
        }
        if (ImGui::MenuItem("OKX Connectivity")) {
        }
        if (ImGui::MenuItem("Coinbase Connectivity")) {
        }
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Sectors")) {
        if (ImGui::MenuItem("Layer 1")) {
        }
        if (ImGui::MenuItem("DeFi Bluechips")) {
        }
        if (ImGui::MenuItem("AI / DePin")) {
        }
        ImGui::EndMenu();
      }
      ImGui::Separator();
      // Quick select top markets
      const std::vector<std::string> top_markets = {"BTC/USDT", "ETH/USDT",
                                                    "SOL/USDT"};
      for (const auto &m : top_markets) {
        if (ImGui::MenuItem(m.c_str())) {
          set_active_symbol(m);
        }
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Charts")) {
      if (ImGui::MenuItem("New Price Chart", "Alt+C")) {
        add_chart(active_symbol_);
      }
      ImGui::Separator();
      if (ImGui::BeginMenu("Layout Mode")) {
        if (ImGui::MenuItem("Single Fullscreen", nullptr,
                            current_layout_mode_ == LayoutMode::Single)) {
          current_layout_mode_ = LayoutMode::Single;
          handle_menu_command("LAYOUT_SINGLE");
        }
        if (ImGui::MenuItem("Grid View (2x2)", nullptr,
                            current_layout_mode_ == LayoutMode::Grid)) {
          current_layout_mode_ = LayoutMode::Grid;
          handle_menu_command("LAYOUT_2X2");
        }
        if (ImGui::MenuItem("Multitask Tabs", nullptr,
                            current_layout_mode_ == LayoutMode::Tabs)) {
          current_layout_mode_ = LayoutMode::Tabs;
        }
        ImGui::EndMenu();
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Synchronize All Charts")) {
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Strategies")) {
      if (ImGui::MenuItem("Strategy Manager", "Ctrl+T")) {
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Backtest Runner", "Alt+B")) {
        show_backtest_dialog_ = true;
      }
      if (ImGui::MenuItem("Optimization Suite")) {
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Live Execution Panel")) {
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Tools")) {
      if (ImGui::MenuItem("Risk Manager")) {
        show_risk_manager_ = true;
      }
      if (ImGui::MenuItem("Alerts Center")) {
      }
      if (ImGui::MenuItem("Market Screener")) {
      }
      ImGui::Separator();
      if (ImGui::MenuItem("System Console", "`")) {
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Window")) {
      for (auto &comp : components_) {
        bool visible = comp->is_visible();
        if (ImGui::MenuItem(comp->get_name().c_str(), nullptr, &visible)) {
          comp->set_visible(visible);
        }
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Performance Overlay", "F3",
                          &show_performance_overlay_)) {
      }
      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Help")) {
      if (ImGui::MenuItem("Documentation")) {
      }
      if (ImGui::MenuItem("About BTQuant")) {
      }
      ImGui::EndMenu();
    }

    // Right-aligned area
    float status_width = 300.0f;
    ImGui::SameLine(ImGui::GetWindowWidth() - status_width);
    ImGui::TextColored(
        ImVec4(theme_.status_connected.x, theme_.status_connected.y,
               theme_.status_connected.z, theme_.status_connected.w),
        "CONNECTED");
    ImGui::SameLine();
    ImGui::Text("| CPU: %.1f%%", current_stats_.cpu_usage_percent);
    ImGui::SameLine();
    ImGui::Text("| %s", active_symbol_.c_str());

    ImGui::EndMainMenuBar();
  }
}

void VulkanDashboard::render_chart_tabs() {
  if (chart_components_.empty())
    return;

  ImGui::SetNextWindowPos(ImVec2(0, 20)); // Below main menu
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, 30));
  ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollWithMouse |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar;

  if (ImGui::Begin("##ChartTabs", nullptr, flags)) {
    if (ImGui::BeginMenuBar()) {
      for (size_t i = 0; i < chart_components_.size(); ++i) {
        std::string label = chart_components_[i]->get_target_symbol() + " [" +
                            std::to_string(i) + "]";
        bool active = (static_cast<int>(i) == active_chart_index_);

        if (active)
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));

        if (ImGui::MenuItem(label.c_str(), nullptr, active)) {
          active_chart_index_ = static_cast<int>(i);
        }

        if (active)
          ImGui::PopStyleColor();

        if (i < chart_components_.size() - 1)
          ImGui::TextDisabled("|");
      }
      ImGui::EndMenuBar();
    }
  }
  ImGui::End();
}

void VulkanDashboard::render_symbol_selector() {
  if (!is_symbol_selector_open_)
    return;

  ImGui::SetNextWindowSize(ImVec2(500, 600), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("Market Search", &is_symbol_selector_open_,
                   ImGuiWindowFlags_NoCollapse)) {
    static char filter[128] = "";
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##Search",
                                 "Search symbols (e.g. BTC, ETH, binance...)",
                                 filter, IM_ARRAYSIZE(filter))) {
    }
    ImGui::Separator();

    if (ImGui::BeginChild("SymbolList")) {
      std::vector<std::string> symbols;
      if (hotspine_bridge_) {
        auto all = hotspine_bridge_->getAllSymbols();
        for (const auto &s : all) {
          symbols.push_back(s.symbol);
        }
      }

      // Fallback/Mock if bridge is empty or not connected
      if (symbols.empty()) {
        symbols = {"BTC/USDT",   "ETH/USDT", "SOL/USDT",  "DOT/USDT",
                   "MATIC/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
                   "LINK/USDT",  "LTC/USDT"};
      }

      for (const auto &s : symbols) {
        std::string lower_s = s;
        std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        std::string lower_filter = filter;
        std::transform(lower_filter.begin(), lower_filter.end(),
                       lower_filter.begin(),
                       [](unsigned char c) { return std::tolower(c); });

        if (strlen(filter) > 0 &&
            lower_s.find(lower_filter) == std::string::npos)
          continue;

        bool is_active = (active_symbol_ == s);
        if (is_active)
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0.8f, 0, 1));

        if (ImGui::Selectable(s.c_str(), is_active)) {
          set_active_symbol(s);
          is_symbol_selector_open_ = false;
        }

        if (is_active)
          ImGui::PopStyleColor();
      }
      ImGui::EndChild();
    }
    ImGui::End();
  }
}

void VulkanDashboard::handle_menu_command(const std::string &cmd,
                                          const std::string &args) {
  if (cmd == "LAYOUT_SINGLE") {
    current_layout_mode_ = LayoutMode::Single;
    for (auto &c : chart_components_)
      c->set_visible(false);
    if (!chart_components_.empty() &&
        active_chart_index_ < (int)chart_components_.size()) {
      auto &active = chart_components_[active_chart_index_];
      active->set_visible(true);
      active->set_position({5, 55}); // Account for menu and tabs
      active->set_size({(float)width_ - 10, (float)height_ - 85});
    }
  } else if (cmd == "LAYOUT_2X2") {
    current_layout_mode_ = LayoutMode::Grid;
    float w = (float)width_ / 2.0f;
    float h = ((float)height_ - 60.0f) / 2.0f;
    for (size_t i = 0; i < chart_components_.size() && i < 4; ++i) {
      chart_components_[i]->set_visible(true);
      chart_components_[i]->set_position({(i % 2) * w, 55 + (i / 2) * h});
      chart_components_[i]->set_size({w - 5, h - 5});
    }
  } else if (cmd.find("TIMEFRAME_") == 0) {
    std::string tf = cmd.substr(10);
    if (active_chart_index_ >= 0 &&
        active_chart_index_ < (int)chart_components_.size()) {
      // Dynamic cast or just use the interface if added to UIComponent
      // For now, we know chart_components_ are RealtimeChartComponent
      auto chart = static_cast<RealtimeChartComponent *>(
          chart_components_[active_chart_index_].get());
      chart->set_timeframe(tf);
      chart->clear_data();
      // In a real app, this would trigger a data reload from HotSpine
      fprintf(stderr,
              "[VulkanDashboard] Timeframe changed to %s for chart %d\n",
              tf.c_str(), active_chart_index_);
    }
  }
}

void VulkanDashboard::add_chart(const std::string &symbol,
                                const std::string &timeframe) {
  // Create a new professional candlestick chart
  auto chart = std::make_unique<RealtimeChartComponent>(
      glm::vec2(0, 0), glm::vec2(width_, height_));
  chart->set_dashboard(this);
  chart->enable_candlestick_mode(true);
  chart->set_target_symbol(symbol);
  chart->set_timeframe(timeframe);

  if (vulkan_core_) {
    chart->initialize_vulkan_resources(vulkan_core_.get());
  }

  chart_components_.push_back(std::move(chart));

  // Switch to the newly added chart tab
  active_chart_index_ = static_cast<int>(chart_components_.size()) - 1;

  if (current_layout_mode_ == LayoutMode::Single) {
    handle_menu_command("LAYOUT_SINGLE");
  } else if (current_layout_mode_ == LayoutMode::Grid) {
    handle_menu_command("LAYOUT_2X2");
  }
}

void VulkanDashboard::apply_layout(const WorkspaceLayout &layout) {
  current_workspace_ = layout;
  // TODO: Rebuild components based on layout
}

void VulkanDashboard::render_status_bar() {
  const float height = 24.0f;
  ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(
      ImVec2(viewport->Pos.x, viewport->Pos.y + viewport->Size.y - height));
  ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, height));
  ImGui::SetNextWindowViewport(viewport->ID);

  ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollWithMouse |
      ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_MenuBar;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(0.02f, 0.02f, 0.02f, 1.00f));

  if (ImGui::Begin("##StatusBar", nullptr, window_flags)) {
    if (ImGui::BeginMenuBar()) {
      ImGui::TextColored(to_imvec4(theme_.accent_primary), " BTQ CORE ");
      ImGui::Separator();

      ImGui::TextColored(to_imvec4(live_execution_active_ ? theme_.price_up
                                                          : theme_.price_down),
                         " %s ", live_execution_active_ ? "LIVE" : "PAUSED");
      ImGui::Separator();

      if (vulkan_core_) {
        auto stats = vulkan_core_->get_memory_manager().get_memory_stats();
        VkDeviceSize total_used = stats.vertex_pool_used +
                                  stats.uniform_pool_used +
                                  stats.storage_pool_used;
        ImGui::Text("GPU MEM: %.1f MB",
                    (float)total_used / (1024.0f * 1024.0f));
        ImGui::Separator();
      }

      ImGui::TextDisabled(" DATA ");
      ImGui::Text("%.2fms", current_stats_.data_latency_ms);
      ImGui::Separator();

      // Right-aligned section
      float right_offset = 200.0f; // Adjusted for better alignment
      float window_width = ImGui::GetWindowWidth();
      if (window_width > right_offset) {
        ImGui::SetCursorPosX(window_width - right_offset);
      }

      ImGui::Separator();
      ImGui::TextDisabled(" FPS ");
      ImGui::Text("%.0f", current_stats_.fps);
      ImGui::Separator();

      // Simple time display
      time_t now = time(nullptr);
      struct tm *timeinfo = localtime(&now);
      if (timeinfo) {
        char time_buf[64];
        strftime(time_buf, sizeof(time_buf), "%H:%M:%S", timeinfo);
        ImGui::TextColored(to_imvec4(theme_.text_secondary), " %s ", time_buf);
      }

      ImGui::EndMenuBar();
    }
  }
  ImGui::End();
  ImGui::PopStyleColor();
  ImGui::PopStyleVar(2);
}

void VulkanDashboard::render_backtest_dialog() {
  ImGui::SetNextWindowSize(ImVec2(400, 350), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Run Strategy Backtest", &show_backtest_dialog_)) {
    ImGui::End();
    return;
  }

  static char strategy_name[64] = "MomentumScalper_v1";
  ImGui::InputText("Strategy Name", strategy_name, 64);

  static int exchange_idx = 0;
  const char *exchanges[] = {"Binance", "OKX", "Bybit", "Kraken"};
  ImGui::Combo("Exchange", &exchange_idx, exchanges, IM_ARRAYSIZE(exchanges));

  static char symbol[32] = "BTC/USDT";
  ImGui::InputText("Symbol", symbol, 32);

  ImGui::Separator();
  ImGui::Text("Parameters");
  static float momentum_threshold = 0.5f;
  ImGui::SliderFloat("Threshold", &momentum_threshold, 0.1f, 1.0f);

  static int timeframe = 1;
  const char *timeframes[] = {"1m", "5m", "15m", "1h", "4h", "1d"};
  ImGui::Combo("Timeframe", &timeframe, timeframes, IM_ARRAYSIZE(timeframes));

  ImGui::Spacing();
  if (ImGui::Button("Run Simulation", ImVec2(120, 30))) {
    // Mock simulation start
    show_backtest_dialog_ = false;
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(120, 30))) {
    show_backtest_dialog_ = false;
  }

  ImGui::End();
}

void VulkanDashboard::render_risk_manager() {
  ImGui::SetNextWindowSize(ImVec2(350, 450), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Risk Manager", &show_risk_manager_)) {
    ImGui::End();
    return;
  }

  ImGui::TextColored(ImVec4(0.7f, 0.7f, 1.0f, 1.0f), "Portfolio Overview");
  ImGui::Separator();

  ImGui::Columns(2, "RiskColumns", false);
  ImGui::Text("Total Equity:");
  ImGui::NextColumn();
  ImGui::Text("$%.2f", risk_metrics_.total_equity);
  ImGui::NextColumn();

  ImGui::Text("Daily PL:");
  ImGui::NextColumn();
  ImGui::TextColored(risk_metrics_.daily_pnl >= 0
                         ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                         : ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                     "$%.2f", risk_metrics_.daily_pnl);
  ImGui::NextColumn();

  ImGui::Text("Exposure:");
  ImGui::NextColumn();
  ImGui::ProgressBar(
      (float)(risk_metrics_.current_exposure / risk_metrics_.total_equity),
      ImVec2(-1, 0));
  ImGui::NextColumn();
  ImGui::Columns(1);

  ImGui::Separator();
  ImGui::Text("Risk Indicators");

  ImGui::Text("Sharpe Ratio:");
  ImGui::SameLine(150);
  ImGui::Text("%.2f", risk_metrics_.sharpe_ratio);

  ImGui::Text("Max Drawdown:");
  ImGui::SameLine(150);
  ImGui::Text("%.2f%%", risk_metrics_.max_drawdown * 100.0);

  ImGui::Text("95%% VaR:");
  ImGui::SameLine(150);
  ImGui::Text("$%.2f", risk_metrics_.var_95);

  ImGui::Spacing();
  if (ImGui::CollapsingHeader("Active Alerts")) {
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                       "[CRITICAL] Margin utilization > 80%%");
    ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.2f, 1.0f),
                       "[WARNING] High volatility on BTC/USDT");
  }

  ImGui::End();
}

void VulkanDashboard::render_performance_overlay() {
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.85f); // Slightly darker for readability

  if (ImGui::Begin(
          "Performance Monitor", &show_performance_overlay_,
          ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
              ImGuiWindowFlags_NoSavedSettings |
              ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav)) {

    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "FPS: %.1f",
                       current_stats_.fps);
    ImGui::Separator();

    auto plot_callback = [](void *data, int idx) {
      auto *deque = (std::deque<float> *)data;
      if (!deque || deque->empty())
        return 0.0f;
      // Clamp index to prevent out of bounds if deque shrinks between calls
      size_t safe_idx = static_cast<size_t>(
          std::max(0, std::min(idx, (int)deque->size() - 1)));
      return (*deque)[safe_idx];
    };

    // Frame Time Graph
    {
      std::lock_guard lock(stats_mutex_);
      if (!frame_times_.empty()) {
        int count = static_cast<int>(frame_times_.size());
        ImGui::PlotLines("Frame (ms)", plot_callback, &frame_times_, count, 0,
                         nullptr, 0.0f, 33.0f, ImVec2(250, 40));
        ImGui::SameLine();
        ImGui::Text("%.2f", current_stats_.frame_time_ms);
      }

      // Granular breakdowns
      if (!event_times_.empty()) {
        int count = static_cast<int>(event_times_.size());
        ImGui::PlotLines("Event (ms)", plot_callback, &event_times_, count, 0,
                         nullptr, 0.0f, 5.0f, ImVec2(250, 30));
        ImGui::SameLine();
        ImGui::Text("%.2f", current_stats_.event_processing_ms);
      }

      if (!data_times_.empty()) {
        int count = static_cast<int>(data_times_.size());
        ImGui::PlotLines("Data (ms) ", plot_callback, &data_times_, count, 0,
                         nullptr, 0.0f, 5.0f, ImVec2(250, 30));
        ImGui::SameLine();
        ImGui::Text("%.2f", current_stats_.data_bridge_update_ms);
      }

      if (!render_times_.empty()) {
        int count = static_cast<int>(render_times_.size());
        ImGui::PlotLines("Render(ms)", plot_callback, &render_times_, count, 0,
                         nullptr, 0.0f, 16.0f, ImVec2(250, 30));
        ImGui::SameLine();
        ImGui::Text("%.2f", current_stats_.render_dispatch_ms);
      }
    }

    ImGui::Separator();
    ImGui::Text("Throughput:");
    ImGui::Text("Trades: %lu", current_stats_.trades_processed);
    ImGui::Text("Books:  %lu", current_stats_.orderbooks_processed);

    ImGui::Separator();
    auto mem_stats = vulkan_core_->get_memory_manager().get_memory_stats();
    ImGui::Text("GPU Mem: V:%0.0f%% U:%0.0f%% S:%0.0f%%",
                mem_stats.vertex_pool_usage, mem_stats.uniform_pool_usage,
                mem_stats.storage_pool_usage);

    ImGui::Separator();
    ImGui::TextDisabled("Press F3 to toggle");
  }
  ImGui::End();
}

void VulkanDashboard::save_layout(const std::string &filename) {
  nlohmann::json j;
  j["version"] = 2.0; // Updated version
  j["theme"] = static_cast<int>(current_theme_);
  j["active_symbol"] = active_symbol_;
  j["active_chart_index"] = active_chart_index_;
  j["layout_mode"] = static_cast<int>(current_layout_mode_);

  // Save Watchlist
  if (watchlist_component_) {
    nlohmann::json wl_json = nlohmann::json::array();
    for (const auto &entry : watchlist_component_->get_entries()) {
      wl_json.push_back(entry.symbol);
    }
    j["watchlist"] = wl_json;
  }

  // Save Alerts
  nlohmann::json alerts_json = nlohmann::json::array();
  for (const auto &alert : alert_manager_.get_alerts()) {
    nlohmann::json a;
    a["symbol"] = alert.symbol;
    a["condition"] = static_cast<int>(alert.condition);
    a["target"] = alert.target_value;
    alerts_json.push_back(a);
  }
  j["alerts"] = alerts_json;

  nlohmann::json components_json = nlohmann::json::array();
  for (const auto &comp : components_) {
    nlohmann::json c;
    c["name"] = comp->get_name();
    c["visible"] = comp->is_visible();
    c["minimized"] = comp->is_minimized();
    c["pos"] = {comp->get_position().x, comp->get_position().y};
    c["size"] = {comp->get_size().x, comp->get_size().y};
    components_json.push_back(c);
  }
  j["components"] = components_json;

  std::ofstream o(filename);
  o << std::setw(4) << j << std::endl;
  fprintf(stderr, "[VulkanDashboard] Layout saved to %s\n", filename.c_str());
}

void VulkanDashboard::load_layout(const std::string &filename) {
  std::ifstream i(filename);
  if (!i.is_open()) {
    fprintf(stderr, "[VulkanDashboard] Could not open layout file %s\n",
            filename.c_str());
    return;
  }

  nlohmann::json j;
  i >> j;

  if (j.contains("theme")) {
    apply_theme(static_cast<AppTheme>(j["theme"].get<int>()));
  }

  // Phase 7: Load Persistence 2.0 fields
  if (j.contains("active_symbol")) {
    set_active_symbol(j["active_symbol"].get<std::string>());
  }
  if (j.contains("active_chart_index")) {
    active_chart_index_ = j["active_chart_index"].get<int>();
  }
  if (j.contains("layout_mode")) {
    current_layout_mode_ = static_cast<LayoutMode>(j["layout_mode"].get<int>());
  }

  if (watchlist_component_ && j.contains("watchlist")) {
    watchlist_component_->clear_symbols();
    for (const auto &sym : j["watchlist"]) {
      watchlist_component_->add_symbol(sym.get<std::string>());
    }
  }

  if (j.contains("alerts")) {
    alert_manager_.clear_alerts();
    for (const auto &a_json : j["alerts"]) {
      AlertRule rule;
      rule.symbol = a_json["symbol"];
      rule.condition =
          static_cast<AlertCondition>(a_json["condition"].get<int>());
      rule.target_value = a_json["target"];
      rule.is_active = true;
      alert_manager_.add_alert(rule);
    }
  }

  if (j.contains("components")) {
    for (const auto &c_json : j["components"]) {
      std::string name = c_json["name"];
      for (auto &comp : components_) {
        if (comp->get_name() == name) {
          comp->set_visible(c_json["visible"]);
          if (c_json.contains("minimized")) {
            comp->set_minimized(c_json["minimized"]);
          }
          comp->set_position({c_json["pos"][0], c_json["pos"][1]});
          comp->set_size({c_json["size"][0], c_json["size"][1]});
          break;
        }
      }
    }
  }
  fprintf(stderr, "[VulkanDashboard] Layout loaded from %s\n",
          filename.c_str());
}

void VulkanDashboard::render_command_palette() {
  // Center the modal
  ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(600, 400));

  if (show_command_palette_)
    ImGui::OpenPopup("Command Palette");

  if (ImGui::BeginPopupModal("Command Palette", &show_command_palette_,
                             ImGuiWindowFlags_NoDecoration |
                                 ImGuiWindowFlags_NoMove)) {

    // Search Bar
    ImGui::SetKeyboardFocusHere(0);
    if (ImGui::InputText("##CommandSearch", command_buffer_,
                         sizeof(command_buffer_),
                         ImGuiInputTextFlags_AutoSelectAll |
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
      // Execute first match logic here if desired
    }

    ImGui::Separator();

    struct Command {
      std::string name;
      std::string shortcut;
      std::function<void()> action;
    };

    std::vector<Command> commands = {
        {"Save Layout", "Ctrl+S",
         [this]() {
           save_layout("config/last_layout.json");
           show_command_palette_ = false;
           ImGui::CloseCurrentPopup();
         }},
        {"Load Layout", "Ctrl+L",
         [this]() {
           load_layout("config/last_layout.json");
           show_command_palette_ = false;
           ImGui::CloseCurrentPopup();
         }},
        {"Reset Layout", "Ctrl+R",
         [this]() {
           load_layout("config/default_layout.json");
           show_command_palette_ = false;
           ImGui::CloseCurrentPopup();
         }},
        {"Toggle Performance Overlay", "F3",
         [this]() {
           show_performance_overlay_ = !show_performance_overlay_;
           show_command_palette_ = false;
           ImGui::CloseCurrentPopup();
         }},
        {"Toggle Fullscreen", "F11",
         [this]() {
           show_command_palette_ = false;
           ImGui::CloseCurrentPopup();
         }},
        {"Toggle Theme", "",
         [this]() {
           // Cycle through themes: TealStreet -> InstitutionalDark -> LightMode
           // -> TealStreet
           if (current_theme_ == AppTheme::TealStreet) {
             apply_theme(AppTheme::InstitutionalDark);
           } else if (current_theme_ == AppTheme::InstitutionalDark) {
             apply_theme(AppTheme::LightMode);
           } else { // current_theme_ == AppTheme::LightMode
             apply_theme(AppTheme::TealStreet);
           }
           show_command_palette_ = false;
           ImGui::CloseCurrentPopup();
         }},
        {"Exit Application", "Alt+F4", [this]() { exit(0); }}};

    if (watchlist_component_) {
      commands.push_back({"Add Symbol to Watchlist...", "", [this]() {
                            show_command_palette_ = false;
                            ImGui::CloseCurrentPopup();
                          }});
    }

    ImGui::BeginChild("CommandList");
    for (const auto &cmd : commands) {
      std::string search = command_buffer_;
      std::string name = cmd.name;
      bool match = true;
      if (!search.empty()) {
        std::string s_lower = search;
        std::string n_lower = name;
        std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(),
                       ::tolower);
        std::transform(n_lower.begin(), n_lower.end(), n_lower.begin(),
                       ::tolower);
        if (n_lower.find(s_lower) == std::string::npos)
          match = false;
      }

      if (match) {
        if (ImGui::Selectable(cmd.name.c_str())) {
          cmd.action();
        }
        ImGui::SameLine(ImGui::GetWindowContentRegionMax().x - 100);
        ImGui::TextDisabled("%s", cmd.shortcut.c_str());
      }
    }
    ImGui::EndChild();

    ImGui::EndPopup();
  }
}

} // namespace BTQuant
