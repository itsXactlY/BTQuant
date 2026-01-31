#pragma once
#include <atomic>
#include <chrono>
#include <functional>  // Added
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// 3rd party
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

struct ImDrawData;

namespace BTQuant {

class VulkanDashboard;

struct VulkanDashboardConfig {
  // Vulkan configuration
  bool enable_validation_layers = false;
  bool enable_msaa = false;  // Disable MSAA for better performance in sub-second charts
  VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT;
  bool enable_hdr = false;  // Disable HDR for better performance
  VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

  // Performance targets
  uint32_t target_fps = 144;         // Higher target FPS for sub-second charts
  uint32_t max_ui_elements = 500;    // Reduce UI element count for performance
  float max_data_latency_ms = 0.5f;  // Lower latency target

  // Memory configuration (increased for higher data rates)
  size_t vertex_pool_size = 128 * 1024 * 1024;  // 128MB
  size_t uniform_pool_size = 32 * 1024 * 1024;  // 32MB
  size_t storage_pool_size = 64 * 1024 * 1024;  // 64MB

  // HotSpine integration
  std::string shm_name = "/btquant_hotspine";
  std::string symbol_registry_path = "/dev/shm/btquant_symbols.json";

  // Performance optimization flags
  bool enable_command_buffer_recycling = true;
  bool enable_gpu_memory_budgeting = true;
  bool enable_low_latency_mode = true;
};

struct BufferAllocation {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped_ptr = nullptr;
  VkDeviceSize size = 0;
  VkDeviceSize offset = 0;
  bool is_mapped = false;
  uint32_t pool_id = 0;  // 0: None, 1: Vertex, 2: Uniform, 3: Storage, 4: Staging
};

class VulkanException : public std::runtime_error {
 public:
  VulkanException(VkResult result, const std::string& operation)
      : std::runtime_error("Vulkan error"), result_(result), operation_(operation) {
    message_ = "Vulkan error in " + operation + ": " + std::to_string(static_cast<int>(result));
  }

  const char* what() const noexcept override { return message_.c_str(); }
  VkResult result() const { return result_; }
  const std::string& operation() const { return operation_; }

 private:
  VkResult result_;
  std::string operation_;
  std::string message_;
};

// Forward declaration of class needed in many headers
class VulkanErrorHandler {
 public:
  static void check_result(VkResult result, const std::string& operation) {
    if (result != VK_SUCCESS) {
      throw VulkanException(result, operation);
    }
  }
  static void setup_debug_messenger(VkInstance instance);
  static void cleanup_debug_messenger(VkInstance instance);
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                 VkDebugUtilsMessageTypeFlagsEXT message_type,
                 const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data);

  static VkDebugUtilsMessengerEXT debug_messenger_;
};

class MemoryPool {
 public:
  MemoryPool(VkDevice device, VkPhysicalDevice physical_device, VkBufferUsageFlags usage,
             VkMemoryPropertyFlags properties, VkDeviceSize pool_size);
  ~MemoryPool();

  BufferAllocation allocate(VkDeviceSize size, VkDeviceSize alignment = 1);
  void deallocate(const BufferAllocation& allocation);

  VkDeviceSize get_total_size() const { return pool_size_; }
  VkDeviceSize get_used_size() const { return used_size_; }
  VkBuffer get_pool_buffer() const { return pool_buffer_; }
  float get_usage_percentage() const { return static_cast<float>(used_size_) / pool_size_; }

 private:
  VkDevice device_;
  VkPhysicalDevice physical_device_;
  VkBufferUsageFlags usage_;
  VkMemoryPropertyFlags properties_;
  VkBuffer pool_buffer_;
  VkDeviceMemory pool_memory_;
  void* mapped_ptr_;
  VkDeviceSize pool_size_;
  std::atomic<VkDeviceSize> used_size_{0};

  struct FreeBlock {
    VkDeviceSize offset;
    VkDeviceSize size;
  };
  std::vector<FreeBlock> free_blocks_;
  std::mutex allocation_mutex_;
  std::vector<VkBuffer> cleanup_buffers_;
  std::vector<VkDeviceMemory> cleanup_memories_;

  uint32_t find_memory_type(VkPhysicalDevice physical_device, uint32_t type_filter,
                            VkMemoryPropertyFlags properties);
};

class GPUMemoryManager {
 public:
  GPUMemoryManager(VkDevice device, VkPhysicalDevice physical_device,
                   const VulkanDashboardConfig& config);
  ~GPUMemoryManager();

  // Specialized allocators for different buffer types
  BufferAllocation allocate_vertex_buffer(VkDeviceSize size);
  BufferAllocation allocate_index_buffer(VkDeviceSize size);
  BufferAllocation allocate_uniform_buffer(VkDeviceSize size);
  BufferAllocation allocate_storage_buffer(VkDeviceSize size);
  BufferAllocation allocate_staging_buffer(VkDeviceSize size);

  void deallocate_buffer(const BufferAllocation& allocation);

  // Memory usage statistics
  struct MemoryStats {
    VkDeviceSize vertex_pool_used;
    VkDeviceSize uniform_pool_used;
    VkDeviceSize storage_pool_used;
    float vertex_pool_usage;
    float uniform_pool_usage;
    float storage_pool_usage;
  };
  MemoryStats get_memory_stats() const;

 private:
  VkDevice device_;
  VkPhysicalDevice physical_device_;
  std::unique_ptr<MemoryPool> vertex_pool_;
  std::unique_ptr<MemoryPool> uniform_pool_;
  std::unique_ptr<MemoryPool> storage_pool_;
};

class VulkanCore {
 public:
  VulkanCore(const VulkanDashboardConfig& config);
  ~VulkanCore();

  // Initialization and cleanup
  void initialize(GLFWwindow* window, uint32_t width, uint32_t height);
  // Robust Frame Rendering API (User Requested)
  VkResult PrepareFrame(uint32_t& imageIndex);
  VkResult PresentFrame(uint32_t imageIndex);
  void RecordCommandBuffer(uint32_t imageIndex, ImDrawData* drawData,
                           std::function<void(VkCommandBuffer)> graphicsCallback = nullptr);
  void RecreateSwapchain();  // Uses internal width_/height_

  // Legacy/Internal frame rendering
  void begin_main_render_pass();
  void end_frame();
  VkCommandBuffer get_current_command_buffer() const { return current_command_buffer_; }

  // Resource access
  VkDevice get_device() const { return device_; }
  VkPhysicalDevice get_physical_device() const { return physical_device_; }
  uint32_t get_current_frame_index() const { return current_frame_; }
  VkQueue get_graphics_queue() const { return graphics_queue_; }
  VkQueue get_present_queue() const { return present_queue_; }
  VkQueue get_compute_queue() const { return compute_queue_; }
  VkRenderPass get_render_pass() const { return render_pass_; }
  VkExtent2D get_swapchain_extent() const { return swapchain_extent_; }

  // Memory management
  GPUMemoryManager& get_memory_manager() { return *memory_manager_; }

  // Synchronization
  VkDescriptorPool get_descriptor_pool() const { return descriptor_pool_; }
  void wait_idle() { vkDeviceWaitIdle(device_); }

  // Swapchain management
  void recreate_swapchain(uint32_t width, uint32_t height);

  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);

  // Helper methods for single-time commands
  VkCommandBuffer begin_single_time_commands();
  void end_single_time_commands(VkCommandBuffer commandBuffer);

  // Performance optimization methods
  void set_low_latency_mode(bool enabled);
  bool is_low_latency_mode() const { return config_.enable_low_latency_mode; }

  // Frame timing metrics
  float get_frame_time_ms() const { return frame_time_ms_; }
  float get_fps() const { return fps_; }

 private:
  VulkanDashboardConfig config_;

  // Core Vulkan objects
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;

  // Queues
  VkQueue graphics_queue_ = VK_NULL_HANDLE;
  VkQueue present_queue_ = VK_NULL_HANDLE;
  VkQueue compute_queue_ = VK_NULL_HANDLE;
  uint32_t graphics_queue_family_ = UINT32_MAX;
  uint32_t present_queue_family_ = UINT32_MAX;
  uint32_t compute_queue_family_ = UINT32_MAX;

  // Swapchain
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  std::vector<VkImage> swapchain_images_;
  std::vector<VkImageView> swapchain_image_views_;
  VkFormat swapchain_image_format_;
  VkExtent2D swapchain_extent_;

  // Render pass and framebuffers
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> framebuffers_;

  // MSAA resources
  VkImage msaa_color_image_ = VK_NULL_HANDLE;
  VkImageView msaa_color_image_view_ = VK_NULL_HANDLE;
  VkDeviceMemory msaa_color_memory_ = VK_NULL_HANDLE;

  // Depth buffer
  VkImage depth_image_ = VK_NULL_HANDLE;
  VkImageView depth_image_view_ = VK_NULL_HANDLE;
  VkDeviceMemory depth_memory_ = VK_NULL_HANDLE;

  // Command buffers (recyclable)
  VkCommandPool command_pool_ = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> command_buffers_;
  std::vector<bool> command_buffer_in_use_;
  VkCommandBuffer current_command_buffer_ = VK_NULL_HANDLE;

  // Descriptor pools
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorPool imgui_descriptor_pool_ = VK_NULL_HANDLE;

  // Synchronization
  std::vector<VkSemaphore> image_available_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> in_flight_fences_;
  uint32_t current_frame_ = 0;
  static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

  // Memory management
  std::unique_ptr<GPUMemoryManager> memory_manager_;

  // Frame timing
  std::chrono::high_resolution_clock::time_point last_frame_time_;
  float delta_time_ = 0.0f;
  float frame_time_ms_ = 0.0f;
  float fps_ = 0.0f;

  // Current frame data
  uint32_t current_image_index_ = 0;

  // Performance monitoring
  std::vector<float> frame_time_history_;
  static constexpr size_t FRAME_TIME_HISTORY_SIZE = 100;

  // Private initialization methods
  void create_instance();
  void select_physical_device();
  void create_logical_device();
  void create_surface(GLFWwindow* window);
  void create_swapchain(uint32_t width, uint32_t height);
  void create_image_views();
  void create_render_pass();
  void create_msaa_resources();
  void create_depth_resources();
  void create_framebuffers();
  void create_command_pool();
  void create_command_buffers();
  void create_descriptor_pool();
  void create_sync_objects();

  // Helper methods
  std::vector<const char*> get_required_extensions();
  bool check_validation_layer_support();
  bool is_device_suitable(VkPhysicalDevice device);
  VkSampleCountFlagBits get_max_usable_sample_count();
  VkFormat find_supported_format(const std::vector<VkFormat>& candidates, VkImageTiling tiling,
                                 VkFormatFeatureFlags features);
  VkFormat find_depth_format();

  // Cleanup helpers
  void cleanup();
  void cleanup_swapchain();

  // ImGui resources
  void init_imgui();
  void cleanup_imgui();

  VkSampler default_sampler_ = VK_NULL_HANDLE;
  void create_default_sampler();

  // Command buffer recycling
  VkCommandBuffer acquire_command_buffer();
  void release_command_buffer(VkCommandBuffer cmd_buf);
};

}  // namespace BTQuant
