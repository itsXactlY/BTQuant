#ifndef BTQUANT_VULKAN_CONTEXT_HPP
#define BTQUANT_VULKAN_CONTEXT_HPP

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace btquant::vulkan {

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
  std::optional<uint32_t> computeFamily;

  [[nodiscard]] bool isComplete() const noexcept {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

struct SwapchainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities{};
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

enum class VulkanError {
  InstanceCreationFailed,
  NoPhysicalDevice,
  NoSuitableDevice,
  QueueFamilyNotFound,
  SwapchainCreationFailed,
  CommandPoolCreationFailed,
  RenderPassCreationFailed,
  WindowCreationFailed,
  ExtensionNotSupported,
  OutOfMemory,
  Unknown
};

class VulkanContext {
public:
  VulkanContext() noexcept;
  ~VulkanContext() noexcept;

  VulkanContext(const VulkanContext &) = delete;
  VulkanContext &operator=(const VulkanContext &) = delete;

  VulkanContext(VulkanContext &&other) noexcept;
  VulkanContext &operator=(VulkanContext &&other) noexcept;

  // Initialize - returns nullptr on success, error string on failure
  [[nodiscard]] std::optional<std::string> initialize() noexcept;

  // Granular init steps used by main
  [[nodiscard]] std::optional<std::string> createInstance() noexcept;
  [[nodiscard]] std::optional<std::string> pickPhysicalDevice() noexcept;
  [[nodiscard]] std::optional<std::string> createLogicalDevice() noexcept;

  void setSurface(VkSurfaceKHR surface) { m_surface = surface; }

  [[nodiscard]] std::optional<std::string>
  createWindow(uint32_t width, uint32_t height,
               const std::string &title) noexcept;
  [[nodiscard]] std::optional<std::string> createSwapchain() noexcept;
  [[nodiscard]] std::optional<std::string> recreateSwapchain() noexcept;
  void cleanupSwapchain() noexcept;

  [[nodiscard]] VkInstance instance() const noexcept { return m_instance; }
  [[nodiscard]] VkPhysicalDevice physicalDevice() const noexcept {
    return m_physicalDevice;
  }
  [[nodiscard]] VkDevice device() const noexcept { return m_device; }
  [[nodiscard]] VkSurfaceKHR surface() const noexcept { return m_surface; }
  [[nodiscard]] VkQueue graphicsQueue() const noexcept {
    return m_graphicsQueue;
  }
  [[nodiscard]] VkQueue presentQueue() const noexcept { return m_presentQueue; }
  [[nodiscard]] VkQueue computeQueue() const noexcept { return m_computeQueue; }
  [[nodiscard]] VkCommandPool commandPool() const noexcept {
    return m_commandPool;
  }
  [[nodiscard]] VkRenderPass renderPass() const noexcept {
    return m_renderPass;
  }
  [[nodiscard]] VkSwapchainKHR swapchain() const noexcept {
    return m_swapchain;
  }

  [[nodiscard]] const std::vector<VkImage> &swapchainImages() const noexcept {
    return m_swapchainImages;
  }
  [[nodiscard]] const std::vector<VkImageView> &
  swapchainImageViews() const noexcept {
    return m_swapchainImageViews;
  }
  [[nodiscard]] VkExtent2D swapchainExtent() const noexcept {
    return m_swapchainExtent;
  }
  [[nodiscard]] VkFormat swapchainImageFormat() const noexcept {
    return m_swapchainImageFormat;
  }

  [[nodiscard]] const QueueFamilyIndices &queueFamilies() const noexcept {
    return m_queueFamilies;
  }
  [[nodiscard]] SwapchainSupportDetails
  querySwapchainSupport(VkPhysicalDevice device) const noexcept;

  [[nodiscard]] uint32_t currentFrame() const noexcept {
    return m_currentFrame;
  }
  [[nodiscard]] uint32_t maxFramesInFlight() const noexcept {
    return m_maxFramesInFlight;
  }

  [[nodiscard]] VkCommandBuffer currentCommandBuffer() const noexcept;
  [[nodiscard]] VkFramebuffer currentFramebuffer() const noexcept;
  [[nodiscard]] uint32_t currentImageIndex() const noexcept { return m_currentImageIndex; }
  [[nodiscard]] VkCommandBuffer beginFrame() noexcept;
  void endFrame() noexcept;

  [[nodiscard]] VkSemaphore imageAvailableSemaphore() const noexcept;
  [[nodiscard]] VkSemaphore renderFinishedSemaphore() const noexcept;
  [[nodiscard]] VkFence inFlightFence() const noexcept;

  [[nodiscard]] std::string deviceName() const noexcept;
  void cleanup() noexcept;
  [[nodiscard]] bool isInitialized() const noexcept { return m_initialized; }

private:
  VkInstance m_instance = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;
  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VkCommandPool m_commandPool = VK_NULL_HANDLE;
  VkRenderPass m_renderPass = VK_NULL_HANDLE;
  VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;

  VkQueue m_graphicsQueue = VK_NULL_HANDLE;
  VkQueue m_presentQueue = VK_NULL_HANDLE;
  VkQueue m_computeQueue = VK_NULL_HANDLE;

  std::vector<VkImage> m_swapchainImages;
  std::vector<VkImageView> m_swapchainImageViews;
  std::vector<VkFramebuffer> m_framebuffers;
  std::vector<VkCommandBuffer> m_commandBuffers;
  std::vector<VkSemaphore> m_imageAvailableSemaphores;
  std::vector<VkSemaphore> m_renderFinishedSemaphores;
  std::vector<VkFence> m_inFlightFences;

  VkExtent2D m_swapchainExtent{};
  VkFormat m_swapchainImageFormat{};
  VkColorSpaceKHR m_swapchainColorSpace{};

  QueueFamilyIndices m_queueFamilies;
  uint32_t m_currentFrame = 0;
  uint32_t m_maxFramesInFlight = 2;
  uint32_t m_currentImageIndex = 0;

  bool m_initialized = false;
  bool m_validationEnabled = false;
  std::vector<const char *> m_validationLayers;

  [[nodiscard]] std::optional<std::string> createCommandPool() noexcept;
  [[nodiscard]] std::optional<std::string> createRenderPass() noexcept;
  [[nodiscard]] std::optional<std::string> createFramebuffers() noexcept;
  [[nodiscard]] std::optional<std::string> createCommandBuffers() noexcept;
  [[nodiscard]] std::optional<std::string> createSyncObjects() noexcept;

  [[nodiscard]] QueueFamilyIndices
  findQueueFamilies(VkPhysicalDevice device) const noexcept;
  [[nodiscard]] bool
  isDeviceSuitable(VkPhysicalDevice device) const noexcept;
  [[nodiscard]] bool
  checkDeviceExtensionSupport(VkPhysicalDevice device) const noexcept;
  [[nodiscard]] VkSurfaceFormatKHR chooseSwapchainSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &available) const noexcept;
  [[nodiscard]] VkPresentModeKHR chooseSwapchainPresentMode(
      const std::vector<VkPresentModeKHR> &available) const noexcept;
  [[nodiscard]] VkExtent2D
  chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR &capabilities,
                        uint32_t width, uint32_t height) const noexcept;
};

} // namespace btquant::vulkan

#endif
