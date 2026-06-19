#include "vulkan_context.hpp"
#include <limits>
#include <set>
#include <iostream>

#ifdef BTQUANT_USE_GLFW
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif

namespace btquant::vulkan {

VulkanContext::VulkanContext() noexcept {}

VulkanContext::~VulkanContext() noexcept { 
    cleanup(); 
}

VulkanContext::VulkanContext(VulkanContext &&other) noexcept {
    *this = std::move(other);
}

VulkanContext &VulkanContext::operator=(VulkanContext &&other) noexcept {
    if (this != &other) {
        cleanup();
        
        m_instance = other.m_instance;
        m_physicalDevice = other.m_physicalDevice;
        m_device = other.m_device;
        m_surface = other.m_surface;
        m_commandPool = other.m_commandPool;
        m_renderPass = other.m_renderPass;
        m_swapchain = other.m_swapchain;
        
        m_graphicsQueue = other.m_graphicsQueue;
        m_presentQueue = other.m_presentQueue;
        m_computeQueue = other.m_computeQueue;
        
        m_swapchainImages = std::move(other.m_swapchainImages);
        m_swapchainImageViews = std::move(other.m_swapchainImageViews);
        m_framebuffers = std::move(other.m_framebuffers);
        m_commandBuffers = std::move(other.m_commandBuffers);
        m_imageAvailableSemaphores = std::move(other.m_imageAvailableSemaphores);
        m_renderFinishedSemaphores = std::move(other.m_renderFinishedSemaphores);
        m_inFlightFences = std::move(other.m_inFlightFences);
        
        m_swapchainExtent = other.m_swapchainExtent;
        m_swapchainImageFormat = other.m_swapchainImageFormat;
        m_swapchainColorSpace = other.m_swapchainColorSpace;
        
        m_queueFamilies = other.m_queueFamilies;
        m_currentFrame = other.m_currentFrame;
        m_maxFramesInFlight = other.m_maxFramesInFlight;
        m_currentImageIndex = other.m_currentImageIndex;
        
        m_initialized = other.m_initialized;
        m_validationEnabled = other.m_validationEnabled;
        m_validationLayers = std::move(other.m_validationLayers);
        
        // Reset other object's resources
        other.m_instance = VK_NULL_HANDLE;
        other.m_physicalDevice = VK_NULL_HANDLE;
        other.m_device = VK_NULL_HANDLE;
        other.m_surface = VK_NULL_HANDLE;
        other.m_commandPool = VK_NULL_HANDLE;
        other.m_renderPass = VK_NULL_HANDLE;
        other.m_swapchain = VK_NULL_HANDLE;
        
        other.m_graphicsQueue = VK_NULL_HANDLE;
        other.m_presentQueue = VK_NULL_HANDLE;
        other.m_computeQueue = VK_NULL_HANDLE;
        
        other.m_initialized = false;
    }
    return *this;
}

std::optional<std::string> VulkanContext::initialize() noexcept {
    // Note: createInstance() and setSurface() are the caller's responsibility —
    // we cannot create the surface here because it needs the GLFW window handle.
    if (m_instance == VK_NULL_HANDLE) {
        return "createInstance() must be called before initialize()";
    }
    if (m_surface == VK_NULL_HANDLE) {
        return "setSurface() must be called before initialize()";
    }
    if (auto err = pickPhysicalDevice())
        return err;
    if (auto err = createLogicalDevice())
        return err;
    if (auto err = createSwapchain())
        return err;
    if (auto err = createRenderPass())
        return err;
    if (auto err = createFramebuffers())
        return err;
    if (auto err = createCommandPool())
        return err;
    if (auto err = createCommandBuffers())
        return err;
    if (auto err = createSyncObjects())
        return err;
    m_initialized = true;
    return std::nullopt;
}

std::optional<std::string>
VulkanContext::createWindow(uint32_t width, uint32_t height,
                            const std::string &title) noexcept {
    // Handled externally in main.cpp for now using GLFW direct pointers.
    return std::nullopt;
}

std::optional<std::string> VulkanContext::createSwapchain() noexcept {
    SwapchainSupportDetails swapChainSupport = querySwapchainSupport(m_physicalDevice);
    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapchainSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode =
        chooseSwapchainPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapchainExtent(swapChainSupport.capabilities, 1280,
                                              720); // TODO dynamically
    m_swapchainImageFormat = surfaceFormat.format;
    m_swapchainExtent = extent;
    m_swapchainColorSpace = surfaceFormat.colorSpace;

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilyIndices[] = {m_queueFamilies.graphicsFamily.value(),
                                     m_queueFamilies.presentFamily.value()};
    if (m_queueFamilies.graphicsFamily != m_queueFamilies.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain) !=
        VK_SUCCESS) {
        return "Failed to create swapchain";
    }

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount,
                            m_swapchainImages.data());

    // Create image views
    m_swapchainImageViews.resize(m_swapchainImages.size());
    for (size_t i = 0; i < m_swapchainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = m_swapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = m_swapchainImageFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &createInfo, nullptr, &m_swapchainImageViews[i]) != VK_SUCCESS) {
            return "Failed to create image views";
        }
    }

    return std::nullopt;
}

std::optional<std::string> VulkanContext::recreateSwapchain() noexcept {
    cleanupSwapchain();
    return createSwapchain();
}

void VulkanContext::cleanupSwapchain() noexcept {
    for (auto framebuffer : m_framebuffers) {
        vkDestroyFramebuffer(m_device, framebuffer, nullptr);
    }
    m_framebuffers.clear();

    for (auto imageView : m_swapchainImageViews) {
        vkDestroyImageView(m_device, imageView, nullptr);
    }
    m_swapchainImageViews.clear();

    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

SwapchainSupportDetails
VulkanContext::querySwapchainSupport(VkPhysicalDevice device) const noexcept {
    SwapchainSupportDetails details;
    if (device == VK_NULL_HANDLE || m_surface == VK_NULL_HANDLE) {
        return details;  // empty — caller treats this as "not suitable"
    }
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface,
                                              &details.capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            device, m_surface, &formatCount, details.formats.data());
    }
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface,
                                              &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface,
                                                  &presentModeCount,
                                                  details.presentModes.data());
    }
    return details;
}

VkCommandBuffer VulkanContext::currentCommandBuffer() const noexcept {
    if (m_commandBuffers.empty()) {
        return VK_NULL_HANDLE;
    }
    return m_commandBuffers[m_currentFrame];
}

VkFramebuffer VulkanContext::currentFramebuffer() const noexcept {
    if (m_framebuffers.empty() || m_currentImageIndex >= m_framebuffers.size()) {
        return VK_NULL_HANDLE;
    }
    return m_framebuffers[m_currentImageIndex];
}

VkCommandBuffer VulkanContext::beginFrame() noexcept {
    vkWaitForFences(m_device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX,
                                            m_imageAvailableSemaphores[m_currentFrame],
                                            VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        (void)recreateSwapchain();
        return VK_NULL_HANDLE;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        return VK_NULL_HANDLE;
    }

    m_currentImageIndex = imageIndex;

    vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);

    VkCommandBuffer commandBuffer = m_commandBuffers[imageIndex];

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    return commandBuffer;
}

void VulkanContext::endFrame() noexcept {
    VkCommandBuffer commandBuffer = m_commandBuffers[m_currentFrame];

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {m_imageAvailableSemaphores[m_currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkSemaphore signalSemaphores[] = {m_renderFinishedSemaphores[m_currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS) {
        return;
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapchains[] = {m_swapchain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;

    presentInfo.pImageIndices = &m_currentImageIndex;
    presentInfo.pResults = nullptr;

    VkResult result = vkQueuePresentKHR(m_presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        (void)recreateSwapchain();
    } else if (result != VK_SUCCESS) {
        return;
    }

    m_currentFrame = (m_currentFrame + 1) % m_maxFramesInFlight;
}

VkSemaphore VulkanContext::imageAvailableSemaphore() const noexcept {
    if (m_imageAvailableSemaphores.empty()) {
        return VK_NULL_HANDLE;
    }
    return m_imageAvailableSemaphores[m_currentFrame];
}

VkSemaphore VulkanContext::renderFinishedSemaphore() const noexcept {
    if (m_renderFinishedSemaphores.empty()) {
        return VK_NULL_HANDLE;
    }
    return m_renderFinishedSemaphores[m_currentFrame];
}

VkFence VulkanContext::inFlightFence() const noexcept {
    if (m_inFlightFences.empty()) {
        return VK_NULL_HANDLE;
    }
    return m_inFlightFences[m_currentFrame];
}

std::string VulkanContext::deviceName() const noexcept {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);
    return std::string(properties.deviceName);
}

void VulkanContext::cleanup() noexcept {
    if (m_device) {
        vkDeviceWaitIdle(m_device);
        
        for (size_t i = 0; i < m_maxFramesInFlight; i++) {
            if (m_imageAvailableSemaphores[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_device, m_imageAvailableSemaphores[i], nullptr);
            }
            if (m_renderFinishedSemaphores[i] != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_device, m_renderFinishedSemaphores[i], nullptr);
            }
            if (m_inFlightFences[i] != VK_NULL_HANDLE) {
                vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
            }
        }

        if (m_commandBuffers.size() > 0) {
            vkFreeCommandBuffers(m_device, m_commandPool, 
                                static_cast<uint32_t>(m_commandBuffers.size()), 
                                m_commandBuffers.data());
        }

        if (m_commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        }

        cleanupSwapchain();

        if (m_renderPass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(m_device, m_renderPass, nullptr);
        }

        vkDestroyDevice(m_device, nullptr);
    }
    if (m_instance) {
        vkDestroyInstance(m_instance, nullptr);
    }
    m_initialized = false;
}

std::optional<std::string> VulkanContext::createInstance() noexcept {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "BTQuant Terminal";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "BTQuant Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;

    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
        return "Failed to create Vulkan instance";
    }
    return std::nullopt;
}

std::optional<std::string> VulkanContext::pickPhysicalDevice() noexcept {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        return "Failed to find GPUs with Vulkan support";
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());
    
    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            m_physicalDevice = device;
            m_queueFamilies = findQueueFamilies(device);
            break;
        }
    }

    if (m_physicalDevice == VK_NULL_HANDLE) {
        return "Failed to find a suitable GPU";
    }
    
    return std::nullopt;
}

bool VulkanContext::isDeviceSuitable(VkPhysicalDevice device) const noexcept {
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapchainSupportDetails swapChainSupport = querySwapchainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

bool VulkanContext::checkDeviceExtensionSupport(VkPhysicalDevice device) const noexcept {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

std::optional<std::string> VulkanContext::createLogicalDevice() noexcept {
    m_queueFamilies = findQueueFamilies(m_physicalDevice);
    if (!m_queueFamilies.isComplete()) {
        return "Failed to find required queue families";
    }

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        m_queueFamilies.graphicsFamily.value(), 
        m_queueFamilies.presentFamily.value()
    };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;

    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) !=
        VK_SUCCESS) {
        return "Failed to create logical device";
    }

    vkGetDeviceQueue(m_device, m_queueFamilies.graphicsFamily.value(), 0,
                     &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_queueFamilies.presentFamily.value(), 0,
                     &m_presentQueue);

    // Try to get a separate compute queue if available
    if (m_queueFamilies.computeFamily.has_value()) {
        vkGetDeviceQueue(m_device, m_queueFamilies.computeFamily.value(), 0,
                         &m_computeQueue);
    } else {
        // Use graphics queue for compute operations if no dedicated compute queue
        m_computeQueue = m_graphicsQueue;
    }

    return std::nullopt;
}

std::optional<std::string> VulkanContext::createCommandPool() noexcept {
    QueueFamilyIndices queueFamiliesIndices = findQueueFamilies(m_physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamiliesIndices.graphicsFamily.value();

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
        return "Failed to create command pool";
    }

    return std::nullopt;
}

std::optional<std::string> VulkanContext::createRenderPass() noexcept {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = m_swapchainImageFormat;
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

    if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
        return "Failed to create render pass";
    }

    return std::nullopt;
}

std::optional<std::string> VulkanContext::createFramebuffers() noexcept {
    m_framebuffers.resize(m_swapchainImageViews.size());

    for (size_t i = 0; i < m_swapchainImageViews.size(); i++) {
        VkImageView attachments[] = {
            m_swapchainImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = m_renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = m_swapchainExtent.width;
        framebufferInfo.height = m_swapchainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffers[i]) != VK_SUCCESS) {
            return "Failed to create framebuffer";
        }
    }

    return std::nullopt;
}

std::optional<std::string> VulkanContext::createCommandBuffers() noexcept {
    m_commandBuffers.resize(m_framebuffers.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());

    if (vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()) != VK_SUCCESS) {
        return "Failed to allocate command buffers";
    }

    return std::nullopt;
}

std::optional<std::string> VulkanContext::createSyncObjects() noexcept {
    m_imageAvailableSemaphores.resize(m_maxFramesInFlight);
    m_renderFinishedSemaphores.resize(m_maxFramesInFlight);
    m_inFlightFences.resize(m_maxFramesInFlight);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < m_maxFramesInFlight; i++) {
        if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFences[i]) != VK_SUCCESS) {
            return "Failed to create sync objects";
        }
    }

    return std::nullopt;
}

QueueFamilyIndices
VulkanContext::findQueueFamilies(VkPhysicalDevice device) const noexcept {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);

        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            indices.computeFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

VkSurfaceFormatKHR VulkanContext::chooseSwapchainSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &available) const noexcept {
    for (const auto& availableFormat : available) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return available[0];
}

VkPresentModeKHR VulkanContext::chooseSwapchainPresentMode(
    const std::vector<VkPresentModeKHR> &available) const noexcept {
    VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

    for (const auto& availableMode : available) {
        if (availableMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availableMode;
        } else if (availableMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            bestMode = availableMode;
        }
    }

    return bestMode;
}

VkExtent2D VulkanContext::chooseSwapchainExtent(
    const VkSurfaceCapabilitiesKHR &capabilities, uint32_t width,
    uint32_t height) const noexcept {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        VkExtent2D actualExtent = {width, height};

        actualExtent.width = std::max(capabilities.minImageExtent.width,
                                      std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height,
                                       std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

} // namespace btquant::vulkan