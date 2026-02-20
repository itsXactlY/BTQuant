#pragma once

/**
 * @file timeline_semaphore.hpp
 * @brief Vulkan Timeline Semaphore Support for Advanced GPU-CPU Synchronization
 * 
 * Timeline semaphores provide a more flexible synchronization mechanism
 * compared to binary semaphores. They allow:
 * - Wait for specific timeline values
 * - Signal from CPU or GPU
 * - Better multi-frame synchronization
 * - Reduced latency in rendering pipeline
 * 
 * Reference: VK_KHR_timeline_semaphore extension
 */

#include <vulkan/vulkan.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <stdexcept>

namespace btq {
namespace vulkan {

/**
 * @brief Timeline semaphore wrapper with value tracking
 */
class TimelineSemaphore {
public:
    TimelineSemaphore() = default;
    
    TimelineSemaphore(VkDevice device, uint64_t initial_value = 0)
        : device_(device)
        , current_value_(initial_value)
    {
        VkSemaphoreTypeCreateInfoKHR timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR;
        timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
        timeline_info.initialValue = initial_value;
        
        VkSemaphoreCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        create_info.pNext = &timeline_info;
        
        VkResult result = vkCreateSemaphore(device, &create_info, nullptr, &semaphore_);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create timeline semaphore");
        }
    }
    
    ~TimelineSemaphore() {
        if (semaphore_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, semaphore_, nullptr);
        }
    }
    
    // Non-copyable, movable
    TimelineSemaphore(const TimelineSemaphore&) = delete;
    TimelineSemaphore& operator=(const TimelineSemaphore&) = delete;
    
    TimelineSemaphore(TimelineSemaphore&& other) noexcept
        : device_(other.device_)
        , semaphore_(other.semaphore_)
        , current_value_(other.current_value_.load())
    {
        other.device_ = VK_NULL_HANDLE;
        other.semaphore_ = VK_NULL_HANDLE;
    }
    
    TimelineSemaphore& operator=(TimelineSemaphore&& other) noexcept {
        if (this != &other) {
            if (semaphore_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, semaphore_, nullptr);
            }
            device_ = other.device_;
            semaphore_ = other.semaphore_;
            current_value_.store(other.current_value_.load());
            other.device_ = VK_NULL_HANDLE;
            other.semaphore_ = VK_NULL_HANDLE;
        }
        return *this;
    }
    
    /**
     * @brief Get the Vulkan semaphore handle
     */
    VkSemaphore get() const { return semaphore_; }
    
    /**
     * @brief Get the current timeline value
     */
    uint64_t getValue() const { return current_value_.load(); }
    
    /**
     * @brief Signal the semaphore from CPU
     * @param value The value to signal
     */
    void signal(uint64_t value) {
        VkSemaphoreSignalInfo signal_info{};
        signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        signal_info.semaphore = semaphore_;
        signal_info.value = value;
        
        vkSignalSemaphore(device_, &signal_info);
        current_value_.store(value);
    }
    
    /**
     * @brief Wait for the semaphore to reach a value
     * @param value The value to wait for
     * @param timeout_ns Timeout in nanoseconds (default: infinite)
     * @return true if the wait succeeded, false if timeout
     */
    bool wait(uint64_t value, uint64_t timeout_ns = UINT64_MAX) {
        VkSemaphoreWaitInfo wait_info{};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.flags = 0;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &semaphore_;
        wait_info.pValues = &value;
        
        VkResult result = vkWaitSemaphores(device_, &wait_info, timeout_ns);
        return result == VK_SUCCESS;
    }
    
    /**
     * @brief Query the current counter value from the GPU
     */
    uint64_t queryCounterValue() const {
        uint64_t value = 0;
        vkGetSemaphoreCounterValue(device_, semaphore_, &value);
        return value;
    }
    
    /**
     * @brief Create a submit info for waiting on this semaphore
     */
    VkPipelineStageFlags getWaitStage(uint64_t wait_value) const {
        // Store for later use in submit info
        wait_value_ = wait_value;
        return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }
    
    /**
     * @brief Get the semaphore handle for submit info
     */
    VkSemaphore getSemaphore() const { return semaphore_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
    std::atomic<uint64_t> current_value_{0};
    mutable uint64_t wait_value_{0};
};

/**
 * @brief Timeline semaphore manager for the rendering engine
 * 
 * Manages multiple timeline semaphores for different synchronization purposes:
 * - Frame synchronization (GPU -> CPU)
 * - Resource synchronization (GPU -> GPU)
 * - Upload synchronization (CPU -> GPU)
 */
class TimelineSemaphoreManager {
public:
    /**
     * @brief Semaphore types for different purposes
     */
    enum class SemaphoreType : uint32_t {
        FRAME_RENDER = 0,       // Signaled when frame rendering completes
        FRAME_PRESENT = 1,      // Signaled when frame is presented
        RESOURCE_UPLOAD = 2,    // Signaled when resource upload completes
        COMPUTE = 3,            // Signaled when compute work completes
        COUNT = 4               // Number of semaphore types
    };
    
    TimelineSemaphoreManager() = default;
    
    /**
     * @brief Initialize the semaphore manager
     */
    void initialize(VkDevice device, VkPhysicalDevice physical_device) {
        device_ = device;
        physical_device_ = physical_device;
        
        // Check for timeline semaphore support
        if (!checkTimelineSemaphoreSupport()) {
            throw std::runtime_error("Timeline semaphores not supported");
        }
        
        // Create semaphores for each type
        for (size_t i = 0; i < static_cast<size_t>(SemaphoreType::COUNT); ++i) {
            semaphores_[i] = TimelineSemaphore(device, 0);
        }
        
        initialized_ = true;
    }
    
    /**
     * @brief Cleanup resources
     */
    void cleanup() {
        if (!initialized_) return;
        
        for (size_t i = 0; i < static_cast<size_t>(SemaphoreType::COUNT); ++i) {
            semaphores_[i] = TimelineSemaphore();
        }
        
        initialized_ = false;
    }
    
    ~TimelineSemaphoreManager() {
        cleanup();
    }
    
    // Non-copyable, non-movable
    TimelineSemaphoreManager(const TimelineSemaphoreManager&) = delete;
    TimelineSemaphoreManager& operator=(const TimelineSemaphoreManager&) = delete;
    
    /**
     * @brief Get a specific semaphore
     */
    const TimelineSemaphore& getSemaphore(SemaphoreType type) const {
        return semaphores_[static_cast<size_t>(type)];
    }
    
    /**
     * @brief Signal a semaphore from CPU
     */
    void signal(SemaphoreType type, uint64_t value) {
        semaphores_[static_cast<size_t>(type)].signal(value);
    }
    
    /**
     * @brief Wait for a semaphore
     */
    bool wait(SemaphoreType type, uint64_t value, uint64_t timeout_ns = UINT64_MAX) {
        return semaphores_[static_cast<size_t>(type)].wait(value, timeout_ns);
    }
    
    /**
     * @brief Get the current value of a semaphore
     */
    uint64_t getValue(SemaphoreType type) const {
        return semaphores_[static_cast<size_t>(type)].getValue();
    }
    
    /**
     * @brief Create a timeline semaphore submit info for command buffer submission
     */
    VkTimelineSemaphoreSubmitInfo createTimelineSubmitInfo(
        uint64_t wait_value, uint64_t signal_value,
        SemaphoreType wait_type = SemaphoreType::FRAME_RENDER,
        SemaphoreType signal_type = SemaphoreType::FRAME_RENDER) const
    {
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        
        static thread_local uint64_t wait_values[1];
        static thread_local uint64_t signal_values[1];
        
        wait_values[0] = wait_value;
        signal_values[0] = signal_value;
        
        timeline_info.waitSemaphoreValueCount = 1;
        timeline_info.pWaitSemaphoreValues = wait_values;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = signal_values;
        
        return timeline_info;
    }
    
    /**
     * @brief Wait for frame completion
     * @param frame_number The frame number to wait for
     * @param timeout_ms Timeout in milliseconds
     */
    bool waitForFrame(uint64_t frame_number, uint64_t timeout_ms = 1000) {
        return wait(SemaphoreType::FRAME_RENDER, frame_number, timeout_ms * 1000000ULL);
    }
    
    /**
     * @brief Signal frame completion from GPU
     * @param frame_number The completed frame number
     */
    void signalFrameComplete(uint64_t frame_number) {
        signal(SemaphoreType::FRAME_RENDER, frame_number);
    }
    
    /**
     * @brief Check if timeline semaphores are supported
     */
    bool checkTimelineSemaphoreSupport() const {
        // Check for VK_KHR_timeline_semaphore extension
        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, nullptr);
        
        std::vector<VkExtensionProperties> extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, extensions.data());
        
        for (const auto& ext : extensions) {
            if (strcmp(ext.extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0) {
                return true;
            }
        }
        
        // Also check Vulkan 1.2+ which has timeline semaphores as core feature
        VkPhysicalDeviceFeatures2 features2{};
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
        
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        timeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        features2.pNext = &timeline_features;
        
        vkGetPhysicalDeviceFeatures2(physical_device_, &features2);
        
        return timeline_features.timelineSemaphore == VK_TRUE;
    }
    
    /**
     * @brief Get the device extension names needed for timeline semaphores
     */
    static std::vector<const char*> getRequiredExtensions() {
        return { VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME };
    }
    
    /**
     * @brief Get the device features structure for timeline semaphores
     */
    static VkPhysicalDeviceTimelineSemaphoreFeatures getRequiredFeatures() {
        VkPhysicalDeviceTimelineSemaphoreFeatures features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        features.timelineSemaphore = VK_TRUE;
        return features;
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    bool initialized_ = false;
    
    std::array<TimelineSemaphore, static_cast<size_t>(SemaphoreType::COUNT)> semaphores_;
};

/**
 * @brief RAII helper for timeline semaphore wait/signal
 */
class TimelineSemaphoreScope {
public:
    TimelineSemaphoreScope(TimelineSemaphoreManager& manager,
                           TimelineSemaphoreManager::SemaphoreType type,
                           uint64_t wait_value, uint64_t signal_value)
        : manager_(manager)
        , type_(type)
        , signal_value_(signal_value)
    {
        manager_.wait(type_, wait_value);
    }
    
    ~TimelineSemaphoreScope() {
        manager_.signal(type_, signal_value_);
    }
    
    // Non-copyable, non-movable
    TimelineSemaphoreScope(const TimelineSemaphoreScope&) = delete;
    TimelineSemaphoreScope& operator=(const TimelineSemaphoreScope&) = delete;

private:
    TimelineSemaphoreManager& manager_;
    TimelineSemaphoreManager::SemaphoreType type_;
    uint64_t signal_value_;
};

/**
 * @brief Frame synchronization helper using timeline semaphores
 */
class FrameSynchronizer {
public:
    FrameSynchronizer() = default;
    
    void initialize(VkDevice device, VkPhysicalDevice physical_device) {
        manager_.initialize(device, physical_device);
    }
    
    void cleanup() {
        manager_.cleanup();
    }
    
    /**
     * @brief Begin a new frame
     * @return The new frame number
     */
    uint64_t beginFrame() {
        return ++current_frame_;
    }
    
    /**
     * @brief Wait for the previous frame to complete
     */
    bool waitForPreviousFrame(uint64_t timeout_ms = 1000) {
        if (current_frame_ <= 1) return true;
        return manager_.waitForFrame(current_frame_ - 1, timeout_ms);
    }
    
    /**
     * @brief Signal that the current frame is complete
     */
    void endFrame() {
        manager_.signalFrameComplete(current_frame_);
    }
    
    /**
     * @brief Get the current frame number
     */
    uint64_t getCurrentFrame() const {
        return current_frame_;
    }
    
    /**
     * @brief Get the semaphore manager
     */
    TimelineSemaphoreManager& getManager() {
        return manager_;
    }

private:
    TimelineSemaphoreManager manager_;
    std::atomic<uint64_t> current_frame_{0};
};

} // namespace vulkan
} // namespace btq
