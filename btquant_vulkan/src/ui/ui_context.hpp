#ifndef BTQUANT_UI_CONTEXT_HPP
#define BTQUANT_UI_CONTEXT_HPP

#include <imgui.h>
#include <implot.h>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <optional>
#include <functional>

class VulkanContext;  // forward for UIContext to reach render-pass + font-upload

namespace btquant::ui {

struct UIConfig {
    ImVec4 background = ImVec4(0.05f, 0.07f, 0.09f, 1.0f);
    ImVec4 bidColor = ImVec4(0.0f, 0.83f, 1.0f, 1.0f);
    ImVec4 askColor = ImVec4(1.0f, 0.28f, 0.34f, 1.0f);
    ImVec4 gridColor = ImVec4(0.12f, 0.15f, 0.19f, 1.0f);
    ImVec4 textColor = ImVec4(0.9f, 0.9f, 0.9f, 1.0f);
};

class UIContext {
public:
    UIContext();
    ~UIContext();

    bool initialize(void* window, VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device, uint32_t graphicsQueueFamily, VkQueue graphicsQueue, VkRenderPass renderPass);
    void shutdown();
    void newFrame();
    void render(VkCommandBuffer commandBuffer);

    [[nodiscard]] const UIConfig& config() const { return m_config; }
    [[nodiscard]] UIConfig& config() { return m_config; }

private:
    UIConfig m_config;
    bool m_initialized = false;
    VkDevice m_imguiDevice = VK_NULL_HANDLE;
    VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;
};

} // namespace btquant::ui

#endif
