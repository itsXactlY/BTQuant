#include <cstdlib>
#include <iostream>
#include <stdexcept>

#ifdef BTQUANT_USE_GLFW
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#else
#error "Only GLFW is fully supported in this backend right now"
#endif

#include "core/vulkan_context.hpp"
#include "ui/ui_context.hpp"
#include "ui/window_manager.hpp"

using namespace btquant;

class BTQuantApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    initUI();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window = nullptr;
  vulkan::VulkanContext vkContext;
  ui::UIContext uiContext;
  ui::WindowManager windowManager;

  uint32_t width = 1280;
  uint32_t height = 720;

  void initWindow() {
    if (!glfwInit()) {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // Headless / offscreen window — works around a NVIDIA+X11+i3 hang in glfwCreateWindow
    // when the window is set to visible. Can be flipped to GLFW_TRUE for a desktop session.
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    window =
        glfwCreateWindow(width, height, "BTQuant Terminal", nullptr, nullptr);
    if (!window) {
      throw std::runtime_error("Failed to create GLFW window");
    }
  }

  void initVulkan() {
    if (auto err = vkContext.createInstance()) {
      throw std::runtime_error("Failed to initialize Vulkan: " + *err);
    }

    // Create surface — must happen AFTER instance creation and BEFORE
    // any function that needs queue family present support.
    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(vkContext.instance(), window, nullptr,
                                &surface) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create window surface");
    }
    vkContext.setSurface(surface);

    // initialize() runs the rest of the chain: pickPhysicalDevice, createLogicalDevice,
    // createSwapchain, createRenderPass, createFramebuffers, createCommandPool,
    // createCommandBuffers, createSyncObjects. createInstance() and setSurface()
    // are caller responsibilities (surface needs the GLFW window handle).
    if (auto err = vkContext.initialize()) {
      throw std::runtime_error("Failed to complete Vulkan init: " + *err);
    }
  }

  void initUI() {
    if (!uiContext.initialize(window, vkContext.instance(),
                              vkContext.physicalDevice(), vkContext.device(),
                              vkContext.queueFamilies().graphicsFamily.value(),
                              vkContext.graphicsQueue(),
                              vkContext.renderPass())) {
      throw std::runtime_error("Failed to initialize UI Context");
    }
    windowManager.initialize();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      // Begin a frame: wait for fence, acquire swapchain image, begin render pass.
      VkCommandBuffer cmd = vkContext.beginFrame();
      if (cmd == VK_NULL_HANDLE) {
        // Swapchain was out of date and got recreated — skip this frame.
        continue;
      }

      // Begin ImGui frame BEFORE starting the render pass (per ImGui convention).
      uiContext.newFrame();

      // DockSpace + menu + 4 widgets.
      ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                   ImGuiDockNodeFlags_PassthruCentralNode);
      windowManager.showMainMenu();
      windowManager.showOrderBookWindow();
      windowManager.showDOMWindow();
      windowManager.showTradesWindow();
      windowManager.showTPOWindow();

      // Begin render pass for this swapchain image, hand the command buffer to ImGui,
      // end render pass, end frame (which submits and presents).
      VkClearValue clearColor = {{{0.031f, 0.035f, 0.039f, 1.0f}}};  // #08090a
      VkRenderPassBeginInfo rpBegin{};
      rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      rpBegin.renderPass = vkContext.renderPass();
      rpBegin.framebuffer = vkContext.currentFramebuffer();
      rpBegin.renderArea.offset = {0, 0};
      rpBegin.renderArea.extent = vkContext.swapchainExtent();
      rpBegin.clearValueCount = 1;
      rpBegin.pClearValues = &clearColor;

      vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
      uiContext.render(cmd);  // ImGui_ImplVulkan_RenderDrawData inside
      vkCmdEndRenderPass(cmd);

      vkContext.endFrame();
    }
  }

  void cleanup() {
    if (window) {
      glfwDestroyWindow(window);
      window = nullptr;
    }
    glfwTerminate();
  }
};

int main() {
  BTQuantApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[BTQuant] fatal: %s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
