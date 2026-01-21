#include "vulkan_dashboard_advanced.hpp"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "components/quant_workspace_component.hpp"
#include "imgui.h"
#include "implot.h"
#include <iostream>

namespace BTQuant {

VulkanDashboard::VulkanDashboard(
    uint32_t width, uint32_t height, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    const VulkanDashboardConfig &config)
    : width_(width), height_(height), hotspine_bridge_(bridge),
      market_data_processor_(processor), config_(config) {}

VulkanDashboard::~VulkanDashboard() { shutdown(); }

void VulkanDashboard::initialize() {
  std::cout << "[VulkanDashboard] Initializing Advanced Terminal Renderer..."
            << std::endl;

  // 1. ImGui Context Lifecycle Setup (MUST BE BEFORE VULKAN INIT)
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();

  // Style Definitions
  ImGui::StyleColorsDark();
  ImPlot::StyleColorsDark();

  // 2. HW Layer Init
  init_window();
  m_vulkanCore = std::make_unique<VulkanCore>(config_);
  m_vulkanCore->initialize(window_, width_, height_);
  std::cout << "[VulkanDashboard] Vulkan initialized." << std::endl;

  // Initialize Glfw ImGui Backend
  std::cout << "[VulkanDashboard] Initializing ImGui GLFW Backend..."
            << std::endl;
  ImGui_ImplGlfw_InitForVulkan(window_, true);
  std::cout << "[VulkanDashboard] ImGui GLFW Backend initialized." << std::endl;

  init_components();
  std::cout << "[VulkanDashboard] Components initialized." << std::endl;
}

void VulkanDashboard::init_components() {
  // Replaces all obsolete discrete components with the unified QuantWorkspace
  m_workspace = std::make_unique<QuantWorkspaceComponent>(
      hotspine_bridge_, market_data_processor_);
}

void VulkanDashboard::render_frame() {
  // 1. Prepare Frame
  VkResult result = m_vulkanCore->PrepareFrame(m_currentImageIndex);

  // 2. Handle Resize IMMEDIATELY
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
    m_vulkanCore->RecreateSwapchain();
    return; // Skip rendering this frame!
  } else if (result != VK_SUCCESS) {
    throw std::runtime_error("Failed to acquire swapchain image!");
  }

  // 3. ImGui/ImPlot Frame
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // 4. Draw Components (QuantWorkspace)
  if (m_workspace) {
    m_workspace->update(ImGui::GetIO().DeltaTime);
    m_workspace->render_gui();
  }

  // 5. Render & Present
  ImGui::Render();
  m_vulkanCore->RecordCommandBuffer(m_currentImageIndex, ImGui::GetDrawData());

  result = m_vulkanCore->PresentFrame(m_currentImageIndex);

  // 6. Handle Resize AFTER Present (Crucial for some drivers)
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
      m_windowResized) {
    int width, height;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width > 0 && height > 0) { // Only recreate if not minimized
      m_vulkanCore->RecreateSwapchain();
      m_windowResized = false;
    }
  }
}

void VulkanDashboard::shutdown() {
  static bool already_shutdown = false;
  if (already_shutdown) {
    return;
  }

  already_shutdown = true;

  std::cout << "[VulkanDashboard] Terminating Rendering Engine..." << std::endl;

  // Shutdown ImGui backends in correct order
  if (ImGui::GetCurrentContext() != nullptr) {
    ImGuiIO &io = ImGui::GetIO();
    if (io.BackendRendererUserData != nullptr) {
      ImGui_ImplVulkan_Shutdown();
    }

    if (io.BackendPlatformUserData != nullptr) {
      ImGui_ImplGlfw_Shutdown();
    }

    ImPlot::DestroyContext();
    ImGui::DestroyContext();
  }

  if (window_) {
    glfwDestroyWindow(window_);
  }

  if (m_vulkanCore) {
    m_vulkanCore.reset();
  }

  glfwTerminate();
}

void VulkanDashboard::handle_events() { glfwPollEvents(); }

bool VulkanDashboard::should_close() const {
  return glfwWindowShouldClose(window_) || !is_running_;
}

void VulkanDashboard::init_window() {
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW!");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  window_ = glfwCreateWindow(width_, height_, "BTQuant Advanced Dashboard",
                             nullptr, nullptr);
  if (!window_) {
    throw std::runtime_error("Failed to create GLFW window!");
  }

  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
}

void VulkanDashboard::framebuffer_size_callback(GLFWwindow *window, int width,
                                                int height) {
  auto app =
      reinterpret_cast<VulkanDashboard *>(glfwGetWindowUserPointer(window));
  app->m_windowResized = true;
  app->width_ = width;
  app->height_ = height;
}

} // namespace BTQuant
