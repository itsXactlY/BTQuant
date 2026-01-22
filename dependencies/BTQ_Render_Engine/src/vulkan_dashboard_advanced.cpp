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

  // Use the established VulkanCore initialization
  m_vulkanCore = std::make_unique<VulkanCore>(config_);
  m_vulkanCore->initialize(window_, width_, height_);
  std::cout << "[VulkanDashboard] Vulkan initialized." << std::endl;

  // Initialize Glfw ImGui Backend
  std::cout << "[VulkanDashboard] Initializing ImGui GLFW Backend..."
            << std::endl;
  ImGui_ImplGlfw_InitForVulkan(window_, true);
  std::cout << "[VulkanDashboard] ImGui GLFW Backend initialized." << std::endl;

  init_components();
}

void VulkanDashboard::init_components() {
  std::cout << "[VulkanDashboard] Initializing Components..." << std::endl;
  m_workspace = std::make_unique<QuantWorkspaceComponent>(
      hotspine_bridge_, market_data_processor_);

  // Workspace doesn't need to be registered with VulkanCore explicitly
  // as VulkanDashboard will handle its rendering in render_frame.
}

void VulkanDashboard::render_frame() {
  if (m_windowResized) {
    m_vulkanCore->recreate_swapchain(width_, height_);
    m_windowResized = false;
  }

  uint32_t imageIndex;
  VkResult result = m_vulkanCore->PrepareFrame(imageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    m_vulkanCore->recreate_swapchain(width_, height_);
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    // Silently ignore or log - swapchain might be rebuilding
    return;
  }

  // Start ImGui frame
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("Tools")) {
      if (ImGui::MenuItem("Clear Dashboard History")) {
        if (market_data_processor_) {
          market_data_processor_->clearHistory();
        }
      }
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  // Process updates and UI
  float dt = m_vulkanCore->get_frame_time_ms() / 1000.0f;
  if (m_workspace) {
    m_workspace->update(dt);
    m_workspace->render_gui();
  }

  // Finalize ImGui
  ImGui::Render();

  // Record and Present through VulkanCore
  m_vulkanCore->RecordCommandBuffer(imageIndex, ImGui::GetDrawData());
  m_vulkanCore->PresentFrame(imageIndex);
}

void VulkanDashboard::handle_events() { glfwPollEvents(); }

bool VulkanDashboard::should_close() const {
  return glfwWindowShouldClose(window_);
}

void VulkanDashboard::shutdown() {
  static bool already_shutdown = false;
  if (already_shutdown) {
    return;
  }
  already_shutdown = true;

  std::cout << "[VulkanDashboard] Shutting down..." << std::endl;

  if (m_vulkanCore) {
    m_vulkanCore->wait_idle();
  }

  m_workspace.reset();

  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  if (window_) {
    glfwDestroyWindow(window_);
    glfwTerminate();
  }
}

void VulkanDashboard::init_window() {
  if (!glfwInit()) {
    std::cerr << "[VulkanDashboard] Failed to initialize GLFW" << std::endl;
    exit(EXIT_FAILURE);
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(width_, height_, "BTQuant Advanced Terminal",
                             nullptr, nullptr);
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
