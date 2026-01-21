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
  init_vulkan();
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

  // 4. Create fullscreen DockSpace
  ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);

  ImGuiWindowFlags dockspace_flags =
      ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
      ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
      ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

  ImGui::Begin("DockSpace Window", nullptr, dockspace_flags);
  ImGui::PopStyleVar(3);

  ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
  ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                   ImGuiDockNodeFlags_PassthruCentralNode);

  ImGui::End();

  // 5. Draw Components (QuantWorkspace)
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
    window_ = nullptr;
  }
  glfwTerminate();

  // m_vulkanCore handles its own cleanup
}

bool VulkanDashboard::should_close() const {
  return glfwWindowShouldClose(window_);
}

void VulkanDashboard::init_window() {
  if (!glfwInit()) {
    throw std::runtime_error("Failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window_ = glfwCreateWindow(width_, height_, "BTQuant Professional Terminal",
                             nullptr, nullptr);
  if (!window_) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }

  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, [](GLFWwindow *window, int width,
                                             int height) {
    auto app =
        reinterpret_cast<VulkanDashboard *>(glfwGetWindowUserPointer(window));
    app->m_windowResized = true;
    app->width_ = width;
    app->height_ = height;
  });
}

void VulkanDashboard::init_vulkan() {
  m_vulkanCore = std::make_unique<VulkanCore>(config_);
  m_vulkanCore->initialize(window_, width_, height_);
}

void VulkanDashboard::handle_events() { glfwPollEvents(); }

} // namespace BTQuant
