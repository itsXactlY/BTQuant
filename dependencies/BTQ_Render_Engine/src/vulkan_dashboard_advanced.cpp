#include "vulkan_dashboard_advanced.hpp"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "components/interaction_manager.hpp"
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

  // Style Definitions (Managed by ThemeManager)
  ThemeManager::getInstance().initialize();

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

  // Register Hotkeys
  auto &im = InteractionManager::getInstance();

  // Ctrl+1 to Ctrl+5 for layout switching or panel focus (Placeholder)
  // Ctrl+1 to Ctrl+5 for layout switching
  auto *workspace = m_workspace.get(); // Capture for lambda
  im.registerHotKey(
      ImGuiKey_1,
      [workspace]() {
        if (workspace->getPanelManager()) {
          // workspace->getPanelManager()->load_layout("layout_desktop_3x5.json");
          std::cout << "[Layout] Switched to Desktop 3x5" << std::endl;
          workspace->getPanelManager()
              ->auto_arrange_panels(); // Simple verification action
        }
      },
      "Layout 1 (3x5 Grid)", true);

  im.registerHotKey(
      ImGuiKey_2,
      [workspace]() {
        if (workspace->getPanelManager()) {
          // workspace->getPanelManager()->load_layout("layout_focus_chart.json");
          std::cout << "[Layout] Switched to Chart Focus" << std::endl;
        }
      },
      "Layout 2 (Chart Focus)", true);

  // Space to Toggle Theme
  im.registerHotKey(
      ImGuiKey_Space,
      []() {
        ThemeManager::getInstance().toggleTheme();
        std::cout << "Hotkey: Theme Toggled" << std::endl;
      },
      "Toggle Theme");

  // Alt+Enter for Fullscreen (Requires window handle, capture this)
  GLFWwindow *win = window_;
  im.registerHotKey(
      ImGuiKey_Enter,
      [win]() {
        if (glfwGetWindowMonitor(win)) {
          glfwSetWindowMonitor(win, nullptr, 100, 100, 1280, 720, 0);
        } else {
          GLFWmonitor *monitor = glfwGetPrimaryMonitor();
          const GLFWvidmode *mode = glfwGetVideoMode(monitor);
          glfwSetWindowMonitor(win, monitor, 0, 0, mode->width, mode->height,
                               mode->refreshRate);
        }
      },
      "Toggle Fullscreen", false, true); // Alt+Enter
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

  // Update Interaction Manager
  InteractionManager::getInstance().update();

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
