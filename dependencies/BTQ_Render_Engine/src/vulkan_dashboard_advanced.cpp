#include "vulkan_dashboard_advanced.hpp"

#include <cstdint>
#include <iostream>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "components/footprint_panel.hpp"
#include "components/interaction_manager.hpp"
#include "components/quant_workspace_component.hpp"
#include "components/tpo_panel.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"
#include "performance_monitor.hpp"
#include "performance/debug_overlay.hpp"
#include "ui/layout_manager.hpp"

namespace BTQuant {

VulkanDashboard::VulkanDashboard(uint32_t width, uint32_t height,
                                 std::shared_ptr<HotSpineDataBridge> bridge,
                                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                                 const VulkanDashboardConfig& config)
    : width_(width),
      height_(height),
      hotspine_bridge_(bridge),
      market_data_processor_(processor),
      config_(config) {}

VulkanDashboard::~VulkanDashboard() { shutdown(); }

std::expected<void, std::string> VulkanDashboard::initialize() {
  std::println("[VulkanDashboard] Initializing Advanced Terminal Renderer...");

  // 1. ImGui Context Lifecycle Setup (MUST BE BEFORE VULKAN INIT)
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Optional:
  // Multi-viewport

  // Style Definitions (Managed by ThemeManager)
  ThemeManager::getInstance().initialize();

  // 2. HW Layer Init
  init_window();

  // Use the established VulkanCore initialization
  vulkan_core_ = std::make_unique<VulkanCore>(config_);
  vulkan_core_->initialize(window_, width_, height_);
  std::println("[VulkanDashboard] Vulkan initialized.");

  timeline_semaphore_ = std::make_unique<TimelineSemaphore>(vulkan_core_->get_device());

  // Initialize Glfw ImGui Backend
  std::println("[VulkanDashboard] Initializing ImGui GLFW Backend...");
  ImGui_ImplGlfw_InitForVulkan(window_, true);
  std::println("[VulkanDashboard] ImGui GLFW Backend initialized.");

  init_components();
  return {};
}

void VulkanDashboard::init_components() {
  std::println("[VulkanDashboard] Initializing Components...");
  workspace_ = std::make_unique<QuantWorkspaceComponent>(hotspine_bridge_, market_data_processor_);

  // LAYOUT BOOTSTRAP - Load saved layout or apply default preset
  if (auto* pm = workspace_->getPanelManager()) {
    std::println("[VulkanDashboard] Loading layout...");
    // Try to load saved layout first (load_layout returns void, so we just call it)
    pm->load_layout("default_layout.json");
    
    // If no panels were loaded, apply default preset
    if (pm->get_panel_count() == 0) {
      std::println("[VulkanDashboard] No saved layout found, applying default preset...");
      pm->apply_layout_preset(LayoutPreset::MODERN_TRADING);
    }
  }

  // Register Hotkeys
  auto& im = InteractionManager::getInstance();

  // Ctrl+1 to Ctrl+5 for layout switching or panel focus (Placeholder)
  // Ctrl+1 to Ctrl+5 for layout switching
  auto* workspace = workspace_.get();  // Capture for lambda
  im.registerHotKey(
      ImGuiKey_1,
      [workspace]() {
        if (workspace->getPanelManager()) {
          // workspace->getPanelManager()->load_layout("layout_desktop_3x5.json");
          std::cout << "[Layout] Switched to Desktop 3x5" << std::endl;
          workspace->getPanelManager()->auto_arrange_panels();  // Simple verification action
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
  GLFWwindow* win = window_;
  im.registerHotKey(
      ImGuiKey_Enter,
      [win]() {
        if (glfwGetWindowMonitor(win)) {
          glfwSetWindowMonitor(win, nullptr, 100, 100, 1280, 720, 0);
        } else {
          GLFWmonitor* monitor = glfwGetPrimaryMonitor();
          const GLFWvidmode* mode = glfwGetVideoMode(monitor);
          glfwSetWindowMonitor(win, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
        }
      },
      "Toggle Fullscreen", false, true);  // Alt+Enter

  // Quick-save hotkeys: F5-F8 to save layouts, Shift+F5-F8 to load layouts
  // Using LayoutManager singleton for proper preset management
  im.registerHotKey(ImGuiKey_F5, []() {
    auto& layoutManager = UI::LayoutManager::getInstance();
    if (ImGui::GetIO().KeyShift) {
      // Shift+F5 - Load layout
      if (layoutManager.quick_load_layout(1)) {
        std::cout << "[Layout] Loaded Quick Save 1 (Shift+F5)" << std::endl;
      } else {
        std::cout << "[Layout] Failed to load Quick Save 1 (Shift+F5)" << std::endl;
      }
    } else {
      // F5 - Save layout
      if (layoutManager.quick_save_layout(1)) {
        std::cout << "[Layout] Saved to Quick Save 1 (F5)" << std::endl;
      } else {
        std::cout << "[Layout] Failed to save to Quick Save 1 (F5)" << std::endl;
      }
    }
  }, "Quick Save/Load Layout 1", false, false, false); // F5/F5+Shift

  im.registerHotKey(ImGuiKey_F6, []() {
    auto& layoutManager = UI::LayoutManager::getInstance();
    if (ImGui::GetIO().KeyShift) {
      // Shift+F6 - Load layout
      if (layoutManager.quick_load_layout(2)) {
        std::cout << "[Layout] Loaded Quick Save 2 (Shift+F6)" << std::endl;
      } else {
        std::cout << "[Layout] Failed to load Quick Save 2 (Shift+F6)" << std::endl;
      }
    } else {
      // F6 - Save layout
      if (layoutManager.quick_save_layout(2)) {
        std::cout << "[Layout] Saved to Quick Save 2 (F6)" << std::endl;
      } else {
        std::cout << "[Layout] Failed to save to Quick Save 2 (F6)" << std::endl;
      }
    }
  }, "Quick Save/Load Layout 2", false, false, false); // F6/F6+Shift

  im.registerHotKey(ImGuiKey_F7, []() {
    auto& layoutManager = UI::LayoutManager::getInstance();
    if (ImGui::GetIO().KeyShift) {
      // Shift+F7 - Load layout
      if (layoutManager.quick_load_layout(3)) {
        std::cout << "[Layout] Loaded Quick Save 3 (Shift+F7)" << std::endl;
      } else {
        std::cout << "[Layout] Failed to load Quick Save 3 (Shift+F7)" << std::endl;
      }
    } else {
      // F7 - Save layout
      if (layoutManager.quick_save_layout(3)) {
        std::cout << "[Layout] Saved to Quick Save 3 (F7)" << std::endl;
      } else {
        std::cout << "[Layout] Failed to save to Quick Save 3 (F7)" << std::endl;
      }
    }
  }, "Quick Save/Load Layout 3", false, false, false); // F7/F7+Shift

  im.registerHotKey(ImGuiKey_F8, []() {
    auto& layoutManager = UI::LayoutManager::getInstance();
    if (ImGui::GetIO().KeyShift) {
      // Shift+F8 - Load layout
      if (layoutManager.quick_load_layout(4)) {
        std::cout << "[Layout] Loaded Quick Save 4 (Shift+F8)" << std::endl;
      } else {
        std::cout << "[Layout] Failed to load Quick Save 4 (Shift+F8)" << std::endl;
      }
    } else {
      // F8 - Save layout
      if (layoutManager.quick_save_layout(4)) {
        std::cout << "[Layout] Saved to Quick Save 4 (F8)" << std::endl;
      } else {
        std::cout << "[Layout] Failed to save to Quick Save 4 (F8)" << std::endl;
      }
    }
  }, "Quick Save/Load Layout 4", false, false, false); // F8/F8+Shift

  // F12 - Toggle Debug Overlay
  im.registerHotKey(ImGuiKey_F12, [this]() {
    g_debug_overlay.toggle_visibility();
    std::cout << "[Debug Overlay] Toggled visibility: "
              << (g_debug_overlay.is_visible() ? "ON" : "OFF") << std::endl;
  }, "Toggle Debug Overlay", false, false, false); // F12

  // Delete key - Remove currently focused panel
  im.registerHotKey(ImGuiKey_Delete, [this]() {
    // Find the currently focused panel and remove it
    // We need to iterate through all panels to find which one currently has focus
    auto panel_manager = workspace_->getPanelManager();
    if (panel_manager) {
      auto all_panel_ids = panel_manager->get_all_panel_ids();

      // Find the currently focused window by checking ImGui's focused window
      ImGuiWindow* focused_window = GImGui->NavWindow;
      const char* focused_window_name = focused_window ? focused_window->Name : nullptr;

      for (uint32_t panel_id : all_panel_ids) {
        auto panel = panel_manager->get_panel_by_id(panel_id);
        if (panel) {
          // Check if this panel's window is currently focused
          std::string expected_window_name =
              panel->get_config().title + "###panel_" +
              std::to_string(reinterpret_cast<uintptr_t>(panel));

          if (focused_window_name &&
              std::string(focused_window_name) == expected_window_name) {
            // Found the focused panel, remove it
            panel_manager->remove_panel(panel_id);
            std::cout << "[Hotkey] Removed focused panel: " << panel->get_config().title << std::endl;
            break;
          }
        }
      }
    }
  }, "Remove Focused Panel", false, false, false); // Delete
}

void VulkanDashboard::render_frame() {
  if (window_resized_) {
    vulkan_core_->recreate_swapchain(width_, height_);
    window_resized_ = false;
  }

  uint32_t imageIndex;
  VkResult result = vulkan_core_->PrepareFrame(imageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    vulkan_core_->recreate_swapchain(width_, height_);
    return;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    // Silently ignore or log - swapchain might be rebuilding
    return;
  }

  // Start ImGui frame
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // Process updates and UI - ONLY workspace_->render_gui() (panel_manager is handled internally)
  float dt = vulkan_core_->get_frame_time_ms() / 1000.0f;
  if (workspace_) {
    workspace_->update(dt);
    workspace_->render_gui();
  }

  // Finalize ImGui and Record Graphics commands
  ImGui::Render();

  vulkan_core_->RecordCommandBuffer(imageIndex, ImGui::GetDrawData(), [](VkCommandBuffer cmd) {
    // No microstructure renderer - panels handle their own rendering
  });
  vulkan_core_->PresentFrame(imageIndex);
}

void VulkanDashboard::handle_events() { glfwPollEvents(); }

bool VulkanDashboard::should_close() const { return glfwWindowShouldClose(window_); }

void VulkanDashboard::shutdown() {
  static bool already_shutdown = false;
  if (already_shutdown) {
    return;
  }
  already_shutdown = true;

  std::cout << "[VulkanDashboard] Shutting down..." << std::endl;

  if (vulkan_core_) {
    vulkan_core_->wait_idle();
  }

  workspace_.reset();

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
  window_ = glfwCreateWindow(width_, height_, "BTQuant Advanced Terminal", nullptr, nullptr);
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
}

void VulkanDashboard::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  auto app = reinterpret_cast<VulkanDashboard*>(glfwGetWindowUserPointer(window));
  app->window_resized_ = true;
  app->width_ = width;
  app->height_ = height;
}


void VulkanDashboard::render_performance_overlay() {
  if (!show_performance_overlay_) {
    return;
  }

  ImGui::SetNextWindowPos(ImVec2(10, 40), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(300, 200), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Performance", nullptr,
                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                       ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize)) {
    double fps = g_performance_monitor.get_fps();
    double frame_time = g_performance_monitor.get_frame_time_ms();

    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Frame Time: %.2f ms", frame_time);

    ImGui::Separator();

    auto metrics = g_performance_monitor.get_metrics();
    for (const auto& m : metrics) {
      ImGui::Text("%s: %.2f %s", m.name.c_str(), m.value, m.unit.c_str());
    }

    ImGui::Separator();
    ImGui::Text("Symbol: %s", active_symbol_.c_str());

    if (market_data_processor_) {
      auto stats = market_data_processor_->getPerformanceMetrics();
      ImGui::Text("Trades/sec: %.0f", stats.trades_per_second);
      ImGui::Text("Orderbook Updates/sec: %.0f", stats.orderbooks_per_second);
    }
  }

  ImGui::End();

  // Update debug overlay with active component counts
  if (workspace_ && workspace_->getPanelManager()) {
    size_t active_panels = workspace_->getPanelManager()->get_panel_count();
    g_debug_overlay.set_active_panels_count(active_panels);
  }

  // TODO: Update active indicators and alerts counts when available
  g_debug_overlay.set_active_indicators_count(0); // Placeholder - update when indicator system is integrated
  g_debug_overlay.set_active_alerts_count(0);     // Placeholder - update when alert system is integrated
}

void VulkanDashboard::render_layout_indicator() {
  // Get the active layout name from the LayoutManager
  auto& layoutManager = UI::LayoutManager::getInstance();
  std::string active_layout = layoutManager.get_active_layout_name();

  // Get the active quick save slot
  int active_slot = layoutManager.get_active_quick_slot();
  std::string slot_info = "";
  if (active_slot > 0) {
      slot_info = " (QS" + std::to_string(active_slot) + ")";
  }

  // Position the layout indicator in the top-right corner
  ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 250, 30), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(240, 40), ImGuiCond_Always);

  // Create a transparent overlay window for the layout indicator
  ImGui::Begin("##LayoutIndicator", nullptr,
               ImGuiWindowFlags_NoTitleBar |
               ImGuiWindowFlags_NoResize |
               ImGuiWindowFlags_NoMove |
               ImGuiWindowFlags_NoScrollbar |
               ImGuiWindowFlags_NoScrollWithMouse |
               ImGuiWindowFlags_NoCollapse |
               ImGuiWindowFlags_AlwaysAutoResize |
               ImGuiWindowFlags_NoSavedSettings |
               ImGuiWindowFlags_NoInputs |
               ImGuiWindowFlags_NoFocusOnAppearing |
               ImGuiWindowFlags_NoNav);

  // Draw the layout indicator with a semi-transparent background
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 0.7f)); // Dark semi-transparent background

  // Change text color based on whether a quick save slot is active
  if (active_slot > 0) {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.9f, 0.5f, 1.0f)); // Greenish color for active quick save
  } else {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f)); // Light text
  }

  ImGui::Text("Layout: %s%s", active_layout.c_str(), slot_info.c_str());

  ImGui::PopStyleColor(2);
  ImGui::End();
}

}  // namespace BTQuant
