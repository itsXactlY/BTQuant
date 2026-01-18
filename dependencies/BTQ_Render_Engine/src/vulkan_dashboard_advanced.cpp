#include "../include/vulkan_dashboard_advanced.hpp"
#include "../include/market_data_processor.hpp"
// #include "imgui_impl_vulkan.h" // Removed as wrapped by VulkanCore

#include <iostream>
#include <stdexcept>

// Ensure we have the implementations linked or included.
// Since the user asked for a specific 4-file set, and we are not modifying
// CMakeLists to add new CPPs, we might need to rely on the build system.
// HOWEVER, if specific functionality is missing, we must ensure it's compiled.
// For now, adhering to the requested file structure.

namespace BTQuant {

// ============================================================================
// VulkanDashboard Implementation
// ============================================================================

VulkanDashboard::VulkanDashboard(
    uint32_t width, uint32_t height,
    std::shared_ptr<RenderEngine::HotSpineDataBridge> bridge,
    const VulkanDashboardConfig &config)
    : config_(config), width_(width), height_(height),
      hotspine_bridge_(bridge) {

  std::cout << "[VulkanDashboard] Constructed with Bridge Ptr: " << bridge.get()
            << std::endl;
}

VulkanDashboard::~VulkanDashboard() { shutdown(); }

void VulkanDashboard::initialize() {
  init_x11();
  init_vulkan();

  // Initialize ImGui specifically here if not done in VulkanCore
  // We assume VulkanCore handles basic ImGui context, but we need to configure
  // Docking
  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  init_components();
}

void VulkanDashboard::init_x11() {
  display_ = XOpenDisplay(nullptr);
  if (!display_)
    throw std::runtime_error("Failed to open X display");

  int screen = DefaultScreen(display_);
  Window root = RootWindow(display_, screen);

  XSetWindowAttributes attrs;
  attrs.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask |
                     ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                     StructureNotifyMask;

  window_ =
      XCreateWindow(display_, root, 0, 0, width_, height_, 0, CopyFromParent,
                    InputOutput, CopyFromParent, CWEventMask, &attrs);

  XSetStandardProperties(display_, window_, "BTQuant Unified Dashboard",
                         "BTQuant", None, nullptr, 0, nullptr);

  // Handle close button
  wm_delete_window_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(display_, window_, &wm_delete_window_, 1);

  XMapWindow(display_, window_);
  XFlush(display_);
}

void VulkanDashboard::init_vulkan() {
  // Create Core (manages Instance, Device, Swapchain, RenderPass)
  vulkan_core_ = std::make_unique<VulkanCore>(config_);
  vulkan_core_->initialize(display_, window_, width_, height_);
}

void VulkanDashboard::init_components() {
  std::cout << "[VulkanDashboard] Initializing Components..." << std::endl;

  // 1. Chart Component (The Core Visualization)
  // We pass the bridge so it can access data internally if needed
  auto chart = std::make_unique<RealtimeChartComponent>(
      glm::vec2(0, 0), glm::vec2(800, 600), hotspine_bridge_);
  chart->initialize_vulkan_resources(vulkan_core_.get());
  components_.push_back(std::move(chart));
}

void VulkanDashboard::shutdown() {
  if (vulkan_core_) {
    vkDeviceWaitIdle(vulkan_core_->get_device());
  }
  components_.clear();
  vulkan_core_.reset();
  cleanup_x11();
}

void VulkanDashboard::cleanup_x11() {
  if (display_) {
    XDestroyWindow(display_, window_);
    XCloseDisplay(display_);
    display_ = nullptr;
  }
}

bool VulkanDashboard::run_frame() {
  handle_x11_events();
  if (!is_running_)
    return false;

  render_frame();
  return true;
}

void VulkanDashboard::handle_x11_events() {
  XEvent event;
  while (XPending(display_) > 0) {
    XNextEvent(display_, &event);
    if (event.type == ClientMessage) {
      if ((Atom)event.xclient.data.l[0] == wm_delete_window_) {
        is_running_ = false;
      }
    }
  }
}

void VulkanDashboard::synchronize_market_data() {
  // Poll the bridge for new updates
  if (hotspine_bridge_) {
    auto updates = hotspine_bridge_->getLatestUpdates();

    // Dispatch to all components
    for (const auto &update : updates) {

      if (update.type == RenderEngine::MarketDataType::TRADE) {
        RenderEngine::TradeData trade;
        trade.price = update.price;
        trade.size = update.size;
        trade.timestamp = update.timestamp;
        trade.is_buy = (update.side == "buy");
        trade.symbol_id = update.symbol_id;

        for (auto &comp : components_) {
          comp->handle_trade(trade);
        }
      }
      // Handle Orderbook...
    }
  }
}

void VulkanDashboard::render_frame() {
  // 1. Begin Vulkan Frame
  if (!vulkan_core_->begin_frame()) {
    return;
  }

  // 2. Begin Main Render Pass (Clear screen)
  vulkan_core_->begin_main_render_pass();

  // 3. ImGui Docking Setup
  VkCommandBuffer cmd = vulkan_core_->get_current_command_buffer();

  // Create the DockSpace
  ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
  ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                   ImGuiDockNodeFlags_PassthruCentralNode);
  // ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(),
  // ImGuiDockNodeFlags_PassthruCentralNode);

  // 4. Render All Components
  for (auto &comp : components_) {
    comp->render(cmd);
    comp->render_gui();
  }

  // 5. End Frame (ImGui Render + Submit)
  vulkan_core_->end_frame();
}

void VulkanDashboard::set_active_symbol(const std::string &symbol) {
  active_symbol_ = symbol;
  for (auto &comp : components_) {
    comp->set_target_symbol(symbol);
  }
}

void VulkanDashboard::add_component(std::unique_ptr<UIComponent> component) {
  component->initialize_vulkan_resources(vulkan_core_.get());
  components_.push_back(std::move(component));
}

} // namespace BTQuant
