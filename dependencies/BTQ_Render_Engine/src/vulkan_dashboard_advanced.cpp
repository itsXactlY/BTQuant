#include "../include/vulkan_dashboard_advanced.hpp"
#include "../include/market_data_processor.hpp"
// #include "imgui_impl_vulkan.h" // Removed as wrapped by VulkanCore

#include "imgui_impl_vulkan.h"
#include <X11/Xutil.h>
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
  if (ImGui::GetCurrentContext() == nullptr) {
    std::cout << "[VulkanDashboard] Creating ImGui Context..." << std::endl;
    ImGui::CreateContext();
  }

  std::cout << "[VulkanDashboard] Initializing X11..." << std::endl;
  init_x11();
  std::cout << "[VulkanDashboard] Initializing Vulkan..." << std::endl;
  init_vulkan();

  // Initialize ImGui specifically here if not done in VulkanCore
  // We assume VulkanCore handles basic ImGui context, but we need to configure
  // Docking
  std::cout << "[VulkanDashboard] Configuring ImGui..." << std::endl;

  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  std::cout << "[VulkanDashboard] Initializing Components..." << std::endl;
  init_components();
  std::cout << "[VulkanDashboard] Initialization Complete." << std::endl;
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
  attrs.background_pixel = BlackPixel(display_, screen);
  attrs.border_pixel = BlackPixel(display_, screen);

  window_ = XCreateWindow(display_, root, 0, 0, width_, height_, 0,
                          CopyFromParent, InputOutput, CopyFromParent,
                          CWEventMask | CWBackPixel | CWBorderPixel, &attrs);

  XSetStandardProperties(display_, window_, "BTQuant Unified Dashboard",
                         "BTQuant", None, nullptr, 0, nullptr);

  // Handle close button
  wm_delete_window_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(display_, window_, &wm_delete_window_, 1);

  XMapWindow(display_, window_);

  // Wait for MapNotify event to ensure window is visible
  XEvent ev;
  while (true) {
    XNextEvent(display_, &ev);
    if (ev.type == MapNotify)
      break;
  }

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
  if (!is_running_) {
    std::cout << "[VulkanDashboard] is_running_ is false, exiting run_frame"
              << std::endl;
    return false;
  }

  render_frame();
  return true;
}

void VulkanDashboard::handle_x11_events() {
  XEvent event;
  ImGuiIO &io = ImGui::GetIO();
  while (XPending(display_) > 0) {
    XNextEvent(display_, &event);
    if (event.type == ClientMessage) {
      if ((Atom)event.xclient.data.l[0] == wm_delete_window_) {
        std::cout << "[VulkanDashboard] Received WM_DELETE_WINDOW" << std::endl;
        is_running_ = false;
      }
    } else if (event.type == DestroyNotify) {
      is_running_ = false;
    } else if (event.type == ConfigureNotify) {
      if ((uint32_t)event.xconfigure.width != width_ ||
          (uint32_t)event.xconfigure.height != height_) {
        width_ = event.xconfigure.width;
        height_ = event.xconfigure.height;
        io.DisplaySize = ImVec2((float)width_, (float)height_);
        vulkan_core_->recreate_swapchain(width_, height_);
      }
    } else if (event.type == MotionNotify) {
      io.AddMousePosEvent((float)event.xmotion.x, (float)event.xmotion.y);
    } else if (event.type == ButtonPress) {
      if (event.xbutton.button == Button1)
        io.AddMouseButtonEvent(0, true);
      if (event.xbutton.button == Button2)
        io.AddMouseButtonEvent(2, true);
      if (event.xbutton.button == Button3)
        io.AddMouseButtonEvent(1, true);
      if (event.xbutton.button == Button4)
        io.AddMouseWheelEvent(0.0f, 1.0f);
      if (event.xbutton.button == Button5)
        io.AddMouseWheelEvent(0.0f, -1.0f);
    } else if (event.type == ButtonRelease) {
      if (event.xbutton.button == Button1)
        io.AddMouseButtonEvent(0, false);
      if (event.xbutton.button == Button2)
        io.AddMouseButtonEvent(2, false);
      if (event.xbutton.button == Button3)
        io.AddMouseButtonEvent(1, false);
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
    static int fail_count = 0;
    if (fail_count++ % 60 == 0) {
      std::cout << "[VulkanDashboard] begin_frame failed (OUT_OF_DATE), "
                   "recreating swapchain..."
                << std::endl;
    }
    vulkan_core_->recreate_swapchain(width_, height_);
    return;
  }

  static int frame_count = 0;
  if (frame_count++ % 60 == 0) {
    std::cout << "[VulkanDashboard] Rendering frame " << frame_count << "..."
              << std::endl;
  }

  // 2. Begin Main Render Pass (Clear screen)
  vulkan_core_->begin_main_render_pass();

  // 3. ImGui Docking Setup
  VkCommandBuffer cmd = vulkan_core_->get_current_command_buffer();

  // ImGui Required Boilerplate
  ImGui_ImplVulkan_NewFrame();
  ImGui::NewFrame();

  // Create the DockSpace
  ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
  ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                   ImGuiDockNodeFlags_PassthruCentralNode);

  // 4. Render All Components
  fprintf(stderr, "[DEBUG] Frame %d: Render All Components (count=%zu)\n",
          frame_count, components_.size());
  ImGui::ShowDemoWindow();
  for (auto &comp : components_) {
    comp->render(cmd);
    comp->render_gui();
  }

  // 5. Finalize ImGui and End Frame
  ImGui::Render();
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
