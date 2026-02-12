#include "vulkan_dashboard_advanced.hpp"

#include <cstdint>
#include <iostream>
#include <ranges>

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "components/MarketMicrostructureRenderer.h"
#include "components/chart_panel.hpp"
#include "components/footprint_panel.hpp"
#include "components/interaction_manager.hpp"
#include "components/panel_manager.hpp"
#include "components/quant_workspace_component.hpp"
#include "components/tpo_panel.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"
#include "performance/debug_overlay.hpp"
#include "performance_monitor.hpp"
#include "ui/layout_manager.hpp"
#include "ui/tutorial.hpp"

// Shorter aliases for commonly used types
using BTQuant::RenderEngine::OrderbookData;
using BTQuant::RenderEngine::TradeData;

namespace BTQuant {

VulkanDashboard::VulkanDashboard(uint32_t width, uint32_t height,
                                 std::shared_ptr<HotSpineDataBridge> bridge,
                                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                                 const VulkanDashboardConfig& config)
    : width_(width),
      height_(height),
      hotspine_bridge_(bridge),
      market_data_processor_(processor),
      config_(config) {
  // Initialize frame budget from config if available
  frame_budget_ms_ = config.frame_budget_ms;
}

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
  micro_renderer_ = std::make_unique<RenderEngine::MarketMicrostructureRenderer>(
      vulkan_core_.get(), hotspine_bridge_, market_data_processor_);

  if (auto result = micro_renderer_->initialize(); !result) [[unlikely]] {
    std::println("[VulkanDashboard] CRITICAL: Micro Renderer failed to initialize: {}",
                 RenderEngine::to_string(result.error()));
  }

  workspace_ = std::make_unique<QuantWorkspaceComponent>(hotspine_bridge_, market_data_processor_,
                                                         micro_renderer_.get());

  // Show tutorial on first run (after components are initialized)
  BTQuant::UI::show_tutorial_if_first_run();

  // Register Hotkeys
  auto& im = InteractionManager::getInstance();

  // Keys 1-4 for layout switching
  auto* workspace = workspace_.get();  // Capture for lambda
  im.registerHotKey(
      ImGuiKey_1,
      [workspace]() {
        workspace->set_layout(LayoutPreset::DEFAULT);
        std::cout << "[Layout] Switched to Default Layout" << std::endl;
      },
      "Layout 1 (Default Layout)");

  im.registerHotKey(
      ImGuiKey_2,
      [workspace]() {
        workspace->set_layout(LayoutPreset::MODERN_TRADING);
        std::cout << "[Layout] Switched to Modern Trading Layout" << std::endl;
      },
      "Layout 2 (Modern Trading Layout)");

  im.registerHotKey(
      ImGuiKey_3,
      [workspace]() {
        workspace->set_layout(LayoutPreset::PRO_QUANT);
        std::cout << "[Layout] Switched to Pro Quant Layout" << std::endl;
      },
      "Layout 3 (Pro Quant Layout)");

  im.registerHotKey(
      ImGuiKey_4,
      [workspace]() {
        workspace->set_layout(LayoutPreset::SCALPER_DOM);
        std::cout << "[Layout] Switched to Scalper DOM Layout" << std::endl;
      },
      "Layout 4 (Scalper DOM Layout)");

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
  im.registerHotKey(
      ImGuiKey_F5,
      []() {
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
      },
      "Quick Save/Load Layout 1", false, false, false);  // F5/F5+Shift

  im.registerHotKey(
      ImGuiKey_F6,
      []() {
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
      },
      "Quick Save/Load Layout 2", false, false, false);  // F6/F6+Shift

  im.registerHotKey(
      ImGuiKey_F7,
      []() {
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
      },
      "Quick Save/Load Layout 3", false, false, false);  // F7/F7+Shift

  im.registerHotKey(
      ImGuiKey_F8,
      []() {
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
      },
      "Quick Save/Load Layout 4", false, false, false);  // F8/F8+Shift

  // F12 - Toggle Debug Overlay
  im.registerHotKey(
      ImGuiKey_F12,
      [this]() {
        g_debug_overlay.toggle_visibility();
        std::cout << "[Debug Overlay] Toggled visibility: "
                  << (g_debug_overlay.is_visible() ? "ON" : "OFF") << std::endl;
      },
      "Toggle Debug Overlay", false, false, false);  // F12

  // Delete key - Remove currently focused panel
  im.registerHotKey(
      ImGuiKey_Delete,
      [this]() {
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
              std::string expected_window_name = panel->get_config().title + "###panel_" +
                                                 std::to_string(reinterpret_cast<uintptr_t>(panel));

              if (focused_window_name && std::string(focused_window_name) == expected_window_name) {
                // Found the focused panel, remove it
                panel_manager->remove_panel(panel_id);
                std::cout << "[Hotkey] Removed focused panel: " << panel->get_config().title
                          << std::endl;
                break;
              }
            }
          }
        }
      },
      "Remove Focused Panel", false, false, false);  // Delete
}

void VulkanDashboard::render_frame() {
  // Start frame timing for budgeting
  auto frame_start_time = std::chrono::high_resolution_clock::now();

  if (!prepare_frame()) {
    return;  // Frame preparation failed (e.g., due to resize)
  }

  // Start ImGui frame
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // Update Interaction Manager
  InteractionManager::getInstance().update();

  render_main_menu_bar();

  // Render overlays and UI elements
  render_overlays();

  // Check frame budget before processing updates
  auto elapsed_before_updates = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::high_resolution_clock::now() - frame_start_time)
                                    .count() /
                                1000.0;  // Convert to ms

  if (elapsed_before_updates >= frame_budget_ms_) {
    // Skip updates if we've already exceeded the frame budget
    finalize_and_present_frame(frame_start_time);
    return;
  }

  // Process updates and UI
  process_updates_and_ui(frame_start_time);

  // Finalize and present frame
  finalize_and_present_frame(frame_start_time);
}

bool VulkanDashboard::prepare_frame() {
  if (window_resized_) {
    vulkan_core_->recreate_swapchain(width_, height_);
    window_resized_ = false;
  }

  uint32_t imageIndex;
  VkResult result = vulkan_core_->PrepareFrame(imageIndex);

  if (result == VK_ERROR_OUT_OF_DATE_KHR) {
    vulkan_core_->recreate_swapchain(width_, height_);
    return false;
  } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    // Silently ignore or log - swapchain might be rebuilding
    return false;
  }

  current_image_index_ = imageIndex;
  return true;
}

void VulkanDashboard::render_main_menu_bar() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("Tools")) {
      if (ImGui::MenuItem("Clear Dashboard History")) {
        if (market_data_processor_) {
          market_data_processor_->clearHistory();
        }
      }
      ImGui::EndMenu();
    }
    if (custom_menubar_callback_) {
      custom_menubar_callback_();
    }
    ImGui::EndMainMenuBar();
  }
}

void VulkanDashboard::render_overlays() {
  // Performance Overlay
  render_performance_overlay();

  // Debug Overlay
  g_debug_overlay.render();

  // Visual indicator for active layout
  render_layout_indicator();

  // Render tutorial if active (moved here to ensure it's within proper frame scope)
  BTQuant::UI::render_tutorial();
}

void VulkanDashboard::process_updates_and_ui(const std::chrono::high_resolution_clock::time_point& frame_start_time) {
  float dt = vulkan_core_->get_frame_time_ms() / 1000.0f;
  if (workspace_) {
    workspace_->update(dt);

    // Clear Screen for Tutorial: Only render GUI if tutorial is NOT active
    if (!BTQuant::UI::get_global_tutorial_manager().is_active()) {
      workspace_->render_gui();
    }

    // Check frame budget before syncing data
    auto elapsed_before_sync = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::high_resolution_clock::now() - frame_start_time)
                                   .count() /
                               1000.0;  // Convert to ms

    if (elapsed_before_sync < frame_budget_ms_) {
      // Sync UI state to Data feed (Fixing the Data Disconnect)
      // Extract UI state from workspace and propagate to data bridge
      if (hotspine_bridge_ && market_data_processor_) {
        // Sync selected symbol from UI to active symbol
        const std::string& selected_symbol = workspace_->getSelectedSymbol();
        if (!selected_symbol.empty() && selected_symbol != active_symbol_) {
          active_symbol_ = selected_symbol;
          // Update the dashboard's active symbol to sync with data feed
          set_active_symbol(selected_symbol);

          // Propagate the symbol change to all relevant components
          auto symbol_id_opt = SymbolRegistry::instance().get_symbol_id("Binance", selected_symbol);
          if (symbol_id_opt) {
            // Update the panel manager with the new active symbol
            if (workspace_->getPanelManager()) {
              workspace_->getPanelManager()->set_active_symbol(*symbol_id_opt, selected_symbol);
            }
          }
        }

        // Sync order state if any changes occurred in the UI
        // double order_qty = workspace_->getOrderQuantity();
        // double order_price = workspace_->getOrderPrice();
        // int order_side = workspace_->getSelectedOrderSide();  // 0 = Buy, 1 = Sell
        // int order_type = workspace_->getSelectedOrderType();  // 0 = Market, 1 = Limit

        // If there are pending orders from UI, submit them to the order manager
        auto order_manager = workspace_->getOrderManager();
        if (order_manager) {
          // Process any UI-initiated order submissions
          // This would typically happen through button clicks in the UI
          // For now, we'll just ensure the state is consistent

          // Sync hierarchical selector state to data feed
          const auto& selector_state = workspace_->getSelectorState();
          if (!selector_state.selected_symbol.empty()) {
            // Ensure the dashboard's active symbol matches the selector
            if (selector_state.selected_symbol != active_symbol_) {
              active_symbol_ = selector_state.selected_symbol;
              set_active_symbol(selector_state.selected_symbol);

              // Propagate to panel manager
              if (workspace_->getPanelManager()) {
                workspace_->getPanelManager()->set_active_symbol(selector_state.selected_symbol_id,
                                                                 selector_state.selected_symbol);
              }
            }
          }
        }

        // Sync mixed UI/Data state

        // ... (existing code) ...

        // Sync any UI-driven configuration changes back to the data bridge
        // REMOVED: hotspine_bridge_->sync(); to prevent race condition with background thread
      }
    }
  }

  // Check frame budget before microstructure rendering
  auto elapsed_before_rendering = std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::high_resolution_clock::now() - frame_start_time)
                                      .count() /
                                  1000.0;  // Convert to ms

  if (elapsed_before_rendering < frame_budget_ms_ && micro_renderer_) {
    // Handle high-performance microstructure rendering (Data Ingestion & Compute
    // Phase)
    pollDataToRenderer();
    auto result = micro_renderer_->prepare();
    if (!result) {
      // Log error if preparation failed
      // std::cerr << "[VulkanDashboard] Micro renderer prepare failed: " <<
      // RenderEngine::to_string(result.error()) << std::endl;
    }
    micro_renderer_->executeCompute(vulkan_core_->get_current_command_buffer());

    // Add pipeline barrier to ensure compute writes are visible to graphics
    VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT};
    vkCmdPipelineBarrier(vulkan_core_->get_current_command_buffer(),
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
  }
}

void VulkanDashboard::finalize_and_present_frame(const std::chrono::high_resolution_clock::time_point& frame_start_time) {
  // Finalize ImGui and Record Graphics commands
  ImGui::Render();

  // Custom hook for microstructure graphics inside the render pass.
  // We need to pass the renderer to RecordCommandBuffer or similar.
  // For now, let's assume we can call executeGraphics inside the render pass.
  // We'll modify RecordCommandBuffer to accept a callback or a renderer.

  vulkan_core_->RecordCommandBuffer(current_image_index_, ImGui::GetDrawData(), [this](VkCommandBuffer cmd) {
    if (micro_renderer_) {
      micro_renderer_->executeGraphics(cmd);
    }
  });

  vulkan_core_->PresentFrame(current_image_index_);
}

void VulkanDashboard::handle_events() { glfwPollEvents(); }

bool VulkanDashboard::should_close() const { return glfwWindowShouldClose(window_); }

void VulkanDashboard::shutdown() {
  static bool already_shutdown = false;
  if (already_shutdown) {
    return;
  }
  already_shutdown = true;

  // Shutdown ImGui first before destroying window or Vulkan resources
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();

  // Cleanup components that might hold Vulkan resources
  // CRITICAL: specific order to prevent use-after-free
  micro_renderer_.reset();
  workspace_.reset();

  // Cleanup Vulkan resources
  if (vulkan_core_) {
    vulkan_core_->wait_idle();
    vulkan_core_.reset();
  }

  // Cleanup GLFW
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

void VulkanDashboard::pollDataToRenderer() {
  using namespace BTQuant::RenderEngine;

  if (!micro_renderer_ || !hotspine_bridge_ || !market_data_processor_) {
    return;
  }

  // 1. Resolve Symbol ID from active_symbol_
  uint32_t symbol_id = 0;

  auto id_opt = SymbolRegistry::instance().get_symbol_id("Binance", active_symbol_);
  if (id_opt) {
    symbol_id = *id_opt;
  } else {
    auto all_symbols = SymbolRegistry::instance().get_all_symbols();
    for (const auto& info : all_symbols) {
      if (info.symbol == active_symbol_) {
        symbol_id = info.id;
        break;
      }
    }
    if (symbol_id == 0 && !all_symbols.empty()) {
      symbol_id = all_symbols[0].id;
      active_symbol_ = all_symbols[0].symbol;
    }
  }

  if (symbol_id == 0) {
    return;
  }

  // 2. Lock-free snapshot consume — zero locks, zero copies
  auto* snapshot_buf = market_data_processor_->getSnapshotBuffer(symbol_id);
  if (!snapshot_buf) return;  // Symbol not yet seen by worker threads
  snapshot_buf->consume();    // Swap middle→front if new data available
  const auto& snap = snapshot_buf->read();

  if (snap.last_update_time == 0) {
    return;
  }

  // 3. Update LOB Heatmap — data is pre-flattened by worker thread
  uint32_t bidsCount = static_cast<uint32_t>(snap.bids.size());
  uint32_t asksCount = static_cast<uint32_t>(snap.asks.size());
  uint32_t totalLevels = bidsCount + asksCount;

  if (totalLevels > 0) {
    size_t bufferSize = RenderEngine::HotspineOrderBookSnapshot::calculateBufferSize(totalLevels);
    std::vector<uint8_t> buffer(bufferSize);
    auto* snapshot = reinterpret_cast<RenderEngine::HotspineOrderBookSnapshot*>(buffer.data());

    snapshot->currentTimeIndex = static_cast<uint32_t>(vulkan_core_->get_current_frame_index());
    snapshot->priceLevelsCount = totalLevels;
    snapshot->basePrice = snap.ob_min_price;
    float range = snap.ob_max_price - snap.ob_min_price;
    snapshot->priceRange = range > 1e-6f ? range : 1.0f;

    uint32_t idx = 0;
    for (const auto& level : snap.bids) {
      snapshot->levels[idx++] = {level.price, 0, static_cast<uint32_t>(level.size), 0};
    }
    for (const auto& level : snap.asks) {
      snapshot->levels[idx++] = {level.price, static_cast<uint32_t>(level.size), 0, 0};
    }

    micro_renderer_->updateLOBData(*snapshot);
  }

  // 4. Update Trade Data — pre-converted by worker thread
  if (!snap.trade_ticks.empty()) {
    micro_renderer_->updateTradeData(snap.trade_ticks);
  }

  // 5. Update Footprint Clusters — pre-aggregated by worker thread
  if (!snap.footprint_clusters.empty()) {
    micro_renderer_->updateFootprintClusters(snap.footprint_clusters);
  }
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

  // Update renderer stats if available
  if (micro_renderer_) {
    auto stats = micro_renderer_->getStats();
    g_debug_overlay.set_renderer_stats(stats.framesRendered, stats.lobUpdates, stats.tradeUpdates,
                                       stats.footprintCellsRendered);
  }

  // Get active indicators count from the workspace components
  size_t active_indicators_count = 0;
  if (auto* workspace = get_workspace_component()) {
    if (auto* panel_manager = workspace->getPanelManager()) {
      // For each panel, if it's a chart panel, count its active indicators
      auto all_panel_ids = panel_manager->get_all_panel_ids();
      for (auto panel_id : all_panel_ids) {
        auto* panel = panel_manager->get_panel_by_id(panel_id);
        if (panel && panel->get_config().type == PanelType::CHART) {
          // Cast to ChartPanel to access its active indicators
          auto* chart_panel = dynamic_cast<ChartPanel*>(panel);
          if (chart_panel) {
            active_indicators_count += chart_panel->get_active_indicators_count();
          }
        }
      }
    }
  }
  g_debug_overlay.set_active_indicators_count(active_indicators_count);

  // For alerts count, we'll set it to 0 for now since accessing it requires
  // restructuring how GlobalAlertManager is instantiated and accessed
  // This addresses the TODO but with a temporary solution
  g_debug_overlay.set_active_alerts_count(0);
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
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs |
                   ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

  // Draw the layout indicator with a semi-transparent background
  ImGui::PushStyleColor(ImGuiCol_WindowBg,
                        ImVec4(0.1f, 0.1f, 0.1f, 0.7f));  // Dark semi-transparent background

  // Change text color based on whether a quick save slot is active
  if (active_slot > 0) {
    ImGui::PushStyleColor(ImGuiCol_Text,
                          ImVec4(0.5f, 0.9f, 0.5f, 1.0f));  // Greenish color for active quick save
  } else {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));  // Light text
  }

  ImGui::Text("Layout: %s%s", active_layout.c_str(), slot_info.c_str());

  ImGui::PopStyleColor(2);
  ImGui::End();
}

void VulkanDashboard::set_always_on_top(bool enabled) {
  if (window_ && glfwGetWindowAttrib(window_, GLFW_VISIBLE)) {
    glfwSetWindowAttrib(window_, GLFW_FLOATING, enabled ? GLFW_TRUE : GLFW_FALSE);
    always_on_top_ = enabled;
  }
}

}  // namespace BTQuant
