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
  micro_renderer_ = std::make_unique<RenderEngine::MarketMicrostructureRenderer>(
      vulkan_core_.get(), hotspine_bridge_, market_data_processor_);

  if (auto result = micro_renderer_->initialize(); !result) [[unlikely]] {
    std::println("[VulkanDashboard] CRITICAL: Micro Renderer failed to initialize: {}",
                 RenderEngine::to_string(result.error()));
  }

  workspace_ = std::make_unique<QuantWorkspaceComponent>(hotspine_bridge_, market_data_processor_,
                                                        micro_renderer_.get());

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
    if (custom_menubar_callback_) {
      custom_menubar_callback_();
    }
    ImGui::EndMainMenuBar();
  }

  // Performance Overlay
  render_performance_overlay();

  // Debug Overlay
  g_debug_overlay.render();

  // Visual indicator for active layout
  render_layout_indicator();

  // Process updates and UI
  float dt = vulkan_core_->get_frame_time_ms() / 1000.0f;
  if (workspace_) {
    workspace_->update(dt);
    workspace_->render_gui();
  }

  // Handle high-performance microstructure rendering (Data Ingestion & Compute
  // Phase)
  if (micro_renderer_) {
    pollDataToRenderer();
    auto result = micro_renderer_->prepare();
    if (!result) {
        // Log error if preparation failed
        std::cout << "[VulkanDashboard] Micro renderer prepare failed: " <<
                     RenderEngine::to_string(result.error()) << std::endl;
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

  // Finalize ImGui and Record Graphics commands
  ImGui::Render();

  // Custom hook for microstructure graphics inside the render pass.
  // We need to pass the renderer to RecordCommandBuffer or similar.
  // For now, let's assume we can call executeGraphics inside the render pass.
  // We'll modify RecordCommandBuffer to accept a callback or a renderer.

  vulkan_core_->RecordCommandBuffer(imageIndex, ImGui::GetDrawData(), [this](VkCommandBuffer cmd) {
    if (micro_renderer_) {
      micro_renderer_->executeGraphics(cmd);
    }
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

void VulkanDashboard::pollDataToRenderer() {
  if (!micro_renderer_ || !hotspine_bridge_ || !market_data_processor_) {
    return;
  }

  // 1. Resolve Symbol ID from active_symbol_
  uint32_t symbol_id = 0;
  auto id_opt = SymbolRegistry::instance().get_symbol_id("Binance", active_symbol_);
  if (!id_opt) {
    auto all_symbols = SymbolRegistry::instance().get_all_symbols();
    for (const auto& info : all_symbols) {
      if (info.symbol == active_symbol_) {
        symbol_id = info.id;
        break;
      }
    }
  } else {
    symbol_id = *id_opt;
  }

  if (symbol_id == 0) {
    return;
  }

  // 2. Fetch Latest Analytics
  auto analytics = market_data_processor_->getSymbolAnalytics(symbol_id);
  if (analytics.last_update_time == 0) {
    return;
  }

  // 3. Update LOB Heatmap Data
  uint32_t bidsCount = static_cast<uint32_t>(analytics.consolidated_bids.size());
  uint32_t asksCount = static_cast<uint32_t>(analytics.consolidated_asks.size());
  uint32_t totalLevels = bidsCount + asksCount;

  if (totalLevels > 0) {
    size_t bufferSize = RenderEngine::HotspineOrderBookSnapshot::calculateBufferSize(totalLevels);
    std::vector<uint8_t> buffer(bufferSize);
    auto* snapshot = reinterpret_cast<RenderEngine::HotspineOrderBookSnapshot*>(buffer.data());

    snapshot->currentTimeIndex = static_cast<uint32_t>(vulkan_core_->get_current_frame_index());
    snapshot->priceLevelsCount = totalLevels;

    // Calculate dynamic price range for the snapshot
    float minPrice = 1e9f, maxPrice = -1e9f;
    if (!analytics.consolidated_bids.empty()) {
      minPrice =
          std::min(minPrice, static_cast<float>(analytics.consolidated_bids.rbegin()->first));
      maxPrice = std::max(maxPrice, static_cast<float>(analytics.consolidated_bids.begin()->first));
    }
    if (!analytics.consolidated_asks.empty()) {
      minPrice = std::min(minPrice, static_cast<float>(analytics.consolidated_asks.begin()->first));
      maxPrice =
          std::max(maxPrice, static_cast<float>(analytics.consolidated_asks.rbegin()->first));
    }

    snapshot->basePrice = minPrice;
    snapshot->priceRange = (maxPrice - minPrice) > 1e-6f ? (maxPrice - minPrice) : 1.0f;

    uint32_t idx = 0;
    for (auto const& [price, size] : analytics.consolidated_bids) {
      snapshot->levels[idx++] = {static_cast<float>(price), 0, static_cast<uint32_t>(size), 0};
    }
    for (auto const& [price, size] : analytics.consolidated_asks) {
      snapshot->levels[idx++] = {static_cast<float>(price), static_cast<uint32_t>(size), 0, 0};
    }

    micro_renderer_->updateLOBData(*snapshot);
  }

  // 4. Update Trade Data
  if (!analytics.recent_trades.empty()) {
    std::vector<RenderEngine::HotspineTradeTick> ticks;
    size_t count = std::min(static_cast<size_t>(1000), analytics.recent_trades.size());
    ticks.reserve(count);

    for (size_t i = analytics.recent_trades.size() - count; i < analytics.recent_trades.size();
         ++i) {
      const auto& t = analytics.recent_trades[i];
      ticks.emplace_back(t.timestamp, static_cast<float>(t.price), static_cast<float>(t.size),
                         t.symbol_id, t.is_buy);
    }
    micro_renderer_->updateTradeData(ticks);
  }

  // 5. Aggregate Footprint Clusters (Exocharts Style)
  // We use current 1s candle to generate clusters for the footprint
  auto candle_opt =
      market_data_processor_->getCurrentCandle(symbol_id, RenderEngine::TimeFrame::TF_1SEC);
  if (candle_opt) {
    // 5. Optimized Aggregate Footprint Clusters (C++26 Zero-Copy)
    const uint64_t now_us = analytics.last_update_time;
    const uint64_t timeframe_us = 1'000'000;  // 1 second bins
    const uint64_t window_us = 30'000'000;    // 30 seconds window
    constexpr float tickSize = 0.5f;

    std::vector<RenderEngine::CandleCluster> clusters;

    // Efficiency: Use an ordered map for aggregation (stable for rendering)
    struct ClusterKey {
      uint64_t time;
      int32_t price_bin;
      auto operator<=>(const ClusterKey&) const = default;
    };

    struct ClusterValue {
      uint32_t bidVol = 0;
      uint32_t askVol = 0;
      uint32_t count = 0;
      uint32_t buyCount = 0;        // Number of buy trades
      uint32_t sellCount = 0;       // Number of sell trades
      float maxTradeVol = 0.0f;     // Maximum single trade volume
      float totalTradeSize = 0.0f;  // For calculating VWAP
    };
    std::map<ClusterKey, ClusterValue> aggregator;

    // Process trades in reverse for window efficiency
    for (const auto& t : std::views::reverse(analytics.recent_trades)) {
      if (t.timestamp <= now_us - window_us) break;

      const uint64_t timeBin = (t.timestamp / timeframe_us) * timeframe_us;
      const int32_t priceBin = static_cast<int32_t>(std::round(t.price / tickSize));

      auto& val = aggregator[ClusterKey{timeBin, priceBin}];
      if (t.is_buy) [[likely]] {
        val.bidVol += static_cast<uint32_t>(t.size);  // Buy trade contributes to bid volume
        val.buyCount++;                               // Increment buy trade count
        if (t.size > val.maxTradeVol)
          val.maxTradeVol = static_cast<float>(t.size);  // Track max trade volume
        val.totalTradeSize += static_cast<float>(t.size);
      } else {
        val.askVol += static_cast<uint32_t>(t.size);  // Sell trade contributes to ask volume
        val.sellCount++;                              // Increment sell trade count
        if (t.size > val.maxTradeVol)
          val.maxTradeVol = static_cast<float>(t.size);  // Track max trade volume
        val.totalTradeSize += static_cast<float>(t.size);
      }
      val.count++;
    }

    clusters.reserve(aggregator.size());
    for (auto const& [key, val] : aggregator) {
      // Use relative seconds from window start for better float precision in
      // coordinates
      const float rel_time_sec = static_cast<float>(key.time - (now_us - window_us)) / 1'000'000.0f;

      clusters.emplace_back(rel_time_sec, static_cast<float>(key.price_bin) * tickSize,
                            static_cast<float>(timeframe_us) / 1'000'000.0f * 0.9f, tickSize * 0.9f,
                            val.bidVol, val.askVol, val.count, 0.0f, true, val.buyCount,
                            val.sellCount, val.maxTradeVol, (key.time - timeframe_us) * 1000,
                            key.time * 1000);  // Convert to nanoseconds
    }

    if (!clusters.empty()) {
      micro_renderer_->updateFootprintClusters(clusters);
    }
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
    g_debug_overlay.set_renderer_stats(stats.framesRendered, stats.lobUpdates,
                                      stats.tradeUpdates, stats.footprintCellsRendered);
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
