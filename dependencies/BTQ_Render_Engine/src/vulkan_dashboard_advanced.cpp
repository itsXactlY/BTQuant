#include "../include/vulkan_dashboard_advanced.hpp"
#include "../../include/components/footprint_panel.hpp"
#include "../../include/components/tpo_panel.hpp"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "components/interaction_manager.hpp"
#include "components/quant_workspace_component.hpp"
#include "components/realtime_dashboard_component.hpp"
#include "imgui.h"
#include "implot.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

namespace BTQuant {

VulkanDashboard::VulkanDashboard(
    uint32_t width, uint32_t height, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    const VulkanDashboardConfig &config)
    : width_(width), height_(height), hotspine_bridge_(bridge),
      market_data_processor_(processor), config_(config) {}

VulkanDashboard::~VulkanDashboard() {
  is_running_ = false;
  if (m_worker_thread.joinable()) {
    m_worker_thread.join();
  }
  shutdown();
}

std::expected<void, std::string> VulkanDashboard::initialize() {
  printf("[VulkanDashboard] Initializing Advanced Terminal Renderer...\n");

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
  printf("[VulkanDashboard] Vulkan initialized.\n");

  m_timeline_semaphore =
      std::make_unique<TimelineSemaphore>(m_vulkanCore->get_device());

  // Initialize Glfw ImGui Backend
  printf("[VulkanDashboard] Initializing ImGui GLFW Backend...\n");
  ImGui_ImplGlfw_InitForVulkan(window_, true);
  printf("[VulkanDashboard] ImGui GLFW Backend initialized.\n");

  init_components();

  // 3. Start Data Microstructure Worker Thread (Parallel Ingestion)
  m_worker_thread = std::thread(&VulkanDashboard::worker_loop, this);

  return {};
}

void VulkanDashboard::worker_loop() {
  while (is_running_.load()) {
    pollDataToRenderer();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void VulkanDashboard::init_components() {
  printf("[VulkanDashboard] Initializing Components...\n");
  m_micro_renderer =
      std::make_unique<RenderEngine::MarketMicrostructureRenderer>(
          m_vulkanCore.get(), hotspine_bridge_, market_data_processor_);

  if (auto result = m_micro_renderer->initialize(); !result) [[unlikely]] {
    printf(
        "[VulkanDashboard] CRITICAL: Micro Renderer failed to initialize: %s\n",
        std::string(RenderEngine::to_string(result.error())).c_str());
  }

  m_workspace = std::make_unique<QuantWorkspaceComponent>(
      hotspine_bridge_, market_data_processor_, m_micro_renderer.get());

  m_modern_dashboard = std::make_unique<RealtimeDashboardComponent>(
      hotspine_bridge_, market_data_processor_);
  m_modern_dashboard->initialize_vulkan_resources(m_vulkanCore.get());

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
          printf("[Layout] Switched to Desktop 3x5\n");
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
          printf("[Layout] Switched to Chart Focus\n");
        }
      },
      "Layout 2 (Chart Focus)", true);

  // Space to Toggle Theme
  im.registerHotKey(
      ImGuiKey_Space,
      []() {
        ThemeManager::getInstance().toggleTheme();
        printf("Hotkey: Theme Toggled\n");
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
    if (ImGui::BeginMenu("Dashboard")) {
      if (ImGui::MenuItem("Modern Dashboard", nullptr,
                          &use_modern_dashboard_)) {
        // Toggle flag
      }
      if (use_modern_dashboard_ && ImGui::MenuItem("Reset Layout")) {
        m_modern_dashboard->reset_layout();
      }
      ImGui::EndMenu();
    }
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
  // Process updates and UI
  float dt = m_vulkanCore->get_frame_time_ms() / 1000.0f;

  if (use_modern_dashboard_ && m_modern_dashboard) {
    m_modern_dashboard->update(dt);
    m_modern_dashboard->render_gui();
  } else if (m_workspace) {
    m_workspace->update(dt);
    m_workspace->render_gui();
  }

  // --- DEBUG OVERLAY (Moved to end for Z-order visibility) ---
  if (true) {
    ImGui::SetNextWindowPos(ImVec2(10, 50), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Debug Info", nullptr,
                     ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
      ImGui::Text("Logic Frame: %lu", m_logic_frame_count);
      if (m_micro_renderer) {
        auto stats = m_micro_renderer->getStats();
        ImGui::Text("Renderer Statistics:");
        ImGui::Text("  LOB Updates: %lu", stats.lobUpdates);
        ImGui::Text("  Trade Updates: %lu", stats.tradeUpdates);

        void *texID = m_micro_renderer->getHeatmapTextureID();
        ImGui::Text("  Texture ID: %p", texID);

        auto bounds = m_micro_renderer->getLOBPriceBounds();
        ImGui::Text("  LOB Bounds: %.2f - %.2f", bounds.first,
                    bounds.first + bounds.second);
      }
    }
    ImGui::End();
  }

  // Handle high-performance microstructure rendering (Data Ingestion & Compute
  // Phase)
  // Handle microstructure rendering execution
  if (m_micro_renderer) {
    // Note: pollDataToRenderer is now handled by m_worker_thread
    m_micro_renderer->prepare();
    m_micro_renderer->executeCompute(
        m_vulkanCore->get_current_command_buffer());

    // Add pipeline barrier to ensure compute writes are visible to graphics
    VkMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                            .dstAccessMask =
                                VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT};
    vkCmdPipelineBarrier(m_vulkanCore->get_current_command_buffer(),
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
  }

  // Finalize ImGui and Record Graphics commands
  ImGui::Render();

  // Custom hook for microstructure graphics inside the render pass.
  // We need to pass the renderer to RecordCommandBuffer or similar.
  // For now, let's assume we can call executeGraphics inside the render pass.
  // We'll modify RecordCommandBuffer to accept a callback or a renderer.

  m_vulkanCore->RecordCommandBuffer(imageIndex, ImGui::GetDrawData(),
                                    [this](VkCommandBuffer cmd) {
                                      if (m_micro_renderer) {
                                        m_micro_renderer->executeGraphics(cmd);
                                      }
                                    });
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

  printf("[VulkanDashboard] Shutting down...\n");

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
    fprintf(stderr, "[VulkanDashboard] Failed to initialize GLFW\n");
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

void VulkanDashboard::pollDataToRenderer() {
  if (!m_micro_renderer || !hotspine_bridge_ || !market_data_processor_) {
    return;
  }

  // 1. Resolve Symbol ID from active_symbol_ (Thread-Safe)
  std::string active_sym;
  {
    std::lock_guard lock(m_configMutex);
    active_sym = active_symbol_;
  }

  uint32_t symbol_id = 0;
  auto id_opt = SymbolRegistry::instance().get_symbol_id("Binance", active_sym);
  if (!id_opt) {
    auto all_symbols = SymbolRegistry::instance().get_all_symbols();
    for (const auto &info : all_symbols) {
      if (info.symbol == active_sym) {
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
  uint32_t bidsCount =
      static_cast<uint32_t>(analytics.consolidated_bids.size());
  uint32_t asksCount =
      static_cast<uint32_t>(analytics.consolidated_asks.size());
  uint32_t totalLevels = bidsCount + asksCount;

  if (totalLevels > 0) {
    size_t bufferSize =
        RenderEngine::HotspineOrderBookSnapshot::calculateBufferSize(
            totalLevels);
    std::vector<uint8_t> buffer(bufferSize);
    auto *snapshot =
        reinterpret_cast<RenderEngine::HotspineOrderBookSnapshot *>(
            buffer.data());

    snapshot->currentTimeIndex = static_cast<uint32_t>(m_logic_frame_count++);
    snapshot->priceLevelsCount = totalLevels;

    // Use mid-price centered window for professional Heatmap scaling (Quantower
    // style)
    float midPrice = 0.0f;
    if (!analytics.consolidated_bids.empty() &&
        !analytics.consolidated_asks.empty()) {
      midPrice =
          (static_cast<float>(analytics.consolidated_bids.begin()->first) +
           static_cast<float>(analytics.consolidated_asks.begin()->first)) /
          2.0f;
    } else if (!analytics.consolidated_bids.empty()) {
      midPrice = static_cast<float>(analytics.consolidated_bids.begin()->first);
    } else if (!analytics.consolidated_asks.empty()) {
      midPrice = static_cast<float>(analytics.consolidated_asks.begin()->first);
    }

    // Use a tighter window for high-resolution depth (Quantower style)
    constexpr float tickSize = 0.5f;
    constexpr float tickWindow =
        50.0f; // ±50 ticks = ±25 USD for BTC (Ultra-Durable & Vibrant)

    snapshot->basePrice = midPrice - (tickWindow * tickSize);
    snapshot->priceRange = (tickWindow * 2.0f) * tickSize;

    float maxVol = 1.0f;
    uint32_t idx = 0;
    for (auto const &[price, size] : analytics.consolidated_bids) {
      if (idx >= totalLevels)
        break;
      float vol = static_cast<float>(size);
      maxVol = std::max(maxVol, vol);
      snapshot->levels[idx++] = {static_cast<float>(price), 0,
                                 static_cast<uint32_t>(vol * 100.0f), 0};
    }
    for (auto const &[price, size] : analytics.consolidated_asks) {
      if (idx >= totalLevels)
        break;
      float vol = static_cast<float>(size);
      maxVol = std::max(maxVol, vol);
      snapshot->levels[idx++] = {static_cast<float>(price),
                                 static_cast<uint32_t>(vol * 100.0f), 0, 0};
    }

    // Dynamic Normalization for Heatmap (Quantower style)
    m_micro_renderer->updateHeatmapParams(maxVol * 100.0f * 0.7f);

    m_micro_renderer->updateLOBData(*snapshot);
  }

  // 4. Update Trade Data
  if (!analytics.recent_trades.empty()) {
    std::vector<RenderEngine::HotspineTradeTick> ticks;
    size_t count =
        std::min(static_cast<size_t>(1000), analytics.recent_trades.size());
    ticks.reserve(count);

    for (size_t i = analytics.recent_trades.size() - count;
         i < analytics.recent_trades.size(); ++i) {
      const auto &t = analytics.recent_trades[i];
      ticks.emplace_back(t.timestamp, static_cast<float>(t.price),
                         static_cast<float>(t.size), t.symbol_id, t.is_buy);
    }
    m_micro_renderer->updateTradeData(ticks);
  }

  // 5. Aggregate Footprint Clusters (Exocharts Style)
  // We use current 1s candle to generate clusters for the footprint
  auto candle_opt = market_data_processor_->getCurrentCandle(
      symbol_id, RenderEngine::TimeFrame::TF_1SEC);
  if (candle_opt) {
    // 5. Optimized Aggregate Footprint Clusters (C++26 Zero-Copy)
    const uint64_t now_us = analytics.last_update_time;
    const uint64_t timeframe_us = 1'000'000; // 1 second bins
    const uint64_t window_us = 30'000'000;   // 30 seconds window
    constexpr float tickSize = 0.5f;

    std::vector<RenderEngine::CandleCluster> clusters;

    // Efficiency: Use an ordered map for aggregation (stable for rendering)
    struct ClusterKey {
      uint64_t time;
      int32_t price_bin;
      auto operator<=>(const ClusterKey &) const = default;
    };

    struct ClusterValue {
      uint32_t bidVol = 0;
      uint32_t askVol = 0;
      uint32_t count = 0;
    };
    std::map<ClusterKey, ClusterValue> aggregator;

    // Process trades in reverse for window efficiency
    for (const auto &t : std::views::reverse(analytics.recent_trades)) {
      if (t.timestamp <= now_us - window_us)
        break;

      const uint64_t timeBin = (t.timestamp / timeframe_us) * timeframe_us;
      const int32_t priceBin =
          static_cast<int32_t>(std::round(t.price / tickSize));

      auto &val = aggregator[ClusterKey{timeBin, priceBin}];
      if (t.is_buy) [[likely]]
        val.askVol += static_cast<uint32_t>(t.size);
      else
        val.bidVol += static_cast<uint32_t>(t.size);
      val.count++;
    }

    clusters.reserve(aggregator.size());
    for (auto const &[key, val] : aggregator) {
      // Use relative seconds from window start for better float precision in
      // coordinates
      const float rel_time_sec =
          static_cast<float>(key.time - (now_us - window_us)) / 1'000'000.0f;

      clusters.emplace_back(
          rel_time_sec, static_cast<float>(key.price_bin) * tickSize,
          static_cast<float>(timeframe_us) / 1'000'000.0f * 0.9f,
          tickSize * 0.9f, val.bidVol, val.askVol, val.count, 0.0f, true);
    }

    if (!clusters.empty()) {
      m_micro_renderer->updateFootprintClusters(clusters);
    }
  }
}

} // namespace BTQuant
