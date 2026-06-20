#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <sys/stat.h>

#ifdef BTQUANT_USE_GLFW
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#else
#error "Only GLFW is fully supported in this backend right now"
#endif

#include "core/vulkan_context.hpp"
#include "ui/ui_context.hpp"
#include "ui/window_manager.hpp"
#include "util/settings.hpp"

#include "data/market_data_processor.hpp"
#include "data/mock_producer.hpp"
#include "renderer/heatmap_compute.hpp"
#include "widgets/heatmap_widget.hpp"
#include "widgets/log_panel.hpp"

using namespace btquant;
using btquant::renderer::HeatmapConfig;

class BTQuantApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    initUI();
    initData();
    initCompute();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window = nullptr;
  vulkan::VulkanContext vkContext;
  ui::UIContext uiContext;
  ui::WindowManager windowManager;

  // Real-time data pipeline (subscribes to /dev/shm/btquant_hotspine).
  MarketDataProcessor marketData;

  // In-process mock data writer. Started by initData() when no external
  // producer (scripts/mock_producer.py) is detected, OR when BTQUANT_DEMO=1.
  data::MockProducer mockProducer;

  // GPU compute pipeline for the heatmap texture.
  renderer::HeatmapCompute heatmapCompute;
  ui::HeatmapWidget heatmapWidget;

  uint32_t width = 1280;
  uint32_t height = 720;
  uint64_t frameCounter = 0;

  void initWindow() {
    if (!glfwInit()) {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // Headless / offscreen window — works around a NVIDIA+X11+i3 hang in glfwCreateWindow
    // when the window is set to visible. Can be flipped to GLFW_TRUE for a desktop session.
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    window =
        glfwCreateWindow(width, height, "BTQuant Terminal", nullptr, nullptr);
    if (!window) {
      throw std::runtime_error("Failed to create GLFW window");
    }

    // Vsync: fpsLimit==0 → no vsync (uncapped render loop), otherwise → vsync on
    // (set swap interval to 1; the renderer itself does not throttle above vsync).
    glfwSwapInterval(1);
  }

  void initVulkan() {
    if (auto err = vkContext.createInstance()) {
      throw std::runtime_error("Failed to initialize Vulkan: " + *err);
    }

    VkSurfaceKHR surface;
    if (glfwCreateWindowSurface(vkContext.instance(), window, nullptr,
                                &surface) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create window surface");
    }
    vkContext.setSurface(surface);

    if (auto err = vkContext.initialize()) {
      throw std::runtime_error("Failed to complete Vulkan init: " + *err);
    }
  }

  void initUI() {
    // Ensure ~/.config/btquant_vulkan exists BEFORE resolving ini/state paths,
    // otherwise ImGui's auto-save will silently fall back to writing "0" in CWD
    // when the parent dir can't be created.
    auto settingsPath = util::Settings::defaultPath();
    std::filesystem::create_directories(settingsPath.parent_path());

    // Resolve ImGui ini path alongside Settings (BTQUANT_INI env override).
    auto iniPath = (settingsPath.parent_path() / "imgui.ini").string();
    if (const char* env = std::getenv("BTQUANT_INI")) iniPath = env;

    // Load persisted settings BEFORE initializing UIContext so the theme is
    // applied on the very first frame (no flash-of-dark-theme).
    auto settings = util::Settings::load(settingsPath);
    std::fprintf(stderr, "[BTQuant] loaded settings from %s\n",
                 settingsPath.c_str());
    BTQ_LOG_INFO("loaded settings from %s", settingsPath.c_str());

    if (!uiContext.initialize(window, vkContext.instance(),
                              vkContext.physicalDevice(), vkContext.device(),
                              vkContext.queueFamilies().graphicsFamily.value(),
                              vkContext.graphicsQueue(),
                              vkContext.renderPass(),
                              iniPath.c_str(),
                              settings.theme == 0 ? ui::UIContext::Theme::Dark
                                                  : ui::UIContext::Theme::Light)) {
      throw std::runtime_error("Failed to initialize UI Context");
    }

    // settings is already loaded — push visibility flags into WindowManager.
    windowManager.initialize();
    windowManager.showOrderBook       = settings.showOrderBook;
    windowManager.showOrderBookDepth  = settings.showOrderBookDepth;
    windowManager.showFootprint       = settings.showFootprint;
    windowManager.showVPVR            = settings.showVPVR;
    windowManager.showMultiVWAP       = settings.showMultiVWAP;
    windowManager.showRiskPanel       = settings.showRiskPanel;
    windowManager.showDOM             = settings.showDOM;
    windowManager.showTrades          = settings.showTrades;
    windowManager.showTPO             = settings.showTPO;
    windowManager.showSettings        = settings.showSettings;
    windowManager.showStatsOverlay    = settings.showStatsOverlay;
    windowManager.theme               = settings.theme;
    windowManager.fpsLimit            = settings.fpsLimit;
    windowManager.heatmapDensity      = settings.heatmapDensity;
    glfwSwapInterval(windowManager.fpsLimit > 0 ? 1 : 0);
  }

  void initData() {
    // Decide whether to start the in-process mock producer. We start it when:
    //   1. BTQUANT_DEMO=1 is set explicitly, OR
    //   2. /dev/shm/btquant_hotspine doesn't exist (no external producer running).
    // The in-process producer uses the same binary format as scripts/mock_producer.py
    // so MarketDataProcessor can read from it transparently.
    const char* demoEnv = std::getenv("BTQUANT_DEMO");
    bool wantDemo = (demoEnv && demoEnv[0] == '1');

    struct stat shmStat{};
    bool shmExists = (::stat("/dev/shm/btquant_hotspine", &shmStat) == 0);

    if (wantDemo || !shmExists) {
      if (auto err = mockProducer.start()) {
        std::fprintf(stderr, "[BTQuant] MockProducer start failed: %s\n",
                     err->c_str());
      }
    }

    // Try /dev/shm/btquant_hotspine first; MarketDataProcessor falls back to
    // a synthetic generator if the spine can't be opened (mock producer not
    // running yet).
    if (auto err = marketData.start("/dev/shm/btquant_hotspine", 16)) {
      std::fprintf(stderr, "[BTQuant] MarketDataProcessor start failed: %s\n",
                   err->c_str());
    }
    // Wire the live data source into all 4 trading widgets. OrderBook, DOM,
    // Trades, TPO all switch from internal synthetic mock data to live
    // snapshots; widgets still fall back to mocks if the producer is offline.
    windowManager.setMarketData(&marketData);
  }

  void initCompute() {
    // Compute queue may be the same as graphics — get whatever compute support
    // VulkanContext found, falling back to graphics queue.
    auto qf = vkContext.queueFamilies();
    uint32_t family = qf.computeFamily.value_or(qf.graphicsFamily.value());
    HeatmapConfig cfg;
    // Apply persisted density at startup (long → uint32_t).
    cfg.image_width = cfg.image_height = static_cast<uint32_t>(windowManager.heatmapDensity);
    if (auto err = heatmapCompute.initialize(
            vkContext.device(), vkContext.physicalDevice(),
            vkContext.commandPool(), vkContext.graphicsQueue(),
            family, cfg)) {
      std::fprintf(stderr, "[BTQuant] HeatmapCompute init failed: %s\n",
                   err->c_str());
    } else {
      (void)heatmapWidget.initialize(heatmapCompute);
      // Record the applied size so the render loop can detect slider changes.
      windowManager.lastAppliedHeatmapDensity =
          static_cast<long>(heatmapCompute.currentSize());
    }
  }

  void pushTradesToHeatmap() {
    // Pull recent trades from the data pipeline and push them to the heatmap.
    // Normalize price to [0,1] using a running min/max window, time to [0,1]
    // across the rolling window.
    auto snap = marketData.snapshot(256);
    if (snap.recent_trades.empty()) return;

    // Compute price range from the snapshot.
    double pmin = snap.metrics.low;
    double pmax = snap.metrics.high;
    if (pmax <= pmin) pmax = pmin + 1e-6;
    double span = pmax - pmin;

    // Time normalization: oldest trade → 0.0, newest → 1.0 (uniformly distributed
    // over the rolling window). We treat the last N trades as time bin = i/N.
    const size_t N = snap.recent_trades.size();
    for (size_t i = 0; i < N; ++i) {
      const auto& t = snap.recent_trades[N - 1 - i];  // newest first → oldest last
      double price_norm = (t.price - pmin) / span;
      float time_norm = static_cast<float>(i) / static_cast<float>(N - 1);
      heatmapWidget.push(static_cast<float>(price_norm), time_norm,
                         static_cast<float>(t.size),
                         t.isBuy ? 0u : 1u);
    }

    // Mirror the most recent trade into the watchlist for the active symbol.
    // (Single-symbol MVP — the wire format has no symbol field yet. When the
    //  multi-symbol spine lands, this becomes a per-symbol dispatcher.)
    const auto& latest = snap.recent_trades.front();
    windowManager.updateWatchlist("BTC/USDT",
                                  latest.price, latest.size,
                                  latest.isBuy, latest.timestamp);
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      // Hotkeys run BEFORE ImGui's newFrame so user input reaches widgets.
      // Suppressed automatically when a text field has focus.
      windowManager.processHotkeys(window);

      // Stats overlay EWMA — one sample per frame.
      windowManager.tickStatsOverlay();

      VkCommandBuffer cmd = vkContext.beginFrame();
      if (cmd == VK_NULL_HANDLE) {
        continue;
      }

      // Pump new trades into the heatmap buffer BEFORE the compute dispatch.
      pushTradesToHeatmap();

      // Apply heatmap density changes from the Settings slider. Cheap when
      // unchanged (setSize is a no-op fast path).
      if (windowManager.heatmapDensity != windowManager.lastAppliedHeatmapDensity) {
        uint32_t target = static_cast<uint32_t>(windowManager.heatmapDensity);
        if (target < 64) target = 64;
        if (target > 512) target = 512;
        heatmapCompute.setSize(target);
        windowManager.lastAppliedHeatmapDensity = static_cast<long>(target);
      }

      // Apply theme changes from the View → Theme menu.
      if (static_cast<long>(uiContext.theme()) != windowManager.theme) {
        uiContext.applyTheme(windowManager.theme == 0
                                 ? ui::UIContext::Theme::Dark
                                 : ui::UIContext::Theme::Light);
      }

      // Record compute dispatch OUTSIDE the render pass. The compute writes
      // to the storage image (GENERAL layout), then transitions back to
      // SHADER_READ_ONLY_OPTIMAL so ImGui can sample it inside the render pass.
      heatmapCompute.dispatch(cmd);

      // ImGui frame + UI.
      uiContext.newFrame();
      ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                   ImGuiDockNodeFlags_PassthruCentralNode);
      windowManager.applyInitialDockLayoutIfNeeded();
      windowManager.showMainMenu();
      windowManager.showOrderBookWindow();
      windowManager.showOrderBookDepthWindow();
      windowManager.showFootprintWindow();
      windowManager.showVPVRWindow();
      windowManager.showMultiVWAPWindow();
      windowManager.showRiskPanelWindow();
      windowManager.showDOMWindow();
      windowManager.showTradesWindow();
      windowManager.showTPOWindow();
      windowManager.showAlertsWindow();
      windowManager.showWatchlistWindow();
      windowManager.showLogWindow();
      windowManager.showConnectionWindow();
      windowManager.showProfileManagerWindow();
      windowManager.showSettingsWindow();
      windowManager.showHotkeyHelpWindow();

      // Stats overlay (top-right). Pass live queue/candle counters so the user
      // can see when MarketDataProcessor is falling behind.
      auto snap = marketData.snapshot(0, 0);  // shallow — just for counts
      windowManager.renderStatsOverlay(
          static_cast<uint64_t>(snap.recent_trades.size()),
          static_cast<uint64_t>(snap.recent_candles.size()));

      heatmapWidget.render();

      // Periodic auto-save of state.ini when any setting is dirty. Save every
      // ~1 s at 60 fps; the dirty flag is cleared after each save so we don't
      // spam disk on every frame the user holds a key down. Atomic write means
      // a crash mid-save can never corrupt the existing file.
      if (windowManager.settingsDirty() && (frameCounter % 60) == 0) {
        auto settingsPath = util::Settings::defaultPath();
        util::Settings s;
        s.showOrderBook       = windowManager.showOrderBook;
        s.showOrderBookDepth  = windowManager.showOrderBookDepth;
        s.showFootprint       = windowManager.showFootprint;
        s.showVPVR            = windowManager.showVPVR;
        s.showMultiVWAP       = windowManager.showMultiVWAP;
        s.showRiskPanel       = windowManager.showRiskPanel;
        s.showDOM             = windowManager.showDOM;
        s.showTrades          = windowManager.showTrades;
        s.showTPO             = windowManager.showTPO;
        s.showSettings        = windowManager.showSettings;
        s.showStatsOverlay    = windowManager.showStatsOverlay;
        s.theme               = windowManager.theme;
        s.fpsLimit            = windowManager.fpsLimit;
        s.heatmapDensity      = windowManager.heatmapDensity;
        s.save(settingsPath);
        windowManager.clearSettingsDirty();
      }
      ++frameCounter;

      VkClearValue clearColor = {{{0.031f, 0.035f, 0.039f, 1.0f}}};  // #08090a
      VkRenderPassBeginInfo rpBegin{};
      rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      rpBegin.renderPass = vkContext.renderPass();
      rpBegin.framebuffer = vkContext.currentFramebuffer();
      rpBegin.renderArea.offset = {0, 0};
      rpBegin.renderArea.extent = vkContext.swapchainExtent();
      rpBegin.clearValueCount = 1;
      rpBegin.pClearValues = &clearColor;

      vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
      uiContext.render(cmd);
      vkCmdEndRenderPass(cmd);

      vkContext.endFrame();
    }
  }

  void cleanup() {
    // Persist user-visible state on shutdown.
    auto settingsPath = util::Settings::defaultPath();
    util::Settings s;
    s.showOrderBook       = windowManager.showOrderBook;
    s.showOrderBookDepth  = windowManager.showOrderBookDepth;
    s.showFootprint       = windowManager.showFootprint;
    s.showVPVR            = windowManager.showVPVR;
    s.showMultiVWAP       = windowManager.showMultiVWAP;
    s.showRiskPanel       = windowManager.showRiskPanel;
    s.showDOM             = windowManager.showDOM;
    s.showTrades          = windowManager.showTrades;
    s.showTPO             = windowManager.showTPO;
    s.showSettings        = windowManager.showSettings;
    s.showStatsOverlay    = windowManager.showStatsOverlay;
    s.fpsLimit            = windowManager.fpsLimit;
    s.heatmapDensity      = windowManager.heatmapDensity;
    s.save(settingsPath);
    std::fprintf(stderr, "[BTQuant] saved settings to %s\n",
                 settingsPath.c_str());

    // UIContext::shutdown() also flushes imgui.ini to disk.
    heatmapCompute.shutdown();
    marketData.stop();
    mockProducer.stop();
    if (window) {
      glfwDestroyWindow(window);
      window = nullptr;
    }
    glfwTerminate();
  }
};

int main() {
  BTQuantApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[BTQuant] fatal: %s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
