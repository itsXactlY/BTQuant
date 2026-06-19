#include <cstdlib>
#include <iostream>
#include <stdexcept>

#ifdef BTQUANT_USE_GLFW
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#else
#error "Only GLFW is fully supported in this backend right now"
#endif

#include "core/vulkan_context.hpp"
#include "ui/ui_context.hpp"
#include "ui/window_manager.hpp"

#include "data/market_data_processor.hpp"
#include "renderer/heatmap_compute.hpp"
#include "widgets/heatmap_widget.hpp"

using namespace btquant;

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

  // GPU compute pipeline for the heatmap texture.
  renderer::HeatmapCompute heatmapCompute;
  ui::HeatmapWidget heatmapWidget;

  uint32_t width = 1280;
  uint32_t height = 720;

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
    if (!uiContext.initialize(window, vkContext.instance(),
                              vkContext.physicalDevice(), vkContext.device(),
                              vkContext.queueFamilies().graphicsFamily.value(),
                              vkContext.graphicsQueue(),
                              vkContext.renderPass())) {
      throw std::runtime_error("Failed to initialize UI Context");
    }
    windowManager.initialize();
  }

  void initData() {
    // Try /dev/shm/btquant_hotspine first; MarketDataProcessor falls back to
    // a synthetic generator if the spine can't be opened (mock producer not
    // running yet).
    if (auto err = marketData.start("/dev/shm/btquant_hotspine", 16)) {
      std::fprintf(stderr, "[BTQuant] MarketDataProcessor start failed: %s\n",
                   err->c_str());
    }
  }

  void initCompute() {
    // Compute queue may be the same as graphics — get whatever compute support
    // VulkanContext found, falling back to graphics queue.
    auto qf = vkContext.queueFamilies();
    uint32_t family = qf.computeFamily.value_or(qf.graphicsFamily.value());
    if (auto err = heatmapCompute.initialize(
            vkContext.device(), vkContext.physicalDevice(),
            vkContext.commandPool(), vkContext.graphicsQueue(),
            family)) {
      std::fprintf(stderr, "[BTQuant] HeatmapCompute init failed: %s\n",
                   err->c_str());
    } else {
      (void)heatmapWidget.initialize(heatmapCompute);
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
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      VkCommandBuffer cmd = vkContext.beginFrame();
      if (cmd == VK_NULL_HANDLE) {
        continue;
      }

      // Pump new trades into the heatmap buffer BEFORE the compute dispatch.
      pushTradesToHeatmap();

      // Record compute dispatch OUTSIDE the render pass. The compute writes
      // to the storage image (GENERAL layout), then transitions back to
      // SHADER_READ_ONLY_OPTIMAL so ImGui can sample it inside the render pass.
      heatmapCompute.dispatch(cmd);

      // ImGui frame + UI.
      uiContext.newFrame();
      ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                   ImGuiDockNodeFlags_PassthruCentralNode);
      windowManager.showMainMenu();
      windowManager.showOrderBookWindow();
      windowManager.showDOMWindow();
      windowManager.showTradesWindow();
      windowManager.showTPOWindow();
      heatmapWidget.render();

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
    heatmapCompute.shutdown();
    marketData.stop();
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
