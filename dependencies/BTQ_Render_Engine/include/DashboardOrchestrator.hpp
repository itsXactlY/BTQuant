#pragma once

#include "vulkan_base_types.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

// Forward declarations
namespace BTQuant {
class OffscreenChartRenderer;
struct CandleData;
class CandlePipeline;
namespace RenderEngine {
class HotSpineDataBridge;
}
} // namespace BTQuant
namespace HotSpine {
// Legacy/Stub forward declarations if needed
}

namespace BTQuant {

struct OrchestratorCamera {
  float zoom = 1.0f;
  float pan_x = 0.0f;
};

class DashboardOrchestrator {
public:
  DashboardOrchestrator(VulkanCore *core);
  ~DashboardOrchestrator();

  // Dependency Injection / Ownership Transfer
  void
  SetBridge(std::unique_ptr<BTQuant::RenderEngine::HotSpineDataBridge> bridge);
  void SetRenderer(std::unique_ptr<OffscreenChartRenderer> renderer);
  void SetPipeline(std::unique_ptr<CandlePipeline> pipeline);

  void Update();
  void Draw(VkCommandBuffer cmd); // Vulkan Pass
  void RenderUI();                // ImGui Pass

private:
  void CalculateViewPort();

  VulkanCore *core_;

  // Components owned by Orchestrator
  std::unique_ptr<BTQuant::RenderEngine::HotSpineDataBridge> bridge_;
  std::unique_ptr<OffscreenChartRenderer> renderer_;
  std::unique_ptr<CandlePipeline> pipeline_;

  // Data
  std::vector<CandleData> candles_;
  OrchestratorCamera camera_;

  // Auto-scroll state
  bool auto_scroll_ = true;

  // Aggregation state
  struct Aggregator {
    long long current_minute = -1;
    float open = 0, high = 0, low = 0, close = 0;
    bool active = false;
  } agg_;

  struct ViewPort {
    float minTime, maxTime;
    float minPrice, maxPrice;
  } viewport_;
};

} // namespace BTQuant
