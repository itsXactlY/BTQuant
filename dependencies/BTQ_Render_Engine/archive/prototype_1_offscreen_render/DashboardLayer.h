#pragma once

#include <vulkan/vulkan.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>
#include <vector>

#include "CandlePipeline.h"
#include "OffscreenChartRenderer.h"
#include "vulkan_base_types.hpp"

namespace BTQuant {

class VulkanCore;

struct ChartCamera {
  double offset_x = 0.0; // Time/Index offset
  double offset_y = 0.0; // Price offset
  double scale_x = 1.0;  // Zoom factor X
  double scale_y = 1.0;  // Zoom factor Y

  // Convert screen pixel delta to chart units
  glm::vec2 ScreenToChart(const glm::vec2 &screen_pos,
                          const glm::vec2 &window_size,
                          const glm::vec2 &range) const {
    float x = (screen_pos.x / window_size.x) * (range.x / scale_x) + offset_x;
    float y = (1.0f - (screen_pos.y / window_size.y)) * (range.y / scale_y) +
              offset_y;
    return {x, y};
  }
};

class DashboardLayer {
public:
  DashboardLayer(VulkanCore *core);
  ~DashboardLayer() = default;

  void OnUpdate(float delta_time);
  void OnUIRender();

private:
  void SetupDockspace();
  void DrawSymbolSelector();
  void DrawChartWindow();
  void HandleInputs(const glm::vec2 &window_pos, const glm::vec2 &window_size);

  // Data Management
  void FetchData(const std::string &symbol);

  VulkanCore *core_;
  std::unique_ptr<OffscreenChartRenderer> chart_renderer_;
  std::unique_ptr<CandlePipeline> candle_pipeline_;

  // State
  ChartCamera camera_;
  std::string current_symbol_ = "BTC-USDT";
  std::vector<CandleData> current_candles_;

  bool is_dragging_ = false;
  glm::vec2 last_mouse_pos_ = {0, 0};

  // Chart limits (for normalization)
  glm::vec2 chart_range_ = {1000.0f,
                            100.0f}; // [Total Index Range, Price Range]
  glm::vec2 data_min_ = {0.0f, 0.0f};
  glm::vec2 data_max_ = {1000.0f, 100.0f};
};

} // namespace BTQuant
