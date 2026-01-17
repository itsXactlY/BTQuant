#pragma once

#include "../../../tests/new/include/hotspine_reader.hpp"
#include "CandlePipeline.h"
#include "ChartMath.hpp"
#include "OffscreenChartRenderer.h"
#include "vulkan_base_types.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <iostream>
#include <memory>
#include <vector>

namespace BTQuant {

/**
 * DashboardOrchestrator - The "Glue" class binding Backend, Renderer, and UI.
 *
 * Responsibilities:
 * - Owns Rendering Pipeline and Data Reader.
 * - Polls HotSpine for real-time market data.
 * - Aggregates trades into candles.
 * - Manages ViewPort (Zoom/Pan) and coordinate mapping.
 * - Orchestrates offscreen rendering to a texture displayed in ImGui.
 */
class DashboardOrchestrator {
public:
  explicit DashboardOrchestrator(VulkanCore *core);
  ~DashboardOrchestrator();

  // Data Pump: Poll shared memory and update GPU buffers
  void UpdateData();

  // Interaction Logic: Handle Zoom and Pan via ImGuiIO
  void HandleInput();

  // Render Bridge: Execute pipeline and draw to ImGui
  void Draw(const char *windowName);

  // Sync window dimensions
  void Resize(uint32_t width, uint32_t height);

private:
  void ProcessNewTrades();
  void UpdateCandle(const HotSpine::HotTrade &trade);

  VulkanCore *core_;
  std::unique_ptr<OffscreenChartRenderer> renderer_;
  std::unique_ptr<CandlePipeline> pipeline_;
  std::unique_ptr<HotSpine::HotSpineReader> reader_;

  ViewPort viewport_;
  std::vector<CandleData> candles_;

  // OHLC State
  double candle_interval_us_ = 60.0 * 1000000.0; // 1 minute default

  uint32_t width_ = 1280;
  uint32_t height_ = 720;

  // Interaction state
  bool is_panning_ = false;
  glm::vec2 pan_velocity_{0.0f};
  double zoom_accel_ = 0.0;

  // Double buffering support (implicit in storage buffer logic for now)
  // but we can track frame indices.
  uint32_t last_update_frame_ = 0;

  // Performance / Backpressure
  uint32_t max_polls_per_update_ = 1000;
};

} // namespace BTQuant
