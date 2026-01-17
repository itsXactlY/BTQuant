#include "DashboardOrchestrator.hpp"
#if 0
#include "../../../tests/new/include/hotspine_reader.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <iostream>

namespace BTQuant {

DashboardOrchestrator::DashboardOrchestrator(VulkanCore *core) : core_(core) {
  renderer_ = std::make_unique<OffscreenChartRenderer>(core);
  pipeline_ =
      std::make_unique<CandlePipeline>(core, renderer_->GetRenderPass());
  reader_ = std::make_unique<HotSpine::HotSpineReader>("/btquant_hotspine");

  // Initialize ViewPort (example ranges)
  viewport_.minTime = 0.0;
  viewport_.maxTime = 100.0;
  viewport_.minPrice = 0.0;
  viewport_.maxPrice = 100.0;

  renderer_->resize(width_, height_);
}

DashboardOrchestrator::~DashboardOrchestrator() = default;

void DashboardOrchestrator::Resize(uint32_t width, uint32_t height) {
  if (width == 0 || height == 0)
    return;
  width_ = width;
  height_ = height;
  renderer_->resize(width_, height_);
}

void DashboardOrchestrator::UpdateData() { ProcessNewTrades(); }

void DashboardOrchestrator::ProcessNewTrades() {
  HotSpine::HotTrade trade;
  while (reader_->pollTrade(trade)) {
    UpdateCandle(trade);
  }
}

void DashboardOrchestrator::UpdateCandle(const HotSpine::HotTrade &trade) {
  double time = (double)trade.ts_exchange;
  double price = trade.price;

  // Fixed interval candle generation
  double candle_idx = std::floor(time / candle_interval_us_);
  float x_pos = (float)candle_idx;

  if (candles_.empty() || candles_.back().x != x_pos) {
    CandleData newCandle;
    newCandle.x = x_pos;
    newCandle.open = (float)price;
    newCandle.high = (float)price;
    newCandle.low = (float)price;
    newCandle.close = (float)price;
    newCandle.color = 0xFF00FF00; // Default green
    candles_.push_back(newCandle);
  } else {
    auto &candle = candles_.back();
    candle.high = std::max(candle.high, (float)price);
    candle.low = std::min(candle.low, (float)price);
    candle.close = (float)price;
    candle.color = (candle.close >= candle.open)
                       ? 0xFF00FF00
                       : 0xFF0000FF; // Green or Red (ABGR)
  }

  // Auto-scroll logic if needed, but let's keep it manual for now per
  // HandleInput
}

void DashboardOrchestrator::HandleInput() {
  ImGuiIO &io = ImGui::GetIO();
  if (!ImGui::IsWindowHovered())
    return;

  // Zoom (Mouse Wheel)
  if (io.MouseWheel != 0.0f) {
    double zoomFactor = 0.1 * io.MouseWheel;
    double range = viewport_.width();
    viewport_.minTime += range * zoomFactor;
    viewport_.maxTime -= range * zoomFactor;
  }

  // Pan (Right Click Drag)
  if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
    ImVec2 delta = io.MouseDelta;
    double rangeX = viewport_.width();
    double rangeY = viewport_.height();

    double shiftX = (delta.x / width_) * rangeX;
    double shiftY = (delta.y / height_) * rangeY;

    viewport_.minTime -= shiftX;
    viewport_.maxTime -= shiftX;
    viewport_.minPrice += shiftY; // Y is inverted in NDC/Screen
    viewport_.maxPrice += shiftY;
  }
}

void DashboardOrchestrator::Draw(const char *windowName) {
  ImGui::Begin(windowName);

  // Update target size if window resized
  ImVec2 size = ImGui::GetContentRegionAvail();
  if (size.x != width_ || size.y != height_) {
    Resize((uint32_t)size.x, (uint32_t)size.y);
  }

  // Prepare Push Constants
  CandlePipeline::PushConstants pc;
  pc.projection =
      ChartMath::CreateProjectionMatrix((float)width_, (float)height_);
  pc.chart_min = glm::vec2(viewport_.minTime, viewport_.minPrice);
  pc.chart_max = glm::vec2(viewport_.maxTime, viewport_.maxPrice);
  pc.candle_width =
      0.8f *
      (float)(width_ / (viewport_.width() > 0 ? viewport_.width() : 1.0));

  // Execute Render Pass
  VkCommandBuffer cmd = core_->get_current_command_buffer();
  if (cmd != VK_NULL_HANDLE) {
    renderer_->begin_render(cmd);
    pipeline_->Render(cmd, candles_, pc);
    renderer_->end_render(cmd);
  }

  // Display Texture
  ImGui::Image((ImTextureID)renderer_->GetDescriptor(), size);

  // Call HandleInput inside the ImGui context for this window
  HandleInput();

  ImGui::End();
}

} // namespace BTQuant
