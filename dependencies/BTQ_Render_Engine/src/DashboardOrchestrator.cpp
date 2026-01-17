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

  // Initialize ViewPort
  viewport_.minTime = 0.0;
  viewport_.maxTime = 1000.0;
  viewport_.minPrice = 0.0;
  viewport_.maxPrice = 100.0;
  viewport_.clamp();

  renderer_->resize(width_, height_);
}

DashboardOrchestrator::~DashboardOrchestrator() = default;

void DashboardOrchestrator::Resize(uint32_t width, uint32_t height) {
  if (width < 1 || height < 1)
    return;
  width_ = width;
  height_ = height;
  renderer_->resize(width_, height_);
}

void DashboardOrchestrator::UpdateData() {
  ProcessNewTrades();

  // Apply interaction momentum
  if (!is_panning_) {
    double dt = ImGui::GetIO().DeltaTime;
    if (glm::length(pan_velocity_) > 0.01f) {
      double shiftX = pan_velocity_.x * dt * viewport_.width();
      double shiftY = pan_velocity_.y * dt * viewport_.height();
      viewport_.minTime -= shiftX;
      viewport_.maxTime -= shiftX;
      viewport_.minPrice += shiftY;
      viewport_.maxPrice += shiftY;
      pan_velocity_ *= std::pow(0.1, dt); // Decay
    } else {
      pan_velocity_ = glm::vec2(0.0f);
    }
  }
}

void DashboardOrchestrator::ProcessNewTrades() {
  if (!reader_ || !reader_->isAttached())
    return;

  HotSpine::HotTrade trade;
  uint32_t count = 0;
  // Backpressure: limit polls per frame
  while (count < max_polls_per_update_ && reader_->pollTrade(trade)) {
    UpdateCandle(trade);
    count++;
  }
}

void DashboardOrchestrator::UpdateCandle(const HotSpine::HotTrade &trade) {
  double time = (double)trade.ts_exchange;
  double price = trade.price;

  double candle_idx = std::floor(time / candle_interval_us_);
  float x_pos = (float)candle_idx;

  if (candles_.empty() || candles_.back().x != x_pos) {
    if (candles_.empty()) {
      // First candle: center viewport if it was at 0
      if (viewport_.minTime == 0.0) {
        viewport_.minTime = time - candle_interval_us_ * 50;
        viewport_.maxTime = time + candle_interval_us_ * 50;
      }
    }
    CandleData newCandle;
    newCandle.x = x_pos;
    newCandle.open = (float)price;
    newCandle.high = (float)price;
    newCandle.low = (float)price;
    newCandle.close = (float)price;
    newCandle.color = 0xFF00FF00;
    candles_.push_back(newCandle);
  } else {
    auto &candle = candles_.back();
    candle.high = std::max(candle.high, (float)price);
    candle.low = std::min(candle.low, (float)price);
    candle.close = (float)price;
    candle.color = (candle.close >= candle.open) ? 0xFF00FF00 : 0xFF0000FF;
  }
}

void DashboardOrchestrator::HandleInput() {
  ImGuiIO &io = ImGui::GetIO();
  ImVec2 win_pos = ImGui::GetWindowPos();
  ImVec2 win_size = ImGui::GetContentRegionAvail();
  ImVec2 mouse_pos = ImGui::GetMousePos();

  // Local mouse position within the chart area
  glm::vec2 local_mouse(mouse_pos.x - win_pos.x,
                        mouse_pos.y - (win_pos.y + ImGui::GetFrameHeight()));

  if (!ImGui::IsWindowHovered()) {
    is_panning_ = false;
    return;
  }

  // Zoom (Towards Mouse Cursor)
  if (io.MouseWheel != 0.0f) {
    glm::vec2 market_before = ChartMath::ScreenToMarket(
        local_mouse, viewport_, (float)width_, (float)height_);

    double zoomFactor = 1.0 - (0.1 * io.MouseWheel);
    double new_width = viewport_.width() * zoomFactor;
    double new_height = viewport_.height() * zoomFactor;

    // Adjust viewport around market_before
    float normX = local_mouse.x / (float)width_;
    float normY = 1.0f - (local_mouse.y / (float)height_);

    viewport_.minTime = market_before.x - normX * new_width;
    viewport_.maxTime = viewport_.minTime + new_width;
    viewport_.minPrice = market_before.y - normY * new_height;
    viewport_.maxPrice = viewport_.minPrice + new_height;
    viewport_.clamp();
  }

  // Pan (Right Click Drag)
  if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
    is_panning_ = true;
    ImVec2 delta = io.MouseDelta;
    double rangeX = viewport_.width();
    double rangeY = viewport_.height();

    double shiftX = (delta.x / width_) * rangeX;
    double shiftY = (delta.y / height_) * rangeY;

    viewport_.minTime -= shiftX;
    viewport_.maxTime -= shiftX;
    viewport_.minPrice += shiftY;
    viewport_.maxPrice += shiftY;

    // Track velocity for momentum
    pan_velocity_ = glm::vec2(delta.x / width_ / io.DeltaTime,
                              delta.y / height_ / io.DeltaTime);
  } else {
    is_panning_ = false;
  }
}

void DashboardOrchestrator::Draw(const char *windowName) {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::Begin(windowName);

  ImVec2 size = ImGui::GetContentRegionAvail();
  if (size.x > 0 && size.y > 0 && (size.x != width_ || size.y != height_)) {
    Resize((uint32_t)size.x, (uint32_t)size.y);
  }

  // Prepare Push Constants
  CandlePipeline::PushConstants pc;
  pc.projection =
      ChartMath::CreateProjectionMatrix((float)width_, (float)height_);
  pc.chart_min = glm::vec2(viewport_.minTime, viewport_.minPrice);
  pc.chart_max = glm::vec2(viewport_.maxTime, viewport_.maxPrice);

  // Scale candle width based on zoom
  float visible_candles = (float)(viewport_.width() / candle_interval_us_);
  pc.candle_width =
      (visible_candles > 0) ? (0.8f * width_ / visible_candles) : 10.0f;

  // Execute Render Pass
  VkCommandBuffer cmd = core_->get_current_command_buffer();
  if (cmd != VK_NULL_HANDLE) {
    renderer_->begin_render(cmd);
    pipeline_->Render(cmd, candles_, pc);
    renderer_->end_render(cmd);
  }

  // Display Texture
  ImGui::Image((ImTextureID)renderer_->GetDescriptor(), size);

  // Interaction
  HandleInput();

  ImGui::End();
  ImGui::PopStyleVar();
}

} // namespace BTQuant
