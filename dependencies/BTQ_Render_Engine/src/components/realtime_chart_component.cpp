/**
 * BTQuant Advanced Vulkan Dashboard - Realtime Chart Component Implementation
 */

#include "../../include/CandlePipeline.h"
#include "../../include/OffscreenChartRenderer.h"
#include "../../include/hotspine_data_bridge.hpp"
#include "../../include/vulkan_dashboard_advanced.hpp"
#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace BTQuant {

RealtimeChartComponent::RealtimeChartComponent(
    const glm::vec2 &position, const glm::vec2 &size,
    std::shared_ptr<RenderEngine::HotSpineDataBridge> bridge)
    : UIComponent(position, size), bridge_(bridge) {

  time_window_ = 60.0f;
  min_y_ = 0.0f;
  max_y_ = 100.0f;
  auto_scale_ = true;
  candlestick_mode_ = true;

  view_zoom_ = 1.0f;
  view_offset_ = 0.0f;
}

RealtimeChartComponent::~RealtimeChartComponent() {
  // Unique pointers handle cleanup
}

void RealtimeChartComponent::initialize_vulkan_resources(
    VulkanCore *vulkan_core) {
  vulkan_core_ = vulkan_core;

  if (!vulkan_core)
    return;

  // Initialize Offscreen Renderer
  offscreen_renderer_ = std::make_unique<OffscreenChartRenderer>(vulkan_core);
  offscreen_renderer_->create_resources(static_cast<uint32_t>(size_.x),
                                        static_cast<uint32_t>(size_.y));

  // Initialize Candle Pipeline
  candle_pipeline_ = std::make_unique<CandlePipeline>(
      vulkan_core, offscreen_renderer_->GetRenderPass());
}

void RealtimeChartComponent::handle_trade(
    const RenderEngine::TradeData &trade) {
  std::lock_guard<std::recursive_mutex> lock(data_mutex_);

  // Aggregation Logic
  // 1-minute candles for now (fixed timeframe)
  uint64_t timeframe_us = 60 * 1000000;
  uint64_t candle_start_time =
      (trade.timestamp_us / timeframe_us) * timeframe_us;

  if (candles_.empty()) {
    candles_.push_back({(float)trade.price, (float)trade.price,
                        (float)trade.price, (float)trade.price,
                        (float)trade.size, candle_start_time});
  } else {
    Candle &last_candle = candles_.back();
    if (last_candle.timestamp_us == candle_start_time) {
      // Update current candle
      last_candle.high = std::max(last_candle.high, (float)trade.price);
      last_candle.low = std::min(last_candle.low, (float)trade.price);
      last_candle.close = (float)trade.price;
      last_candle.volume += (float)trade.size;
    } else if (trade.timestamp_us > last_candle.timestamp_us) {
      // New candle
      candles_.push_back({(float)trade.price, (float)trade.price,
                          (float)trade.price, (float)trade.price,
                          (float)trade.size, candle_start_time});
    }
  }
}

void RealtimeChartComponent::update(float delta_time) {
  // Handling inputs via handle_input, no continuous update needed for now
}

void RealtimeChartComponent::render(VkCommandBuffer cmd) {
  if (!offscreen_renderer_ || !candle_pipeline_)
    return;

  std::lock_guard<std::recursive_mutex> lock(data_mutex_);

  // Prepare data for pipeline
  std::vector<CandleData> pipeline_candles;
  pipeline_candles.reserve(candles_.size());

  float min_price = 1e9f;
  float max_price = -1e9f;

  for (size_t i = 0; i < candles_.size(); ++i) {
    const auto &c = candles_[i];

    // Color: Green (Cyan) for up, Red (Pink) for down
    uint32_t color =
        (c.close >= c.open)
            ? 0xFF00FFFF
            : 0xFF0000FF; // ABGR: Cyan, Red? Adjust as needed. 0xAABBGGRR
    // 0xFF00FFFF -> R=FF, G=FF, B=00, A=FF (Yellow/Cyan depending on
    // interpretation) Vulkan typically expects packed uint32. Let's assume
    // pipeline handles it. TealStreet theme: Up=Green (0xFF00FF00), Down=Red
    // (0xFF0000FF)
    if (c.close >= c.open)
      color = 0xFF00FF00;
    else
      color = 0xFF0000FF;

    pipeline_candles.push_back({(float)i, // X is index
                                c.open, c.high, c.low, c.close, color});

    if (c.low < min_price)
      min_price = c.low;
    if (c.high > max_price)
      max_price = c.high;
  }

  // Auto-scale Y
  if (auto_scale_ && min_price < max_price) {
    min_y_ = min_price - (max_price - min_price) * 0.1f;
    max_y_ = max_price + (max_price - min_price) * 0.1f;
  }

  // Camera Logic
  CandlePipeline::PushConstants pc;
  pc.chart_min = glm::vec2(0, min_y_);
  pc.chart_max =
      glm::vec2(candles_.size(), max_y_); // View entire range if no zoom

  // Apply zoom/pan
  float visible_candles = 50.0f * view_zoom_;
  float start_idx =
      std::max(0.0f, (float)candles_.size() - visible_candles - view_offset_);
  float end_idx = start_idx + visible_candles;

  pc.chart_min.x = start_idx;
  pc.chart_max.x = end_idx;
  pc.projection =
      glm::ortho(pc.chart_min.x, pc.chart_max.x, pc.chart_max.y,
                 pc.chart_min.y); // Flip Y? Usually Price up is Y up.
                                  // ortho(left, right, bottom, top).
  // If Y grows up, bottom=min_y, top=max_y.
  pc.projection = glm::ortho(pc.chart_min.x, pc.chart_max.x, pc.chart_min.y,
                             pc.chart_max.y);

  pc.candle_width = 0.8f;
  pc.padding = 0.0f;

  // Render
  offscreen_renderer_->begin_render(cmd);
  candle_pipeline_->Render(cmd, pipeline_candles, pc);
  offscreen_renderer_->end_render(cmd);
}

void RealtimeChartComponent::render_gui() {
  if (!offscreen_renderer_)
    return;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  if (ImGui::Begin("Realtime Chart", nullptr,
                   ImGuiWindowFlags_NoScrollbar |
                       ImGuiWindowFlags_NoScrollWithMouse)) {

    ImVec2 size = ImGui::GetContentRegionAvail();
    // Resize offscreen buffer if needed (simple check)
    if (size.x != size_.x || size.y != size_.y) {
      size_ = glm::vec2(size.x, size.y);
      // Recreate resources might be expensive every frame if resizing.
      // Ideally trigger on resize event. But for now:
      vkDeviceWaitIdle(vulkan_core_->get_device());
      offscreen_renderer_->create_resources(size_.x, size_.y);
      // Need to recreate framebuffer in renderer
    }

    ImTextureID tex_id = (ImTextureID)offscreen_renderer_->GetDescriptor();
    ImGui::Image(tex_id, size);

    // Input Handling for interactions (Zoom/Pan)
    if (ImGui::IsItemHovered()) {
      float wheel = ImGui::GetIO().MouseWheel;
      if (wheel != 0) {
        view_zoom_ -= wheel * 0.1f;
        if (view_zoom_ < 0.1f)
          view_zoom_ = 0.1f;
      }

      if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        view_offset_ += drag.x * 0.1f; // Adjust sensitivity
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
      }
    }
  }
  ImGui::End();
  ImGui::PopStyleVar();
}

void RealtimeChartComponent::handle_input(const InputEvent &event) {
  // Handled in render_gui via ImGui directly
}

void RealtimeChartComponent::handle_orderbook(
    const RenderEngine::OrderbookData &ob) {
  // Chart doesn't display orderbook yet
}

void RealtimeChartComponent::add_data_point(float timestamp, float value,
                                            float volume) {
  // Legacy support
}

void RealtimeChartComponent::set_time_window(float seconds) {
  time_window_ = seconds;
}
void RealtimeChartComponent::set_y_range(float min_y, float max_y) {
  min_y_ = min_y;
  max_y_ = max_y;
}
void RealtimeChartComponent::enable_candlestick_mode(bool enable) {
  candlestick_mode_ = enable;
}
void RealtimeChartComponent::clear_data() {
  std::lock_guard<std::recursive_mutex> lock(data_mutex_);
  candles_.clear();
}

} // namespace BTQuant