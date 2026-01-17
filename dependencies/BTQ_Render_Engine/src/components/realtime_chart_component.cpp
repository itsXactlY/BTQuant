/**
 * BTQuant Advanced Vulkan Dashboard - Realtime Chart Component
 */

// Use the corrected header which now contains the class definition
#include "../../include/vulkan_dashboard_advanced.hpp"

#include "../../include/CandlePipeline.h"
#include "../../include/OffscreenChartRenderer.h"
#include "imgui.h"

#include <algorithm>
#include <cmath>

namespace BTQuant {

// Constructor
RealtimeChartComponent::RealtimeChartComponent(
    const glm::vec2 &position, const glm::vec2 &size,
    std::shared_ptr<RenderEngine::HotSpineDataBridge> bridge)
    : UIComponent(position, size), bridge_(bridge) {

  view_zoom_ = 1.0f;
  view_offset_ = 0.0f;
}

RealtimeChartComponent::~RealtimeChartComponent() = default;

void RealtimeChartComponent::initialize_vulkan_resources(
    VulkanCore *vulkan_core) {
  vulkan_core_ = vulkan_core;

  if (!vulkan_core)
    return;

  // Initialize Renderer and Pipeline
  // Start with a default small size, will resize in render_gui
  offscreen_renderer_ = std::make_unique<OffscreenChartRenderer>(vulkan_core);
  // Initial size
  offscreen_renderer_->create_resources(800, 600);

  candle_pipeline_ = std::make_unique<CandlePipeline>(
      vulkan_core, offscreen_renderer_->GetRenderPass());
}

void RealtimeChartComponent::render_gui() {
  if (!offscreen_renderer_ || !candle_pipeline_)
    return;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  if (ImGui::Begin("Realtime Chart", nullptr, 0)) {

    // 1. Handling Input (Zoom/Pan)
    if (ImGui::IsWindowHovered()) {
      float wheel = ImGui::GetIO().MouseWheel;
      if (wheel != 0) {
        view_zoom_ *= (wheel > 0 ? 1.1f : 0.9f);
        view_zoom_ = std::max(0.01f, std::min(view_zoom_, 10.0f));
      }

      if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
        ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
        view_offset_ -= delta.x * 0.01f; // Sensitivity
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
      }
    }

    ImVec2 content_size = ImGui::GetContentRegionAvail();

    // 2. Resize Check
    if (content_size.x > 0 && content_size.y > 0 &&
        (std::abs(content_size.x - size_.x) > 1.0f ||
         std::abs(content_size.y - size_.y) > 1.0f)) {

      size_ = glm::vec2(content_size.x, content_size.y);
      vkDeviceWaitIdle(vulkan_core_->get_device());
      offscreen_renderer_->create_resources((uint32_t)size_.x,
                                            (uint32_t)size_.y);
    }

    // 3. Render Setup
    VkCommandBuffer cmd = vulkan_core_->begin_single_time_commands();
    offscreen_renderer_->begin_render(cmd);

    // 4. Prepare Data & Pipeline
    std::vector<CandleData> pipeline_candles;
    {
      std::lock_guard<std::recursive_mutex> lock(data_mutex_);
      // Convert internal candles to pipeline format
      float min_y = 1e9f, max_y = -1e9f;
      for (size_t i = 0; i < raw_candles_.size(); ++i) {
        const auto &c = raw_candles_[i];
        // Green if close >= open, Red otherwise
        uint32_t color = (c.close >= c.open) ? 0xFF00FF00 : 0xFF0000FF;
        pipeline_candles.push_back(
            {(float)i, c.open, c.high, c.low, c.close, color});

        // Calculate bounds for auto-scale
        min_y = std::min(min_y, c.low);
        max_y = std::max(max_y, c.high);
      }

      if (pipeline_candles.empty()) {
        // Dummy data just to show something if empty
        min_y = 0;
        max_y = 100;
      }

      // Camera / Push Constants
      CandlePipeline::PushConstants pc;
      // Y-axis mapping
      if (max_y == min_y)
        max_y += 1.0f;
      pc.chart_min = glm::vec2(0, min_y);
      pc.chart_max =
          glm::vec2(std::max(10.0f, (float)raw_candles_.size()), max_y);

      // X-axis mapping (Zoom/Pan)
      float view_width = pipeline_candles.size() / view_zoom_;
      if (view_width == 0)
        view_width = 100.0f;

      pc.projection =
          glm::ortho(view_offset_, view_offset_ + view_width, min_y, max_y);
      pc.candle_width = 0.8f;

      candle_pipeline_->Render(cmd, pipeline_candles, pc);
    }

    offscreen_renderer_->end_render(cmd);
    vulkan_core_->end_single_time_commands(cmd);

    // 5. Display Texture
    ImGui::Image((ImTextureID)offscreen_renderer_->GetDescriptor(),
                 content_size);
  }
  ImGui::End();
  ImGui::PopStyleVar();
}

void RealtimeChartComponent::handle_trade(
    const RenderEngine::TradeData &trade) {
  std::lock_guard<std::recursive_mutex> lock(data_mutex_);

  // Aggregate trades into candles (1 Minute fixed for now)
  uint64_t interval = 60 * 1000000; // 1 min in microseconds
  uint64_t candle_ts = (trade.timestamp_us / interval) * interval;

  if (raw_candles_.empty() || raw_candles_.back().timestamp_us != candle_ts) {
    // New Candle
    Candle new_c;
    new_c.timestamp_us = candle_ts;
    new_c.open = (float)trade.price;
    new_c.high = (float)trade.price;
    new_c.low = (float)trade.price;
    new_c.close = (float)trade.price;
    new_c.volume = (float)trade.size;
    raw_candles_.push_back(new_c);
  } else {
    // Update existing
    auto &c = raw_candles_.back();
    c.high = std::max(c.high, (float)trade.price);
    c.low = std::min(c.low, (float)trade.price);
    c.close = (float)trade.price;
    c.volume += (float)trade.size;
  }

  // Prune old candles if necessary
  if (raw_candles_.size() > 1000) {
    raw_candles_.pop_front();
  }
}

void RealtimeChartComponent::handle_orderbook(
    const RenderEngine::OrderbookData &data) {
  // Optional: Visualize orderbook depth on chart
}

void RealtimeChartComponent::clear_data() {
  std::lock_guard<std::recursive_mutex> lock(data_mutex_);
  raw_candles_.clear();
  candles_.clear();
}

void RealtimeChartComponent::add_data_point(float time, float price,
                                            float volume) {
  // Stub
}

void RealtimeChartComponent::set_time_window(float seconds) {
  // Stub
}

void RealtimeChartComponent::set_y_range(float min_y, float max_y) {
  // Stub
}

void RealtimeChartComponent::enable_candlestick_mode(bool enable) {
  // Stub
}

// Required overrides
void RealtimeChartComponent::update(float delta_time) { (void)delta_time; }
void RealtimeChartComponent::render(VkCommandBuffer cmd) {
  (void)cmd;
  // Logic is in render_gui for this component as it uses offscreen rendering
  // displayed via ImGui::Image
}
void RealtimeChartComponent::handle_input(const InputEvent &event) {
  (void)event;
}

} // namespace BTQuant