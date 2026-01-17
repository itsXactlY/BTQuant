/**
 * BTQuant Advanced Vulkan Dashboard - Realtime Chart Component (Surgical Patch)
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
  offscreen_renderer_->create_resources(100, 100);

  candle_pipeline_ = std::make_unique<CandlePipeline>(
      vulkan_core, offscreen_renderer_->GetRenderPass());
}

void RealtimeChartComponent::render_gui() {
  if (!offscreen_renderer_ || !candle_pipeline_)
    return;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  if (ImGui::Begin("Realtime Chart", nullptr,
                   ImGuiWindowFlags_NoScrollbar |
                       ImGuiWindowFlags_NoScrollWithMouse)) {

    ImVec2 content_size = ImGui::GetContentRegionAvail();

    // Handle Resize
    // We check if size changed significantly to avoid spamming recreation
    if (content_size.x > 0 && content_size.y > 0 &&
        (std::abs(content_size.x - size_.x) > 1.0f ||
         std::abs(content_size.y - size_.y) > 1.0f)) {

      size_ = glm::vec2(content_size.x, content_size.y);
      vkDeviceWaitIdle(vulkan_core_->get_device());
      offscreen_renderer_->create_resources((uint32_t)size_.x,
                                            (uint32_t)size_.y);
    }

    // Perform Offscreen Rendering
    // Note: Doing this inside GUI render is unconventional but requested.
    // We typically need a command buffer.
    // We'll obtain a single-use command buffer here. This is expensive per
    // frame but follows instruction. OR we use the one passed to render(), but
    // render_gui doesn't have it. Assuming we are just queuing commands if the
    // renderer supports it, OR we grab one. To be safe and functional, we use
    // the single time command buffer pattern.

    VkCommandBuffer cmd = vulkan_core_->begin_single_time_commands();

    offscreen_renderer_->begin_render(cmd);

    // Prepare Pipeline Data
    std::vector<CandleData> pipeline_candles;
    {
      std::lock_guard<std::recursive_mutex> lock(data_mutex_);
      float min_y = 1e9f, max_y = -1e9f;
      for (size_t i = 0; i < candles_.size(); ++i) {
        const auto &c = candles_[i];
        uint32_t color = (c.close >= c.open) ? 0xFF00FF00 : 0xFF0000FF;
        pipeline_candles.push_back(
            {(float)i, c.open, c.high, c.low, c.close, color});
        min_y = std::min(min_y, c.low);
        max_y = std::max(max_y, c.high);
      }
      if (min_y > max_y) {
        min_y = 0;
        max_y = 100;
      }

      // Camera
      CandlePipeline::PushConstants pc;
      pc.chart_min = glm::vec2(0, min_y);
      pc.chart_max = glm::vec2(std::max(1.0f, (float)candles_.size()), max_y);
      // Apply simple zoom/pan logic
      float visible = 50.0f * view_zoom_;
      float start =
          std::max(0.0f, (float)candles_.size() - visible - view_offset_);
      pc.chart_min.x = start;
      pc.chart_max.x = start + visible;
      pc.projection = glm::ortho(pc.chart_min.x, pc.chart_max.x, pc.chart_min.y,
                                 pc.chart_max.y);
      pc.candle_width = 0.8f;

      candle_pipeline_->Render(cmd, pipeline_candles, pc);
    }

    offscreen_renderer_->end_render(cmd);

    vulkan_core_->end_single_time_commands(cmd); // Submit and wait

    // Display Texture
    // Cast to ImTextureID as required
    ImGui::Image((ImTextureID)offscreen_renderer_->GetDescriptor(),
                 content_size);

    // Inputs
    if (ImGui::IsItemHovered()) {
      float wheel = ImGui::GetIO().MouseWheel;
      if (wheel != 0) {
        view_zoom_ -= wheel * 0.1f;
        if (view_zoom_ < 0.1f)
          view_zoom_ = 0.1f;
      }
      if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        view_offset_ +=
            ImGui::GetMouseDragDelta(ImGuiMouseButton_Left).x * 0.1f;
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
      }
    }
  }
  ImGui::End();
  ImGui::PopStyleVar();
}

void RealtimeChartComponent::handle_trade(
    const RenderEngine::TradeData &trade) {
  std::lock_guard<std::recursive_mutex> lock(data_mutex_);
  // Simple 1-minute aggregation logic
  uint64_t tf = 60000000;
  uint64_t start = (trade.timestamp_us / tf) * tf;

  if (candles_.empty() || candles_.back().timestamp_us != start) {
    if (!candles_.empty() && trade.timestamp_us < candles_.back().timestamp_us)
      return; // Ignore old
    candles_.push_back({(float)trade.price, (float)trade.price,
                        (float)trade.price, (float)trade.price,
                        (float)trade.size, start});
  } else {
    auto &c = candles_.back();
    c.high = std::max(c.high, (float)trade.price);
    c.low = std::min(c.low, (float)trade.price);
    c.close = (float)trade.price;
    c.volume += (float)trade.size;
  }
}

// Stubs for required overrides
void RealtimeChartComponent::update(float) {}
void RealtimeChartComponent::render(VkCommandBuffer) {
  // Logic moved to render_gui as per surgical specific instructions
}
void RealtimeChartComponent::handle_input(const InputEvent &) {}
void RealtimeChartComponent::handle_orderbook(
    const RenderEngine::OrderbookData &) {}
void RealtimeChartComponent::add_data_point(float, float, float) {}
void RealtimeChartComponent::set_time_window(float) {}
void RealtimeChartComponent::set_y_range(float, float) {}
void RealtimeChartComponent::enable_candlestick_mode(bool) {}
void RealtimeChartComponent::clear_data() {}

} // namespace BTQuant