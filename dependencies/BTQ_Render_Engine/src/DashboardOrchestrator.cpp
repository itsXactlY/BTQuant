#include "DashboardOrchestrator.hpp"
#include "CandlePipeline.h"
#include "OffscreenChartRenderer.h"
#include "hotspine_data_bridge.hpp"
#include "imgui.h"
#include <algorithm>
#include <iostream>
#include <limits>

namespace BTQuant {

DashboardOrchestrator::DashboardOrchestrator(VulkanCore *core) : core_(core) {}

DashboardOrchestrator::~DashboardOrchestrator() {}

void DashboardOrchestrator::SetBridge(
    std::unique_ptr<BTQuant::RenderEngine::HotSpineDataBridge> bridge) {
  bridge_ = std::move(bridge);
}

void DashboardOrchestrator::SetRenderer(
    std::unique_ptr<OffscreenChartRenderer> renderer) {
  renderer_ = std::move(renderer);
}

void DashboardOrchestrator::SetPipeline(
    std::unique_ptr<CandlePipeline> pipeline) {
  pipeline_ = std::move(pipeline);
}

void DashboardOrchestrator::Update() {
  if (!bridge_)
    return;

  // Use the bridge's reader directly
  auto *reader = bridge_->getReader();
  if (!reader)
    return;

  HotSpine::HotTrade trade;
  bool new_data = false;

  // Poll until buffer empty
  while (reader->pollTrade(trade)) {
    long long ts_min = trade.ts_exchange / 60000000; // Microseconds to minutes

    if (!agg_.active) {
      agg_.active = true;
      agg_.current_minute = ts_min;
      agg_.open = (float)trade.price;
      agg_.high = (float)trade.price;
      agg_.low = (float)trade.price;
      agg_.close = (float)trade.price;
    } else if (ts_min > agg_.current_minute) {
      // Commit candle
      CandleData c;
      c.x = (float)agg_.current_minute;
      c.open = agg_.open;
      c.high = agg_.high;
      c.low = agg_.low;
      c.close = agg_.close;
      c.color =
          (c.close >= c.open) ? 0xFF00FF00 : 0xFF0000FF; // Green : Red packed
      candles_.push_back(c);
      new_data = true;

      // Start new
      agg_.current_minute = ts_min;
      agg_.open = (float)trade.price;
      agg_.high = (float)trade.price;
      agg_.low = (float)trade.price;
      agg_.close = (float)trade.price;
    } else {
      // Update current
      agg_.high = std::max(agg_.high, (float)trade.price);
      agg_.low = std::min(agg_.low, (float)trade.price);
      agg_.close = (float)trade.price;
    }
  }

  if (new_data && auto_scroll_) {
    // Keep view at end
    // Logic handled in CalculateViewPort
  }
}

void DashboardOrchestrator::CalculateViewPort() {
  if (candles_.empty()) {
    viewport_ = {0, 10, 0, 100};
    return;
  }

  float width = 100.0f * camera_.zoom; // Window width in time units
  float last_x = candles_.back().x;

  if (auto_scroll_) {
    viewport_.maxTime = last_x + 1.0f;
    viewport_.minTime = viewport_.maxTime - width;
  } else {
    float center = last_x - camera_.pan_x;
    viewport_.minTime = center - width / 2.0f;
    viewport_.maxTime = center + width / 2.0f;
  }

  // Y-Axis Auto-Scale
  float min_p = std::numeric_limits<float>::max();
  float max_p = std::numeric_limits<float>::lowest();

  for (const auto &c : candles_) {
    if (c.x >= viewport_.minTime && c.x <= viewport_.maxTime) {
      min_p = std::min(min_p, c.low);
      max_p = std::max(max_p, c.high);
    }
  }

  if (min_p >= max_p) {
    min_p = 0;
    max_p = 100;
  }

  // Padding
  float range = max_p - min_p;
  viewport_.minPrice = min_p - range * 0.1f;
  viewport_.maxPrice = max_p + range * 0.1f;
}

void DashboardOrchestrator::Draw(VkCommandBuffer cmd) {
  if (!renderer_ || !pipeline_ || candles_.empty())
    return;

  // 1. Calculate View
  CalculateViewPort();

  CandlePipeline::PushConstants pc = {};
  pc.chart_min = glm::vec2(viewport_.minTime, viewport_.minPrice);
  pc.chart_max = glm::vec2(viewport_.maxTime, viewport_.maxPrice);
  pc.projection = glm::mat4(1.0f); // Shader handles mapping using min/max
  pc.candle_width = 0.8f;

  // 2. Offscreen Rendering Pass
  renderer_->begin_render(cmd);
  pipeline_->Render(cmd, candles_, pc);
  renderer_->end_render(cmd);
}

void DashboardOrchestrator::RenderUI() {
  if (!renderer_)
    return;

  ImGui::Begin("Realtime Chart");

  ImVec2 size = ImGui::GetContentRegionAvail();
  if (size.x > 0 && size.y > 0) {
    // Update renderer size if needed (expensive, skip for now or handle smart)
    // renderer_->resize((uint32_t)size.x, (uint32_t)size.y);

    // Image
    ImGui::Image((ImTextureID)renderer_->GetDescriptor(), size);

    // Interaction
    if (ImGui::IsItemHovered()) {
      float wheel = ImGui::GetIO().MouseWheel;
      if (wheel != 0) {
        camera_.zoom *= (wheel > 0 ? 0.9f : 1.1f);
        auto_scroll_ = false;
      }

      if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        camera_.pan_x += ImGui::GetIO().MouseDelta.x * 0.1f; // Scale speed
        auto_scroll_ = false;
      }

      if (ImGui::IsMouseDoubleClicked(0)) {
        auto_scroll_ = true;
      }
    }
  }

  ImGui::End();
}

} // namespace BTQuant
