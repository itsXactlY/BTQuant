// #pragma once
// #include <memory>
// #include <string>
// #include <vector>
// #include <unordered_map>
// #include "panel_base.hpp"
// #include "chart_manager.hpp"
// #include "../market_data_processor.hpp"

// namespace BTQuant {
// class PanelManager;
// class ChartPanel : public PanelBase {
// public:
//     ChartPanel(const PanelConfig& config,
//                std::shared_ptr<MarketDataProcessor> processor,
//                ChartManager* manager,
//                PanelManager* panel_manager);
//     ~ChartPanel() override = default;
//     void update(float dt) override;
//     void render_content() override;
//     void initialize() override;
//     void set_symbol(const std::string& symbol, const std::string& exchange);
//     void set_timeframe(RenderEngine::TimeFrame timeframe);
//     void center_on_timestamp(uint64_t timestamp);
//     void set_global_crosshair_position(double x_pos, bool active);
//     std::pair<double, bool> get_global_crosshair_state() const;
//     void set_global_crosshair_price(double price);
// private:
//     std::shared_ptr<MarketDataProcessor> processor_;
//     ChartManager* chart_manager_;
//     PanelManager* panel_manager_;
//     std::string symbol_;
//     std::string exchange_;
//     uint32_t current_symbol_id_ = 0;
//     RenderEngine::TimeFrame timeframe_ = RenderEngine::TimeFrame::TF_1MIN;
//     uint32_t chart_id_ = 0;
// };
// } // namespace BTQuant
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../market_data_processor.hpp"
#include "chart_manager.hpp"
#include "panel_base.hpp"

// Vulkan & Rendering Headers
#include <vulkan/vulkan.h>

#include "../rendering/candlestick_instancing.hpp"
#include "../rendering/vulkan_chart_pipeline.hpp"

namespace BTQuant {

class PanelManager;

// Daten, die wir an den asynchronen ImGui Draw-Callback übergeben
struct ChartDrawData {
  Rendering::VulkanChartPipeline* pipeline = nullptr;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;  // Enthält UBO & SSBO
  uint32_t instance_count = 0;
  Rendering::ChartPushConstants push_constants{};
};

class ChartPanel : public PanelBase {
 public:
  ChartPanel(const PanelConfig& config, std::shared_ptr<MarketDataProcessor> processor,
             ChartManager* manager, PanelManager* panel_manager);

  ~ChartPanel() override = default;

  void update(float dt) override;
  void render_content() override;
  void initialize() override;

  void set_symbol(const std::string& symbol) override;
  void set_timeframe(RenderEngine::TimeFrame timeframe);
  void center_on_timestamp(uint64_t timestamp);

  void set_global_crosshair_position(double x_pos, bool active);
  std::pair<double, bool> get_global_crosshair_state() const;
  void set_global_crosshair_price(double price);

  // --- NEUE VULKAN SETTER ---
  void set_vulkan_pipeline(Rendering::VulkanChartPipeline* pipeline, VkDescriptorSet dset) {
    chart_pipeline_ = pipeline;
    descriptor_set_ = dset;
  }

 private:
  std::shared_ptr<MarketDataProcessor> processor_;
  ChartManager* chart_manager_;
  PanelManager* panel_manager_;

  std::string symbol_;
  std::string exchange_;
  RenderEngine::TimeFrame timeframe_ = RenderEngine::TimeFrame::TF_1MIN;

  // Vulkan State
  Rendering::VulkanChartPipeline* chart_pipeline_ = nullptr;
  VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
  ChartDrawData current_draw_data_;
};

}  // namespace BTQuant