#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../vulkan_dashboard_advanced.hpp"
#include "chart_manager.hpp"
#include "imgui.h"
#include "implot.h"
#include "indicator_renderer.hpp"
#include <memory>
#include <unordered_map>

namespace BTQuant {

struct IndicatorConfig {
  bool show_sma_10 = true;
  bool show_sma_20 = true;
  bool show_sma_50 = false;
  bool show_ema_10 = false;
  bool show_ema_20 = false;
  bool show_ema_50 = false;
  bool show_rsi = true;
  bool show_macd = true;
  bool show_bollinger = false;
  bool show_stochastic = false;
  bool show_waddah_explosion = false;
};

struct CandlePushConstants {
  glm::mat4 projection;
  glm::vec2 chart_min;
  glm::vec2 chart_max;
  float candle_width;
  uint32_t chart_offset;
  glm::vec2 viewport_size;
  glm::vec2 viewport_offset;
};

class QuantWorkspaceComponent : public UIComponent {
public:
  explicit QuantWorkspaceComponent(
      std::shared_ptr<HotSpineDataBridge> bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  virtual ~QuantWorkspaceComponent() = default;

  void update(float dt) override;
  void render_gui() override;
  void render(VkCommandBuffer cmd) override;

  void initialize_vulkan_resources(VulkanCore *core) override;
  void clear_data() override;
  void set_viz_engine(
      std::shared_ptr<RenderEngine::DataVisualizationEngine> engine) {
    viz_engine_ = engine;
  }

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<RenderEngine::DataVisualizationEngine> viz_engine_;
  std::unique_ptr<ChartManager> chart_manager_;
  std::unique_ptr<IndicatorRenderer> indicator_renderer_;
  std::map<uint32_t, IndicatorConfig> indicator_configs_;

  void render_instrument_chart(uint32_t chart_id, const std::string &symbol,
                               const InstrumentStore &inst,
                               RenderEngine::TimeFrame timeframe);
  void render_timeframe_selector();
  void render_indicator_selector();
  void render_chart_controls();

  RenderEngine::TimeFrame selected_timeframe_ =
      RenderEngine::TimeFrame::TF_1MIN;
  bool show_chart_controls_ = true;
  bool show_indicator_selector_ = true;

  // Custom Vulkan Pipeline for High-Performance Rendering
  VkPipeline candle_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout candle_pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout candle_descriptor_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet candle_descriptor_set_ = VK_NULL_HANDLE;

  // WAE Compute Pipeline
  VkPipeline wae_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout wae_pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout wae_descriptor_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet wae_descriptor_set_ = VK_NULL_HANDLE;

  // WAE Graphics Pipeline
  VkPipeline wae_graphics_pipeline_ = VK_NULL_HANDLE;
  VkPipelineLayout wae_graphics_pipeline_layout_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout wae_graphics_descriptor_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorSet wae_graphics_descriptor_set_ = VK_NULL_HANDLE;

  VulkanCore *core_ = nullptr;
};

} // namespace BTQuant
