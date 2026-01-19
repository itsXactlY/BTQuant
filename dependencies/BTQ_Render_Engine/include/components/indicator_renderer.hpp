#pragma once

#include "data_visualization_engine.hpp"
#include "imgui.h"
#include "market_data_processor.hpp"
#include "vulkan_base_types.hpp"
#include <string>
#include <vector>

namespace BTQuant {

enum class IndicatorType {
  SMA_10,
  SMA_20,
  SMA_50,
  EMA_10,
  EMA_20,
  EMA_50,
  RSI_14,
  MACD,
  MACD_SIGNAL,
  MACD_HISTOGRAM,
  BOLLINGER_MID,
  BOLLINGER_UPPER,
  BOLLINGER_LOWER,
  STOCHASTIC_K,
  STOCHASTIC_D,
  WADDAH_ATTAR_EXPLOSION
};

struct IndicatorParams {
  IndicatorType type;
  int period1 = 10;
  int period2 = 20;
  int period3 = 9;
  double std_dev = 2.0;
  ImVec4 color = {0.0f, 0.94f, 1.0f, 1.0f};
  float line_width = 1.0f;
  bool visible = true;
};

class IndicatorRenderer {
public:
  IndicatorRenderer(
      VulkanCore *vulkan_core,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void render_indicators(uint32_t symbol_id, RenderEngine::TimeFrame timeframe,
                         const std::vector<IndicatorParams> &indicators);

  void initialize_vulkan_resources();
  void cleanup_vulkan_resources();

private:
  VulkanCore *vulkan_core_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  RenderEngine::DataVisualizationEngine *visualization_engine_;

  std::vector<IndicatorParams> default_indicators_;

  void render_sma(uint32_t symbol_id, RenderEngine::TimeFrame timeframe,
                  const IndicatorParams &params);
  void render_ema(uint32_t symbol_id, RenderEngine::TimeFrame timeframe,
                  const IndicatorParams &params);
  void render_rsi(uint32_t symbol_id, RenderEngine::TimeFrame timeframe,
                  const IndicatorParams &params);
  void render_macd(uint32_t symbol_id, RenderEngine::TimeFrame timeframe,
                   const IndicatorParams &params);
  void render_bollinger(uint32_t symbol_id, RenderEngine::TimeFrame timeframe,
                        const IndicatorParams &params);
  void render_stochastic(uint32_t symbol_id, RenderEngine::TimeFrame timeframe,
                         const IndicatorParams &params);
  void render_waddah_attar_explosion(uint32_t symbol_id,
                                     RenderEngine::TimeFrame timeframe,
                                     const IndicatorParams &params);
};

} // namespace BTQuant
