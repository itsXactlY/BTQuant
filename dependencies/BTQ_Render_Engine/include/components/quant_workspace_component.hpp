#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../trading/order_manager.hpp"
#include "../trading/position_manager.hpp"
#include "../trading/risk_assessment.hpp"
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
};

class QuantWorkspaceComponent : public UIComponent {
public:
  explicit QuantWorkspaceComponent(
      std::shared_ptr<HotSpineDataBridge> bridge,
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  virtual ~QuantWorkspaceComponent() = default;

  void update(float dt) override;
  void render_gui() override;

  void initialize_vulkan_resources(VulkanCore *core) override;
  void clear_data() override;

private:
  // Data \u0026 Chart Management
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::unique_ptr<ChartManager> chart_manager_;
  std::unique_ptr<IndicatorRenderer> indicator_renderer_;
  std::unordered_map<std::string, IndicatorConfig> indicator_configs_;

  // Trading Management (Phase 3)
  std::unique_ptr<OrderManager> order_manager_;
  std::unique_ptr<PositionManager> position_manager_;
  std::unique_ptr<RiskAssessment> risk_assessment_;

  // UI State
  RenderEngine::TimeFrame selected_timeframe_ =
      RenderEngine::TimeFrame::TF_1MIN;
  bool show_chart_controls_ = true;
  bool show_indicator_selector_ = true;
  bool show_trading_panel_ = true;
  std::string selected_symbol_ = "BTC-USDT";

  // Order Entry State
  double order_price_ = 0.0;
  double order_quantity_ = 0.0;
  int selected_order_type_ = 0; // 0=Market, 1=Limit
  int selected_order_side_ = 0; // 0=Buy, 1=Sell

  // Rendering Methods
  void render_instrument_chart(const std::string &symbol,
                               RenderEngine::TimeFrame timeframe,
                               const ChartInstance &chart);
  void render_timeframe_selector();
  void render_indicator_selector();
  void render_chart_controls();
  void render_trading_panel();   // New: Trading order entry & management
  void render_positions_panel(); // New: Active positions display
  void render_orders_panel();    // New: Active orders display
};

} // namespace BTQuant
