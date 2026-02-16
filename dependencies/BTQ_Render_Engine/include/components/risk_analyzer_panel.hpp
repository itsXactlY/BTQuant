#pragma once

#include <imgui.h>
#include <implot.h>

#include <memory>
#include <vector>
#include <functional>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"

namespace BTQuant {

/**
 * RiskAnalyzerPanel - P/L vs Underlying Price chart
 *
 * Displays a 2D coordinate system plotting Profit/Loss (Y-axis) vs Underlying Price (X-axis)
 * This chart helps visualize risk exposure across different price levels for options strategies
 */
class RiskAnalyzerPanel : public PanelBase {
 public:
  // DEPRECATED - Legacy hotspine
  RiskAnalyzerPanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                    std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void render() override;
  void update(float dt) override;
  void set_symbol(const std::string& symbol, const std::string& exchange = "Binance");

 private:
  // DEPRECATED - Legacy hotspine
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  std::string symbol_ = "BTC-USDT";
  std::string exchange_ = "Binance";

  // Data for the risk analysis chart
  std::vector<double> underlying_prices_;
  std::vector<double> profit_losses_;
  
  // Configuration for the risk analyzer
  double current_price_ = 0.0;
  double min_underlying_price_ = 0.0;
  double max_underlying_price_ = 0.0;
  double price_range_multiplier_ = 0.2; // 20% range around current price by default
  
  // What-if simulation parameters
  int days_to_expiration_ = 30;  // Default 30 days to expiration
  double volatility_ = 0.30;     // Default 30% volatility

  // Cached analytics data
  RenderEngine::SymbolAnalytics cached_analytics_;
  
  void compute_risk_data();
  void render_risk_chart();
  void render_controls();
  void update_current_price();
};

}  // namespace BTQuant