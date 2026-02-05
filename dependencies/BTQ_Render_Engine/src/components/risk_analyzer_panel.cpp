#include "../../include/components/risk_analyzer_panel.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

RiskAnalyzerPanel::RiskAnalyzerPanel(const PanelConfig& config,
                                   std::shared_ptr<HotSpineDataBridge> bridge,
                                   std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
  // Initialize with some default data points
  config_.title = "Risk Analyzer - " + symbol_;
}

void RiskAnalyzerPanel::update(float dt) {
  (void)dt; // Unused parameter
  
  // Update current price periodically
  update_current_price();
  
  // Recompute risk data if needed
  compute_risk_data();
}

void RiskAnalyzerPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  // Auto-select first available symbol if none set
  if (processor_ && symbol_.empty()) {
    auto active_symbols = bridge_ ? bridge_->getActiveSymbols() : std::vector<uint32_t>{};
    if (!active_symbols.empty()) {
      auto symbol_id = active_symbols[0];
      symbol_ = bridge_ ? bridge_->getSymbolName(symbol_id) : "BTC-USDT";
      config_.title = "Risk Analyzer - " + symbol_;
    }
  }

  // Render controls
  render_controls();
  ImGui::Separator();

  // Render the risk chart
  render_risk_chart();

  end_panel_window();
}

void RiskAnalyzerPanel::set_symbol(const std::string& symbol, const std::string& exchange) {
  symbol_ = symbol;
  exchange_ = exchange;
  config_.title = "Risk Analyzer - " + symbol_;
  
  // Clear existing data
  underlying_prices_.clear();
  profit_losses_.clear();
  
  // Update current price
  update_current_price();
  
  // Recompute risk data
  compute_risk_data();
}

void RiskAnalyzerPanel::update_current_price() {
  if (!bridge_ || symbol_.empty()) return;
  
  // Get symbol ID from bridge since processor doesn't have getSymbolId method
  uint32_t symbol_id = 0;
  
  // Find the symbol ID by iterating through active symbols
  auto active_symbols = bridge_->getActiveSymbols();
  for (auto id : active_symbols) {
    if (bridge_->getSymbolName(id) == symbol_) {
      symbol_id = id;
      break;
    }
  }
  
  if (symbol_id == 0) return;
  
  auto analytics = processor_->getSymbolAnalytics(symbol_id);
  
  if (analytics.last_trade_price > 0) {
    current_price_ = analytics.last_trade_price;
    
    // Update price range based on current price
    min_underlying_price_ = current_price_ * (1.0 - price_range_multiplier_);
    max_underlying_price_ = current_price_ * (1.0 + price_range_multiplier_);
  }
}

void RiskAnalyzerPanel::compute_risk_data() {
  if (current_price_ <= 0) {
    update_current_price();
    if (current_price_ <= 0) return; // Still no valid price
  }

  // Clear existing data
  underlying_prices_.clear();
  profit_losses_.clear();

  // Generate price points across the range
  const int num_points = 200; // Higher resolution for smoother curves
  double step = (max_underlying_price_ - min_underlying_price_) / (num_points - 1);

  // Calculate time decay factor based on days to expiration
  // Shorter time to expiration means steeper P/L curve near the money
  double time_decay_factor = std::max(0.01, static_cast<double>(days_to_expiration_) / 365.0);
  
  // Calculate volatility factor - higher volatility flattens the curve
  double vol_factor = volatility_ / 0.30; // Normalize to 30% baseline

  for (int i = 0; i < num_points; i++) {
    double price = min_underlying_price_ + i * step;
    underlying_prices_.push_back(price);

    // Realistic risk profile calculation
    // This simulates a basic options strategy (e.g., a short straddle)
    // In a real implementation, this would calculate based on actual positions
    double pl = 0.0;

    // Example: Simulate a short straddle position (short call + short put at current price)
    // This creates a profit if price stays near current price, loss if it moves significantly
    double strike = current_price_; // For simplicity, assume ATM options
    
    // Adjust max profit based on volatility and time to expiration
    // Higher volatility = higher premiums, shorter time = lower premiums
    double max_profit = 100.0 * vol_factor * time_decay_factor;
    double max_loss = 1000.0; // Maximum theoretical loss

    // Calculate profit/loss based on options payoff
    if (price < strike) {
      // Put option payoff: profit decreases as price goes down
      double put_payoff = std::max(0.0, strike - price) - max_profit;
      
      // Apply time decay effect - closer to expiration has sharper curve
      if (time_decay_factor < 0.5) {
        // Near expiration - steeper curve
        put_payoff = std::pow(std::abs(put_payoff), 1.2) * (put_payoff < 0 ? -1 : 1);
      }
      
      pl = std::max(-max_loss, put_payoff); // Cap the loss
    } else {
      // Call option payoff: profit decreases as price goes up
      double call_payoff = std::max(0.0, price - strike) - max_profit;
      
      // Apply time decay effect - closer to expiration has sharper curve
      if (time_decay_factor < 0.5) {
        // Near expiration - steeper curve
        call_payoff = std::pow(std::abs(call_payoff), 1.2) * (call_payoff < 0 ? -1 : 1);
      }
      
      pl = std::max(-max_loss, -call_payoff); // Negative because we're short
    }

    // Apply volatility effect - higher volatility flattens the curve
    pl = pl / vol_factor;

    // Add some realistic scaling based on the underlying asset
    pl = pl * (price / current_price_); // Adjust for price scaling

    profit_losses_.push_back(pl);
  }
}

void RiskAnalyzerPanel::render_risk_chart() {
  if (underlying_prices_.empty() || profit_losses_.empty()) {
    ImGui::Text("Calculating risk profile...");
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.x < 50 || region.y < 50) return;

  // Unique plot ID
  char plot_id[64];
  snprintf(plot_id, sizeof(plot_id), "##RiskAnalyzer_%s", symbol_.c_str());

  // Styling: Use theme colors
  const auto& colors = ThemeManager::getInstance().getColors();
  ImVec4 col_profit = colors.accent_green;
  ImVec4 col_loss = colors.accent_red;
  ImVec4 col_current = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow for current price line

  // Setup Plot
  if (ImPlot::BeginPlot(plot_id, region,
                        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText |
                            ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMenus)) {
    
    // Set axis labels
    ImPlot::SetupAxes("Underlying Price", "Profit/Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
    
    // Set axis limits
    if (!underlying_prices_.empty()) {
      double x_min = underlying_prices_.front();
      double x_max = underlying_prices_.back();
      double y_min = *std::min_element(profit_losses_.begin(), profit_losses_.end());
      double y_max = *std::max_element(profit_losses_.begin(), profit_losses_.end());
      
      // Add some padding
      double y_range = y_max - y_min;
      if (y_range == 0) y_range = 1.0;
      y_min -= y_range * 0.1;
      y_max += y_range * 0.1;
      
      ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImPlotCond_Always);
      ImPlot::SetupAxisLimits(ImAxis_Y1, y_min, y_max, ImPlotCond_Always);
    }

    // Plot the P/L curve
    ImPlot::PushStyleColor(ImPlotCol_Line, col_profit);
    ImPlot::PlotLine("P/L Curve", underlying_prices_.data(), profit_losses_.data(),
                     static_cast<int>(underlying_prices_.size()));
    ImPlot::PopStyleColor();

    // Draw current price line
    if (current_price_ > 0) {
      double current_line_x[2] = {current_price_, current_price_};
      double y_min = ImPlot::GetPlotLimits().Y.Min;
      double y_max = ImPlot::GetPlotLimits().Y.Max;
      double current_line_y[2] = {y_min, y_max};
      
      ImPlot::PushStyleColor(ImPlotCol_Line, col_current);
      ImPlot::PlotLine("Current Price", current_line_x, current_line_y, 2);
      ImPlot::PopStyleColor();
      
      // Add annotation for current price
      char current_label[64];
      snprintf(current_label, sizeof(current_label), "Current: %.2f", current_price_);
      ImPlot::Annotation(current_price_, y_max * 0.9, ImVec4(1, 1, 0, 1), ImVec2(5, -5), true,
                         "%s", current_label);
    }

    ImPlot::EndPlot();
  }
}

void RiskAnalyzerPanel::render_controls() {
  ImGui::Text("Symbol: %s", symbol_.c_str());
  ImGui::SameLine();

  if (current_price_ > 0) {
    ImGui::Text("| Current Price: %.2f", current_price_);
  }

  // What-if simulation controls section
  ImGui::Separator();
  ImGui::Text("What-if Simulation Parameters:");
  
  // Days to expiration slider
  ImGui::Text("Days to Expiration:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(150);
  if (ImGui::SliderInt("##DaysExp", &days_to_expiration_, 1, 365, "%d days")) {
    // Parameter changed, recompute data
    compute_risk_data();
  }

  // Volatility slider
  ImGui::Text("Volatility (%):");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(150);
  float vol_pct = static_cast<float>(volatility_ * 100.0); // Convert to percentage for display
  if (ImGui::SliderFloat("##VolPct", &vol_pct, 1.0f, 200.0f, "%.1f%%")) {
    volatility_ = static_cast<double>(vol_pct) / 100.0; // Convert back to decimal
    // Parameter changed, recompute data
    compute_risk_data();
  }

  ImGui::Separator();

  // Price range multiplier control
  ImGui::Text("Price Range Multiplier:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(100);
  float temp_multiplier = static_cast<float>(price_range_multiplier_);
  if (ImGui::SliderFloat("##RangeMult", &temp_multiplier, 0.05f, 0.5f, "%.2f")) {
    price_range_multiplier_ = static_cast<double>(temp_multiplier);
    // Range changed, recompute data
    min_underlying_price_ = current_price_ * (1.0 - price_range_multiplier_);
    max_underlying_price_ = current_price_ * (1.0 + price_range_multiplier_);
    compute_risk_data();
  }

  // Button to refresh data
  if (ImGui::Button("Refresh")) {
    update_current_price();
    compute_risk_data();
  }

  ImGui::SameLine();

  // Button to reset to default range
  if (ImGui::Button("Reset Range")) {
    price_range_multiplier_ = 0.2f;
    if (current_price_ > 0) {
      min_underlying_price_ = current_price_ * (1.0 - price_range_multiplier_);
      max_underlying_price_ = current_price_ * (1.0 + price_range_multiplier_);
      compute_risk_data();
    }
  }
}

}  // namespace BTQuant