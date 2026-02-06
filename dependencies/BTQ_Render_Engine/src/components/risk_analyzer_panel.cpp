#include "../../include/components/risk_analyzer_panel.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "imgui.h"
#include "implot.h"
#include "../../include/symbol_registry.hpp"

namespace BTQuant {

RiskAnalyzerPanel::RiskAnalyzerPanel(const PanelConfig& config,
                                   std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), processor_(processor) {
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
    auto active_symbols = processor_ ? processor_->getActiveSymbols() : std::vector<uint32_t>{};
    if (!active_symbols.empty()) {
      auto symbol_id = active_symbols[0];
      // For now, use a default name since we don't have direct access to symbol names from processor
      // In a real implementation, this would come from SymbolRegistry or similar
      auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
      symbol_ = symbol_info.has_value() ? symbol_info->symbol : "SYMBOL_" + std::to_string(symbol_id);
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
  if (!processor_ || symbol_.empty()) return;
  
  // Get symbol ID from SymbolRegistry since we don't have direct access through processor
  uint32_t symbol_id = 0;
  
  // Find the symbol ID by iterating through active symbols from processor
  auto active_symbols = processor_->getActiveSymbols();
  for (auto id : active_symbols) {
    // Get symbol info from registry to match the symbol name
    auto symbol_info = SymbolRegistry::instance().get_symbol_info(id);
    if (symbol_info.has_value() && symbol_info->symbol == symbol_) {
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

  // Calculate time fraction (years to expiration)
  double time_to_expiry = static_cast<double>(days_to_expiration_) / 365.0;

  for (int i = 0; i < num_points; i++) {
    double price = min_underlying_price_ + i * step;
    underlying_prices_.push_back(price);

    // More realistic options risk calculation
    // Simulate a simple strategy: short ATM straddle (short 1 call + short 1 put)
    double strike = current_price_; // At-the-money strike
    
    // Calculate option values at expiration (intrinsic value only)
    double call_value_at_expiry = std::max(0.0, price - strike);  // Value of short call at expiry
    double put_value_at_expiry = std::max(0.0, strike - price);   // Value of short put at expiry
    
    // Calculate approximate premium received (using simplified Black-Scholes approximation)
    // Premium ~ S * σ * sqrt(T) * 0.4 for ATM options
    double approx_premium = current_price_ * volatility_ * std::sqrt(time_to_expiry) * 0.4;
    
    // Total premium collected for straddle
    double total_premium = 2.0 * approx_premium;
    
    // Calculate P/L at expiration
    // For short straddle: P/L = Premium received - (Call payoff + Put payoff)
    double pl = total_premium - (call_value_at_expiry + put_value_at_expiry);
    
    // Apply time decay effect (theta) - as time passes, the curve becomes more like expiry
    // At t=0 (now), we're further from expiry, so less extreme P/L
    // As time approaches expiry, the curve approaches the expiry payoff
    if (time_to_expiry > 0) {
      // Interpolate between current theoretical value and expiry value based on time remaining
      // As time_to_expiry approaches 0, we get closer to expiry payoff
      double time_weight = std::min(1.0, 0.1 / time_to_expiry); // Weight towards expiry as time decreases
      
      // For current theoretical value (before expiry), adjust based on time value
      double current_theoretical_pl = total_premium * (1.0 - std::exp(-time_to_expiry * 2.0));
      
      // Blend current theoretical value with expiry value
      pl = current_theoretical_pl * (1.0 - time_weight) + pl * time_weight;
    }
    
    // Apply volatility adjustment
    // Higher volatility increases premium received but also affects the shape
    pl *= (1.0 + (volatility_ - 0.30)); // Baseline at 30% volatility
    
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
  ImVec4 col_current = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow for current price line

  // Setup Plot with enhanced styling
  if (ImPlot::BeginPlot(plot_id, region,
                        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_CanvasOnly)) {

    // Set axis labels (grid lines are enabled by default)
    ImPlot::SetupAxes("Underlying Price", "Profit/Loss", 
                      ImPlotAxisFlags_None, ImPlotAxisFlags_None);

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

    // Plot the P/L curve with enhanced styling
    ImPlot::PushStyleColor(ImPlotCol_Line, col_profit);
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f); // Thicker line for better visibility
    ImPlot::PlotLine("P/L Curve", underlying_prices_.data(), profit_losses_.data(),
                     static_cast<int>(underlying_prices_.size()));
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor();

    // Draw current price line
    if (current_price_ > 0) {
      double current_line_x[2] = {current_price_, current_price_};
      double y_min = ImPlot::GetPlotLimits().Y.Min;
      double y_max = ImPlot::GetPlotLimits().Y.Max;
      double current_line_y[2] = {y_min, y_max};

      ImPlot::PushStyleColor(ImPlotCol_Line, col_current);
      ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.5f);
      ImPlot::PlotLine("Current Price", current_line_x, current_line_y, 2);
      ImPlot::PopStyleVar();
      ImPlot::PopStyleColor();

      // Add annotation for current price
      char current_label[64];
      snprintf(current_label, sizeof(current_label), "Current: %.2f", current_price_);
      ImPlot::Annotation(current_price_, y_max * 0.9, ImVec4(1, 1, 0, 1), ImVec2(5, -5), true,
                         "%s", current_label);
    }

    // Add zero P/L reference line
    double zero_line_x[2] = {underlying_prices_.front(), underlying_prices_.back()};
    double zero_line_y[2] = {0.0, 0.0};
    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.5f, 0.5f, 0.5f, 0.5f)); // Gray for zero line
    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 1.0f);
    ImPlot::PlotLine("Zero P/L", zero_line_x, zero_line_y, 2);
    ImPlot::PopStyleVar();
    ImPlot::PopStyleColor();

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
  ImGui::Text("Risk Analysis Parameters:");
  ImGui::TextDisabled("(Simulating short ATM straddle strategy)");

  // Days to expiration slider
  ImGui::Text("Days to Expiration:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(150);
  if (ImGui::SliderInt("##DaysExp", &days_to_expiration_, 1, 365, "%d days")) {
    // Parameter changed, recompute data
    compute_risk_data();
  }

  // Volatility slider
  ImGui::Text("Volatility (%%):");
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
    if (current_price_ > 0) {
      min_underlying_price_ = current_price_ * (1.0 - price_range_multiplier_);
      max_underlying_price_ = current_price_ * (1.0 + price_range_multiplier_);
      compute_risk_data();
    }
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
  
  // Show strategy summary
  ImGui::Separator();
  ImGui::Text("Strategy Summary:");
  if (!profit_losses_.empty()) {
    double max_pl = *std::max_element(profit_losses_.begin(), profit_losses_.end());
    double min_pl = *std::min_element(profit_losses_.begin(), profit_losses_.end());
    ImGui::Text("Max Profit: %.2f | Max Loss: %.2f", max_pl, min_pl);
  }
}

}  // namespace BTQuant