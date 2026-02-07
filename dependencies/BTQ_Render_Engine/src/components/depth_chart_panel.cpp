#include "../../include/components/depth_chart_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

DepthChartPanel::DepthChartPanel(const PanelConfig& config,
                                 std::shared_ptr<HotSpineDataBridge> bridge,
                                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor), visualization_mode_(DepthChartVisualizationMode::CUMULATIVE_AREA) {
  // Pre-allocate vectors for typical orderbook depth
  bid_prices_.reserve(50);
  bid_cumulative_.reserve(50);
  ask_prices_.reserve(50);
  ask_cumulative_.reserve(50);

  // C++26: Subscribe to push notifications instead of polling
  subscribe_to_updates();
}

DepthChartPanel::~DepthChartPanel() {
  // C++26: Clean unsubscription on destruction
  if (processor_ && subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }
}

void DepthChartPanel::subscribe_to_updates() {
  if (!processor_ || symbol_id_ == 0) return;

  // Unsubscribe from previous symbol if any
  if (subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }

  // Subscribe to ORDERBOOK notifications for this symbol
  subscription_id_ = processor_->subscribe(
      symbol_id_, RenderEngine::NotificationType::ORDERBOOK,
      [this](uint32_t /*symbol_id*/, RenderEngine::NotificationType /*type*/) {
        // Thread-safe: atomic flag set from worker thread
        this->markDirty();
      });
}

void DepthChartPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();

  // Auto-select first available symbol if none set (like OrderbookPanel)
  if (processor_ && bridge_ && symbol_id_ == 0) {
    auto active_symbols = processor_->getActiveSymbols();
    for (uint32_t sym_id : active_symbols) {
      auto ob_opt = processor_->getOrderbookData(sym_id);
      if (ob_opt.has_value()) {
        symbol_id_ = sym_id;
        symbol_name_ = bridge_->getSymbolName(sym_id);
        config_.title = "Depth Chart - " + symbol_name_;
        subscribe_to_updates();
        markDirty();
        break;
      }
    }
  }

  render_stats();
  ImGui::Separator();

  // Render visualization mode selector
  render_visualization_mode_selector();
  ImGui::Separator();

  // C++26 Reactive: Refresh data when dirty or on first frame with valid symbol
  if (processor_ && symbol_id_ != 0) {
    // Always check orderbook if we have a valid symbol
    // consumeDirty() returns true on first call or when notified
    if (consumeDirty() || cached_orderbook_.bids.empty()) {
      auto orderbook_opt = processor_->getOrderbookData(symbol_id_);
      if (orderbook_opt.has_value()) {
        cached_orderbook_ = orderbook_opt.value();
        compute_depth_data();
      }
    }
  }

  // Render based on selected visualization mode
  switch (visualization_mode_) {
    case DepthChartVisualizationMode::CUMULATIVE_AREA:
      render_depth_chart_implot();
      break;
    case DepthChartVisualizationMode::SEPARATE_SIDES:
      render_depth_chart_separate_sides();
      break;
    case DepthChartVisualizationMode::BID_ASK_SPLIT:
      render_depth_chart_bid_ask_split();
      break;
  }

  end_panel_window();
}

void DepthChartPanel::compute_depth_data() {
  const auto& bids = cached_orderbook_.bids;
  const auto& asks = cached_orderbook_.asks;

  // Clear and repopulate
  bid_prices_.clear();
  bid_cumulative_.clear();
  ask_prices_.clear();
  ask_cumulative_.clear();

  // --- BIDS (Left Side, Green) ---
  // Visualize as: High Depth at Low Price (Left), dropping to 0 at Best Bid
  // (Right)
  if (!bids.empty()) {
    std::vector<std::pair<double, double>> temp_points;
    double cumulative = 0.0;

    // 1. Calculate cumulative depth from Best -> Worst (Descending Price)
    //    Best Bid (High Price) has small depth. Worst Bid (Low Price) has max
    //    depth.
    for (const auto& level : bids) {
      cumulative += level.size;
      temp_points.push_back({level.price, cumulative});
    }

    if (!temp_points.empty()) {
      double max_depth = temp_points.back().second;
      double worst_price = temp_points.back().first;
      double best_price = temp_points.front().first;

      // 2. Build vectors for ImPlot (X must be Ascending: Low Price -> High
      // Price)
      //    So we add points in this order:
      //    A. Extension to Left (Price < Worst Bid, Depth = Max)
      //    B. Worst Bid -> Best Bid (Reverse of temp_points)
      //    C. Drop to Zero at Best Bid

      // A. Extension
      bid_prices_.push_back(worst_price * 0.995);
      bid_cumulative_.push_back(max_depth);

      // B. Points (Reverse iteration of temp_points to get ascending Price)
      for (auto it = temp_points.rbegin(); it != temp_points.rend(); ++it) {
        bid_prices_.push_back(it->first);
        bid_cumulative_.push_back(it->second);
      }

      // C. Drop to Zero
      bid_prices_.push_back(best_price);
      bid_cumulative_.push_back(0.0);
    }
  }

  // --- ASKS (Right Side, Red) ---
  // Visualize as: 0 at Best Ask (Left), rising to High Depth at High Price
  // (Right)
  if (!asks.empty()) {
    double cumulative = 0.0;
    std::vector<std::pair<double, double>> temp_points;

    // 1. Calculate cumulative depth from Best -> Worst (Ascending Price)
    //    Best Ask (Low Price) has small depth. Worst Ask (High Price) has max
    //    depth.
    for (const auto& level : asks) {
      cumulative += level.size;
      temp_points.push_back({level.price, cumulative});
    }

    if (!temp_points.empty()) {
      double max_depth = temp_points.back().second;
      double worst_price = temp_points.back().first;
      double best_price = temp_points.front().first;

      // 2. Build vectors for ImPlot (X is Ascending: Low Price -> High Price)
      //    Order:
      //    A. Start at Zero at Best Ask
      //    B. Best Ask -> Worst Ask (Normal order)
      //    C. Extension to Right

      // A. Start at Zero
      ask_prices_.push_back(best_price);
      ask_cumulative_.push_back(0.0);

      // B. Points
      for (const auto& p : temp_points) {
        ask_prices_.push_back(p.first);
        ask_cumulative_.push_back(p.second);
      }

      // C. Extension
      ask_prices_.push_back(worst_price * 1.005);
      ask_cumulative_.push_back(max_depth);
    }
  }

  // Compute mid price and max depth for axis scaling
  if (!bids.empty() && !asks.empty()) {
    mid_price_ = (bids.front().price + asks.front().price) / 2.0;
  }

  max_depth_ = 1.0;
  // Check cumulative vectors for max depth (ignoring the 0 points)
  for (double d : bid_cumulative_) max_depth_ = std::max(max_depth_, d);
  for (double d : ask_cumulative_) max_depth_ = std::max(max_depth_, d);
}

void DepthChartPanel::set_symbol(uint32_t symbol_id, const std::string& symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  cached_orderbook_ = {};
  bid_prices_.clear();
  bid_cumulative_.clear();
  ask_prices_.clear();
  ask_cumulative_.clear();

  // Re-subscribe to new symbol
  subscribe_to_updates();
  markDirty();  // Force immediate rebuild
  
  // Notify the panel manager about the symbol change to trigger symbol linking
  if (get_panel_manager()) {
    get_panel_manager()->propagate_symbol_to_linked_panels(get_panel_id(), symbol_name_);
  }
}

void DepthChartPanel::render_stats() {
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();

  // Color-coded spread
  ImVec4 spread_color = cached_orderbook_.spread_percent < 0.1f ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)
                                                                : ImVec4(0.8f, 0.8f, 0.2f, 1.0f);
  ImGui::TextColored(spread_color, "| Spread: %.4f (%.3f%%)", cached_orderbook_.spread,
                     cached_orderbook_.spread_percent);
  ImGui::SameLine();

  // Color-coded imbalance
  ImVec4 imbalance_color = cached_orderbook_.imbalance > 0 ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)
                                                           : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
  ImGui::TextColored(imbalance_color, "| Imbalance: %+.2f", cached_orderbook_.imbalance);
}

void DepthChartPanel::render_depth_chart_implot() {
  if (bid_prices_.empty() && ask_prices_.empty()) {
    ImGui::Text("Waiting for orderbook data...");
    ImGui::Text("(Bids: %zu, Asks: %zu)", cached_orderbook_.bids.size(),
                cached_orderbook_.asks.size());
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.x < 50 || region.y < 50) return;

  // Calculate axis limits
  double price_min = mid_price_ * 0.995;
  double price_max = mid_price_ * 1.005;

  if (!bid_prices_.empty()) {
    price_min = std::min(price_min, bid_prices_.back());
  }
  if (!ask_prices_.empty()) {
    price_max = std::max(price_max, ask_prices_.back());
  }

  // Unique plot ID
  char plot_id[64];
  snprintf(plot_id, sizeof(plot_id), "##DepthChart_%s", config_.title.c_str());

  // Styling: Neon Financial Colors (Green/Red) from ThemeManager
  const auto& colors = ThemeManager::getInstance().getColors();
  ImVec4 col_bid_fill = colors.accent_green;
  col_bid_fill.w = 0.2f;
  ImVec4 col_bid_line = colors.accent_green;
  ImVec4 col_ask_fill = colors.accent_red;
  col_ask_fill.w = 0.2f;
  ImVec4 col_ask_line = colors.accent_red;

  // Setup Plot Flags for clean look
  if (ImPlot::BeginPlot(plot_id, region,
                        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText |
                            ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMenus)) {
    // Set axis limits - cleaner look without labels
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoLabel,
                      ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_Opposite);
    ImPlot::SetupAxisLimits(ImAxis_X1, price_min, price_max, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_depth_ * 1.1, ImPlotCond_Always);

    // Plot bid depth (Neon Green)
    if (!bid_prices_.empty()) {
      ImPlot::PushStyleColor(ImPlotCol_Fill, col_bid_fill);
      ImPlot::PushStyleColor(ImPlotCol_Line, col_bid_line);
      ImPlot::PlotShaded("Bids", bid_prices_.data(), bid_cumulative_.data(),
                         static_cast<int>(bid_prices_.size()), 0.0);
      ImPlot::PlotLine("Bids", bid_prices_.data(), bid_cumulative_.data(),
                       static_cast<int>(bid_prices_.size()));
      ImPlot::PopStyleColor(2);
    }

    // Plot ask depth (Neon Red)
    if (!ask_prices_.empty()) {
      ImPlot::PushStyleColor(ImPlotCol_Fill, col_ask_fill);
      ImPlot::PushStyleColor(ImPlotCol_Line, col_ask_line);
      ImPlot::PlotShaded("Asks", ask_prices_.data(), ask_cumulative_.data(),
                         static_cast<int>(ask_prices_.size()), 0.0);
      ImPlot::PlotLine("Asks", ask_prices_.data(), ask_cumulative_.data(),
                       static_cast<int>(ask_prices_.size()));
      ImPlot::PopStyleColor(2);
    }

    // Draw mid-price vertical line
    if (mid_price_ > 0) {
      double mid_line_x[2] = {mid_price_, mid_price_};
      double mid_line_y[2] = {0, max_depth_ * 1.1};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 1.0f, 1.0f, 0.5f));
      ImPlot::PlotLine("Mid", mid_line_x, mid_line_y, 2);
      ImPlot::PopStyleColor();

      // Annotation for mid price
      char mid_label[32];
      snprintf(mid_label, sizeof(mid_label), "Mid: %.2f", mid_price_);
      ImPlot::Annotation(mid_price_, max_depth_ * 0.9, ImVec4(1, 1, 1, 1), ImVec2(5, -5), true,
                         "%s", mid_label);
    }

    ImPlot::EndPlot();
  }
}

void DepthChartPanel::render_visualization_mode_selector() {
  const char* items[] = {"Cumulative Area", "Separate Sides", "Bid/Ask Split"};
  int current_item = static_cast<int>(visualization_mode_);

  ImGui::SetNextItemWidth(ImGui::GetFontSize() * 12);
  if (ImGui::Combo("##DepthChartMode", &current_item, items, IM_ARRAYSIZE(items))) {
    visualization_mode_ = static_cast<DepthChartVisualizationMode>(current_item);
    markDirty(); // Trigger recomputation if needed
  }

  ImGui::SameLine();
  ImGui::Text("Visualization Mode:");
}

void DepthChartPanel::render_depth_chart_separate_sides() {
  if (bid_prices_.empty() && ask_prices_.empty()) {
    ImGui::Text("Waiting for orderbook data...");
    ImGui::Text("(Bids: %zu, Asks: %zu)", cached_orderbook_.bids.size(),
                cached_orderbook_.asks.size());
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.x < 50 || region.y < 50) return;

  // Calculate axis limits
  double price_min = mid_price_ * 0.995;
  double price_max = mid_price_ * 1.005;

  if (!bid_prices_.empty()) {
    price_min = std::min(price_min, bid_prices_.front()); // Use front for min bid price
  }
  if (!ask_prices_.empty()) {
    price_max = std::max(price_max, ask_prices_.back()); // Use back for max ask price
  }

  // Unique plot ID
  char plot_id[64];
  snprintf(plot_id, sizeof(plot_id), "##DepthChartSeparate_%s", config_.title.c_str());

  // Styling: Neon Financial Colors (Green/Red) from ThemeManager
  const auto& colors = ThemeManager::getInstance().getColors();
  ImVec4 col_bid_fill = colors.accent_green;
  col_bid_fill.w = 0.2f;
  ImVec4 col_bid_line = colors.accent_green;
  ImVec4 col_ask_fill = colors.accent_red;
  col_ask_fill.w = 0.2f;
  ImVec4 col_ask_line = colors.accent_red;

  // Setup Plot Flags for clean look
  if (ImPlot::BeginPlot(plot_id, region,
                        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText |
                            ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMenus)) {
    // Set axis limits - cleaner look without labels
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoLabel,
                      ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_Opposite);
    ImPlot::SetupAxisLimits(ImAxis_X1, price_min, price_max, ImPlotCond_Always);

    // Calculate max depth for Y-axis scaling
    double max_depth_for_axis = max_depth_ * 1.1;
    ImPlot::SetupAxisLimits(ImAxis_Y1, -max_depth_for_axis, max_depth_for_axis, ImPlotCond_Always);

    // Plot bid depth (Neon Green) - on the negative side of Y-axis
    if (!bid_prices_.empty()) {
      // Create inverted data for bids (negative values)
      std::vector<double> bid_negative_values(bid_cumulative_.size());
      for (size_t i = 0; i < bid_cumulative_.size(); ++i) {
        bid_negative_values[i] = -bid_cumulative_[i];
      }

      ImPlot::PushStyleColor(ImPlotCol_Fill, col_bid_fill);
      ImPlot::PushStyleColor(ImPlotCol_Line, col_bid_line);
      ImPlot::PlotShaded("Bids", bid_prices_.data(), bid_negative_values.data(),
                         static_cast<int>(bid_prices_.size()), 0.0);
      ImPlot::PlotLine("Bids", bid_prices_.data(), bid_negative_values.data(),
                       static_cast<int>(bid_prices_.size()));
      ImPlot::PopStyleColor(2);
    }

    // Plot ask depth (Neon Red) - on the positive side of Y-axis
    if (!ask_prices_.empty()) {
      ImPlot::PushStyleColor(ImPlotCol_Fill, col_ask_fill);
      ImPlot::PushStyleColor(ImPlotCol_Line, col_ask_line);
      ImPlot::PlotShaded("Asks", ask_prices_.data(), ask_cumulative_.data(),
                         static_cast<int>(ask_prices_.size()), 0.0);
      ImPlot::PlotLine("Asks", ask_prices_.data(), ask_cumulative_.data(),
                       static_cast<int>(ask_prices_.size()));
      ImPlot::PopStyleColor(2);
    }

    // Draw zero-depth horizontal line
    double zero_line_x[2] = {price_min, price_max};
    double zero_line_y[2] = {0, 0};
    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.8f, 0.8f, 0.8f, 0.3f));
    ImPlot::PlotLine("Zero", zero_line_x, zero_line_y, 2);
    ImPlot::PopStyleColor();

    // Draw mid-price vertical line
    if (mid_price_ > 0) {
      double mid_line_x[2] = {mid_price_, mid_price_};
      double mid_line_y[2] = {-max_depth_for_axis, max_depth_for_axis};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 1.0f, 1.0f, 0.5f));
      ImPlot::PlotLine("Mid", mid_line_x, mid_line_y, 2);
      ImPlot::PopStyleColor();

      // Annotation for mid price
      char mid_label[32];
      snprintf(mid_label, sizeof(mid_label), "Mid: %.2f", mid_price_);
      ImPlot::Annotation(mid_price_, max_depth_for_axis * 0.9, ImVec4(1, 1, 1, 1), ImVec2(5, -5), true,
                         "%s", mid_label);
    }

    ImPlot::EndPlot();
  }
}

void DepthChartPanel::render_depth_chart_bid_ask_split() {
  if (bid_prices_.empty() && ask_prices_.empty()) {
    ImGui::Text("Waiting for orderbook data...");
    ImGui::Text("(Bids: %zu, Asks: %zu)", cached_orderbook_.bids.size(),
                cached_orderbook_.asks.size());
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.x < 50 || region.y < 50) return;

  // Calculate axis limits
  double price_min = mid_price_ * 0.995;
  double price_max = mid_price_ * 1.005;

  if (!bid_prices_.empty()) {
    price_min = std::min(price_min, bid_prices_.front()); // Use front for min bid price
  }
  if (!ask_prices_.empty()) {
    price_max = std::max(price_max, ask_prices_.back()); // Use back for max ask price
  }

  // Unique plot ID
  char plot_id[64];
  snprintf(plot_id, sizeof(plot_id), "##DepthChartSplit_%s", config_.title.c_str());

  // Styling: Neon Financial Colors (Green/Red) from ThemeManager
  const auto& colors = ThemeManager::getInstance().getColors();
  ImVec4 col_bid_fill = colors.accent_green;
  col_bid_fill.w = 0.2f;
  ImVec4 col_bid_line = colors.accent_green;
  ImVec4 col_ask_fill = colors.accent_red;
  col_ask_fill.w = 0.2f;
  ImVec4 col_ask_line = colors.accent_red;

  // Setup Plot Flags for clean look
  if (ImPlot::BeginPlot(plot_id, region,
                        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText |
                            ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMenus)) {
    // Set axis limits - cleaner look without labels
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoLabel,
                      ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_Opposite);
    ImPlot::SetupAxisLimits(ImAxis_X1, price_min, price_max, ImPlotCond_Always);

    // Calculate max depth for Y-axis scaling
    double max_depth_for_axis = max_depth_ * 1.1;
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_depth_for_axis, ImPlotCond_Always);

    // Plot bid depth (Neon Green) - only show bid data on the left side of the chart
    if (!bid_prices_.empty()) {
      // Calculate the midpoint between min and max price to determine left/right split
      double mid_point = (price_min + price_max) / 2.0;

      // Transform bid prices to the left half of the chart
      std::vector<double> transformed_bid_prices(bid_prices_.size());
      // Bids are typically sorted in descending order (highest bid first)
      double bid_min = bid_prices_.back();  // Lowest bid price
      double bid_max = bid_prices_.front(); // Highest bid price

      for (size_t i = 0; i < bid_prices_.size(); ++i) {
        // Map bid prices from their original range to the left half of the chart
        double normalized = (bid_prices_[i] - bid_min) / (bid_max - bid_min);
        transformed_bid_prices[i] = price_min + normalized * (mid_point - price_min);
      }

      ImPlot::PushStyleColor(ImPlotCol_Fill, col_bid_fill);
      ImPlot::PushStyleColor(ImPlotCol_Line, col_bid_line);
      ImPlot::PlotShaded("Bids", transformed_bid_prices.data(), bid_cumulative_.data(),
                         static_cast<int>(transformed_bid_prices.size()), 0.0);
      ImPlot::PlotLine("Bids", transformed_bid_prices.data(), bid_cumulative_.data(),
                       static_cast<int>(transformed_bid_prices.size()));
      ImPlot::PopStyleColor(2);
    }

    // Plot ask depth (Neon Red) - only show ask data on the right side of the chart
    if (!ask_prices_.empty()) {
      // Calculate the midpoint between min and max price to determine left/right split
      double mid_point = (price_min + price_max) / 2.0;

      // Transform ask prices to the right half of the chart
      std::vector<double> transformed_ask_prices(ask_prices_.size());
      // Asks are typically sorted in ascending order (lowest ask first)
      double ask_min = ask_prices_.front(); // Lowest ask price
      double ask_max = ask_prices_.back();  // Highest ask price

      for (size_t i = 0; i < ask_prices_.size(); ++i) {
        // Map ask prices from their original range to the right half of the chart
        double normalized = (ask_prices_[i] - ask_min) / (ask_max - ask_min);
        transformed_ask_prices[i] = mid_point + normalized * (price_max - mid_point);
      }

      ImPlot::PushStyleColor(ImPlotCol_Fill, col_ask_fill);
      ImPlot::PushStyleColor(ImPlotCol_Line, col_ask_line);
      ImPlot::PlotShaded("Asks", transformed_ask_prices.data(), ask_cumulative_.data(),
                         static_cast<int>(transformed_ask_prices.size()), 0.0);
      ImPlot::PlotLine("Asks", transformed_ask_prices.data(), ask_cumulative_.data(),
                       static_cast<int>(transformed_ask_prices.size()));
      ImPlot::PopStyleColor(2);
    }

    // Draw mid-price vertical line (the dividing line between bid and ask areas)
    if (mid_price_ > 0) {
      double mid_line_x[2] = {mid_price_, mid_price_};
      double mid_line_y[2] = {0, max_depth_for_axis};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 1.0f, 1.0f, 0.5f));
      ImPlot::PlotLine("Mid", mid_line_x, mid_line_y, 2);
      ImPlot::PopStyleColor();

      // Annotation for mid price
      char mid_label[32];
      snprintf(mid_label, sizeof(mid_label), "Mid: %.2f", mid_price_);
      ImPlot::Annotation(mid_price_, max_depth_for_axis * 0.9, ImVec4(1, 1, 1, 1), ImVec2(5, -5), true,
                         "%s", mid_label);
    }

    ImPlot::EndPlot();
  }
}

}  // namespace BTQuant
