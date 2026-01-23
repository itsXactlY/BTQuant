#include "../../include/components/depth_chart_panel.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace BTQuant {

DepthChartPanel::DepthChartPanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
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
  if (!processor_ || symbol_id_ == 0)
    return;

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
  render_stats();
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

  render_depth_chart_implot();

  end_panel_window();
}

void DepthChartPanel::compute_depth_data() {
  const auto &bids = cached_orderbook_.bids;
  const auto &asks = cached_orderbook_.asks;

  // Clear and repopulate (vectors keep capacity)
  bid_prices_.clear();
  bid_cumulative_.clear();
  ask_prices_.clear();
  ask_cumulative_.clear();

  // Compute cumulative bid depth
  double cumulative = 0.0;
  for (const auto &level : bids) {
    cumulative += level.size;
    bid_prices_.push_back(level.price);
    bid_cumulative_.push_back(cumulative);
  }

  // Compute cumulative ask depth
  cumulative = 0.0;
  for (const auto &level : asks) {
    cumulative += level.size;
    ask_prices_.push_back(level.price);
    ask_cumulative_.push_back(cumulative);
  }

  // Compute mid price and max depth for axis scaling
  if (!bids.empty() && !asks.empty()) {
    mid_price_ = (bids.front().price + asks.front().price) / 2.0;
  }

  max_depth_ = 1.0;
  if (!bid_cumulative_.empty()) {
    max_depth_ = std::max(max_depth_, bid_cumulative_.back());
  }
  if (!ask_cumulative_.empty()) {
    max_depth_ = std::max(max_depth_, ask_cumulative_.back());
  }
}

void DepthChartPanel::set_symbol(uint32_t symbol_id,
                                 const std::string &symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  cached_orderbook_ = {};
  bid_prices_.clear();
  bid_cumulative_.clear();
  ask_prices_.clear();
  ask_cumulative_.clear();

  // Re-subscribe to new symbol
  subscribe_to_updates();
  markDirty(); // Force immediate rebuild
}

void DepthChartPanel::render_stats() {
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();

  // Color-coded spread
  ImVec4 spread_color = cached_orderbook_.spread_percent < 0.1f
                            ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)
                            : ImVec4(0.8f, 0.8f, 0.2f, 1.0f);
  ImGui::TextColored(spread_color, "| Spread: %.4f (%.3f%%)",
                     cached_orderbook_.spread,
                     cached_orderbook_.spread_percent);
  ImGui::SameLine();

  // Color-coded imbalance
  ImVec4 imbalance_color = cached_orderbook_.imbalance > 0
                               ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)
                               : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
  ImGui::TextColored(imbalance_color, "| Imbalance: %+.2f",
                     cached_orderbook_.imbalance);
}

void DepthChartPanel::render_depth_chart_implot() {
  if (bid_prices_.empty() && ask_prices_.empty()) {
    ImGui::Text("Waiting for orderbook data...");
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.x < 50 || region.y < 50)
    return;

  // Calculate axis limits
  double price_min = mid_price_ * 0.995;
  double price_max = mid_price_ * 1.005;

  if (!bid_prices_.empty()) {
    price_min = std::min(price_min, bid_prices_.back());
  }
  if (!ask_prices_.empty()) {
    price_max = std::max(price_max, ask_prices_.back());
  }

  // Unique plot ID per panel instance to avoid ImGui ID conflicts
  char plot_id[64];
  snprintf(plot_id, sizeof(plot_id), "##DepthChart_%s", config_.title.c_str());

  if (ImPlot::BeginPlot(plot_id, region,
                        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend |
                            ImPlotFlags_NoMouseText)) {

    // Set axis limits
    ImPlot::SetupAxes("Price", "Cumulative Size", ImPlotAxisFlags_AutoFit,
                      ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisLimits(ImAxis_X1, price_min, price_max, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_depth_ * 1.1, ImPlotCond_Always);

    // Plot bid depth (green shaded area)
    if (bid_prices_.size() >= 2) {
      ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.1f, 0.7f, 0.1f, 0.4f));
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.9f, 0.2f, 1.0f));
      ImPlot::PlotShaded("Bids", bid_prices_.data(), bid_cumulative_.data(),
                         static_cast<int>(bid_prices_.size()), 0.0);
      ImPlot::PlotLine("Bids", bid_prices_.data(), bid_cumulative_.data(),
                       static_cast<int>(bid_prices_.size()));
      ImPlot::PopStyleColor(2);
    }

    // Plot ask depth (red shaded area)
    if (ask_prices_.size() >= 2) {
      ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.7f, 0.1f, 0.1f, 0.4f));
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));
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
      ImPlot::Annotation(mid_price_, max_depth_ * 0.9, ImVec4(1, 1, 1, 1),
                         ImVec2(5, -5), true, "%s", mid_label);
    }

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant
