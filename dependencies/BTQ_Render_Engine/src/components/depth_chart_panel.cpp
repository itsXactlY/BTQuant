#include "../../include/components/depth_chart_panel.hpp"
#include "imgui.h"
#include <algorithm>
#include <cmath>

namespace BTQuant {

DepthChartPanel::DepthChartPanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {}

void DepthChartPanel::update(float dt) {
  update_timer_ += dt;
  if (update_timer_ >= UPDATE_INTERVAL) {
    if (processor_ && symbol_id_ != 0) {
      auto orderbook_opt = processor_->getOrderbookData(symbol_id_);
      if (orderbook_opt.has_value()) {
        cached_orderbook_ = orderbook_opt.value();
      }
    }
    update_timer_ = 0.0f;
  }
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
  render_depth_chart();

  end_panel_window();
}

void DepthChartPanel::set_symbol(uint32_t symbol_id,
                                 const std::string &symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  cached_orderbook_ = {};
}

void DepthChartPanel::render_stats() {
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();
  ImGui::Text("| Spread: %.4f (%.2f%%)", cached_orderbook_.spread,
              cached_orderbook_.spread_percent);
  ImGui::SameLine();
  ImGui::Text("| Imbalance: %.2f", cached_orderbook_.imbalance);
}

void DepthChartPanel::render_depth_chart() {
  const auto &bids = cached_orderbook_.bids;
  const auto &asks = cached_orderbook_.asks;

  if (bids.empty() && asks.empty()) {
    ImGui::Text("No orderbook data available");
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  ImVec2 cursor = ImGui::GetCursorScreenPos();

  // Calculate price range
  double min_price = bids.empty() ? 0.0 : bids.back().price;
  double max_price = asks.empty() ? 0.0 : asks.back().price;

  if (!bids.empty() && !asks.empty()) {
    min_price = bids.back().price;
    max_price = asks.back().price;
  } else if (!bids.empty()) {
    min_price = bids.back().price * 0.99;
    max_price = bids.front().price * 1.01;
  } else if (!asks.empty()) {
    min_price = asks.front().price * 0.99;
    max_price = asks.back().price * 1.01;
  }

  double price_range = max_price - min_price;
  if (price_range <= 0)
    price_range = 1.0;

  // Calculate cumulative depths and max
  std::vector<double> bid_cumulative, ask_cumulative;
  std::vector<double> bid_prices, ask_prices;

  double cumulative = 0.0;
  for (const auto &level : bids) {
    cumulative += level.size;
    bid_cumulative.push_back(cumulative);
    bid_prices.push_back(level.price);
  }

  cumulative = 0.0;
  for (const auto &level : asks) {
    cumulative += level.size;
    ask_cumulative.push_back(cumulative);
    ask_prices.push_back(level.price);
  }

  double max_depth = 0.0;
  if (!bid_cumulative.empty())
    max_depth = std::max(max_depth, bid_cumulative.back());
  if (!ask_cumulative.empty())
    max_depth = std::max(max_depth, ask_cumulative.back());

  if (max_depth <= 0)
    max_depth = 1.0;

  // Helper lambda: price to X coordinate
  auto price_to_x = [&](double price) -> float {
    return cursor.x +
           static_cast<float>((price - min_price) / price_range) * region.x;
  };

  // Helper lambda: depth to Y coordinate (inverted, 0 at bottom)
  auto depth_to_y = [&](double depth) -> float {
    return cursor.y + region.y -
           static_cast<float>(depth / max_depth) * region.y;
  };

  float baseline_y = cursor.y + region.y;

  // Draw bid area (green)
  if (bid_cumulative.size() >= 2) {
    std::vector<ImVec2> bid_points;
    bid_points.push_back(ImVec2(price_to_x(bid_prices[0]), baseline_y));
    for (size_t i = 0; i < bid_prices.size(); ++i) {
      bid_points.push_back(
          ImVec2(price_to_x(bid_prices[i]), depth_to_y(bid_cumulative[i])));
    }
    bid_points.push_back(ImVec2(price_to_x(bid_prices.back()), baseline_y));

    draw_list->AddConvexPolyFilled(bid_points.data(),
                                   static_cast<int>(bid_points.size()),
                                   IM_COL32(50, 200, 50, 100));
    // Draw line
    for (size_t i = 1; i < bid_prices.size(); ++i) {
      draw_list->AddLine(
          ImVec2(price_to_x(bid_prices[i - 1]),
                 depth_to_y(bid_cumulative[i - 1])),
          ImVec2(price_to_x(bid_prices[i]), depth_to_y(bid_cumulative[i])),
          IM_COL32(50, 200, 50, 255), 2.0f);
    }
  }

  // Draw ask area (red)
  if (ask_cumulative.size() >= 2) {
    std::vector<ImVec2> ask_points;
    ask_points.push_back(ImVec2(price_to_x(ask_prices[0]), baseline_y));
    for (size_t i = 0; i < ask_prices.size(); ++i) {
      ask_points.push_back(
          ImVec2(price_to_x(ask_prices[i]), depth_to_y(ask_cumulative[i])));
    }
    ask_points.push_back(ImVec2(price_to_x(ask_prices.back()), baseline_y));

    draw_list->AddConvexPolyFilled(ask_points.data(),
                                   static_cast<int>(ask_points.size()),
                                   IM_COL32(200, 50, 50, 100));
    // Draw line
    for (size_t i = 1; i < ask_prices.size(); ++i) {
      draw_list->AddLine(
          ImVec2(price_to_x(ask_prices[i - 1]),
                 depth_to_y(ask_cumulative[i - 1])),
          ImVec2(price_to_x(ask_prices[i]), depth_to_y(ask_cumulative[i])),
          IM_COL32(200, 50, 50, 255), 2.0f);
    }
  }

  // Draw mid price line
  if (!bids.empty() && !asks.empty()) {
    double mid_price = (bids.front().price + asks.front().price) / 2.0;
    float mid_x = price_to_x(mid_price);
    draw_list->AddLine(ImVec2(mid_x, cursor.y), ImVec2(mid_x, baseline_y),
                       IM_COL32(200, 200, 200, 150), 1.0f);
  }

  // Reserve space
  ImGui::Dummy(region);
}

} // namespace BTQuant
