#include "components/orderbook_panel.hpp"
#include <algorithm>
#include <cmath>

namespace BTQuant {

OrderbookPanel::OrderbookPanel(const PanelConfig &config,
                               std::shared_ptr<HotSpineDataBridge> bridge,
                               std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(std::move(bridge)), processor_(std::move(processor)) {}

void OrderbookPanel::update(float dt) {
  // Update logic here
}

void OrderbookPanel::render() {
  begin_panel_window();
  
  // Render logic here (will be implemented later)
  
  end_panel_window();
}

void OrderbookPanel::set_symbol(uint32_t symbol_id, const std::string &symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
}

void OrderbookPanel::render_orderbook_ladder(const RenderEngine::OrderbookData &orderbook) {
  // Orderbook ladder rendering logic here (will be implemented later)
}

void OrderbookPanel::render_market_depth_chart(const RenderEngine::OrderbookData &orderbook) {
  if (orderbook.bids.empty() || orderbook.asks.empty())
    return;

  // Always update chart data with latest orderbook information
  cached_bx_.clear();
  cached_by_.clear();
  cached_ax_.clear();
  cached_ay_.clear();

  // Handle Bids - Cumulative depth from best bid down
  if (!orderbook.bids.empty()) {
    double cumulative_depth = 0.0;

    // Add points from worst bid to best bid for increasing X-axis
    for (auto it = orderbook.bids.rbegin(); it != orderbook.bids.rend();
         ++it) {
      cumulative_depth += it->size;
      cached_bx_.push_back(it->price);
      cached_by_.push_back(cumulative_depth);
    }

    // Extend to left for visual completeness
    const auto &worst_bid = orderbook.bids.back();
    cached_bx_.insert(cached_bx_.begin(), worst_bid.price * 0.995);
    cached_by_.insert(cached_by_.begin(), cumulative_depth);

    // Add point at best bid with 0 depth for shading
    const auto &best_bid = orderbook.bids[0];
    cached_bx_.push_back(best_bid.price);
    cached_by_.push_back(0.0);
  }

  // Handle Asks - Cumulative depth from best ask up
  if (!orderbook.asks.empty()) {
    double cumulative_depth = 0.0;

    // Add points from best ask to worst ask
    for (const auto &ask : orderbook.asks) {
      cumulative_depth += ask.size;
      cached_ax_.push_back(ask.price);
      cached_ay_.push_back(cumulative_depth);
    }

    // Extend to right for visual completeness
    const auto &worst_ask = orderbook.asks.back();
    cached_ax_.push_back(worst_ask.price * 1.005);
    cached_ay_.push_back(cumulative_depth);

    // Add point at best ask with 0 depth for shading
    const auto &best_ask = orderbook.asks[0];
    cached_ax_.insert(cached_ax_.begin(), best_ask.price);
    cached_ay_.insert(cached_ay_.begin(), 0.0);
  }

  if (ImPlot::BeginPlot("##Depth", ImVec2(-1, 150), ImPlotFlags_CanvasOnly)) {
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_AutoFit,
                      ImPlotAxisFlags_AutoFit);

    const auto &colors = ThemeManager::getInstance().getColors();
    ImPlot::SetNextFillStyle(colors.accent_green);
    ImPlot::PlotShaded("Bids", cached_bx_.data(), cached_by_.data(), (int)cached_bx_.size(), 0);

    ImPlot::SetNextFillStyle(colors.accent_red);
    ImPlot::PlotShaded("Asks", cached_ax_.data(), cached_ay_.data(), (int)cached_ax_.size(), 0);

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant