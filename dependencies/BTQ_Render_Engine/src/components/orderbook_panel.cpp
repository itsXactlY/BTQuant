#include "../../include/components/orderbook_panel.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

namespace BTQuant {

OrderbookPanel::OrderbookPanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {}

void OrderbookPanel::set_symbol(uint32_t symbol_id,
                                const std::string &symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
}

void OrderbookPanel::update(float dt) { (void)dt; }

void OrderbookPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  // Symbol selector for this orderbook panel
  auto active_symbols = processor_->getActiveSymbols();
  if (!active_symbols.empty()) {

    // Build symbol names for combo
    // Use panel pointer as unique ID for the combo itself to avoid conflicts
    // between panels
    ImGui::PushID(this);
    if (ImGui::BeginCombo("Symbol", symbol_name_.c_str())) {
      for (size_t i = 0; i < active_symbols.size(); ++i) {
        uint32_t sym_id = active_symbols[i];

        // Push unique ID for this item
        ImGui::PushID(static_cast<int>(sym_id));

        std::string sym_name = bridge_->getSymbolName(sym_id);
        std::string exchange = bridge_->getExchangeName(sym_id);
        std::string display_name = "[" + exchange + "] " + sym_name;

        bool is_selected = (sym_id == symbol_id_);

        if (ImGui::Selectable(display_name.c_str(), is_selected)) {
          symbol_id_ = sym_id;
          symbol_name_ = sym_name;
        }

        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }

        ImGui::PopID();
      }
      ImGui::EndCombo();
    }
    ImGui::PopID();
    ImGui::Separator();
  }

  // Get orderbook data for the selected symbol
  auto orderbook_opt = processor_->getOrderbookData(symbol_id_);

  if (!orderbook_opt.has_value()) {
    if (symbol_id_ == 0) {
      ImGui::Text("Select a symbol above");
    } else {
      ImGui::Text("Waiting for orderbook: %s (ID: %u)", symbol_name_.c_str(),
                  symbol_id_);
    }
    end_panel_window();
    return;
  }

  const auto &orderbook = orderbook_opt.value();

  // Debug: Show actual data counts
  ImGui::Text("Debug: Bids: %zu, Asks: %zu, Spread: %.4f",
              orderbook.bids.size(), orderbook.asks.size(), orderbook.spread);

  if (!orderbook.asks.empty()) {
    ImGui::Text("Debug: Top Ask: %.4f, Count (min): %d",
                orderbook.asks[0].price,
                std::min((int)orderbook.asks.size(), MAX_LEVELS));
  } else {
    ImGui::Text("Debug: Asks Empty!");
  }

  if (orderbook.bids.empty() && orderbook.asks.empty()) {
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                       "Empty Orderbook Vectors!");
  }

  // Render Orderbook Ladder
  render_orderbook_ladder(orderbook);
  ImGui::Separator();
  render_market_depth_chart(orderbook);

  end_panel_window();
}

void OrderbookPanel::render_orderbook_ladder(
    const RenderEngine::OrderbookData &orderbook) {
  // Simple Ladder View
  ImGui::Columns(3, "Ladder");
  ImGui::Text("Size");
  ImGui::NextColumn();
  ImGui::Text("Price");
  ImGui::NextColumn();
  ImGui::Text("Size");
  ImGui::NextColumn();
  ImGui::Separator();

  // Draw Asks (Sell) - Top down
  int ask_count = std::min((int)orderbook.asks.size(), MAX_LEVELS);
  for (int i = ask_count - 1; i >= 0; --i) {
    const auto &level = orderbook.asks[i];
    ImGui::NextColumn();
    ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.4f, 1.0f), "%.4f", level.price);
    ImGui::NextColumn();
    ImGui::Text("%.4f", level.size);
    ImGui::NextColumn();
  }

  // Spread
  if (!orderbook.asks.empty() && !orderbook.bids.empty()) {
    float spread = orderbook.asks[0].price - orderbook.bids[0].price;
    ImGui::NextColumn();
    ImGui::Text("Spread:");
    ImGui::NextColumn();
    ImGui::Text("%.4f", spread);
    ImGui::NextColumn();
    ImGui::Separator();
  }

  // Draw Bids (Buy)
  int bid_count = std::min((int)orderbook.bids.size(), MAX_LEVELS);
  for (int i = 0; i < bid_count; ++i) {
    const auto &level = orderbook.bids[i];
    ImGui::Text("%.4f", level.size);
    ImGui::NextColumn();
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "%.4f", level.price);
    ImGui::NextColumn();
    ImGui::NextColumn();
  }

  ImGui::Columns(1);
}

void OrderbookPanel::render_market_depth_chart(
    const RenderEngine::OrderbookData &orderbook) {
  if (orderbook.bids.empty() || orderbook.asks.empty())
    return;

  if (ImPlot::BeginPlot("Market Depth", ImVec2(-1, 200),
                        ImPlotFlags_NoMouseText)) {
    ImPlot::SetupAxes("Price", "Cumulative Size", ImPlotAxisFlags_AutoFit,
                      ImPlotAxisFlags_AutoFit);

    // Calculate cumulative bid depth
    std::vector<double> bid_prices, bid_depth;
    double cum_bid = 0;
    for (const auto &bid : orderbook.bids) {
      cum_bid += bid.size;
      bid_prices.push_back(bid.price);
      bid_depth.push_back(cum_bid);
    }

    // Calculate cumulative ask depth (cumulative from spread out)
    std::vector<double> ask_prices, ask_depth;
    double cum_ask = 0;
    for (const auto &ask : orderbook.asks) {
      cum_ask += ask.size;
      ask_prices.push_back(ask.price);
      ask_depth.push_back(cum_ask);
    }

    ImPlot::SetNextFillStyle(ImVec4(0.0f, 1.0f, 0.0f, 0.3f));
    ImPlot::PlotShaded("Bids", bid_prices.data(), bid_depth.data(),
                       (int)bid_prices.size(), 0);

    ImPlot::SetNextFillStyle(ImVec4(1.0f, 0.0f, 0.0f, 0.3f));
    ImPlot::PlotShaded("Asks", ask_prices.data(), ask_depth.data(),
                       (int)ask_prices.size(), 0);

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant
