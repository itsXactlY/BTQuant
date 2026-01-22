#include "../../include/components/orderbook_panel.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
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
  config_.title = symbol_name + " Orderbook";
}

void OrderbookPanel::update(float dt) {
    // Request data update from data bridge
    bridge_->sync();
    
    // The MarketDataProcessor will handle the actual data processing
    // through its internal worker threads and lock-free queue
}

void OrderbookPanel::render() {
  begin_panel_window();

  // If panel is hidden via X button, we still need to call end
  if (!is_visible()) {
    end_panel_window();
    return;
  }

  // Symbol selector for this orderbook panel
  auto active_symbols = processor_->getActiveSymbols();

  // Debug Info
  static int frame_count = 0;
  if (frame_count++ % 300 == 0) {
    std::cout << "[OrderbookPanel] Rendering. SymID=" << symbol_id_
              << " ActiveSyms=" << active_symbols.size() << std::endl;
  }

  if (!active_symbols.empty()) {
    // Check if current symbol has orderbook data, re-select if not
    auto current_ob = processor_->getOrderbookData(symbol_id_);
    bool need_reselect = (symbol_id_ == 0) || !current_ob.has_value();

    if (need_reselect) {
      for (uint32_t sym_id : active_symbols) {
        auto ob_opt = processor_->getOrderbookData(sym_id);
        if (ob_opt.has_value()) {
          if (symbol_id_ != sym_id) {
            symbol_id_ = sym_id;
            symbol_name_ = bridge_->getSymbolName(symbol_id_);
            config_.title = symbol_name_ + " Orderbook";
            std::cout << "[OrderbookPanel] Auto-selected: " << symbol_name_
                      << " (ID=" << symbol_id_ << ")" << std::endl;
          }
          break;
        }
      }
      // If still 0, just pick first to show "waiting"
      if (symbol_id_ == 0 && !active_symbols.empty()) {
        symbol_id_ = active_symbols[0];
        symbol_name_ = bridge_->getSymbolName(symbol_id_);
      }
    }

    // Build symbol names for combo
    ImGui::PushID(this);
    if (ImGui::BeginCombo("Select Symbol", symbol_name_.c_str())) {
      for (size_t i = 0; i < active_symbols.size(); ++i) {
        uint32_t sym_id = active_symbols[i];
        ImGui::PushID(static_cast<int>(sym_id));

        std::string sym_name = bridge_->getSymbolName(sym_id);
        std::string exchange = bridge_->getExchangeName(sym_id);
        std::string display_name = "[" + exchange + "] " + sym_name;

        bool is_selected = (sym_id == symbol_id_);
        if (ImGui::Selectable(display_name.c_str(), is_selected)) {
          symbol_id_ = sym_id;
          symbol_name_ = sym_name;
          config_.title = symbol_name_ + " Orderbook";
          std::cout << "[OrderbookPanel] Title updated to: " << config_.title
                    << std::endl;
        }
        if (is_selected)
          ImGui::SetItemDefaultFocus();
        ImGui::PopID();
      }
      ImGui::EndCombo();
    }
    ImGui::PopID();
  } else {
    ImGui::TextColored(ImVec4(1, 0, 0, 1),
                       "No active symbols detected in SHM!");
  }

  ImGui::Separator();

  // Get orderbook data
  auto orderbook_opt = processor_->getOrderbookData(symbol_id_);

  if (!orderbook_opt.has_value()) {
    ImGui::Text("Waiting for Orderbook: %s", symbol_name_.c_str());
    ImGui::Text("ID: %u", symbol_id_);
    ImGui::ProgressBar(((frame_count % 100) / 100.0f), ImVec2(-1, 0),
                       "Polling Data Processor...");
    end_panel_window();
    return;
  }

  const auto &orderbook = orderbook_opt.value();

  // Statistics Header
  ImGui::Columns(2, "Stats", false);
  ImGui::Text("Spread: %.4f", orderbook.spread);
  ImGui::NextColumn();
  ImGui::Text("Imbalance: %.2f", orderbook.imbalance);
  ImGui::Columns(1);
  ImGui::Separator();

  // Render Orderbook Ladder
  render_orderbook_ladder(orderbook);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Text("Market Depth (Cumulative)");
  render_market_depth_chart(orderbook);

  end_panel_window();
}

void OrderbookPanel::render_orderbook_ladder(
    const RenderEngine::OrderbookData &orderbook) {

  ImGui::Columns(3, "Ladder", true);
  ImGui::SetColumnWidth(0, 80);
  ImGui::SetColumnWidth(1, 100);
  // Col 2 uses remaining space

  ImGui::Text("Bid Size");
  ImGui::NextColumn();
  ImGui::Text("Price");
  ImGui::NextColumn();
  ImGui::Text("Ask Size");
  ImGui::NextColumn();
  ImGui::Separator();

  // Draw Asks (Sell) - Top down (descending price)
  int ask_count = std::min((int)orderbook.asks.size(), MAX_LEVELS);
  for (int i = ask_count - 1; i >= 0; --i) {
    const auto &level = orderbook.asks[i];
    ImGui::NextColumn(); // Skip Bid Size
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "%.4f", level.price);
    ImGui::NextColumn();
    ImGui::Text("%.4f", level.size);
    ImGui::NextColumn();
  }

  // Best Bid/Ask Highlight or Spread
  ImGui::Separator();
  ImGui::NextColumn();
  ImGui::TextColored(ImVec4(1, 1, 1, 0.5f), "SPREAD: %.4f", orderbook.spread);
  ImGui::NextColumn();
  ImGui::NextColumn();
  ImGui::Separator();

  // Draw Bids (Buy) - (descending price)
  int bid_count = std::min((int)orderbook.bids.size(), MAX_LEVELS);
  for (int i = 0; i < bid_count; ++i) {
    const auto &level = orderbook.bids[i];
    ImGui::Text("%.4f", level.size);
    ImGui::NextColumn();
    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%.4f", level.price);
    ImGui::NextColumn();
    ImGui::NextColumn();
  }

  ImGui::Columns(1);
}

void OrderbookPanel::render_market_depth_chart(
    const RenderEngine::OrderbookData &orderbook) {
  if (orderbook.bids.empty() || orderbook.asks.empty())
    return;

  if (ImPlot::BeginPlot("##Depth", ImVec2(-1, 150), ImPlotFlags_CanvasOnly)) {
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_AutoFit,
                      ImPlotAxisFlags_AutoFit);

    // Handle Bids - Cumulative depth from best bid down
    std::vector<double> bx, by;
    if (!orderbook.bids.empty()) {
      double cumulative_depth = 0.0;
      
      // Add points from worst bid to best bid for increasing X-axis
      for (auto it = orderbook.bids.rbegin(); it != orderbook.bids.rend(); ++it) {
        cumulative_depth += it->size;
        bx.push_back(it->price);
        by.push_back(cumulative_depth);
      }

      // Extend to left for visual completeness
      const auto &worst_bid = orderbook.bids.back();
      bx.insert(bx.begin(), worst_bid.price * 0.995);
      by.insert(by.begin(), cumulative_depth);

      // Add point at best bid with 0 depth for shading
      const auto &best_bid = orderbook.bids[0];
      bx.push_back(best_bid.price);
      by.push_back(0.0);
    }

    ImPlot::SetNextFillStyle(ImVec4(0, 1, 0, 0.2f));
    ImPlot::PlotShaded("Bids", bx.data(), by.data(), (int)bx.size(), 0);

    // Handle Asks - Cumulative depth from best ask up
    std::vector<double> ax, ay;
    if (!orderbook.asks.empty()) {
      double cumulative_depth = 0.0;
      
      // Add points from best ask to worst ask
      for (const auto &ask : orderbook.asks) {
        cumulative_depth += ask.size;
        ax.push_back(ask.price);
        ay.push_back(cumulative_depth);
      }

      // Extend to right for visual completeness
      const auto &worst_ask = orderbook.asks.back();
      ax.push_back(worst_ask.price * 1.005);
      ay.push_back(cumulative_depth);

      // Add point at best ask with 0 depth for shading
      const auto &best_ask = orderbook.asks[0];
      ax.insert(ax.begin(), best_ask.price);
      ay.insert(ay.begin(), 0.0);
    }

    ImPlot::SetNextFillStyle(ImVec4(1, 0, 0, 0.2f));
    ImPlot::PlotShaded("Asks", ax.data(), ay.data(), (int)ax.size(), 0);

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant
