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

  if (bridge_) {
    auto trades = bridge_->getTradeBuffer();
    // Simple linear scan. In production, use monotonic index or similar.
    for (const auto &trade : trades) {
      // Skip potential empty slots
      if (trade.ts_local == 0)
        continue;

      if (trade.ts_local > last_processed_trade_ts_ &&
          trade.symbol_id == symbol_id_) {
        auto &vol = volume_profile_[trade.price];
        if (trade.side == 0)
          vol.bought += trade.size; // Buy
        else
          vol.sold += trade.size; // Sell

        if (trade.ts_local > last_processed_trade_ts_) {
          last_processed_trade_ts_ = trade.ts_local;
        }
      }
    }
  }
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
            config_.title = symbol_name_ + " DOM";
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
    const auto &colors = ThemeManager::getInstance().getColors();
    ImGui::TextColored(colors.accent_red, "No active symbols detected in SHM!");
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

  // Calculate max volume for relative scaling
  double max_vol = 1.0;
  for (const auto &level : orderbook.bids)
    max_vol = std::max(max_vol, level.size);
  for (const auto &level : orderbook.asks)
    max_vol = std::max(max_vol, level.size);
  if (max_vol < 1.0)
    max_vol = 1.0;

  // Use Table instead of Columns for modern layout (C++26 style UI)
  if (ImGui::BeginTable("OrderbookTable", 7,
                        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_SizingStretchSame)) {

    // Setup Columns
    ImGui::TableSetupColumn("Bid", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Sold", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 80);
    ImGui::TableSetupColumn("Bought", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Ask", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Delta", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Vol", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableHeadersRow();

    const auto &colors = ThemeManager::getInstance().getColors();

    // Render Asks (Sell) - Top down
    int ask_count = std::min((int)orderbook.asks.size(), MAX_LEVELS);
    for (int i = ask_count - 1; i >= 0; --i) {
      const auto &level = orderbook.asks[i];
      ImGui::TableNextRow();
      ImGui::PushID(i); // Unique ID for this row/side

      // 1. Bid (Empty)
      ImGui::TableSetColumnIndex(0);

      // 2. Sold (Accumulated)
      ImGui::TableSetColumnIndex(1);
      if (volume_profile_.contains(level.price)) {
        double sold = volume_profile_[level.price].sold;
        if (sold > 0)
          ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
      }

      // 3. Price
      ImGui::TableSetColumnIndex(2);
      // Center Price text
      float cursor_check =
          ImGui::GetCursorPosX() +
          (ImGui::GetContentRegionAvail().x -
           ImGui::CalcTextSize(std::to_string(level.price).c_str()).x) *
              0.5f;
      ImGui::SetCursorPosX(cursor_check);

      ImGui::Selectable(std::format("{:.2f}", level.price).c_str(), false,
                        ImGuiSelectableFlags_SpanAllColumns);
      if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        ImGui::SetDragDropPayload("PRICE_LEVEL", &level.price, sizeof(double));
        ImGui::Text("Price: %.2f", level.price);
        ImGui::EndDragDropSource();
      }
      ImGui::SameLine();
      ImGui::TextColored(colors.accent_red, "%.2f", level.price);

      // 4. Bought (Accumulated)
      ImGui::TableSetColumnIndex(3);
      if (volume_profile_.contains(level.price)) {
        double bought = volume_profile_[level.price].bought;
        if (bought > 0)
          ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
      }

      // 5. Ask Size (with Bar and Heatmap)
      ImGui::TableSetColumnIndex(4);
      {
        float width = ImGui::GetContentRegionAvail().x;
        float bar_width = width * (float)(level.size / max_vol);
        ImVec2 pos = ImGui::GetCursorScreenPos();

        // Liquidity Heatmap Background for the whole row
        float intensity = std::clamp((float)(level.size / max_vol), 0.0f, 1.0f);
        if (intensity > 0.05f) {
          ImU32 bg_color =
              ImGui::GetColorU32(ImVec4(1.0f, 0.5f, 0.0f, intensity * 0.3f));
          ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, bg_color);
        }

        ImGui::GetWindowDrawList()->AddRectFilled(
            pos,
            ImVec2(pos.x + bar_width,
                   pos.y + ImGui::GetTextLineHeightWithSpacing()),
            ImGui::GetColorU32(ImVec4(colors.accent_red.x, colors.accent_red.y,
                                      colors.accent_red.z, 0.2f)));
        ImGui::Text("%.4f", level.size);
      }

      // 6. Delta (Accumulated)
      ImGui::TableSetColumnIndex(5);
      if (volume_profile_.contains(level.price)) {
        const auto &vol = volume_profile_[level.price];
        double delta = vol.bought - vol.sold;
        if (delta != 0) {
          ImVec4 color =
              delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);
          ImGui::TextColored(color, "%+.0f", delta);
        }
      }

      // 7. Volume (Accumulated)
      ImGui::TableSetColumnIndex(6);
      if (volume_profile_.contains(level.price)) {
        const auto &vol = volume_profile_[level.price];
        double total = vol.bought + vol.sold;
        if (total > 0)
          ImGui::Text("%.0f", total);
      }

      ImGui::PopID();
    }

    // Spread Row
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(2);
    ImGui::TextColored(ImVec4(1, 1, 1, 0.5f), "--- %.1f ---", orderbook.spread);

    // Render Bids (Buy)
    int bid_count = std::min((int)orderbook.bids.size(), MAX_LEVELS);
    for (int i = 0; i < bid_count; ++i) {
      const auto &level = orderbook.bids[i];
      ImGui::TableNextRow();
      ImGui::PushID(i + 1000); // Offset to ensure uniqueness from Asks

      // 1. Bid Size (with Bar)
      ImGui::TableSetColumnIndex(0);
      {
        // Draw bar from right to left? Standard is Left or Right aligned.
        // Image 1 implies Right aligned for Bid? No, standard is bars grow from
        // center spine (Price). But here Columns are separated. Let's do
        // Standard Left-to-Right for now, or Right-to-Left if it looks better
        // next to Price. Let's do Right-to-Left for Bid to "point" to Price.
        float width = ImGui::GetContentRegionAvail().x;
        float bar_width = width * (float)(level.size / max_vol);
        ImVec2 pos = ImGui::GetCursorScreenPos();

        // Liquidity Heatmap Background for the whole row
        float intensity = std::clamp((float)(level.size / max_vol), 0.0f, 1.0f);
        if (intensity > 0.05f) {
          ImU32 bg_color =
              ImGui::GetColorU32(ImVec4(0.0f, 0.6f, 1.0f, intensity * 0.3f));
          ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, bg_color);
        }

        ImGui::GetWindowDrawList()->AddRectFilled(
            ImVec2(pos.x + width - bar_width, pos.y),
            ImVec2(pos.x + width,
                   pos.y + ImGui::GetTextLineHeightWithSpacing()),
            ImGui::GetColorU32(ImVec4(colors.accent_green.x,
                                      colors.accent_green.y,
                                      colors.accent_green.z, 0.2f)));

        // Text Right Aligned
        auto text = std::format("{:.4f}", level.size);
        float text_width = ImGui::CalcTextSize(text.c_str()).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + width - text_width);
        ImGui::TextUnformatted(text.c_str());
      }

      // 2. Sold
      ImGui::TableSetColumnIndex(1);
      if (volume_profile_.contains(level.price)) {
        double sold = volume_profile_[level.price].sold;
        if (sold > 0)
          ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
      }

      // 3. Price
      ImGui::TableSetColumnIndex(2);
      float cursor_check =
          ImGui::GetCursorPosX() +
          (ImGui::GetContentRegionAvail().x -
           ImGui::CalcTextSize(std::to_string(level.price).c_str()).x) *
              0.5f;
      ImGui::SetCursorPosX(cursor_check);

      ImGui::Selectable(std::format("{:.2f}", level.price).c_str(), false,
                        ImGuiSelectableFlags_SpanAllColumns);
      if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        ImGui::SetDragDropPayload("PRICE_LEVEL", &level.price, sizeof(double));
        ImGui::Text("Price: %.2f", level.price);
        ImGui::EndDragDropSource();
      }
      ImGui::SameLine();
      ImGui::TextColored(colors.accent_green, "%.2f",
                         level.price); // Green for Bid Price

      // 4. Bought
      ImGui::TableSetColumnIndex(3);
      if (volume_profile_.contains(level.price)) {
        double bought = volume_profile_[level.price].bought;
        if (bought > 0)
          ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
      }

      // 5. Ask (Empty)
      ImGui::TableSetColumnIndex(4);

      // 6. Delta
      ImGui::TableSetColumnIndex(5);
      if (volume_profile_.contains(level.price)) {
        const auto &vol = volume_profile_[level.price];
        double delta = vol.bought - vol.sold;
        if (delta != 0) {
          ImVec4 color =
              delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);
          ImGui::TextColored(color, "%+.0f", delta);
        }
      }

      // 7. Vol
      ImGui::TableSetColumnIndex(6);
      if (volume_profile_.contains(level.price)) {
        const auto &vol = volume_profile_[level.price];
        double total = vol.bought + vol.sold;
        if (total > 0)
          ImGui::Text("%.0f", total);
      }

      ImGui::PopID();
    }

    ImGui::EndTable();
  }
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
      for (auto it = orderbook.bids.rbegin(); it != orderbook.bids.rend();
           ++it) {
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

    const auto &colors = ThemeManager::getInstance().getColors();
    ImPlot::SetNextFillStyle(colors.accent_green);
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

    ImPlot::SetNextFillStyle(colors.accent_red);
    ImPlot::PlotShaded("Asks", ax.data(), ay.data(), (int)ax.size(), 0);

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant
