#include "../../include/components/watchlist_panel.hpp"
#include "imgui.h"
#include <algorithm>
#include <cstring>

namespace BTQuant {

WatchlistPanel::WatchlistPanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {}

void WatchlistPanel::update(float dt) {
  update_timer_ += dt;
  if (update_timer_ >= UPDATE_INTERVAL) {
    update_watchlist_data();
    update_timer_ = 0.0f;
  }
}

void WatchlistPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();

  // Symbol Selector Dropdown - add symbols from all available in bridge
  if (bridge_) {
    auto active_symbols = bridge_->getActiveSymbols();
    if (!active_symbols.empty()) {
      ImGui::Text("Add:");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(200);
      if (ImGui::BeginCombo("##SymbolSelector", "Select Symbol...")) {
        for (uint32_t sym_id : active_symbols) {
          std::string sym_name = bridge_->getSymbolName(sym_id);
          std::string exchange = bridge_->getExchangeName(sym_id);
          if (sym_name.empty())
            continue;

          // Check if already in watchlist
          bool already_added = watchlist_.contains(sym_id);

          std::string label = exchange + "/" + sym_name;
          if (already_added) {
            ImGui::PushStyleColor(ImGuiCol_Text,
                                  ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            ImGui::Selectable(label.c_str(), false,
                              ImGuiSelectableFlags_Disabled);
            ImGui::PopStyleColor();
          } else {
            if (ImGui::Selectable(label.c_str())) {
              add_symbol(sym_id, sym_name, exchange);
            }
          }
        }
        ImGui::EndCombo();
      }
      ImGui::SameLine();
      ImGui::Text("(%zu symbols)", active_symbols.size());
    }
  }

  // Filter input
  render_filter_input();
  ImGui::Separator();

  // Table
  if (ImGui::BeginTable("WatchlistTable", 7,
                        ImGuiTableFlags_Resizable | ImGuiTableFlags_Sortable |
                            ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_BordersInnerV)) {

    render_table_header();

    auto filtered_symbols = get_filtered_symbols();
    for (uint32_t symbol_id : filtered_symbols) {
      auto it = watchlist_.find(symbol_id);
      if (it != watchlist_.end()) {
        render_table_row(it->second);
      }
    }

    ImGui::EndTable();
  }

  // Context menu for adding/removing symbols
  if (ImGui::BeginPopupContextWindow()) {
    if (ImGui::MenuItem("Add All Symbols")) {
      if (bridge_) {
        for (uint32_t sym_id : bridge_->getActiveSymbols()) {
          std::string sym_name = bridge_->getSymbolName(sym_id);
          std::string exchange = bridge_->getExchangeName(sym_id);
          if (!sym_name.empty() && !watchlist_.contains(sym_id)) {
            add_symbol(sym_id, sym_name, exchange);
          }
        }
      }
    }
    if (ImGui::MenuItem("Clear Watchlist")) {
      clear_watchlist();
    }
    ImGui::EndPopup();
  }

  end_panel_window();
}

void WatchlistPanel::add_symbol(uint32_t symbol_id, const std::string &symbol,
                                const std::string &exchange) {
  if (watchlist_.find(symbol_id) != watchlist_.end()) {
    return; // Already exists
  }

  WatchlistEntry entry;
  entry.symbol_id = symbol_id;
  entry.symbol = symbol;
  entry.exchange = exchange;
  entry.is_active = true;

  watchlist_[symbol_id] = entry;
  display_order_.push_back(symbol_id);
}

void WatchlistPanel::remove_symbol(uint32_t symbol_id) {
  watchlist_.erase(symbol_id);
  display_order_.erase(
      std::remove(display_order_.begin(), display_order_.end(), symbol_id),
      display_order_.end());
}

void WatchlistPanel::clear_watchlist() {
  watchlist_.clear();
  display_order_.clear();
}

void WatchlistPanel::update_watchlist_data() {
  if (!processor_)
    return;

  for (auto &[symbol_id, entry] : watchlist_) {
    auto analytics = processor_->getSymbolAnalytics(symbol_id);
    if (analytics.symbol_id != 0) {
      entry.price = analytics.last_trade_price;
      entry.vwap = analytics.vwap;
      entry.last_update_ts = analytics.last_trade_time;

      // Calculate 24h change using longest available timeframe candles
      // Note: Ideally we want TF_1DAY or TF_1HOUR, but we use the longest
      // available from the processor as a proxy/placeholder until the processor
      // supports longer history. We'll use the oldest candle from the longest
      // timeframe to estimate change.
      auto candles =
          processor_->getCandles(symbol_id, RenderEngine::TimeFrame::TF_15SEC);
      if (!candles.empty()) {
        const auto &oldest_candle = candles.front();
        const auto &newest_candle = candles.back(); // Or just use current price
        entry.change_24h = calculate_24h_change(newest_candle, oldest_candle);

        // Estimate 24h volume by summing available candles (best effort)
        double total_vol = 0.0;
        for (const auto &c : candles)
          total_vol += c.volume;
        entry.volume_24h = total_vol;
      } else {
        entry.change_24h = 0.0;
        entry.volume_24h = analytics.volume_1m; // Fallback
      }
    }
  }

  // Apply sort after updating data
  sort_watchlist();
}

void WatchlistPanel::render_filter_input() {
  ImGui::Text("Filter:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1);
  ImGui::InputText("##Filter", filter_buffer_, sizeof(filter_buffer_));
}

void WatchlistPanel::render_table_header() {
  ImGui::TableSetupColumn("Symbol",
                          ImGuiTableColumnFlags_DefaultSort |
                              ImGuiTableColumnFlags_WidthFixed,
                          80.0f);
  ImGui::TableSetupColumn("Exchange", ImGuiTableColumnFlags_DefaultSort);
  ImGui::TableSetupColumn("Price",
                          ImGuiTableColumnFlags_DefaultSort |
                              ImGuiTableColumnFlags_PreferSortDescending);
  ImGui::TableSetupColumn("Change %",
                          ImGuiTableColumnFlags_DefaultSort |
                              ImGuiTableColumnFlags_PreferSortDescending);
  ImGui::TableSetupColumn("Volume",
                          ImGuiTableColumnFlags_DefaultSort |
                              ImGuiTableColumnFlags_PreferSortDescending);
  ImGui::TableSetupColumn("VWAP", ImGuiTableColumnFlags_DefaultSort);
  ImGui::TableSetupColumn("Last Update", ImGuiTableColumnFlags_DefaultSort);
  ImGui::TableHeadersRow();

  ImGuiTableSortSpecs *sorts_specs = ImGui::TableGetSortSpecs();
  if (sorts_specs && sorts_specs->SpecsDirty) {
    if (sorts_specs->SpecsCount > 0) {
      const auto &spec = sorts_specs->Specs[0];
      sort_column_ = spec.ColumnIndex;
      sort_ascending_ = (spec.SortDirection == ImGuiSortDirection_Ascending);
      sort_watchlist();
    }
    sorts_specs->SpecsDirty = false;
  }
}

void WatchlistPanel::render_table_row(const WatchlistEntry &entry) {
  ImGui::TableNextRow();

  // Make entire row selectable for click-to-chart
  ImGui::TableSetColumnIndex(0);
  bool is_selected = (entry.symbol_id == selected_symbol_id_);

  ImGui::PushID(static_cast<int>(entry.symbol_id)); // Fix ID conflict

  // Use Selectable spanning all columns
  if (ImGui::Selectable(entry.symbol.c_str(), is_selected,
                        ImGuiSelectableFlags_SpanAllColumns |
                            ImGuiSelectableFlags_AllowDoubleClick)) {
    selected_symbol_id_ = entry.symbol_id;

    // Trigger symbol selection callback (e.g., to open chart)
    if (on_symbol_selected_) {
      on_symbol_selected_(entry.symbol_id, entry.symbol);
    }
  }

  ImGui::PopID();

  // Show tooltip on hover
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Click to view %s chart", entry.symbol.c_str());
    ImGui::Text("Exchange: %s", entry.exchange.c_str());
    ImGui::Text("24h High/Low: %.4f / %.4f", entry.high_24h, entry.low_24h);
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(1);
  ImGui::Text("%s", entry.exchange.c_str());

  ImGui::TableSetColumnIndex(2);
  ImGui::Text("%.4f", entry.price);

  ImGui::TableSetColumnIndex(3);
  ImVec4 change_color = entry.change_24h >= 0 ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)
                                              : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
  ImGui::TextColored(change_color, "%+.2f%%", entry.change_24h);

  ImGui::TableSetColumnIndex(4);
  // Format volume with K/M suffix for readability
  if (entry.volume_24h >= 1e6) {
    ImGui::Text("%.2fM", entry.volume_24h / 1e6);
  } else if (entry.volume_24h >= 1e3) {
    ImGui::Text("%.2fK", entry.volume_24h / 1e3);
  } else {
    ImGui::Text("%.0f", entry.volume_24h);
  }

  ImGui::TableSetColumnIndex(5);
  ImGui::Text("%.4f", entry.vwap);

  ImGui::TableSetColumnIndex(6);
  if (entry.last_update_ts > 0) {
    time_t time =
        entry.last_update_ts / 1000000; // Convert microseconds to seconds
    char time_str[9];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", localtime(&time));
    ImGui::Text("%s", time_str);
  } else {
    ImGui::Text("-");
  }
}

std::vector<uint32_t> WatchlistPanel::get_filtered_symbols() const {
  std::vector<uint32_t> filtered;

  std::string filter(filter_buffer_);
  std::transform(filter.begin(), filter.end(), filter.begin(), ::tolower);

  for (uint32_t symbol_id : display_order_) {
    auto it = watchlist_.find(symbol_id);
    if (it != watchlist_.end()) {
      if (filter.empty()) {
        filtered.push_back(symbol_id);
      } else {
        std::string symbol = it->second.symbol;
        std::transform(symbol.begin(), symbol.end(), symbol.begin(), ::tolower);
        if (symbol.find(filter) != std::string::npos) {
          filtered.push_back(symbol_id);
        }
      }
    }
  }

  return filtered;
}

const char *WatchlistPanel::get_sort_column_name(int column) {
  switch (column) {
  case 0:
    return "Symbol";
  case 1:
    return "Exchange";
  case 2:
    return "Price";
  case 3:
    return "Change %";
  case 4:
    return "Volume";
  case 5:
    return "VWAP";
  case 6:
    return "Last Update";
  default:
    return "Unknown";
  }
}

double WatchlistPanel::calculate_24h_change(
    const RenderEngine::OHLCVCandle &current,
    const RenderEngine::OHLCVCandle &old) const {
  if (old.close == 0.0)
    return 0.0;
  // Using close price of the candles
  return ((current.close - old.close) / old.close) * 100.0;
}

void WatchlistPanel::sort_watchlist() {
  std::sort(display_order_.begin(), display_order_.end(),
            [this](uint32_t a_id, uint32_t b_id) {
              const auto &a = watchlist_.at(a_id);
              const auto &b = watchlist_.at(b_id);

              bool result = false;
              switch (sort_column_) {
              case 0: // Symbol
                result = a.symbol < b.symbol;
                break;
              case 1: // Exchange
                result = a.exchange < b.exchange;
                break;
              case 2: // Price
                result = a.price < b.price;
                break;
              case 3: // Change %
                result = a.change_24h < b.change_24h;
                break;
              case 4: // Volume
                result = a.volume_24h < b.volume_24h;
                break;
              case 5: // VWAP
                result = a.vwap < b.vwap;
                break;
              case 6: // Last Update
                result = a.last_update_ts < b.last_update_ts;
                break;
              default:
                result = a.symbol < b.symbol;
              }
              return sort_ascending_ ? result : !result;
            });
}

} // namespace BTQuant