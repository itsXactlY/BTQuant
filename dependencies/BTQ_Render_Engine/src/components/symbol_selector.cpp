#include "../../include/components/symbol_selector.hpp"
#include "imgui.h"
#include <algorithm>

namespace BTQuant {

bool SymbolSelector::render(SymbolSelectorState &state) {
  bool changed = false;

  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));

  // Exchange dropdown
  if (!state.available_exchanges.empty()) {
    if (ImGui::BeginCombo("Exchange", state.selected_exchange.c_str())) {
      for (int i = 0; i < static_cast<int>(state.available_exchanges.size());
           ++i) {
        bool is_selected = (state.selected_exchange_idx == i);
        if (ImGui::Selectable(state.available_exchanges[i].c_str(),
                              is_selected)) {
          state.selected_exchange_idx = i;
          state.selected_exchange = state.available_exchanges[i];
          state.needs_refresh = true;
          changed = true;
        }
        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
  }

  ImGui::SameLine();

  // Symbol dropdown
  if (!state.available_symbols.empty()) {
    if (ImGui::BeginCombo("Symbol", state.selected_symbol.c_str())) {
      for (int i = 0; i < static_cast<int>(state.available_symbols.size());
           ++i) {
        bool is_selected = (state.selected_symbol_idx == i);
        if (ImGui::Selectable(state.available_symbols[i].c_str(),
                              is_selected)) {
          state.selected_symbol_idx = i;
          state.selected_symbol = state.available_symbols[i];
          changed = true;
        }
        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
  }

  ImGui::SameLine();

  // Timeframe dropdown
  const char *timeframes[] = {"1m", "5m", "15m", "1h",  "4h",    "1d",
                              "1s", "5s", "15s", "30s", "500ms", "100ms"};

  if (ImGui::BeginCombo("TF", timeframes[state.selected_timeframe_idx])) {
    for (int i = 0; i < get_timeframe_count(); ++i) {
      bool is_selected = (state.selected_timeframe_idx == i);
      if (ImGui::Selectable(timeframes[i], is_selected)) {
        state.selected_timeframe_idx = i;
        state.selected_timeframe = get_timeframe_from_index(i);
        changed = true;
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  ImGui::PopStyleVar(2);

  return changed;
}

void SymbolSelector::refresh_symbols(
    SymbolSelectorState &state, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor) {
  state.available_exchanges = get_default_exchanges();
  state.available_symbols.clear();

  // Try to get active symbols from the bridge
  if (bridge) {
    auto active_ids = bridge->getActiveSymbols();
    for (auto id : active_ids) {
      std::string sym = bridge->getSymbolName(id);
      if (!sym.empty()) {
        state.available_symbols.push_back(sym);
      }
    }
  }

  // If no live symbols, use defaults
  if (state.available_symbols.empty()) {
    state.available_symbols = get_default_symbols();
  }

  // Ensure selected symbol is in the list
  auto it = std::find(state.available_symbols.begin(),
                      state.available_symbols.end(), state.selected_symbol);
  if (it == state.available_symbols.end() && !state.available_symbols.empty()) {
    state.selected_symbol = state.available_symbols[0];
    state.selected_symbol_idx = 0;
  } else if (it != state.available_symbols.end()) {
    state.selected_symbol_idx =
        static_cast<int>(it - state.available_symbols.begin());
  }

  // Ensure selected exchange is in the list
  auto ex_it =
      std::find(state.available_exchanges.begin(),
                state.available_exchanges.end(), state.selected_exchange);
  if (ex_it == state.available_exchanges.end() &&
      !state.available_exchanges.empty()) {
    state.selected_exchange = state.available_exchanges[0];
    state.selected_exchange_idx = 0;
  } else if (ex_it != state.available_exchanges.end()) {
    state.selected_exchange_idx =
        static_cast<int>(ex_it - state.available_exchanges.begin());
  }

  state.needs_refresh = false;
  (void)processor; // Unused for now
}

const char *SymbolSelector::get_timeframe_name(RenderEngine::TimeFrame tf) {
  switch (tf) {
  case RenderEngine::TimeFrame::TF_1MIN:
    return "1m";
  case RenderEngine::TimeFrame::TF_5MIN:
    return "5m";
  case RenderEngine::TimeFrame::TF_15MIN:
    return "15m";
  case RenderEngine::TimeFrame::TF_1HOUR:
    return "1h";
  case RenderEngine::TimeFrame::TF_4HOUR:
    return "4h";
  case RenderEngine::TimeFrame::TF_1DAY:
    return "1d";
  case RenderEngine::TimeFrame::TF_1SEC:
    return "1s";
  case RenderEngine::TimeFrame::TF_5SEC:
    return "5s";
  case RenderEngine::TimeFrame::TF_15SEC:
    return "15s";
  case RenderEngine::TimeFrame::TF_30SEC:
    return "30s";
  case RenderEngine::TimeFrame::TF_500MS:
    return "500ms";
  case RenderEngine::TimeFrame::TF_100MS:
    return "100ms";
  default:
    return "1m";
  }
}

RenderEngine::TimeFrame SymbolSelector::get_timeframe_from_index(int index) {
  switch (index) {
  case 0:
    return RenderEngine::TimeFrame::TF_1MIN;
  case 1:
    return RenderEngine::TimeFrame::TF_5MIN;
  case 2:
    return RenderEngine::TimeFrame::TF_15MIN;
  case 3:
    return RenderEngine::TimeFrame::TF_1HOUR;
  case 4:
    return RenderEngine::TimeFrame::TF_4HOUR;
  case 5:
    return RenderEngine::TimeFrame::TF_1DAY;
  case 6:
    return RenderEngine::TimeFrame::TF_1SEC;
  case 7:
    return RenderEngine::TimeFrame::TF_5SEC;
  case 8:
    return RenderEngine::TimeFrame::TF_15SEC;
  case 9:
    return RenderEngine::TimeFrame::TF_30SEC;
  case 10:
    return RenderEngine::TimeFrame::TF_500MS;
  case 11:
    return RenderEngine::TimeFrame::TF_100MS;
  default:
    return RenderEngine::TimeFrame::TF_1MIN;
  }
}

int SymbolSelector::get_timeframe_count() { return 12; }

const std::vector<std::string> &SymbolSelector::get_default_symbols() {
  static std::vector<std::string> defaults = {
      "BTC-USDT",   "ETH-USDT", "BNB-USDT",  "SOL-USDT", "XRP-USDT",
      "DOGE-USDT",  "ADA-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT",
      "MATIC-USDT", "UNI-USDT", "ATOM-USDT", "LTC-USDT", "ETC-USDT"};
  return defaults;
}

const std::vector<std::string> &SymbolSelector::get_default_exchanges() {
  static std::vector<std::string> defaults = {"Binance", "OKX", "Bybit",
                                              "Coinbase", "Kraken"};
  return defaults;
}

} // namespace BTQuant
