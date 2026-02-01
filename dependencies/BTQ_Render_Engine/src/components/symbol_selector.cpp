/**
 * @file symbol_selector.cpp
 * @brief Symbol Selector Component (C++23/26)
 *
 * Multi-asset symbol selection UI with modern C++ features:
 * - [[nodiscard]] attributes
 * - constexpr constants
 * - std::array for fixed-size collections
 *
 * @version 2.0.0 (C++23/26)
 */

#include "../../include/components/symbol_selector.hpp"

#include <algorithm>
#include <array>

#include "imgui.h"

namespace BTQuant {

// Constants
namespace {
constexpr int TIMEFRAME_COUNT = 21;
constexpr std::array<const char*, TIMEFRAME_COUNT> TIMEFRAME_NAMES = {
    "1ms", "10ms", "100ms", "500ms", "1s", "3s", "5s", "15s", "30s", "1m", "2m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"};
}  // namespace

bool SymbolSelector::render(SymbolSelectorState& state) {
  bool changed = false;

  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));

  // Exchange dropdown
  if (!state.available_exchanges.empty()) {
    if (ImGui::BeginCombo("Exchange", state.selected_exchange.c_str())) {
      for (int i = 0; i < static_cast<int>(state.available_exchanges.size()); ++i) {
        bool is_selected = (state.selected_exchange_idx == i);
        if (ImGui::Selectable(state.available_exchanges[i].c_str(), is_selected)) {
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
      for (int i = 0; i < static_cast<int>(state.available_symbols.size()); ++i) {
        bool is_selected = (state.selected_symbol_idx == i);
        if (ImGui::Selectable(state.available_symbols[i].c_str(), is_selected)) {
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

  // Timeframe dropdown (Extended to include higher timeframes for multi-timeframe analysis)
  const char* timeframes[] = {"1ms", "10ms", "100ms", "500ms", "1s", "3s", "5s", "15s", "30s", "1m", "2m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"};

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

void SymbolSelector::refresh_symbols(SymbolSelectorState& state,
                                     std::shared_ptr<HotSpineDataBridge> bridge,
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
  auto it = std::find(state.available_symbols.begin(), state.available_symbols.end(),
                      state.selected_symbol);
  if (it == state.available_symbols.end() && !state.available_symbols.empty()) {
    state.selected_symbol = state.available_symbols[0];
    state.selected_symbol_idx = 0;
  } else if (it != state.available_symbols.end()) {
    state.selected_symbol_idx = static_cast<int>(it - state.available_symbols.begin());
  }

  // Ensure selected exchange is in the list
  auto ex_it = std::find(state.available_exchanges.begin(), state.available_exchanges.end(),
                         state.selected_exchange);
  if (ex_it == state.available_exchanges.end() && !state.available_exchanges.empty()) {
    state.selected_exchange = state.available_exchanges[0];
    state.selected_exchange_idx = 0;
  } else if (ex_it != state.available_exchanges.end()) {
    state.selected_exchange_idx = static_cast<int>(ex_it - state.available_exchanges.begin());
  }

  state.needs_refresh = false;
  (void)processor;  // Unused for now
}

const char* SymbolSelector::get_timeframe_name(RenderEngine::TimeFrame tf) {
  switch (tf) {
    case RenderEngine::TimeFrame::TF_1MS:
      return "1ms";
    case RenderEngine::TimeFrame::TF_10MS:
      return "10ms";
    case RenderEngine::TimeFrame::TF_100MS:
      return "100ms";
    case RenderEngine::TimeFrame::TF_500MS:
      return "500ms";
    case RenderEngine::TimeFrame::TF_1SEC:
      return "1s";
    case RenderEngine::TimeFrame::TF_3SEC:
      return "3s";
    case RenderEngine::TimeFrame::TF_5SEC:
      return "5s";
    case RenderEngine::TimeFrame::TF_15SEC:
      return "15s";
    case RenderEngine::TimeFrame::TF_30SEC:
      return "30s";
    case RenderEngine::TimeFrame::TF_1MIN:
      return "1m";
    case RenderEngine::TimeFrame::TF_2MIN:
      return "2m";
    case RenderEngine::TimeFrame::TF_5MIN:
      return "5m";
    case RenderEngine::TimeFrame::TF_15MIN:
      return "15m";
    case RenderEngine::TimeFrame::TF_30MIN:
      return "30m";
    case RenderEngine::TimeFrame::TF_1HOUR:
      return "1h";
    case RenderEngine::TimeFrame::TF_2HOUR:
      return "2h";
    case RenderEngine::TimeFrame::TF_4HOUR:
      return "4h";
    case RenderEngine::TimeFrame::TF_6HOUR:
      return "6h";
    case RenderEngine::TimeFrame::TF_12HOUR:
      return "12h";
    case RenderEngine::TimeFrame::TF_1DAY:
      return "1d";
    case RenderEngine::TimeFrame::TF_1WEEK:
      return "1w";
    default:
      return "1s";
  }
}

RenderEngine::TimeFrame SymbolSelector::get_timeframe_from_index(int index) {
  switch (index) {
    case 0: return RenderEngine::TimeFrame::TF_1MS;
    case 1: return RenderEngine::TimeFrame::TF_10MS;
    case 2: return RenderEngine::TimeFrame::TF_100MS;
    case 3: return RenderEngine::TimeFrame::TF_500MS;
    case 4: return RenderEngine::TimeFrame::TF_1SEC;
    case 5: return RenderEngine::TimeFrame::TF_3SEC;
    case 6: return RenderEngine::TimeFrame::TF_5SEC;
    case 7: return RenderEngine::TimeFrame::TF_15SEC;
    case 8: return RenderEngine::TimeFrame::TF_30SEC;
    case 9: return RenderEngine::TimeFrame::TF_1MIN;
    case 10: return RenderEngine::TimeFrame::TF_2MIN;
    case 11: return RenderEngine::TimeFrame::TF_5MIN;
    case 12: return RenderEngine::TimeFrame::TF_15MIN;
    case 13: return RenderEngine::TimeFrame::TF_30MIN;
    case 14: return RenderEngine::TimeFrame::TF_1HOUR;
    case 15: return RenderEngine::TimeFrame::TF_2HOUR;
    case 16: return RenderEngine::TimeFrame::TF_4HOUR;
    case 17: return RenderEngine::TimeFrame::TF_6HOUR;
    case 18: return RenderEngine::TimeFrame::TF_12HOUR;
    case 19: return RenderEngine::TimeFrame::TF_1DAY;
    case 20: return RenderEngine::TimeFrame::TF_1WEEK;
    default: return RenderEngine::TimeFrame::TF_1SEC;
  }
}

int SymbolSelector::get_timeframe_count() { return 21; }

const std::vector<std::string>& SymbolSelector::get_default_symbols() {
  static std::vector<std::string> defaults = {"BTC-USDT",  "ETH-USDT",  "BNB-USDT",   "SOL-USDT",
                                              "XRP-USDT",  "DOGE-USDT", "ADA-USDT",   "AVAX-USDT",
                                              "DOT-USDT",  "LINK-USDT", "MATIC-USDT", "UNI-USDT",
                                              "ATOM-USDT", "LTC-USDT",  "ETC-USDT"};
  return defaults;
}

const std::vector<std::string>& SymbolSelector::get_default_exchanges() {
  static std::vector<std::string> defaults = {"Binance", "OKX", "Bybit", "Coinbase", "Kraken"};
  return defaults;
}

}  // namespace BTQuant
