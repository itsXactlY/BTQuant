#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "imgui.h"

namespace BTQuant {

// State for symbol selection UI
struct SymbolSelectorState {
  std::string selected_symbol = "BTC-USDT";
  std::string selected_exchange = "Binance";
  std::vector<std::string> available_symbols;
  std::vector<std::string> available_exchanges;
  RenderEngine::TimeFrame selected_timeframe = RenderEngine::TimeFrame::TF_1SEC;
  bool needs_refresh = true;
  int selected_symbol_idx = 0;
  int selected_exchange_idx = 0;
  int selected_timeframe_idx = 0;
};

// Unified symbol selector component for all charts
class SymbolSelector {
 public:
  // Renders combo boxes for exchange, symbol, and timeframe
  // Returns true if selection changed
  bool render(SymbolSelectorState& state);

  // Refresh available symbols from the data bridge
  void refresh_symbols(SymbolSelectorState& state, std::shared_ptr<HotSpineDataBridge> bridge,
                       std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  // Get timeframe display names
  static const char* get_timeframe_name(RenderEngine::TimeFrame tf);
  static RenderEngine::TimeFrame get_timeframe_from_index(int index);
  static int get_timeframe_count();

 private:
  // Default symbols for when no live data available
  static const std::vector<std::string>& get_default_symbols();
  static const std::vector<std::string>& get_default_exchanges();
};

}  // namespace BTQuant
