#pragma once

#include <imgui.h>

#include <memory>
#include <string>
#include <vector>


#include "../market_data_processor.hpp"
#include "../symbol_registry.hpp"
#include "chart_manager.hpp"

namespace BTQuant {

// State for hierarchical exchange -> symbol -> chart selection
struct HierarchicalSelectorState {
  // Exchange level (populated from SymbolRegistry)
  std::vector<std::string> available_exchanges;
  std::string selected_exchange = "";
  int selected_exchange_idx = -1;

  // Symbol level (filtered by exchange via
  // SymbolRegistry::get_exchange_symbols)
  std::vector<SymbolInfo> available_symbols;
  std::string selected_symbol = "";
  int selected_symbol_idx = -1;
  uint32_t selected_symbol_id = 0;

  // Chart level (multiple charts per symbol managed by ChartManager)
  std::vector<uint32_t> chart_ids;
  uint32_t selected_chart_id = 0;
  int selected_chart_idx = -1;

  // Timeframe (1ms-15sec only)
  RenderEngine::TimeFrame selected_timeframe = RenderEngine::TimeFrame::TF_1SEC;
  int selected_timeframe_idx = 4;  // Default to 1sec (index 4)

  bool needs_refresh = true;
};

// Hierarchical selector widget: Exchange -> Symbol -> Chart + Timeframe
// Uses SymbolRegistry singleton for all exchange/symbol data from
// /dev/shm/btquant_symbols.json
class HierarchicalSelector {
 public:
  // Returns true if selection changed
  bool render(HierarchicalSelectorState& state);

  // Refresh exchanges/symbols from SymbolRegistry (reads
  // /dev/shm/btquant_symbols.json)
  void refresh_data(HierarchicalSelectorState& state, ChartManager* chart_manager = nullptr);

 private:
  // Timeframe helpers
  static const char* get_timeframe_name(RenderEngine::TimeFrame tf);
  static RenderEngine::TimeFrame get_timeframe_from_index(int index);
  static int get_timeframe_count();
};

}  // namespace BTQuant
