#include "../../include/components/hierarchical_selector.hpp"
#include "../../include/symbol_registry.hpp"
#include <algorithm>
#include <imgui.h>

namespace BTQuant {

bool HierarchicalSelector::render(HierarchicalSelectorState &state) {
  bool changed = false;

  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 4));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));

  // Exchange dropdown
  if (!state.available_exchanges.empty()) {
    const char *current_exchange = state.selected_exchange.empty()
                                       ? "Select Exchange"
                                       : state.selected_exchange.c_str();

    if (ImGui::BeginCombo("Exchange", current_exchange)) {
      for (int i = 0; i < static_cast<int>(state.available_exchanges.size());
           ++i) {
        bool is_selected = (state.selected_exchange_idx == i);
        if (ImGui::Selectable(state.available_exchanges[i].c_str(),
                              is_selected)) {
          state.selected_exchange_idx = i;
          state.selected_exchange = state.available_exchanges[i];
          state.needs_refresh = true; // Trigger symbol list refresh
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

  // Symbol dropdown (filtered by selected exchange)
  if (!state.available_symbols.empty() && !state.selected_exchange.empty()) {
    const char *current_symbol = state.selected_symbol.empty()
                                     ? "Select Symbol"
                                     : state.selected_symbol.c_str();

    if (ImGui::BeginCombo("Symbol", current_symbol)) {
      for (int i = 0; i < static_cast<int>(state.available_symbols.size());
           ++i) {
        bool is_selected = (state.selected_symbol_idx == i);
        const auto &sym_info = state.available_symbols[i];
        if (ImGui::Selectable(sym_info.symbol.c_str(), is_selected)) {
          state.selected_symbol_idx = i;
          state.selected_symbol = sym_info.symbol;
          state.selected_symbol_id = sym_info.id;
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

  // Chart dropdown (shows available charts for selected symbol)
  if (!state.chart_ids.empty()) {
    char chart_label[32];
    snprintf(chart_label, sizeof(chart_label), "Chart #%u",
             state.selected_chart_id);

    if (ImGui::BeginCombo("Chart", chart_label)) {
      for (int i = 0; i < static_cast<int>(state.chart_ids.size()); ++i) {
        bool is_selected = (state.selected_chart_idx == i);
        uint32_t chart_id = state.chart_ids[i];
        snprintf(chart_label, sizeof(chart_label), "Chart #%u", chart_id);

        if (ImGui::Selectable(chart_label, is_selected)) {
          state.selected_chart_idx = i;
          state.selected_chart_id = chart_id;
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

  // Timeframe dropdown (1ms-15sec only)
  const char *timeframes[] = {"1ms", "10ms", "100ms", "500ms",
                              "1s",  "3s",   "5s",    "15s"};

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

void HierarchicalSelector::refresh_data(HierarchicalSelectorState &state,
                                        ChartManager *chart_manager) {
  // Get exchanges from SymbolRegistry (reads /dev/shm/btquant_symbols.json)
  state.available_exchanges = SymbolRegistry::instance().get_exchanges();

  // If exchange selected, get symbols for that exchange
  if (!state.selected_exchange.empty()) {
    state.available_symbols = SymbolRegistry::instance().get_exchange_symbols(
        state.selected_exchange);
  } else if (!state.available_exchanges.empty()) {
    // Auto-select first exchange if none selected
    state.selected_exchange = state.available_exchanges[0];
    state.selected_exchange_idx = 0;
    state.available_symbols = SymbolRegistry::instance().get_exchange_symbols(
        state.selected_exchange);
  }

  // Auto-select first symbol if none selected
  if (state.selected_symbol.empty() && !state.available_symbols.empty()) {
    state.selected_symbol = state.available_symbols[0].symbol;
    state.selected_symbol_id = state.available_symbols[0].id;
    state.selected_symbol_idx = 0;
  }

  // Get charts for selected symbol from ChartManager
  if (chart_manager && state.selected_symbol_id > 0) {
    auto charts = chart_manager->get_charts_for_symbol(state.selected_symbol);
    state.chart_ids.clear();
    for (const auto &chart : charts) {
      state.chart_ids.push_back(chart.chart_id);
    }

    // Auto-select first chart if none selected
    if (!state.chart_ids.empty() && state.selected_chart_id == 0) {
      state.selected_chart_id = state.chart_ids[0];
      state.selected_chart_idx = 0;
    }
  }

  state.needs_refresh = false;
}

const char *
HierarchicalSelector::get_timeframe_name(RenderEngine::TimeFrame tf) {
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
  default:
    return "1s";
  }
}

RenderEngine::TimeFrame
HierarchicalSelector::get_timeframe_from_index(int index) {
  switch (index) {
  case 0:
    return RenderEngine::TimeFrame::TF_1MS;
  case 1:
    return RenderEngine::TimeFrame::TF_10MS;
  case 2:
    return RenderEngine::TimeFrame::TF_100MS;
  case 3:
    return RenderEngine::TimeFrame::TF_500MS;
  case 4:
    return RenderEngine::TimeFrame::TF_1SEC;
  case 5:
    return RenderEngine::TimeFrame::TF_3SEC;
  case 6:
    return RenderEngine::TimeFrame::TF_5SEC;
  case 7:
    return RenderEngine::TimeFrame::TF_15SEC;
  default:
    return RenderEngine::TimeFrame::TF_1SEC;
  }
}

int HierarchicalSelector::get_timeframe_count() { return 8; }

} // namespace BTQuant
