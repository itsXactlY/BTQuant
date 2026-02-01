#include "../../include/components/dashboard_controls.hpp"

#include <imgui.h>
#include <imgui_internal.h>  // For ImGui::InputTextWithHint

#include <iostream>
#include <algorithm>
#include <cctype>

#include "../../include/components/panel_manager.hpp"
#include "../../include/ui/ui_base.hpp"
#include "../../include/symbol_registry.hpp"
#include "../../include/hotspine_data_bridge.hpp"

namespace BTQuant {

DashboardControls::DashboardControls(PanelManager* panel_manager)
    : UIComponent({0, 0}, {0, 0}), panel_manager_(panel_manager) {
  if (!panel_manager_) {
    std::cerr << "[DashboardControls] Error: PanelManager is null" << std::endl;
  }
}

void DashboardControls::render_gui() {
  render_dashboard_controls();
}

void DashboardControls::render_dashboard_controls() {
  // Floating Dashboard Controls Panel
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(350, 500), ImGuiCond_FirstUseEver);

  if (ImGui::Begin("Dashboard Controls", nullptr)) {
    ImGui::Text("Trading Dashboard Controls");
    ImGui::Separator();

    // Symbol selection section
    if (ImGui::CollapsingHeader("Symbol Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Refresh symbols button
      if (ImGui::Button("Refresh Symbols")) {
        needs_refresh_ = true;
      }

      ImGui::SameLine();

      // Static buffer for symbol search input
      static char symbol_search_buffer[256] = "";

      // Load symbols if needed
      if (needs_refresh_ || !symbols_loaded_) {
        if (panel_manager_) {
          // Get all available symbols from the symbol registry
          all_symbols_.clear();

          // Get symbols from the symbol registry
          auto all_symbol_infos = SymbolRegistry::instance().get_all_symbols();
          for (const auto& symbol_info : all_symbol_infos) {
            all_symbols_.push_back(symbol_info.symbol);
          }

          // Also get active symbols from the bridge if possible through the chart manager
          auto chart_manager = panel_manager_->get_chart_manager();
          if (chart_manager) {
            // Access the bridge through the chart manager
            auto bridge = chart_manager->get_bridge();
            if (bridge) {
              auto active_symbols = bridge->getActiveSymbols();
              for (auto symbol_id : active_symbols) {
                std::string symbol_name = bridge->getSymbolName(symbol_id);
                if (!symbol_name.empty()) {
                  // Check if symbol is already in the list
                  bool found = false;
                  for (const auto& existing_symbol : all_symbols_) {
                    if (existing_symbol == symbol_name) {
                      found = true;
                      break;
                    }
                  }
                  if (!found) {
                    all_symbols_.push_back(symbol_name);
                  }
                }
              }
            }
          }

          // Sort symbols alphabetically
          std::sort(all_symbols_.begin(), all_symbols_.end());

          // Update filtered symbols
          filtered_symbols_ = all_symbols_;

          symbols_loaded_ = true;
          needs_refresh_ = false;

          // Update the search buffer to reflect current input
          strncpy(symbol_search_buffer, symbol_input_buffer_.c_str(), sizeof(symbol_search_buffer) - 1);
          symbol_search_buffer[sizeof(symbol_search_buffer) - 1] = '\0';
        }
      }

      // Search input for symbol selection
      ImGui::Text("Select Symbol:");
      if (ImGui::InputTextWithHint("##symbol_search", "Search symbols...", symbol_search_buffer, sizeof(symbol_search_buffer))) {
        symbol_input_buffer_ = std::string(symbol_search_buffer);
        // Filter symbols based on search input
        filtered_symbols_.clear();

        std::string search_lower = symbol_input_buffer_;
        std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);

        for (const auto& symbol : all_symbols_) {
          std::string symbol_lower = symbol;
          std::transform(symbol_lower.begin(), symbol_lower.end(), symbol_lower.begin(), ::tolower);

          if (symbol_lower.find(search_lower) != std::string::npos) {
            filtered_symbols_.push_back(symbol);
          }
        }
      }

      // Display filtered symbols in a selectable list
      if (!filtered_symbols_.empty()) {
        ImGui::BeginChild("SymbolList", ImVec2(0, 150), true);

        for (int i = 0; i < static_cast<int>(filtered_symbols_.size()); ++i) {
          const std::string& symbol = filtered_symbols_[i];
          bool is_selected = (selected_symbol_idx_ >= 0 && selected_symbol_idx_ < static_cast<int>(all_symbols_.size()) &&
                              all_symbols_[selected_symbol_idx_] == symbol);

          if (ImGui::Selectable(symbol.c_str(), is_selected)) {
            // Find the index in the all_symbols_ vector
            for (int j = 0; j < static_cast<int>(all_symbols_.size()); ++j) {
              if (all_symbols_[j] == symbol) {
                selected_symbol_idx_ = j;

                // Get the symbol ID from the registry
                auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
                if (symbol_info_opt) {
                  uint32_t symbol_id = symbol_info_opt->id;

                  // Set the active symbol for all panels
                  if (panel_manager_) {
                    panel_manager_->set_active_symbol(symbol_id, symbol);
                  }
                }
                break;
              }
            }
          }
        }

        ImGui::EndChild();
      } else if (!symbol_input_buffer_.empty()) {
        ImGui::Text("No symbols match your search.");
      }

      // Display currently selected symbol
      if (selected_symbol_idx_ >= 0 && selected_symbol_idx_ < static_cast<int>(all_symbols_.size())) {
        ImGui::Text("Selected: %s", all_symbols_[selected_symbol_idx_].c_str());
      } else {
        ImGui::Text("No symbol selected");
      }
    }

    // Panel management section with all requested panel types
    if (ImGui::CollapsingHeader("Add Panels", ImGuiTreeNodeFlags_DefaultOpen)) {

      // Add Chart Button
      if (ImGui::Button("Add Chart", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::CHART);
        }
      }
      ImGui::Spacing();

      // Add Footprint Button
      if (ImGui::Button("Add Footprint", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::FOOTPRINT_CHART);
        }
      }
      ImGui::Spacing();

      // Add Volume Profile Button
      if (ImGui::Button("Add Volume Profile", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::VOLUME_PROFILE);
        }
      }
      ImGui::Spacing();

      // Add Order Book Button
      if (ImGui::Button("Add Order Book", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ORDERBOOK);
        }
      }
      ImGui::Spacing();

      // Add Time&Sales Button
      if (ImGui::Button("Add Time&Sales", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::TIME_AND_SALES);
        }
      }
      ImGui::Spacing();

      // Add Watchlist Button
      if (ImGui::Button("Add Watchlist", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::WATCHLIST);
        }
      }
      ImGui::Spacing();

      // Add News Button (using Alerts panel as news feed)
      if (ImGui::Button("Add News", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ALERTS);
        }
      }
      ImGui::Spacing();

      // Additional useful panels
      if (ImGui::Button("Add Metrics", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::METRICS);
        }
      }
      ImGui::Spacing();

      if (ImGui::Button("Add Screener", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::SCREENER);
        }
      }
      ImGui::Spacing();
    }

    // Layout controls
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Layout Management")) {
      if (ImGui::Button("Reset Layout")) {
        if (panel_manager_) {
          // Clear all panels and set up default layout
          panel_manager_->clear_panels();

          // Add default panels
          panel_manager_->add_panel(PanelType::CHART, "Default Chart", 0, 0, 4, 3);
          panel_manager_->add_panel(PanelType::ORDERBOOK, "Order Book", 4, 0, 2, 2);
          panel_manager_->add_panel(PanelType::WATCHLIST, "Watchlist", 0, 3, 2, 1);
          panel_manager_->add_panel(PanelType::TIME_AND_SALES, "Time & Sales", 2, 3, 2, 1);
        }
      }

      ImGui::Spacing();

      if (ImGui::Button("Auto Arrange")) {
        if (panel_manager_) {
          panel_manager_->auto_arrange_panels();
        }
      }

      ImGui::Spacing();

      if (ImGui::Button("Clear All Panels")) {
        if (panel_manager_) {
          panel_manager_->clear_panels();
        }
      }
    }

    // Status information
    ImGui::Separator();
    if (panel_manager_) {
      auto panel_count = panel_manager_->get_panel_count();
      ImGui::Text("Active Panels: %zu", panel_count);
    }
    ImGui::Text("Ready to add panels");
  }
  ImGui::End();
}

void DashboardControls::update(float dt) {
  // No update logic needed for controls panel
  (void)dt;
}

void DashboardControls::initialize_vulkan_resources(VulkanCore* core) {
  // No Vulkan resources needed for controls panel
  (void)core;
}

void DashboardControls::clear_data() {
  // No data to clear for controls panel
}

}  // namespace BTQuant