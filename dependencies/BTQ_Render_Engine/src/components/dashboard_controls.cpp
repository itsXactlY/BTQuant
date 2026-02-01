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

    // Exchange selection section
    if (ImGui::CollapsingHeader("Exchange Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Load exchanges if needed
      if (!exchanges_loaded_) {
        all_exchanges_ = SymbolRegistry::instance().get_exchanges();

        // Initialize selected_exchanges_ vector with all exchanges selected by default
        selected_exchanges_.resize(all_exchanges_.size());
        std::fill(selected_exchanges_.begin(), selected_exchanges_.end(), 1); // 1 means true/selected

        exchanges_loaded_ = true;
        needs_refresh_ = true; // Refresh symbols after exchange selection changes
      }

      // Multi-select dropdown for exchanges
      ImGui::Text("Select Active Exchanges:");

      // Create a temporary window to show the multi-select list
      static bool show_exchange_selector = false;
      static char exchange_preview[256] = "All Exchanges";

      if (ImGui::Button("Select Exchanges")) {
        show_exchange_selector = !show_exchange_selector;
      }

      // Store the button position for the popup
      ImVec2 button_pos = ImGui::GetItemRectMin(); // Get top-left corner of the button

      // Update preview text to show selected exchanges
      std::string preview_text = "";
      int selected_count = 0;
      for (size_t i = 0; i < all_exchanges_.size(); ++i) {
        if (selected_exchanges_[i]) {
          if (selected_count > 0) preview_text += ", ";
          preview_text += all_exchanges_[i];
          selected_count++;
        }
      }

      if (selected_count == 0) {
        strcpy(exchange_preview, "No Exchanges Selected");
      } else if (selected_count == static_cast<int>(all_exchanges_.size())) {
        strcpy(exchange_preview, "All Exchanges");
      } else {
        strncpy(exchange_preview, preview_text.c_str(), sizeof(exchange_preview) - 1);
        exchange_preview[sizeof(exchange_preview) - 1] = '\0';
      }

      ImGui::SameLine();
      ImGui::Text("%s", exchange_preview);

      // Show exchange selection popup
      if (show_exchange_selector) {
        // Position the popup near the button
        ImVec2 window_pos = ImVec2(button_pos.x, button_pos.y + ImGui::GetItemRectSize().y); // Position directly below the button
        ImGui::SetNextWindowPos(window_pos);
        ImGui::SetNextWindowSize(ImVec2(250, 300));

        if (ImGui::Begin("Exchange Selector", &show_exchange_selector,
                         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {

          bool any_changes = false;

          // Select All / Deselect All buttons
          if (ImGui::Button("Select All")) {
            std::fill(selected_exchanges_.begin(), selected_exchanges_.end(), 1);
            any_changes = true;
          }
          ImGui::SameLine();
          if (ImGui::Button("Deselect All")) {
            std::fill(selected_exchanges_.begin(), selected_exchanges_.end(), 0);
            any_changes = true;
          }
          ImGui::SameLine();
          if (ImGui::Button("Apply")) {
            show_exchange_selector = false;
            needs_refresh_ = true; // Refresh symbols when exchange selection changes
          }

          ImGui::Separator();

          // Individual exchange checkboxes with scrollable area
          ImGui::BeginChild("ExchangeList", ImVec2(0, 200), true);

          for (size_t i = 0; i < all_exchanges_.size(); ++i) {
            bool temp_selected = selected_exchanges_[i] != 0;
            if (ImGui::Checkbox(all_exchanges_[i].c_str(), &temp_selected)) {
              selected_exchanges_[i] = temp_selected ? 1 : 0;
              any_changes = true;
            }

            // Add a small indent to show hierarchy if needed in the future
            // ImGui::Indent(20.0f);
          }

          ImGui::EndChild();

          if (any_changes) {
            // Update preview text immediately when changes occur
            std::string updated_preview_text = "";
            int updated_selected_count = 0;
            for (size_t i = 0; i < all_exchanges_.size(); ++i) {
              if (selected_exchanges_[i]) {
                if (updated_selected_count > 0) updated_preview_text += ", ";
                updated_preview_text += all_exchanges_[i];
                updated_selected_count++;
              }
            }

            if (updated_selected_count == 0) {
              strcpy(exchange_preview, "No Exchanges Selected");
            } else if (updated_selected_count == static_cast<int>(all_exchanges_.size())) {
              strcpy(exchange_preview, "All Exchanges");
            } else {
              strncpy(exchange_preview, updated_preview_text.c_str(), sizeof(exchange_preview) - 1);
              exchange_preview[sizeof(exchange_preview) - 1] = '\0';
            }
          }

          ImGui::End();
        } else {
          // Window was closed (by clicking outside), so set the flag to false and refresh
          show_exchange_selector = false;
          needs_refresh_ = true; // Refresh symbols when exchange selection changes
        }
      }
    }

    // Symbol selection section
    if (ImGui::CollapsingHeader("Symbol Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Refresh symbols button
      if (ImGui::Button("Refresh Symbols")) {
        needs_refresh_ = true;
      }

      ImGui::SameLine();

      // Add API fetch button to get symbols directly from exchange APIs
      if (ImGui::Button("Fetch from Exchange API")) {
        fetch_symbols_from_api_ = true;
        needs_refresh_ = true;
      }

      // Load symbols if needed
      if (needs_refresh_ || !symbols_loaded_ || fetch_symbols_from_api_) {
        if (panel_manager_) {
          // Get all available symbols from the symbol registry
          all_symbols_.clear();

          if (fetch_symbols_from_api_) {
            // Fetch symbols directly from exchange APIs
            fetch_symbols_from_exchange_api();
            fetch_symbols_from_api_ = false;
          } else {
            // Get symbols from the symbol registry, filtered by selected exchanges
            auto all_symbol_infos = SymbolRegistry::instance().get_all_symbols();
            for (const auto& symbol_info : all_symbol_infos) {
              // Check if this symbol's exchange is in the selected exchanges
              if (is_exchange_selected(symbol_info.exchange)) {
                // Only add symbol if its exchange is selected
                // Check if symbol is already in the list
                bool found = false;
                for (const auto& existing_symbol : all_symbols_) {
                  if (existing_symbol == symbol_info.symbol) {
                    found = true;
                    break;
                  }
                }
                if (!found) {
                  all_symbols_.push_back(symbol_info.symbol);
                }
              }
            }
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
                  // Try to get exchange information from the symbol registry
                  std::string exchange_name = "";
                  auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
                  if (symbol_info.has_value()) {
                    exchange_name = symbol_info->exchange;
                  } else {
                    // If not in registry, try to get from bridge
                    exchange_name = bridge->getExchangeName(symbol_id);
                  }

                  // Check if this symbol's exchange is in the selected exchanges
                  bool exchange_selected = !exchange_name.empty() ? is_exchange_selected(exchange_name) : true;

                  if (exchange_selected) {
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
          }

          // Sort symbols alphabetically
          std::sort(all_symbols_.begin(), all_symbols_.end());

          // Update filtered symbols to match the newly loaded symbols
          filtered_symbols_ = all_symbols_;

          symbols_loaded_ = true;
          needs_refresh_ = false;
        }
      }

      // Search-enabled symbol selection dropdown
      ImGui::Text("Select Symbol:");

      // Search input for filtering the dropdown options
      if (ImGui::InputTextWithHint("##symbol_search", "Search symbols...", symbol_input_buffer_.data(), symbol_input_buffer_.size())) {
        // Filter symbols based on search input
        filtered_symbols_.clear();

        std::string search_lower = symbol_input_buffer_.data();
        std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);

        for (const auto& symbol : all_symbols_) {
          std::string symbol_lower = symbol;
          std::transform(symbol_lower.begin(), symbol_lower.end(), symbol_lower.begin(), ::tolower);

          if (symbol_lower.find(search_lower) != std::string::npos) {
            filtered_symbols_.push_back(symbol);
          }
        }
      } else if (needs_refresh_) {
        // If symbols were refreshed due to exchange selection, update the filtered list too
        filtered_symbols_ = all_symbols_;
      }

      // Create a unique ID for the combo box
      char preview_value[256];
      if (selected_symbol_idx_ >= 0 && selected_symbol_idx_ < static_cast<int>(all_symbols_.size())) {
        strncpy(preview_value, all_symbols_[selected_symbol_idx_].c_str(), sizeof(preview_value) - 1);
        preview_value[sizeof(preview_value) - 1] = '\0';
      } else {
        strcpy(preview_value, "Select a symbol...");
      }

      if (ImGui::BeginCombo("##symbol_combo", preview_value, ImGuiComboFlags_HeightLarge)) {
        // Display filtered symbols in the combo box
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

            // Close the combo box after selection
            ImGui::CloseCurrentPopup();
          }

          if (is_selected) {
            ImGui::SetItemDefaultFocus();
          }
        }

        ImGui::EndCombo();
      }

      // Add a clear button to reset the symbol selection
      ImGui::SameLine();
      if (ImGui::Button("Clear")) {
        selected_symbol_idx_ = -1;
        strcpy(preview_value, "Select a symbol...");

        // Optionally notify panels that no symbol is selected
        if (panel_manager_) {
          // We could pass a special value to indicate no symbol is selected
          // For now, we'll just reset the selection
        }
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

      // First row of panel buttons
      if (ImGui::Button("Add Chart", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::CHART);
        }
      }
      ImGui::Spacing();

      if (ImGui::Button("Add Footprint", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::FOOTPRINT_CHART);
        }
      }
      ImGui::Spacing();

      if (ImGui::Button("Add Volume Profile", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::VOLUME_PROFILE);
        }
      }
      ImGui::Spacing();

      // Second row of panel buttons
      if (ImGui::Button("Add Order Book", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ORDERBOOK);
        }
      }
      ImGui::Spacing();

      if (ImGui::Button("Add Time&Sales", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::TIME_AND_SALES);
        }
      }
      ImGui::Spacing();

      if (ImGui::Button("Add Watchlist", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::WATCHLIST);
        }
      }
      ImGui::Spacing();

      if (ImGui::Button("Add News", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ALERTS);
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

bool DashboardControls::is_exchange_selected(const std::string& exchange_name) const {
  // Check if the given exchange is in the selected exchanges list
  for (size_t i = 0; i < all_exchanges_.size(); ++i) {
    if (all_exchanges_[i] == exchange_name && selected_exchanges_[i] != 0) {
      return true;
    }
  }
  return false;
}

void DashboardControls::fetch_symbols_from_exchange_api() {
  // This method integrates with exchange APIs to fetch live symbols
  if (!panel_manager_) {
    std::cerr << "[DashboardControls] Error: PanelManager is null, cannot fetch symbols from API" << std::endl;
    return;
  }

  // Get the chart manager to access the bridge
  auto chart_manager = panel_manager_->get_chart_manager();
  if (!chart_manager) {
    std::cerr << "[DashboardControls] Error: ChartManager is null, cannot fetch symbols from API" << std::endl;
    return;
  }

  auto bridge = chart_manager->get_bridge();
  if (!bridge) {
    std::cerr << "[DashboardControls] Error: Data bridge is null, cannot fetch symbols from API" << std::endl;
    return;
  }

  // Clear current symbols before fetching fresh ones
  all_symbols_.clear();

  // Iterate through selected exchanges and fetch symbols from each
  for (size_t i = 0; i < all_exchanges_.size(); ++i) {
    if (selected_exchanges_[i] != 0) {  // If exchange is selected
      const std::string& exchange_name = all_exchanges_[i];

      std::cout << "[DashboardControls] Fetching symbols from exchange: " << exchange_name << std::endl;

      // In a real implementation, this would call the actual exchange API
      // For now, we'll simulate API calls by fetching from the symbol registry
      // and the bridge, but in a more structured way that mimics real API integration

      // Simulate API call to exchange to get available symbols
      std::vector<SymbolInfo> api_symbols;

      // Placeholder for actual API call - in real implementation this would be:
      // api_symbols = exchange_api_client.get_symbols(exchange_name);
      // For now, we'll get symbols from registry but simulate the API call
      api_symbols = SymbolRegistry::instance().get_exchange_symbols(exchange_name);

      // Process symbols returned from API simulation
      for (const auto& symbol_info : api_symbols) {
        // Check if symbol is already in the list
        bool found = false;
        for (const auto& existing_symbol : all_symbols_) {
          if (existing_symbol == symbol_info.symbol) {
            found = true;
            break;
          }
        }
        if (!found) {
          all_symbols_.push_back(symbol_info.symbol);
        }
      }

      // Additionally, get any active symbols from the bridge for this exchange
      auto active_symbols = bridge->getActiveSymbols();
      for (auto symbol_id : active_symbols) {
        std::string symbol_name = bridge->getSymbolName(symbol_id);
        std::string bridge_exchange_name = bridge->getExchangeName(symbol_id);

        if (!symbol_name.empty() && bridge_exchange_name == exchange_name) {
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

  // Sort symbols alphabetically for better UX
  std::sort(all_symbols_.begin(), all_symbols_.end());

  // Update filtered symbols to match the newly loaded symbols
  filtered_symbols_ = all_symbols_;

  std::cout << "[DashboardControls] Fetched " << all_symbols_.size() << " symbols from exchange APIs" << std::endl;
}

}  // namespace BTQuant