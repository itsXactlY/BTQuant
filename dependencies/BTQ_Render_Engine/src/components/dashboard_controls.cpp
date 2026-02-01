#include "../../include/components/dashboard_controls.hpp"

#include <imgui.h>
#include <imgui_internal.h>  // For ImGui::InputTextWithHint

#include <iostream>
#include <algorithm>
#include <cctype>

#include "../../include/components/panel_manager.hpp"
#include "../../include/components/chart_panel.hpp"
#include "../../include/ui/ui_base.hpp"
#include "../../include/symbol_registry.hpp"
#include "../../include/hotspine_data_bridge.hpp"

namespace BTQuant {

// Helper function to convert timeframe to string representation
static std::string timeframe_to_string(RenderEngine::TimeFrame tf) {
  switch (tf) {
    case RenderEngine::TimeFrame::TF_1MIN:  return "1m";
    case RenderEngine::TimeFrame::TF_5MIN:  return "5m";
    case RenderEngine::TimeFrame::TF_15MIN: return "15m";
    case RenderEngine::TimeFrame::TF_30MIN: return "30m";
    case RenderEngine::TimeFrame::TF_1HOUR: return "1h";
    case RenderEngine::TimeFrame::TF_4HOUR: return "4h";
    case RenderEngine::TimeFrame::TF_1DAY:  return "1d";
    case RenderEngine::TimeFrame::TF_1WEEK: return "1w";
    case RenderEngine::TimeFrame::TF_1MS:   return "1ms";
    case RenderEngine::TimeFrame::TF_10MS:  return "10ms";
    case RenderEngine::TimeFrame::TF_100MS: return "100ms";
    case RenderEngine::TimeFrame::TF_500MS: return "500ms";
    case RenderEngine::TimeFrame::TF_1SEC:  return "1s";
    case RenderEngine::TimeFrame::TF_3SEC:  return "3s";
    case RenderEngine::TimeFrame::TF_5SEC:  return "5s";
    case RenderEngine::TimeFrame::TF_15SEC: return "15s";
    case RenderEngine::TimeFrame::TF_30SEC: return "30s";
    case RenderEngine::TimeFrame::TF_2MIN:  return "2m";
    case RenderEngine::TimeFrame::TF_2HOUR: return "2h";
    case RenderEngine::TimeFrame::TF_6HOUR: return "6h";
    case RenderEngine::TimeFrame::TF_12HOUR: return "12h";
    default: return "1m";
  }
}

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

        // Refresh symbols after exchange selection changes
        refresh_symbols_for_selected_exchanges();
      }

      // Multi-select dropdown for exchanges
      ImGui::Text("Select Active Exchanges:");

      // Create a more compact multi-select dropdown
      static bool show_exchange_selector = false;
      static char exchange_preview[256] = "All Exchanges";

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

      // Button that acts as a dropdown
      if (ImGui::Button(exchange_preview, ImVec2(-1, 0))) {
        show_exchange_selector = !show_exchange_selector;
      }

      // Show exchange selection popup as a proper dropdown
      if (show_exchange_selector) {
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetItemRectMin().x, ImGui::GetItemRectMax().y));
        ImGui::SetNextWindowSize(ImVec2(ImGui::GetItemRectSize().x, 300));

        bool any_changes = false; // Move this declaration outside the Begin/End block

        if (ImGui::Begin("##ExchangeSelectorPopup", &show_exchange_selector,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                         ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoSavedSettings)) {

          // Select All / Deselect All buttons
          if (ImGui::Button("Select All", ImVec2(ImGui::GetContentRegionAvail().x * 0.45f, 0))) {
            std::fill(selected_exchanges_.begin(), selected_exchanges_.end(), 1);
            any_changes = true;
          }
          ImGui::SameLine();
          if (ImGui::Button("Deselect All", ImVec2(ImGui::GetContentRegionAvail().x, 0))) {
            std::fill(selected_exchanges_.begin(), selected_exchanges_.end(), 0);
            any_changes = true;
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
          }

          ImGui::EndChild();

          // Apply button to close the popup
          if (ImGui::Button("Apply", ImVec2(-1, 0))) {
            show_exchange_selector = false;

            // Refresh symbols when exchange selection changes
            if (any_changes) {
              refresh_symbols_for_selected_exchanges();
            }
          }

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

          // Refresh symbols when exchange selection changes
          if (any_changes) {
            refresh_symbols_for_selected_exchanges();
          }
        }
      }

      // Add a small info text showing how many exchanges are selected
      ImGui::TextDisabled("(%d/%zu exchanges)", selected_count, all_exchanges_.size());
    }

    // Prominent Symbol selection section - moved to top for better UX
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("ACTIVE SYMBOL:");

    // Search-enabled symbol selection dropdown - more prominent
    // Search input for filtering the dropdown options
    if (ImGui::InputTextWithHint("##symbol_search_top", "Search symbols...", symbol_input_buffer_.data(), symbol_input_buffer_.size())) {
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

    if (ImGui::BeginCombo("##symbol_combo_top", preview_value, ImGuiComboFlags_HeightLarge)) {
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

                  // Log the symbol change for debugging
                  std::cout << "[DashboardControls] Setting active symbol to: " << symbol
                            << " (ID: " << symbol_id << ")" << std::endl;
                }
              } else {
                // If symbol not found in registry, try to register it
                std::string exchange_name = "Unknown"; // Default exchange

                // Try to determine exchange from selected exchanges
                if (!selected_exchanges_.empty() && !all_exchanges_.empty()) {
                  for (size_t idx = 0; idx < selected_exchanges_.size(); ++idx) {
                    if (selected_exchanges_[idx] != 0) {
                      exchange_name = all_exchanges_[idx];
                      break;
                    }
                  }
                }

                uint32_t new_symbol_id = SymbolRegistry::instance().register_symbol(exchange_name, symbol);

                if (panel_manager_) {
                  panel_manager_->set_active_symbol(new_symbol_id, symbol);

                  // Log the symbol registration and change for debugging
                  std::cout << "[DashboardControls] Registered and set active symbol: " << symbol
                            << " (ID: " << new_symbol_id << ") on exchange: " << exchange_name << std::endl;
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

      // Optionally notify panels that no symbol is selected
      if (panel_manager_) {
        // Pass a special value to indicate no symbol is selected
        // Using 0 as a special symbol ID for "no symbol"
        panel_manager_->set_active_symbol(0, "");
        std::cout << "[DashboardControls] Cleared active symbol selection" << std::endl;
      }
    }

    // Add an Apply to All button to ensure all panels get the current symbol
    ImGui::SameLine();
    if (ImGui::Button("Apply to All")) {
      if (selected_symbol_idx_ >= 0 && selected_symbol_idx_ < static_cast<int>(all_symbols_.size())) {
        const std::string& symbol = all_symbols_[selected_symbol_idx_];
        auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
        if (symbol_info_opt) {
          uint32_t symbol_id = symbol_info_opt->id;
          if (panel_manager_) {
            panel_manager_->set_active_symbol(symbol_id, symbol);
            std::cout << "[DashboardControls] Applied symbol " << symbol
                      << " (ID: " << symbol_id << ") to all panels" << std::endl;
          }
        }
      }
    }

    // Display currently selected symbol
    if (selected_symbol_idx_ >= 0 && selected_symbol_idx_ < static_cast<int>(all_symbols_.size())) {
      ImGui::Text("Current Symbol: %s", all_symbols_[selected_symbol_idx_].c_str());
    } else {
      ImGui::Text("No symbol selected");
    }

    // Add a refresh button to update symbols from exchange API
    ImGui::Spacing();
    if (ImGui::Button("Refresh Symbols from Exchange API")) {
      fetch_symbols_from_api_ = true;
      needs_refresh_ = true;

      // Trigger the API fetch
      if (panel_manager_) {
        fetch_symbols_from_exchange_api();
        fetch_symbols_from_api_ = false;
        symbols_loaded_ = true;
        needs_refresh_ = false;

        // Update filtered symbols to match the newly loaded symbols
        filtered_symbols_ = all_symbols_;
      }
    }

    // Show status of symbol count
    ImGui::SameLine();
    ImGui::TextDisabled("(%zu symbols)", all_symbols_.size());

    // Panel management section with all requested panel types
    if (ImGui::CollapsingHeader("Add Panels", ImGuiTreeNodeFlags_DefaultOpen)) {

      // Create a grid layout for panel buttons (2 columns)
      ImGui::Columns(2, "panel_buttons", true);

      if (ImGui::Button("Add Chart", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::CHART);
        }
      }
      ImGui::NextColumn();

      if (ImGui::Button("Add Footprint", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::FOOTPRINT_CHART);
        }
      }
      ImGui::NextColumn();

      if (ImGui::Button("Add Volume Profile", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::VOLUME_PROFILE);
        }
      }
      ImGui::NextColumn();

      if (ImGui::Button("Add Order Book", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ORDERBOOK);
        }
      }
      ImGui::NextColumn();

      if (ImGui::Button("Add Time&Sales", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::TIME_AND_SALES);
        }
      }
      ImGui::NextColumn();

      if (ImGui::Button("Add Watchlist", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::WATCHLIST);
        }
      }
      ImGui::NextColumn();

      if (ImGui::Button("Add News", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ALERTS);
        }
      }
      ImGui::NextColumn();

      ImGui::Columns(1); // Reset to single column

      ImGui::Spacing();
    }

    // Timeframe selection section
    if (ImGui::CollapsingHeader("Timeframe Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("Select Chart Timeframe:");

      // Define the target timeframes for the buttons
      const std::vector<std::pair<RenderEngine::TimeFrame, const char*>> timeframes = {
        {RenderEngine::TimeFrame::TF_1MIN, "1m"},
        {RenderEngine::TimeFrame::TF_5MIN, "5m"},
        {RenderEngine::TimeFrame::TF_15MIN, "15m"},
        {RenderEngine::TimeFrame::TF_30MIN, "30m"},
        {RenderEngine::TimeFrame::TF_1HOUR, "1h"},
        {RenderEngine::TimeFrame::TF_4HOUR, "4h"},
        {RenderEngine::TimeFrame::TF_1DAY, "1d"},
        {RenderEngine::TimeFrame::TF_1WEEK, "1w"}
      };

      // Create a row of buttons for each timeframe
      for (size_t i = 0; i < timeframes.size(); ++i) {
        const auto& [timeframe, label] = timeframes[i];

        // Highlight the currently selected timeframe button
        if (current_timeframe_ == timeframe) {
          ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }

        if (ImGui::Button(label, ImVec2(60, 30))) {
          // Update the current timeframe
          current_timeframe_ = timeframe;

          // Update all chart panels with the new timeframe
          update_all_chart_timeframes(timeframe);
        }

        if (current_timeframe_ == timeframe) {
          ImGui::PopStyleColor(2); // Pop the two pushed colors
        }

        // Add spacing between buttons, except for the last one in each row
        if ((i + 1) % 4 != 0 && i < timeframes.size() - 1) {  // 4 buttons per row
          ImGui::SameLine();
        } else if ((i + 1) % 4 == 0 || i == timeframes.size() - 1) {
          ImGui::Spacing(); // Add vertical spacing after each row
        }
      }

      // Show currently selected timeframe
      ImGui::Spacing();
      ImGui::Text("Current Timeframe: %s", timeframe_to_string(current_timeframe_).c_str());
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

void DashboardControls::refresh_symbols_for_selected_exchanges() {
  if (!panel_manager_) {
    std::cerr << "[DashboardControls] Error: PanelManager is null, cannot refresh symbols" << std::endl;
    return;
  }

  // Clear current symbols
  all_symbols_.clear();

  // Get all symbols from the symbol registry, filtered by selected exchanges
  auto all_symbol_infos = SymbolRegistry::instance().get_all_symbols();
  for (const auto& symbol_info : all_symbol_infos) {
    // Check if this symbol's exchange is in the selected exchanges
    if (is_exchange_selected(symbol_info.exchange)) {
      // Only add symbol if its exchange is selected and it's not already in the list
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

  std::cout << "[DashboardControls] Refreshed symbols for selected exchanges. Total symbols: " << all_symbols_.size() << std::endl;

  // Update the selected symbol index if the previously selected symbol is no longer available
  if (selected_symbol_idx_ >= static_cast<int>(all_symbols_.size())) {
    selected_symbol_idx_ = -1; // Reset to no selection
  }
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

          // Register the symbol in the registry if it doesn't exist
          SymbolRegistry::instance().register_symbol(symbol_info.exchange, symbol_info.symbol);
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

            // Register the symbol in the registry if it doesn't exist
            SymbolRegistry::instance().register_symbol(bridge_exchange_name, symbol_name);
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

  // Update the symbol input buffer to clear any previous search
  std::fill(symbol_input_buffer_.begin(), symbol_input_buffer_.end(), 0);

  // Update the selected symbol index if the previously selected symbol is no longer available
  if (selected_symbol_idx_ >= static_cast<int>(all_symbols_.size())) {
    selected_symbol_idx_ = -1; // Reset to no selection
  }
}

void DashboardControls::update_all_chart_timeframes(RenderEngine::TimeFrame timeframe) {
  if (!panel_manager_) {
    std::cerr << "[DashboardControls] Error: PanelManager is null, cannot update chart timeframes" << std::endl;
    return;
  }

  // Update all charts managed by the chart manager
  auto chart_manager = panel_manager_->get_chart_manager();
  if (chart_manager) {
    chart_manager->update_all_chart_timeframes(timeframe);
  }

  // Get all panel IDs
  auto panel_ids = panel_manager_->get_all_panel_ids();

  // Iterate through all panels and update chart panels specifically
  for (uint32_t panel_id : panel_ids) {
    auto panel = panel_manager_->get_panel_by_id(panel_id);
    if (!panel) continue;

    // Only ChartPanel has a direct set_timeframe method, so we only update those directly
    if (panel->get_config().type == PanelType::CHART) {
      ChartPanel* chart_panel = dynamic_cast<ChartPanel*>(panel);
      if (chart_panel) {
        chart_panel->set_timeframe(timeframe);
      }
    }
  }
}

}  // namespace BTQuant