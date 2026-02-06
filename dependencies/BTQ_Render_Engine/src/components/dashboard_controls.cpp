#include "../../include/components/dashboard_controls.hpp"

#include <imgui.h>
#include <imgui_internal.h>  // For ImGui::InputTextWithHint

#include <algorithm>
#include <cctype>
#include <iostream>
#include <unordered_set>

#include "../../include/components/alerts_panel.hpp"
#include "../../include/components/chart_panel.hpp"
#include "../../include/components/chart_replay_panel.hpp"
#include "../../include/components/footprint_panel.hpp"
#include "../../include/components/panel_manager.hpp"
#include "../../include/symbol_registry.hpp"
#include "../../include/ui/tooltips.hpp"
#include "../../include/ui/ui_base.hpp"

namespace BTQuant {

// Helper function to convert timeframe to string representation
static std::string timeframe_to_string(RenderEngine::TimeFrame tf) {
  switch (tf) {
    case RenderEngine::TimeFrame::TF_1MIN:
      return "1m";
    case RenderEngine::TimeFrame::TF_5MIN:
      return "5m";
    case RenderEngine::TimeFrame::TF_15MIN:
      return "15m";
    case RenderEngine::TimeFrame::TF_30MIN:
      return "30m";
    case RenderEngine::TimeFrame::TF_1HOUR:
      return "1h";
    case RenderEngine::TimeFrame::TF_4HOUR:
      return "4h";
    case RenderEngine::TimeFrame::TF_1DAY:
      return "1d";
    case RenderEngine::TimeFrame::TF_1WEEK:
      return "1w";
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
    case RenderEngine::TimeFrame::TF_2MIN:
      return "2m";
    case RenderEngine::TimeFrame::TF_2HOUR:
      return "2h";
    case RenderEngine::TimeFrame::TF_6HOUR:
      return "6h";
    case RenderEngine::TimeFrame::TF_12HOUR:
      return "12h";
    default:
      return "1m";
  }
}

DashboardControls::DashboardControls(PanelManager* panel_manager)
    : UIComponent({0, 0}, {0, 0}), panel_manager_(panel_manager) {
  if (!panel_manager_) {
    std::cerr << "[DashboardControls] Error: PanelManager is null" << std::endl;
  }

  // Initialize the Global Alert Manager
  if (panel_manager_) {
    auto chart_manager = panel_manager_->get_chart_manager();
    if (chart_manager) {
      // Get the processor directly from chart_manager instead of using bridge
      auto processor = chart_manager->get_processor();
      if (processor) {
        // Find the alerts panel to connect to it
        auto panel_ids = panel_manager_->get_all_panel_ids();
        AlertsPanel* alerts_panel_ptr = nullptr;

        for (uint32_t panel_id : panel_ids) {
          auto panel = panel_manager_->get_panel_by_id(panel_id);
          if (panel && panel->get_config().type == PanelType::ALERTS) {
            alerts_panel_ptr = dynamic_cast<AlertsPanel*>(panel);
            break;
          }
        }

        // Create the Global Alert Manager with the processor
        global_alert_manager_ = std::make_shared<GlobalAlertManager>(
            processor,
            alerts_panel_ptr ? std::shared_ptr<AlertsPanel>(alerts_panel_ptr, [](AlertsPanel*) {})
                             : nullptr);

        std::cout << "[DashboardControls] Initialized Global Alert Manager" << std::endl;
      } else {
        std::cerr << "[DashboardControls] Warning: Could not get processor for Global Alert Manager"
                  << std::endl;
      }
    } else {
      std::cerr
          << "[DashboardControls] Warning: Could not get chart manager for Global Alert Manager"
          << std::endl;
    }
  } else {
    std::cerr << "[DashboardControls] Warning: PanelManager is null, cannot initialize Global "
                 "Alert Manager"
              << std::endl;
  }
}

void DashboardControls::render_gui() { render_dashboard_controls(); }

void DashboardControls::render_dashboard_controls() {
  // Floating Dashboard Controls Panel - positioned in top-right corner and stays on top
  ImGuiIO& io = ImGui::GetIO();
  ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 370, 20), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(350, 500), ImGuiCond_FirstUseEver);

  // Use ImGuiWindowFlags_TopMost to ensure it stays on top of other windows
  if (ImGui::Begin("Dashboard Controls", nullptr, ImGuiWindowFlags_TopMost)) {
    ImGui::Text("Trading Dashboard Controls");
    ImGui::Separator();

    // Exchange selection section
    if (ImGui::CollapsingHeader("Exchange Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Load exchanges if needed
      if (!exchanges_loaded_) {
        all_exchanges_ = SymbolRegistry::instance().get_exchanges();

        // Initialize selected_exchanges_ vector with all exchanges selected by default
        selected_exchanges_.resize(all_exchanges_.size());
        std::fill(selected_exchanges_.begin(), selected_exchanges_.end(),
                  1);  // 1 means true/selected

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

      // Define the search buffer in the outer scope to be accessible in both branches
      static char exchange_search_buffer[128] = "";

      // Show exchange selection popup as a proper dropdown
      bool any_changes = false;  // Move this declaration outside the Begin/End block
      if (show_exchange_selector) {
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetItemRectMin().x, ImGui::GetItemRectMax().y));
        ImGui::SetNextWindowSize(
            ImVec2(ImGui::GetItemRectSize().x, 350));  // Increased height for better UX

        if (ImGui::Begin("##ExchangeSelectorPopup", &show_exchange_selector,
                         ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings)) {
          // Select All / Deselect All buttons with better layout
          ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                              ImVec2(5, 5));  // Reduce spacing between buttons
          if (ImGui::Button("Select All", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 0))) {
            std::fill(selected_exchanges_.begin(), selected_exchanges_.end(), 1);
            any_changes = true;
          }
          ImGui::SameLine();
          if (ImGui::Button("Deselect All", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 0))) {
            std::fill(selected_exchanges_.begin(), selected_exchanges_.end(), 0);
            any_changes = true;
          }

          // Invert Selection button
          ImGui::SameLine();
          if (ImGui::Button("Invert", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 0))) {
            for (auto& selection : selected_exchanges_) {
              selection = !selection;
            }
            any_changes = true;
          }
          ImGui::PopStyleVar();  // Restore item spacing

          ImGui::Separator();

          // Search input for filtering exchanges
          ImGui::InputTextWithHint("##exchange_search", "Filter exchanges...",
                                   exchange_search_buffer, sizeof(exchange_search_buffer));

          ImGui::Separator();

          // Show selected exchanges count
          int selected_count = std::count_if(selected_exchanges_.begin(), selected_exchanges_.end(),
                                             [](int val) { return val != 0; });
          ImGui::Text("Selected: %d/%zu", selected_count, all_exchanges_.size());

          // Individual exchange checkboxes with scrollable area
          ImGui::BeginChild("ExchangeList", ImVec2(0, 220), true);  // Increased height

          // Convert search term to lowercase for case-insensitive comparison
          std::string search_term = exchange_search_buffer;
          std::transform(search_term.begin(), search_term.end(), search_term.begin(), ::tolower);

          for (size_t i = 0; i < all_exchanges_.size(); ++i) {
            // Skip exchanges that don't match the search term (if search is not empty)
            if (!search_term.empty()) {
              std::string exchange_lower = all_exchanges_[i];
              std::transform(exchange_lower.begin(), exchange_lower.end(), exchange_lower.begin(),
                             ::tolower);

              if (exchange_lower.find(search_term) == std::string::npos) {
                continue;  // Skip this exchange if it doesn't match the search
              }
            }

            bool temp_selected = selected_exchanges_[i] != 0;

            // Highlight selected exchanges with different color
            if (temp_selected) {
              ImGui::PushStyleColor(
                  ImGuiCol_CheckMark,
                  ImVec4(0.2f, 0.8f, 0.2f, 1.0f));  // Green checkmark for selected
            }

            if (ImGui::Checkbox(all_exchanges_[i].c_str(), &temp_selected)) {
              selected_exchanges_[i] = temp_selected ? 1 : 0;
              any_changes = true;
            }

            if (temp_selected) {
              ImGui::PopStyleColor();  // Restore checkmark color
            }
          }

          ImGui::EndChild();

          // Apply and Cancel buttons
          ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                              ImVec2(5, 5));  // Reduce spacing between buttons
          if (ImGui::Button("Apply", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 0))) {
            show_exchange_selector = false;

            // Refresh symbols when exchange selection changes
            if (any_changes) {
              refresh_symbols_for_selected_exchanges();
            }

            // Clear the search buffer when closing
            memset(exchange_search_buffer, 0, sizeof(exchange_search_buffer));
          }
          ImGui::SameLine();
          if (ImGui::Button("Cancel", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 0))) {
            // Revert changes by reloading the exchanges - restore original selections
            if (show_exchange_selector) {
              // Reload original state by resetting to saved values (we don't have a backup, so just
              // close)
              show_exchange_selector = false;
            }

            // Clear the search buffer when closing
            memset(exchange_search_buffer, 0, sizeof(exchange_search_buffer));
          }

          // Reset to Default button
          ImGui::SameLine();
          if (ImGui::Button("Reset", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 0))) {
            // Reset to select all exchanges by default
            std::fill(selected_exchanges_.begin(), selected_exchanges_.end(), 1);
            any_changes = true;

            show_exchange_selector = false;

            // Refresh symbols when exchange selection changes
            if (any_changes) {
              refresh_symbols_for_selected_exchanges();
            }

            // Clear the search buffer when closing
            memset(exchange_search_buffer, 0, sizeof(exchange_search_buffer));
          }
          ImGui::PopStyleVar();  // Restore item spacing

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

          // Clear the search buffer when closing
          memset(exchange_search_buffer, 0, sizeof(exchange_search_buffer));
        }
      }

      // Add a small info text showing how many exchanges are selected
      ImGui::TextDisabled("(%d/%zu exchanges)", selected_count, all_exchanges_.size());
    }

    // Prominent Symbol selection section - moved to top for better UX
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("ACTIVE SYMBOL:");

    // Enhanced search-enabled symbol selection dropdown - more prominent
    // Search input for filtering the dropdown options
    if (ImGui::InputTextWithHint("##symbol_search_top", "Search symbols...",
                                 symbol_input_buffer_.data(), symbol_input_buffer_.size())) {
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
        bool is_selected = (selected_symbol_idx_ >= 0 &&
                            selected_symbol_idx_ < static_cast<int>(all_symbols_.size()) &&
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

                  // Ensure all panels receive the symbol update
                  sync_symbol_to_all_panels(symbol_id, symbol);
                }
              } else {
                // If symbol not found in registry, try to register it
                std::string exchange_name = "Unknown";  // Default exchange

                // Try to determine exchange from selected exchanges
                if (!selected_exchanges_.empty() && !all_exchanges_.empty()) {
                  for (size_t idx = 0; idx < selected_exchanges_.size(); ++idx) {
                    if (selected_exchanges_[idx] != 0) {
                      exchange_name = all_exchanges_[idx];
                      break;
                    }
                  }
                }

                uint32_t new_symbol_id =
                    SymbolRegistry::instance().register_symbol(exchange_name, symbol);

                if (panel_manager_) {
                  panel_manager_->set_active_symbol(new_symbol_id, symbol);

                  // Log the symbol registration and change for debugging
                  std::cout << "[DashboardControls] Registered and set active symbol: " << symbol
                            << " (ID: " << new_symbol_id << ") on exchange: " << exchange_name
                            << std::endl;

                  // Ensure all panels receive the symbol update
                  sync_symbol_to_all_panels(new_symbol_id, symbol);
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
      if (selected_symbol_idx_ >= 0 &&
          selected_symbol_idx_ < static_cast<int>(all_symbols_.size())) {
        const std::string& symbol = all_symbols_[selected_symbol_idx_];
        auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
        if (symbol_info_opt) {
          uint32_t symbol_id = symbol_info_opt->id;
          if (panel_manager_) {
            panel_manager_->set_active_symbol(symbol_id, symbol);
            std::cout << "[DashboardControls] Applied symbol " << symbol << " (ID: " << symbol_id
                      << ") to all panels" << std::endl;
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

    // Enhanced symbol selection controls with exchange API integration
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Symbol Management", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Add a button to force refresh symbols from exchange API
      if (ImGui::Button("Force Refresh All Symbols")) {
        fetch_symbols_from_api_ = true;
        needs_refresh_ = true;

        if (panel_manager_) {
          fetch_symbols_from_exchange_api();
          fetch_symbols_from_api_ = false;
          symbols_loaded_ = true;
          needs_refresh_ = false;

          // Update filtered symbols to match the newly loaded symbols
          filtered_symbols_ = all_symbols_;

          std::cout << "[DashboardControls] Force refreshed " << all_symbols_.size()
                    << " symbols from exchange API" << std::endl;
        }
      }

      ImGui::SameLine();
      if (ImGui::Button("Sync All Panels")) {
        // Synchronize the currently selected symbol to all panels
        if (selected_symbol_idx_ >= 0 &&
            selected_symbol_idx_ < static_cast<int>(all_symbols_.size())) {
          const std::string& symbol = all_symbols_[selected_symbol_idx_];
          auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(symbol);
          if (symbol_info_opt) {
            uint32_t symbol_id = symbol_info_opt->id;
            if (panel_manager_) {
              panel_manager_->set_active_symbol(symbol_id, symbol);
              std::cout << "[DashboardControls] Synced symbol " << symbol << " (ID: " << symbol_id
                        << ") to all panels" << std::endl;
            }
          }
        } else {
          // If no symbol is selected, clear the symbol for all panels
          if (panel_manager_) {
            panel_manager_->set_active_symbol(0, "");
            std::cout << "[DashboardControls] Cleared symbol for all panels" << std::endl;
          }
        }
      }

      // Show detailed information about the current symbol selection
      ImGui::Spacing();
      ImGui::Text("Symbol Selection Info:");
      ImGui::Indent();
      if (selected_symbol_idx_ >= 0 &&
          selected_symbol_idx_ < static_cast<int>(all_symbols_.size())) {
        const std::string& current_symbol = all_symbols_[selected_symbol_idx_];
        auto symbol_info_opt = SymbolRegistry::instance().get_symbol_by_name(current_symbol);
        if (symbol_info_opt) {
          ImGui::Text("- Current Symbol: %s", current_symbol.c_str());
          ImGui::Text("- Symbol ID: %u", symbol_info_opt->id);
          ImGui::Text("- Exchange: %s", symbol_info_opt->exchange.c_str());
        } else {
          ImGui::Text("- Current Symbol: %s (Not in registry)", current_symbol.c_str());
        }
      } else {
        ImGui::Text("- Current Symbol: None selected");
      }
      ImGui::Text("- Total Available Symbols: %zu", all_symbols_.size());
      ImGui::Text("- Selected Exchanges: %zu",
                  std::count_if(selected_exchanges_.begin(), selected_exchanges_.end(),
                                [](int val) { return val != 0; }));
      ImGui::Unindent();
    }

    // Panel management section with all requested panel types
    if (ImGui::CollapsingHeader("Add Panels", ImGuiTreeNodeFlags_DefaultOpen)) {
      // Create a grid layout for panel buttons (2 columns)
      ImGui::Columns(2, "panel_buttons", true);

      // First column buttons
      if (ImGui::Button("Add Chart", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::CHART);
        }
      }
      BTQuant::UI::show_control_tooltip("add_chart_panel");
      ImGui::NextColumn();

      if (ImGui::Button("Add Footprint", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::FOOTPRINT_CHART);
        }
      }
      BTQuant::UI::show_control_tooltip("add_footprint_panel");
      ImGui::NextColumn();

      if (ImGui::Button("Add Volume Profile", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::VOLUME_PROFILE);
        }
      }
      BTQuant::UI::show_control_tooltip("add_volume_profile_panel");
      ImGui::NextColumn();

      if (ImGui::Button("Add Order Book", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ORDERBOOK);
        }
      }
      BTQuant::UI::show_control_tooltip("add_order_book_panel");
      ImGui::NextColumn();

      // Second column buttons
      if (ImGui::Button("Add Time&Sales", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::TIME_AND_SALES);
        }
      }
      BTQuant::UI::show_control_tooltip("add_time_sales_panel");
      ImGui::NextColumn();

      if (ImGui::Button("Add Watchlist", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::WATCHLIST);
        }
      }
      BTQuant::UI::show_control_tooltip("add_watchlist_panel");
      ImGui::NextColumn();

      if (ImGui::Button("Add News", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::ALERTS);
        }
      }
      BTQuant::UI::show_control_tooltip("add_news_panel");
      ImGui::NextColumn();

      if (ImGui::Button("Add Risk Analyzer", ImVec2(-1, 30))) {
        if (panel_manager_) {
          panel_manager_->add_panel(PanelType::RISK_ANALYZER);
        }
      }
      // Add tooltip for Risk Analyzer
      if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Add a Risk Analyzer panel to visualize P/L vs Underlying Price");
        ImGui::EndTooltip();
      }
      ImGui::NextColumn();

      ImGui::Columns(1);  // Reset to single column

      ImGui::Spacing();

      // Add a separator and status information
      ImGui::Separator();
      if (panel_manager_) {
        auto panel_count = panel_manager_->get_panel_count();
        ImGui::Text("Active Panels: %zu", panel_count);
      }
    }

    // Timeframe selection section
    if (ImGui::CollapsingHeader("Timeframe Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("Select Chart Timeframe:");

      // Define the target timeframes for the buttons
      const std::vector<std::pair<RenderEngine::TimeFrame, const char*>> timeframes = {
          {RenderEngine::TimeFrame::TF_1MIN, "1m"},   {RenderEngine::TimeFrame::TF_5MIN, "5m"},
          {RenderEngine::TimeFrame::TF_15MIN, "15m"}, {RenderEngine::TimeFrame::TF_30MIN, "30m"},
          {RenderEngine::TimeFrame::TF_1HOUR, "1h"},  {RenderEngine::TimeFrame::TF_4HOUR, "4h"},
          {RenderEngine::TimeFrame::TF_1DAY, "1d"},   {RenderEngine::TimeFrame::TF_1WEEK, "1w"}};

      // Create a row of buttons for each timeframe
      for (size_t i = 0; i < timeframes.size(); ++i) {
        const auto& [timeframe, label] = timeframes[i];

        // Highlight the currently selected timeframe button
        if (current_timeframe_ == timeframe) {
          ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }

        if (ImGui::Button(label, ImVec2(60, 30))) {
          // Update the current timeframe
          current_timeframe_ = timeframe;

          // Update all chart panels with the new timeframe
          update_all_chart_timeframes(timeframe);
        }

        if (current_timeframe_ == timeframe) {
          ImGui::PopStyleColor(2);  // Pop the two pushed colors
        }

        // Add spacing between buttons, except for the last one in each row
        if ((i + 1) % 4 != 0 && i < timeframes.size() - 1) {  // 4 buttons per row
          ImGui::SameLine();
        } else if ((i + 1) % 4 == 0 || i == timeframes.size() - 1) {
          ImGui::Spacing();  // Add vertical spacing after each row
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

    // Global Alerts Section
    render_global_alerts_section();

    // Status information
    ImGui::Separator();
    if (panel_manager_) {
      auto panel_count = panel_manager_->get_panel_count();
      ImGui::Text("Active Panels: %zu", panel_count);
    }
    ImGui::Text("Ready to add panels");
  }

  // Render the create alert modal if needed
  render_create_alert_modal();

  ImGui::End();
}

void DashboardControls::update(float dt) {
  // Update the global alert manager to check for triggered alerts
  if (global_alert_manager_) {
    global_alert_manager_->update_alerts();
  }
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
    std::cerr << "[DashboardControls] Error: PanelManager is null, cannot refresh symbols"
              << std::endl;
    return;
  }

  // Clear current symbols
  all_symbols_.clear();

  // Get all symbols from the symbol registry, filtered by selected exchanges
  auto all_symbol_infos = SymbolRegistry::instance().get_all_symbols();

  // Use a set to efficiently track unique symbols
  std::unordered_set<std::string> unique_symbols;

  for (const auto& symbol_info : all_symbol_infos) {
    // Check if this symbol's exchange is in the selected exchanges
    if (is_exchange_selected(symbol_info.exchange)) {
      unique_symbols.insert(symbol_info.symbol);
    }
  }

  // Also get active symbols from the processor if possible through the chart manager
  auto chart_manager = panel_manager_->get_chart_manager();
  if (chart_manager) {
    // Get the processor directly from chart_manager instead of using bridge
    auto processor = chart_manager->get_processor();
    if (processor) {
      auto active_symbols = processor->getActiveSymbols();
      for (auto symbol_id : active_symbols) {
        // For now, we'll use a default name since we don't have direct access to symbol names from processor
        // In a real implementation, this would come from SymbolRegistry or similar
        std::string symbol_name = "SYMBOL_" + std::to_string(symbol_id);

        // Try to get exchange information from the symbol registry
        std::string exchange_name = "";
        auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
        if (symbol_info.has_value()) {
          exchange_name = symbol_info->exchange;
        }

        // Check if this symbol's exchange is in the selected exchanges
        bool exchange_selected =
            !exchange_name.empty() ? is_exchange_selected(exchange_name) : true;

        if (exchange_selected) {
          unique_symbols.insert(symbol_name);
        }
      }
    }
  }

  // Transfer unique symbols to the vector
  all_symbols_.assign(unique_symbols.begin(), unique_symbols.end());

  // Sort symbols alphabetically
  std::sort(all_symbols_.begin(), all_symbols_.end());

  // Update filtered symbols to match the newly loaded symbols
  filtered_symbols_ = all_symbols_;

  std::cout << "[DashboardControls] Refreshed symbols for selected exchanges. Total symbols: "
            << all_symbols_.size() << std::endl;

  // Update the selected symbol index if the previously selected symbol is no longer available
  if (selected_symbol_idx_ >= static_cast<int>(all_symbols_.size())) {
    selected_symbol_idx_ = -1;  // Reset to no selection
  }
}

void DashboardControls::fetch_symbols_from_exchange_api() {
  // This method integrates with exchange APIs to fetch live symbols
  if (!panel_manager_) {
    std::cerr << "[DashboardControls] Error: PanelManager is null, cannot fetch symbols from API"
              << std::endl;
    return;
  }

  // Get the chart manager to access the processor
  auto chart_manager = panel_manager_->get_chart_manager();
  if (!chart_manager) {
    std::cerr << "[DashboardControls] Error: ChartManager is null, cannot fetch symbols from API"
              << std::endl;
    return;
  }

  // Get the processor directly from chart_manager instead of using bridge
  auto processor = chart_manager->get_processor();
  if (!processor) {
    std::cerr << "[DashboardControls] Error: Data processor is null, cannot fetch symbols from API"
              << std::endl;
    return;
  }

  // Clear current symbols before fetching fresh ones
  all_symbols_.clear();

  // Iterate through selected exchanges and fetch symbols from each
  for (size_t i = 0; i < all_exchanges_.size(); ++i) {
    if (selected_exchanges_[i] != 0) {  // If exchange is selected
      const std::string& exchange_name = all_exchanges_[i];

      std::cout << "[DashboardControls] Fetching symbols from exchange: " << exchange_name
                << std::endl;

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

      // Additionally, get any active symbols from the processor for this exchange
      auto active_symbols = processor->getActiveSymbols();
      for (auto symbol_id : active_symbols) {
        // For now, we'll use a default name since we don't have direct access to symbol names from processor
        // In a real implementation, this would come from SymbolRegistry or similar
        std::string symbol_name = "SYMBOL_" + std::to_string(symbol_id);
        
        // Get exchange name from symbol registry
        std::string processor_exchange_name = "";
        auto symbol_info = SymbolRegistry::instance().get_symbol_info(symbol_id);
        if (symbol_info.has_value()) {
          processor_exchange_name = symbol_info->exchange;
        }

        if (!symbol_name.empty() && processor_exchange_name == exchange_name) {
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
            SymbolRegistry::instance().register_symbol(processor_exchange_name, symbol_name);
          }
        }
      }
    }
  }

  // Sort symbols alphabetically for better UX
  std::sort(all_symbols_.begin(), all_symbols_.end());

  // Update filtered symbols to match the newly loaded symbols
  filtered_symbols_ = all_symbols_;

  std::cout << "[DashboardControls] Fetched " << all_symbols_.size()
            << " symbols from exchange APIs" << std::endl;

  // Update the symbol input buffer to clear any previous search
  std::fill(symbol_input_buffer_.begin(), symbol_input_buffer_.end(), 0);

  // Update the selected symbol index if the previously selected symbol is no longer available
  if (selected_symbol_idx_ >= static_cast<int>(all_symbols_.size())) {
    selected_symbol_idx_ = -1;  // Reset to no selection
  }
}

void DashboardControls::update_all_chart_timeframes(RenderEngine::TimeFrame timeframe) {
  if (!panel_manager_) {
    std::cerr << "[DashboardControls] Error: PanelManager is null, cannot update chart timeframes"
              << std::endl;
    return;
  }

  // Update all charts managed by the chart manager
  auto chart_manager = panel_manager_->get_chart_manager();
  if (chart_manager) {
    chart_manager->update_all_chart_timeframes(timeframe);
  }

  // Get all panel IDs
  auto panel_ids = panel_manager_->get_all_panel_ids();

  // Iterate through all panels and update chart-based panels
  for (uint32_t panel_id : panel_ids) {
    auto panel = panel_manager_->get_panel_by_id(panel_id);
    if (!panel) continue;

    // Update all chart-based panels with the new timeframe
    // Check for all chart-related panel types that might be affected by timeframe changes
    if (panel->get_config().type == PanelType::CHART ||
        panel->get_config().type == PanelType::FOOTPRINT_CHART ||
        panel->get_config().type == PanelType::VOLUME_PROFILE ||
        panel->get_config().type == PanelType::DEPTH_CHART ||
        panel->get_config().type == PanelType::TIME_SERIES ||
        panel->get_config().type == PanelType::TIME_STATISTICS ||
        panel->get_config().type == PanelType::TIME_HISTOGRAM ||
        panel->get_config().type == PanelType::TIME_AND_SALES ||
        panel->get_config().type == PanelType::HISTORICAL_TIME_SALES ||
        panel->get_config().type == PanelType::TPO_PROFILE ||
        panel->get_config().type == PanelType::CHART_REPLAY) {
      // Handle different chart panel types that have specific timeframe methods
      if (panel->get_config().type == PanelType::CHART) {
        // Regular chart panel - use set_timeframe method
        ChartPanel* chart_panel = dynamic_cast<ChartPanel*>(panel);
        if (chart_panel) {
          chart_panel->set_timeframe(timeframe);
        }
      } else if (panel->get_config().type == PanelType::CHART_REPLAY) {
        // Chart replay panel - update replay config
        auto* chart_replay_panel = dynamic_cast<class ChartReplayPanel*>(panel);
        if (chart_replay_panel) {
          chart_replay_panel->set_timeframe(timeframe);
        }
      } else if (panel->get_config().type == PanelType::FOOTPRINT_CHART) {
        // Footprint chart panel - update time aggregation type
        auto* footprint_panel = dynamic_cast<class FootprintPanel*>(panel);
        if (footprint_panel) {
          // Map the timeframe to the corresponding TimeAggregationType
          Data::TimeAggregationType time_agg_type;
          switch (timeframe) {
            case RenderEngine::TimeFrame::TF_1MIN:
              time_agg_type = Data::TimeAggregationType::T_1MIN;
              break;
            case RenderEngine::TimeFrame::TF_5MIN:
              time_agg_type = Data::TimeAggregationType::T_5MIN;
              break;
            case RenderEngine::TimeFrame::TF_15MIN:
              time_agg_type = Data::TimeAggregationType::T_15MIN;
              break;
            case RenderEngine::TimeFrame::TF_30MIN:
              time_agg_type = Data::TimeAggregationType::T_30MIN;
              break;
            case RenderEngine::TimeFrame::TF_1HOUR:
              time_agg_type = Data::TimeAggregationType::T_1HOUR;
              break;
            case RenderEngine::TimeFrame::TF_2HOUR:
              time_agg_type = Data::TimeAggregationType::T_2HOUR;
              break;
            case RenderEngine::TimeFrame::TF_4HOUR:
              time_agg_type = Data::TimeAggregationType::T_4HOUR;
              break;
            case RenderEngine::TimeFrame::TF_1DAY:
            case RenderEngine::TimeFrame::TF_1WEEK:
            default:
              time_agg_type = Data::TimeAggregationType::T_1MIN;
              break;
          }

          footprint_panel->setTimeAggregationType(time_agg_type);
        }
      }
      // For other chart-like panels that don't have direct timeframe setters,
      // they will automatically receive updated data when the underlying chart manager
      // updates the data sources with the new timeframe
      // The panels will refresh their data on the next render cycle due to the reactive
      // architecture
    }
  }
}

void DashboardControls::sync_symbol_to_all_panels(uint32_t symbol_id, const std::string& symbol) {
  if (!panel_manager_) {
    std::cerr << "[DashboardControls] Error: PanelManager is null, cannot sync symbol to panels"
              << std::endl;
    return;
  }

  // Get all panel IDs
  auto panel_ids = panel_manager_->get_all_panel_ids();

  // Iterate through all panels and update their symbols
  for (uint32_t panel_id : panel_ids) {
    auto panel = panel_manager_->get_panel_by_id(panel_id);
    if (!panel) continue;

    // Update the panel with the new symbol
    panel_manager_->set_panel_symbol(panel_id, symbol);

    // Log the update for debugging
    std::cout << "[DashboardControls] Updated panel " << panel_id << " with symbol: " << symbol
              << " (ID: " << symbol_id << ")" << std::endl;
  }

  std::cout << "[DashboardControls] Successfully synced symbol " << symbol << " (ID: " << symbol_id
            << ") to all " << panel_ids.size() << " panels" << std::endl;
}

void DashboardControls::render_global_alerts_section() {
  if (ImGui::CollapsingHeader("Global Alerts", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (global_alert_manager_) {
      // Show alert statistics
      ImGui::Text("Alert Statistics:");
      ImGui::Indent();
      ImGui::Text("- Total Alerts: %zu", global_alert_manager_->get_total_alerts_count());
      ImGui::Text("- Active Alerts: %zu", global_alert_manager_->get_active_alerts_count());
      ImGui::Text("- Triggered Alerts: %zu", global_alert_manager_->get_triggered_alerts_count());
      ImGui::Unindent();

      ImGui::Spacing();

      // Buttons for managing alerts
      if (ImGui::Button("Create New Alert")) {
        show_create_alert_modal_ = true;
      }

      ImGui::SameLine();
      if (ImGui::Button("View All Alerts")) {
        // This could open a detailed view of all alerts
        // For now, we'll just show a simple list in the same window
      }

      ImGui::Spacing();

      // Show active alerts table
      auto all_alerts = global_alert_manager_->get_all_alerts();
      if (!all_alerts.empty()) {
        ImGui::Text("Active Alerts:");
        ImGui::BeginTable("AlertsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg);
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Symbol");
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("Threshold");
        ImGui::TableSetupColumn("Status");
        ImGui::TableHeadersRow();

        for (const auto& [id, alert] : all_alerts) {
          ImGui::TableNextRow();

          ImGui::TableSetColumnIndex(0);
          ImGui::Text("%s", alert.name.c_str());

          ImGui::TableSetColumnIndex(1);
          ImGui::Text("%s", alert.symbol_name.c_str());

          ImGui::TableSetColumnIndex(2);
          switch (alert.type) {
            case GlobalAlertType::PRICE_ABOVE:
              ImGui::Text("Price >");
              break;
            case GlobalAlertType::PRICE_BELOW:
              ImGui::Text("Price <");
              break;
            case GlobalAlertType::VOLUME_ABOVE:
              ImGui::Text("Volume >");
              break;
            case GlobalAlertType::VOLUME_BELOW:
              ImGui::Text("Volume <");
              break;
            default:
              ImGui::Text("Custom");
              break;
          }

          ImGui::TableSetColumnIndex(3);
          ImGui::Text("%.2f", alert.threshold_value);

          ImGui::TableSetColumnIndex(4);
          switch (alert.status) {
            case AlertStatus::ACTIVE:
              ImGui::Text("Active");
              break;
            case AlertStatus::TRIGGERED:
              ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Triggered");
              break;
            case AlertStatus::DISABLED:
              ImGui::Text("Disabled");
              break;
            case AlertStatus::COOLDOWN:
              ImGui::Text("Cooldown");
              break;
            default:
              ImGui::Text("Unknown");
              break;
          }
        }
        ImGui::EndTable();
      } else {
        ImGui::Text("No alerts configured. Click 'Create New Alert' to add one.");
      }
    } else {
      ImGui::Text("Global Alert Manager not initialized.");
    }
  }
}

void DashboardControls::render_create_alert_modal() {
  if (show_create_alert_modal_) {
    ImGui::OpenPopup("Create New Alert");
  }

  if (ImGui::BeginPopupModal("Create New Alert", &show_create_alert_modal_,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Create a new price or volume alert");
    ImGui::Separator();

    // Alert name
    ImGui::InputTextWithHint("Alert Name", "Enter alert name...", new_alert_name_,
                             sizeof(new_alert_name_));

    // Symbol selection
    if (selected_symbol_idx_ >= 0 && selected_symbol_idx_ < static_cast<int>(all_symbols_.size())) {
      strncpy(new_alert_symbol_, all_symbols_[selected_symbol_idx_].c_str(),
              sizeof(new_alert_symbol_) - 1);
      new_alert_symbol_[sizeof(new_alert_symbol_) - 1] = '\0';
    }
    ImGui::InputTextWithHint("Symbol", "Enter symbol...", new_alert_symbol_,
                             sizeof(new_alert_symbol_));

    // Alert type selection
    const char* alert_types[] = {"Price Above", "Price Below", "Volume Above", "Volume Below"};
    ImGui::Combo("Alert Type", &new_alert_type_, alert_types, IM_ARRAYSIZE(alert_types));

    // Threshold value
    ImGui::InputTextWithHint("Threshold Value", "Enter threshold...", new_alert_threshold_,
                             sizeof(new_alert_threshold_));

    ImGui::Spacing();

    // Buttons
    if (ImGui::Button("Create Alert", ImVec2(120, 0))) {
      create_alert_from_modal();
      show_create_alert_modal_ = false;
      // Clear the input fields
      memset(new_alert_name_, 0, sizeof(new_alert_name_));
      memset(new_alert_symbol_, 0, sizeof(new_alert_symbol_));
      memset(new_alert_threshold_, 0, sizeof(new_alert_threshold_));
      new_alert_type_ = 0;
    }

    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      show_create_alert_modal_ = false;
      // Clear the input fields
      memset(new_alert_name_, 0, sizeof(new_alert_name_));
      memset(new_alert_symbol_, 0, sizeof(new_alert_symbol_));
      memset(new_alert_threshold_, 0, sizeof(new_alert_threshold_));
      new_alert_type_ = 0;
    }

    ImGui::EndPopup();
  }
}

void DashboardControls::create_alert_from_modal() {
  if (!global_alert_manager_) {
    std::cerr << "[DashboardControls] Error: Global Alert Manager not initialized" << std::endl;
    return;
  }

  // Validate inputs
  if (strlen(new_alert_symbol_) == 0) {
    std::cerr << "[DashboardControls] Error: Symbol is required" << std::endl;
    return;
  }

  if (strlen(new_alert_threshold_) == 0) {
    std::cerr << "[DashboardControls] Error: Threshold is required" << std::endl;
    return;
  }

  // Convert threshold to double
  double threshold_value = 0.0;
  try {
    threshold_value = std::stod(new_alert_threshold_);
  } catch (const std::exception& e) {
    std::cerr << "[DashboardControls] Error parsing threshold value: " << e.what() << std::endl;
    return;
  }

  // Get symbol ID from registry
  uint32_t symbol_id = 0;
  auto symbol_info_opt =
      SymbolRegistry::instance().get_symbol_by_name(std::string(new_alert_symbol_));
  if (symbol_info_opt) {
    symbol_id = symbol_info_opt->id;
  } else {
    // If not found, try to register it
    symbol_id =
        SymbolRegistry::instance().register_symbol("Default", std::string(new_alert_symbol_));
  }

  if (symbol_id == 0) {
    std::cerr << "[DashboardControls] Error: Could not get or register symbol ID" << std::endl;
    return;
  }

  // Determine alert type and create the alert
  std::string alert_name = strlen(new_alert_name_) > 0
                               ? std::string(new_alert_name_)
                               : std::string("Alert: ") + std::string(new_alert_symbol_);

  GlobalAlertType alert_type;
  switch (new_alert_type_) {
    case 0:  // Price Above
      alert_type = GlobalAlertType::PRICE_ABOVE;
      break;
    case 1:  // Price Below
      alert_type = GlobalAlertType::PRICE_BELOW;
      break;
    case 2:  // Volume Above
      alert_type = GlobalAlertType::VOLUME_ABOVE;
      break;
    case 3:  // Volume Below
      alert_type = GlobalAlertType::VOLUME_BELOW;
      break;
    default:
      alert_type = GlobalAlertType::PRICE_ABOVE;
      break;
  }

  std::string alert_id;
  if (alert_type == GlobalAlertType::PRICE_ABOVE || alert_type == GlobalAlertType::PRICE_BELOW) {
    alert_id = global_alert_manager_->add_price_alert(symbol_id, std::string(new_alert_symbol_),
                                                      threshold_value, alert_type, alert_name);
  } else {
    alert_id = global_alert_manager_->add_volume_alert(symbol_id, std::string(new_alert_symbol_),
                                                       threshold_value, alert_type, alert_name);
  }

  if (!alert_id.empty()) {
    std::cout << "[DashboardControls] Created new alert: " << alert_id << " - " << alert_name
              << std::endl;
  } else {
    std::cerr << "[DashboardControls] Failed to create alert" << std::endl;
  }
}

}  // namespace BTQuant