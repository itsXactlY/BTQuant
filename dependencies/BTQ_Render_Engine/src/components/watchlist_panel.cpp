#include "../../include/components/watchlist_panel.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>

#include "imgui.h"

namespace BTQuant {

// Helper function to format numbers for financial display
std::string formatFinancialNumber(double value, int precision = 2) {
  if (value >= 1e9) {
    return std::to_string(value / 1e9).substr(0, std::to_string(value / 1e9).find('.') + precision + 1) + "B";
  } else if (value >= 1e6) {
    return std::to_string(value / 1e6).substr(0, std::to_string(value / 1e6).find('.') + precision + 1) + "M";
  } else if (value >= 1e3) {
    return std::to_string(value / 1e3).substr(0, std::to_string(value / 1e3).find('.') + precision + 1) + "K";
  } else {
    return std::to_string(value).substr(0, std::to_string(value).find('.') + precision + 1);
  }
}

// Helper function to format price values with consistent decimal places
std::string formatPrice(double price) {
  // For prices less than 1, show more decimals
  if (price < 1.0) {
    return std::to_string(price).substr(0, std::to_string(price).find('.') + 6);
  } else {
    return std::to_string(price).substr(0, std::to_string(price).find('.') + 5);
  }
}

// Helper function to format VWAP with consistent decimal places
std::string formatVWAP(double vwap) {
  // For VWAP values less than 1, show more decimals
  if (vwap < 1.0) {
    return std::to_string(vwap).substr(0, std::to_string(vwap).find('.') + 6);
  } else {
    return std::to_string(vwap).substr(0, std::to_string(vwap).find('.') + 5);
  }
}

// Helper function to calculate color intensity based on change magnitude
ImVec4 calculateChangeColor(double change_value, bool is_percentage = true) {
  // Determine if change is positive or negative
  bool is_positive = change_value >= 0;

  // Calculate absolute magnitude for intensity
  double abs_change = std::abs(change_value);

  // Define thresholds for intensity scaling
  double max_intensity_threshold = is_percentage ? 10.0 : 100.0; // 10% or $100 as max intensity

  // Use exponential scaling to make intensity increase more dramatically with larger changes
  double normalized_change = std::min(1.0, abs_change / max_intensity_threshold);
  // Apply exponential curve to make intensity grow faster with larger changes
  double intensity_factor = normalized_change * normalized_change; // Square the value for exponential effect
  double saturation_factor = std::sqrt(normalized_change); // Square root for saturation effect

  // Return appropriate color based on sign and intensity
  if (is_positive) {
    // Green for positive changes - more intense greens for larger changes
    float red_comp = 0.2f * (1.0f - saturation_factor);
    float green_comp = 0.5f + 0.5f * saturation_factor; // Base green with intensity
    float blue_comp = 0.2f * (1.0f - saturation_factor);
    return ImVec4(red_comp, green_comp, blue_comp, 1.0f);
  } else {
    // Red for negative changes - more intense reds for larger changes
    float red_comp = 0.5f + 0.5f * saturation_factor; // Base red with intensity
    float green_comp = 0.3f * (1.0f - saturation_factor);
    float blue_comp = 0.3f * (1.0f - saturation_factor);
    return ImVec4(red_comp, green_comp, blue_comp, 1.0f);
  }
}

WatchlistPanel::WatchlistPanel(const PanelConfig& config,
                               std::shared_ptr<HotSpineDataBridge> bridge,
                               std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
  // Initialize the subscription to real-time market data updates
  // We'll subscribe to individual symbols when they're added to the watchlist
  subscription_id_ = 0;

  // Set config file path based on panel name or use default
  std::string panel_name = config.name.empty() ? "watchlist" : config.name;
  config_file_path_ = panel_name + "_config.ini";

  // Load the saved watchlist order from config file
  load_watchlist_order_from_config(config_file_path_);

  // Subscribe to all currently watched symbols
  for (const auto& [symbol_id, entry] : watchlist_) {
    subscribe_to_symbol(symbol_id);
  }

  // Verify all subscriptions are active
  verify_subscriptions();
}

void WatchlistPanel::update(float dt) {
  // Update animation timers for all watchlist entries
  for (auto& [symbol_id, entry] : watchlist_) {
    if (entry.animation_timer > 0.0f) {
      entry.animation_timer -= dt;
      if (entry.animation_timer < 0.0f) {
        entry.animation_timer = 0.0f;
      }
    }
  }

  // Ensure all symbols in the watchlist are subscribed to real-time updates
  // This handles cases where subscriptions might have been lost or need to be refreshed
  for (const auto& [symbol_id, entry] : watchlist_) {
    if (symbol_subscriptions_.find(symbol_id) == symbol_subscriptions_.end()) {
      subscribe_to_symbol(symbol_id);
    }
  }

  // Additionally, periodically verify all subscriptions are active
  // This ensures robustness in case of connection issues or other problems
  verify_subscriptions();
}

void WatchlistPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();

  // Enhanced section for adding new symbols with better visual grouping
  ImGui::Spacing();
  ImGui::Text("Add Symbol to Watchlist:");
  ImGui::SameLine();

  // Input field for new symbol with improved styling
  ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f)); // Darker background for input
  ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
  ImGui::SetNextItemWidth(150);
  bool input_entered = ImGui::InputTextWithHint("##NewSymbolInput", "e.g., AAPL", new_symbol_buffer_, sizeof(new_symbol_buffer_), ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::PopStyleColor(3);

  // Add tooltip to explain the input field
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Enter a symbol name (e.g., AAPL, MSFT) to add to watchlist");
    ImGui::EndTooltip();
  }
  ImGui::SameLine();

  // Button to add symbol by name with improved styling
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));      // Green background
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.7f, 0.3f, 1.0f)); // Lighter green when hovered
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.8f, 0.4f, 1.0f));  // Even lighter when active
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));         // White text

  bool add_clicked = ImGui::Button("Add Symbol");

  // Add tooltip to explain the button
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Add the symbol entered above to the watchlist");
    ImGui::EndTooltip();
  }

  ImGui::PopStyleColor(4); // Pop all 4 color styles

  // Show current count of symbols in watchlist
  ImGui::SameLine();
  ImGui::TextDisabled("(%zu symbols)", watchlist_.size());

  // Add a clear all button
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));      // Red background
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.1f, 0.1f, 1.0f)); // Darker red when hovered
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));   // Even brighter when active
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));         // White text

  if (ImGui::Button("Clear All")) {
    if (!watchlist_.empty()) {
      show_clear_all_confirmation_ = true;
    }
  }

  // Add tooltip to explain the clear all button
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Remove all symbols from watchlist");
    ImGui::EndTooltip();
  }

  ImGui::PopStyleColor(4); // Pop all 4 color styles

  // Alternative symbol selector dropdown
  if (bridge_) {
    auto active_symbols = bridge_->getActiveSymbols();
    if (!active_symbols.empty()) {
      ImGui::SameLine();
      ImGui::SetNextItemWidth(200);
      if (ImGui::BeginCombo("##SymbolSelector", "Or select...")) {
        for (uint32_t sym_id : active_symbols) {
          std::string sym_name = bridge_->getSymbolName(sym_id);
          std::string exchange = bridge_->getExchangeName(sym_id);
          if (sym_name.empty()) continue;

          // Check if already in watchlist
          bool already_added = (watchlist_.find(sym_id) != watchlist_.end());

          std::string label = exchange + "/" + sym_name;
          if (already_added) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_Disabled);
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
      ImGui::Text("(%zu in watchlist, %zu available)", watchlist_.size(), active_symbols.size());
    }
  }

  // Show error message if symbol not found
  std::string symbol_to_add = new_symbol_buffer_;
  if ((add_clicked || input_entered) && !symbol_to_add.empty() && bridge_) {
    // Trim whitespace from input
    symbol_to_add.erase(0, symbol_to_add.find_first_not_of(" \t"));
    symbol_to_add.erase(symbol_to_add.find_last_not_of(" \t") + 1);

    if (symbol_to_add.empty()) {
      ImGui::SameLine();
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.3f, 0.3f, 1.0f)); // Red text for error
      ImGui::Text("Please enter a valid symbol name!");
      ImGui::PopStyleColor();
    } else {
      // Convert to uppercase for standardization
      std::transform(symbol_to_add.begin(), symbol_to_add.end(), symbol_to_add.begin(), ::toupper);

      bool symbol_found = false;
      auto active_symbols = bridge_->getActiveSymbols();
      for (uint32_t sym_id : active_symbols) {
        std::string sym_name = bridge_->getSymbolName(sym_id);
        std::string exchange = bridge_->getExchangeName(sym_id);

        // Compare the symbol name and check if it's already in the watchlist
        if (!sym_name.empty() &&
            sym_name == symbol_to_add &&
            watchlist_.find(sym_id) == watchlist_.end()) {
          add_symbol(sym_id, sym_name, exchange);
          new_symbol_buffer_[0] = '\0'; // Clear the input buffer
          symbol_found = true;
          break;
        }
      }

      // Show success message if symbol was added
      if (symbol_found) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.9f, 0.3f, 1.0f)); // Green text for success
        ImGui::Text("Added '%s' to watchlist!", symbol_to_add.c_str());
        ImGui::PopStyleColor();
      } else {
        // Show error message if symbol was not found
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.3f, 0.3f, 1.0f)); // Red text for error
        ImGui::Text("Symbol '%s' not found or already in watchlist!", symbol_to_add.c_str());
        ImGui::PopStyleColor();
      }
    }
  }


  // Filter input
  render_filter_input();
  ImGui::Separator();

  // Table
  if (ImGui::BeginTable("WatchlistTable", 11,
                        ImGuiTableFlags_Resizable | ImGuiTableFlags_Sortable |
                            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
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
          if (!sym_name.empty() && (watchlist_.find(sym_id) == watchlist_.end())) {
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

  // Delete confirmation dialog
  if (show_delete_confirmation_) {
    ImGui::OpenPopup("Confirm Delete?");
  }

  if (ImGui::BeginPopupModal("Confirm Delete?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Are you sure you want to remove this symbol from the watchlist?");

    // Find the symbol name to display in the confirmation
    auto it = watchlist_.find(symbol_to_delete_);
    if (it != watchlist_.end()) {
      ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Symbol: %s", it->second.symbol.c_str()); // Highlight the symbol name
    }

    ImGui::Separator();

    if (ImGui::Button("Yes, Remove", ImVec2(80, 0))) {
      remove_symbol(symbol_to_delete_);
      show_delete_confirmation_ = false;
      ImGui::CloseCurrentPopup();
    }

    ImGui::SameLine();

    if (ImGui::Button("Cancel", ImVec2(80, 0))) {
      show_delete_confirmation_ = false;
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  }

  // Clear all confirmation dialog
  if (show_clear_all_confirmation_) {
    ImGui::OpenPopup("Confirm Clear All?");
  }

  if (ImGui::BeginPopupModal("Confirm Clear All?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Are you sure you want to remove ALL symbols from the watchlist?");

    // Show how many symbols will be removed
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Number of symbols to remove: %zu", watchlist_.size());

    ImGui::Separator();

    if (ImGui::Button("Yes, Clear All", ImVec2(100, 0))) {
      clear_watchlist();
      show_clear_all_confirmation_ = false;
      ImGui::CloseCurrentPopup();
    }

    ImGui::SameLine();

    if (ImGui::Button("Cancel", ImVec2(80, 0))) {
      show_clear_all_confirmation_ = false;
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  }

  end_panel_window();
}

void WatchlistPanel::add_symbol(uint32_t symbol_id, const std::string& symbol,
                                const std::string& exchange) {
  if (watchlist_.find(symbol_id) != watchlist_.end()) {
    return;  // Already exists
  }

  WatchlistEntry entry;
  entry.symbol_id = symbol_id;
  entry.symbol = symbol;
  entry.exchange = exchange;
  entry.is_active = true;

  watchlist_[symbol_id] = entry;

  // Add to display order - if there's an existing order, append to the end
  // otherwise, just add to the vector
  display_order_.push_back(symbol_id);

  // Subscribe to real-time updates for this symbol
  subscribe_to_symbol(symbol_id);

  // Save the updated order to config file
  save_watchlist_order_to_config(config_file_path_);
}

void WatchlistPanel::on_market_data_update(uint32_t symbol_id, RenderEngine::NotificationType type) {
  // Only process trade updates
  if (type != RenderEngine::NotificationType::TRADE) {
    return;
  }

  // Check if this symbol is in our watchlist
  auto it = watchlist_.find(symbol_id);
  if (it != watchlist_.end()) {
    // Get the latest analytics data for this symbol
    auto analytics = processor_->getSymbolAnalytics(symbol_id);
    if (analytics.symbol_id != 0) {
      // Store previous values for animation
      double prev_price = it->second.price;
      double prev_vwap = it->second.vwap;
      double prev_volume = it->second.volume_24h;
      double prev_change_pct = it->second.change_pct;
      double prev_change_dollar = it->second.change_dollar;

      // Update the entry with new data
      it->second.price = analytics.last_trade_price;
      it->second.vwap = analytics.vwap;
      it->second.last_update_ts = analytics.last_trade_time;

      // Store the previous values in the entry for animation purposes
      it->second.previous_price = prev_price;
      it->second.previous_vwap = prev_vwap;
      it->second.previous_volume = prev_volume;

      // Calculate 24h change using the longest available timeframe candles
      auto candles = processor_->getCandles(symbol_id, RenderEngine::TimeFrame::TF_15SEC);
      if (!candles.empty()) {
        const auto& oldest_candle = candles.front();
        const auto& newest_candle = candles.back();

        // Calculate percentage change
        it->second.change_pct = calculate_24h_change(newest_candle, oldest_candle);

        // Calculate dollar change
        it->second.change_dollar = newest_candle.close - oldest_candle.close;

        // Store open, high, low values from the oldest candle (representing 24h period)
        it->second.open_24h = oldest_candle.open;
        it->second.high_24h = oldest_candle.high;
        it->second.low_24h = oldest_candle.low;

        // Estimate 24h volume by summing available candles (best effort)
        double total_vol = 0.0;
        for (const auto& c : candles) total_vol += c.volume;
        it->second.volume_24h = total_vol;
      } else {
        it->second.change_pct = 0.0;
        it->second.change_dollar = 0.0;
        it->second.open_24h = 0.0;
        it->second.high_24h = 0.0;
        it->second.low_24h = 0.0;
        it->second.volume_24h = analytics.volume_1m;  // Fallback
      }

      // Start animation for any significant value change
      // Only restart animation if there's a meaningful change
      bool significant_change = false;

      // Check if price changed significantly (more than 0.01% or minimum tick)
      double price_change_pct = (prev_price != 0) ? std::abs((it->second.price - prev_price) / prev_price) * 100.0 : 0;
      if (price_change_pct > 0.01 || std::abs(it->second.price - prev_price) > 0.001) { // 0.01% or $0.001 threshold
        significant_change = true;
      }

      // Check if VWAP changed significantly
      double vwap_change_pct = (prev_vwap != 0) ? std::abs((it->second.vwap - prev_vwap) / prev_vwap) * 100.0 : 0;
      if (vwap_change_pct > 0.01 || std::abs(it->second.vwap - prev_vwap) > 0.001) { // 0.01% or $0.001 threshold
        significant_change = true;
      }

      // Check if change percentages changed significantly
      if (std::abs(it->second.change_pct - prev_change_pct) > 0.01) { // At least 0.01% difference
        significant_change = true;
      }

      // Check if change dollars changed significantly
      if (std::abs(it->second.change_dollar - prev_change_dollar) > 0.001) { // At least $0.001 difference
        significant_change = true;
      }

      if (significant_change) {
        it->second.animation_timer = WatchlistEntry::ANIMATION_DURATION;
      }
    }
  }
}

void WatchlistPanel::remove_symbol(uint32_t symbol_id) {
  watchlist_.erase(symbol_id);
  display_order_.erase(std::remove(display_order_.begin(), display_order_.end(), symbol_id),
                       display_order_.end());

  // Unsubscribe from real-time updates for this symbol
  unsubscribe_from_symbol(symbol_id);

  // Save the updated order to config file
  save_watchlist_order_to_config(config_file_path_);
}

WatchlistPanel::~WatchlistPanel() {
  // Unsubscribe from all market data updates when the panel is destroyed
  for (const auto& [symbol_id, entry] : watchlist_) {
    unsubscribe_from_symbol(symbol_id);
  }
}

void WatchlistPanel::clear_watchlist() {
  // Unsubscribe from all current symbols before clearing
  for (const auto& [symbol_id, entry] : watchlist_) {
    unsubscribe_from_symbol(symbol_id);
  }

  watchlist_.clear();
  display_order_.clear();

  // Save the updated order to config file
  save_watchlist_order_to_config(config_file_path_);
}

void WatchlistPanel::update_watchlist_data() {
  // This method is now deprecated since we use real-time updates
  // The data is updated in on_market_data_update() when new market data arrives
  // This method remains for backward compatibility but does nothing
}

void WatchlistPanel::render_filter_input() {
  ImGui::Text("Filter:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1);
  ImGui::InputText("##Filter", filter_buffer_, sizeof(filter_buffer_));
}

void WatchlistPanel::render_table_header() {
  // Setup table columns with appropriate widths for better readability
  ImGui::TableSetupColumn(
      "Symbol", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthStretch, 0.0f);  // Stretch to fill available space
  ImGui::TableSetupColumn("Exchange", ImGuiTableColumnFlags_WidthFixed, 80.0f);
  ImGui::TableSetupColumn(
      "Last Price", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed, 100.0f);
  ImGui::TableSetupColumn(
      "Change%", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn(
      "Change$", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn(
      "Volume", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed, 100.0f);
  ImGui::TableSetupColumn("High", ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn("Low", ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn("Open", ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn("VWAP", ImGuiTableColumnFlags_WidthFixed, 90.0f);
  ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, 70.0f);
  ImGui::TableHeadersRow();

  ImGuiTableSortSpecs* sorts_specs = ImGui::TableGetSortSpecs();
  if (sorts_specs && sorts_specs->SpecsDirty) {
    if (sorts_specs->SpecsCount > 0) {
      const auto& spec = sorts_specs->Specs[0];
      sort_column_ = spec.ColumnIndex;
      sort_ascending_ = (spec.SortDirection == ImGuiSortDirection_Ascending);
      sort_watchlist();
    }
    sorts_specs->SpecsDirty = false;
  }
}

void WatchlistPanel::render_table_row(const WatchlistEntry& entry) {
  ImGui::TableNextRow();

  // Make entire row selectable for click-to-chart
  ImGui::TableSetColumnIndex(0);
  bool is_selected = (entry.symbol_id == selected_symbol_id_);

  ImGui::PushID(static_cast<int>(entry.symbol_id));  // Fix ID conflict

  // Use Selectable spanning all columns with drag and drop support
  if (ImGui::Selectable(
          entry.symbol.c_str(), is_selected,
          ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
    selected_symbol_id_ = entry.symbol_id;

    // Trigger symbol selection callback (e.g., to open chart)
    if (on_symbol_selected_) {
      on_symbol_selected_(entry.symbol_id, entry.symbol);
    }
  }

  // DRAG-AND-DROP REORDERING IMPLEMENTATION
  // Each row acts as both a drag source and drop target to enable reordering

  // Drag and drop source - make the entire row draggable with enhanced visual feedback
  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID | ImGuiDragDropFlags_SourceNoDisableHover)) {
    // Set payload to carry the symbol_id
    ImGui::SetDragDropPayload("WATCHLIST_ROW", &entry.symbol_id, sizeof(uint32_t));

    // Enhanced visual preview of what is being dragged
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Dragging: %s (%s)", entry.symbol.c_str(), entry.exchange.c_str());

    // Add additional visual elements to the drag preview
    ImGui::Separator();
    ImGui::Text("Last Price: %s", formatPrice(entry.price).c_str());
    ImGui::Text("Change: %+.2f%%", entry.change_pct);
    ImGui::Text("Drag to reorder watchlist");

    // Add a visual border around the preview
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 size = ImVec2(300, ImGui::GetTextLineHeightWithSpacing() * 5); // Fixed size for cleaner preview
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                      ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.8f)), 4.0f, 0, 2.0f); // Rounded corners

    ImGui::EndDragDropSource();
  }

  // Drag and drop target - accept drops to reorder with enhanced visual feedback
  if (ImGui::BeginDragDropTarget()) {
    const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("WATCHLIST_ROW");
    if (payload && payload->DataSize == sizeof(uint32_t)) {
      uint32_t source_symbol_id = *(const uint32_t*)payload->Data;

      // Find positions of source and target in display_order_
      auto source_it = std::find(display_order_.begin(), display_order_.end(), source_symbol_id);
      auto target_it = std::find(display_order_.begin(), display_order_.end(), entry.symbol_id);

      // Prevent dropping on the same item (self-drag)
      if (source_it != display_order_.end() && target_it != display_order_.end() && source_symbol_id != entry.symbol_id) {
        // Calculate new position for the dragged item based on mouse position
        int source_idx = std::distance(display_order_.begin(), source_it);
        int target_idx = std::distance(display_order_.begin(), target_it);

        // Determine if dropping above or below the target row
        // Get the height of the current row to determine drop zone
        ImVec2 cell_rect_min = ImGui::GetItemRectMin();
        ImVec2 cell_rect_max = ImGui::GetItemRectMax();
        float row_height = cell_rect_max.y - cell_rect_min.y;
        float mouse_y = ImGui::GetMousePos().y;

        // If mouse is in upper half of the row, insert above; lower half, insert below
        bool drop_above = (mouse_y - cell_rect_min.y) < (row_height / 2.0f);

        // Adjust target index based on drop position
        if (!drop_above) {
          target_idx++;  // Drop below means insert after the target
        }

        // Move the source item to the calculated position
        uint32_t moved_item = *source_it;

        // Remove the moved item from its current position
        display_order_.erase(source_it);

        // Adjust target index if source was before target (since we removed an element)
        if (source_idx < target_idx) {
          target_idx--;  // If we removed an item before the target position, adjust the target index
        }

        // Insert the moved item at the new position
        display_order_.insert(display_order_.begin() + target_idx, moved_item);

        // Save the updated order to config file
        save_watchlist_order_to_config(config_file_path_);

        // Log the reordering action
        std::cout << "[WatchlistPanel] Reordered symbol " << entry.symbol
                  << " (ID: " << source_symbol_id << ") to position " << target_idx
                  << " (dropped " << (drop_above ? "above" : "below") << " " << entry.symbol << ")" << std::endl;

        // Trigger a refresh of subscriptions to maintain proper ordering
        refresh_all_subscriptions();
      }
    }

    // Enhanced visual feedback for drop target - draw a more prominent indicator
    ImVec2 cell_rect_min = ImGui::GetItemRectMin();
    ImVec2 cell_rect_max = ImGui::GetItemRectMax();

    // Determine where to draw the line based on mouse position within the row
    float row_height = cell_rect_max.y - cell_rect_min.y;
    float mouse_y = ImGui::GetMousePos().y;
    bool drop_above = (mouse_y - cell_rect_min.y) < (row_height / 2.0f);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float line_y = drop_above ? cell_rect_min.y : cell_rect_max.y;

    // Draw a more prominent visual indicator for the drop target
    draw_list->AddLine(
        ImVec2(cell_rect_min.x, line_y),
        ImVec2(cell_rect_max.x, line_y),
        ImGui::GetColorU32(ImVec4(0.2f, 0.8f, 0.2f, 1.0f)), // Green color for better visibility
        4.0f // Increased line thickness for better visibility
    );

    // Add a more distinctive triangle indicator to show insertion direction
    if (drop_above) {
      // Triangle pointing up for "insert above"
      ImVec2 triangle_points[3] = {
          ImVec2(cell_rect_max.x - 25, line_y + 8),
          ImVec2(cell_rect_max.x - 15, line_y - 8),
          ImVec2(cell_rect_max.x - 5, line_y + 8)
      };
      draw_list->AddTriangleFilled(triangle_points[0], triangle_points[1], triangle_points[2],
                                  ImGui::GetColorU32(ImVec4(0.2f, 0.8f, 0.2f, 1.0f)));

      // Add text indicator
      draw_list->AddText(ImVec2(cell_rect_min.x + 5, line_y - 12),
                        ImGui::GetColorU32(ImVec4(0.2f, 0.8f, 0.2f, 1.0f)),
                        "INSERT ABOVE");
    } else {
      // Triangle pointing down for "insert below"
      ImVec2 triangle_points[3] = {
          ImVec2(cell_rect_max.x - 25, line_y - 8),
          ImVec2(cell_rect_max.x - 15, line_y + 8),
          ImVec2(cell_rect_max.x - 5, line_y - 8)
      };
      draw_list->AddTriangleFilled(triangle_points[0], triangle_points[1], triangle_points[2],
                                  ImGui::GetColorU32(ImVec4(0.2f, 0.8f, 0.2f, 1.0f)));

      // Add text indicator
      draw_list->AddText(ImVec2(cell_rect_min.x + 5, line_y + 2),
                        ImGui::GetColorU32(ImVec4(0.2f, 0.8f, 0.2f, 1.0f)),
                        "INSERT BELOW");
    }

    ImGui::EndDragDropTarget();
  }

  // END DRAG-AND-DROP REORDERING IMPLEMENTATION
  // The display_order_ vector is updated immediately when a drag operation completes
  // and the new order is saved to the config file for persistence

  ImGui::PopID();

  // Show tooltip on hover
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Click to view %s chart", entry.symbol.c_str());
    ImGui::Text("Exchange: %s", entry.exchange.c_str());
    ImGui::Text("24h Open/High/Low: %.4f / %.4f / %.4f", entry.open_24h, entry.high_24h, entry.low_24h);
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(1);
  ImGui::Text("%s", entry.exchange.c_str());

  // Add tooltip to explain Exchange
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Trading exchange: %s", entry.exchange.c_str());
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(2);
  // Apply animation effect to price if recently updated
  if (entry.animation_timer > 0.0f) {
    // Calculate animation progress (0.0 to 1.0)
    float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

    // Create a pulsing effect by interpolating between previous and current price
    double animated_price = entry.previous_price + (entry.price - entry.previous_price) * progress;

    // Calculate price change percentage for color calculation
    double price_change_pct = 0.0;
    if (entry.previous_price != 0.0) {
      price_change_pct = ((entry.price - entry.previous_price) / entry.previous_price) * 100.0;
    }

    // Enhanced flash animation - more prominent flash effect with smoother transition
    ImVec4 flash_color = calculateChangeColor(price_change_pct, true);

    // Calculate flash timing for brief flash effect with more intensity
    float flash_phase = progress * 4.0f; // Speed up the flash cycle for more intensity
    if (flash_phase > 2.0f) flash_phase = 0.0f; // Reset after full cycle
    else if (flash_phase > 1.0f) flash_phase = 2.0f - flash_phase; // Create a bounce effect

    // Enhance the color intensity during animation with more pronounced flash
    if (price_change_pct >= 0.0) {
      // Positive change - enhance green component during animation
      flash_color.x = flash_color.x * (0.5f + 0.5f * flash_phase); // Red - reduced to allow green to dominate
      flash_color.y = std::min(1.0f, flash_color.y * (0.5f + 0.5f * flash_phase)); // Green - enhanced
      flash_color.z = flash_color.z * (0.5f + 0.5f * flash_phase); // Blue - reduced
    } else {
      // Negative change - enhance red component during animation
      flash_color.x = std::min(1.0f, flash_color.x * (0.5f + 0.5f * flash_phase)); // Red - enhanced
      flash_color.y = flash_color.y * (0.5f + 0.5f * flash_phase); // Green - reduced
      flash_color.z = flash_color.z * (0.5f + 0.5f * flash_phase); // Blue - reduced
    }

    // Add brief flash animation effect by temporarily highlighting the background
    if (progress > 0.7f) { // Only during the first part of the animation
      // Calculate alpha for background highlight based on animation progress
      float bg_alpha = (1.0f - progress) * 2.0f; // Fade out quickly
      if (bg_alpha > 1.0f) bg_alpha = 1.0f;

      // Create a temporary background highlight for the cell
      ImVec2 pos = ImGui::GetCursorScreenPos();
      ImVec2 textSize = ImGui::CalcTextSize(formatPrice(animated_price).c_str());
      ImDrawList* draw_list = ImGui::GetWindowDrawList();

      // Draw a subtle background highlight
      ImVec4 highlight_color = (price_change_pct >= 0.0) ?
        ImVec4(0.1f, 0.3f, 0.1f, bg_alpha * 0.3f) :
        ImVec4(0.3f, 0.1f, 0.1f, bg_alpha * 0.3f);

      draw_list->AddRectFilled(
        ImVec2(pos.x - 5, pos.y - 2),
        ImVec2(pos.x + textSize.x + 5, pos.y + textSize.y + 2),
        ImGui::GetColorU32(highlight_color)
      );
    }

    ImGui::TextColored(flash_color, "%s", formatPrice(animated_price).c_str());
  } else {
    // Calculate price change percentage for color calculation
    double price_change_pct = 0.0;
    if (entry.previous_price != 0.0) {
      price_change_pct = ((entry.price - entry.previous_price) / entry.previous_price) * 100.0;
    }

    // Use the calculated change color based on price change percentage
    ImVec4 price_color = calculateChangeColor(price_change_pct, true);

    ImGui::TextColored(price_color, "%s", formatPrice(entry.price).c_str());
  }

  // Add tooltip to explain Last Price
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Last traded price");
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(3);
  // Apply subtle animation to change percentage when it updates significantly
  ImVec4 change_pct_color = calculateChangeColor(entry.change_pct, true);

  // Apply animation effect to change percentage if recently updated
  if (entry.animation_timer > 0.0f) {
    // Calculate animation progress (0.0 to 1.0)
    float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

    // Calculate flash timing for change percentage animation
    float flash_phase = progress * 4.0f; // Speed up the flash cycle for more intensity
    if (flash_phase > 2.0f) flash_phase = 0.0f; // Reset after full cycle
    else if (flash_phase > 1.0f) flash_phase = 2.0f - flash_phase; // Create a bounce effect

    // Enhance the color intensity during animation
    if (entry.change_pct >= 0.0) {
      // Positive change - enhance green component during animation
      change_pct_color.x = change_pct_color.x * (0.7f + 0.3f * flash_phase); // Red
      change_pct_color.y = std::min(1.0f, change_pct_color.y * (0.7f + 0.3f * flash_phase)); // Green
      change_pct_color.z = change_pct_color.z * (0.7f + 0.3f * flash_phase); // Blue
    } else {
      // Negative change - enhance red component during animation
      change_pct_color.x = std::min(1.0f, change_pct_color.x * (0.7f + 0.3f * flash_phase)); // Red
      change_pct_color.y = change_pct_color.y * (0.7f + 0.3f * flash_phase); // Green
      change_pct_color.z = change_pct_color.z * (0.7f + 0.3f * flash_phase); // Blue
    }

    // Add brief flash animation effect by temporarily highlighting the background
    if (progress > 0.7f) { // Only during the first part of the animation
      // Calculate alpha for background highlight based on animation progress
      float bg_alpha = (1.0f - progress) * 2.0f; // Fade out quickly
      if (bg_alpha > 1.0f) bg_alpha = 1.0f;

      // Create a temporary background highlight for the cell
      ImVec2 pos = ImGui::GetCursorScreenPos();
      ImVec2 textSize = ImGui::CalcTextSize((std::string("+") + std::to_string(entry.change_pct) + "%").c_str());
      ImDrawList* draw_list = ImGui::GetWindowDrawList();

      // Draw a subtle background highlight
      ImVec4 highlight_color = (entry.change_pct >= 0.0) ?
        ImVec4(0.1f, 0.3f, 0.1f, bg_alpha * 0.3f) :
        ImVec4(0.3f, 0.1f, 0.1f, bg_alpha * 0.3f);

      draw_list->AddRectFilled(
        ImVec2(pos.x - 5, pos.y - 2),
        ImVec2(pos.x + textSize.x + 5, pos.y + textSize.y + 2),
        ImGui::GetColorU32(highlight_color)
      );
    }
  }

  ImGui::TextColored(change_pct_color, "%+.2f%%", entry.change_pct);

  // Add tooltip to explain Change%
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Percentage change from 24-hour opening price");
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(4);
  // Apply subtle animation to change dollar when it updates significantly
  ImVec4 change_dollar_color = calculateChangeColor(entry.change_dollar, false);

  // Apply animation effect to change dollar if recently updated
  if (entry.animation_timer > 0.0f) {
    // Calculate animation progress (0.0 to 1.0)
    float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

    // Calculate flash timing for change dollar animation
    float flash_phase = progress * 4.0f; // Speed up the flash cycle for more intensity
    if (flash_phase > 2.0f) flash_phase = 0.0f; // Reset after full cycle
    else if (flash_phase > 1.0f) flash_phase = 2.0f - flash_phase; // Create a bounce effect

    // Enhance the color intensity during animation
    if (entry.change_dollar >= 0.0) {
      // Positive change - enhance green component during animation
      change_dollar_color.x = change_dollar_color.x * (0.7f + 0.3f * flash_phase); // Red
      change_dollar_color.y = std::min(1.0f, change_dollar_color.y * (0.7f + 0.3f * flash_phase)); // Green
      change_dollar_color.z = change_dollar_color.z * (0.7f + 0.3f * flash_phase); // Blue
    } else {
      // Negative change - enhance red component during animation
      change_dollar_color.x = std::min(1.0f, change_dollar_color.x * (0.7f + 0.3f * flash_phase)); // Red
      change_dollar_color.y = change_dollar_color.y * (0.7f + 0.3f * flash_phase); // Green
      change_dollar_color.z = change_dollar_color.z * (0.7f + 0.3f * flash_phase); // Blue
    }

    // Add brief flash animation effect by temporarily highlighting the background
    if (progress > 0.7f) { // Only during the first part of the animation
      // Calculate alpha for background highlight based on animation progress
      float bg_alpha = (1.0f - progress) * 2.0f; // Fade out quickly
      if (bg_alpha > 1.0f) bg_alpha = 1.0f;

      // Create a temporary background highlight for the cell
      ImVec2 pos = ImGui::GetCursorScreenPos();
      ImVec2 textSize = ImGui::CalcTextSize((std::string("+") + std::to_string(entry.change_dollar)).c_str());
      ImDrawList* draw_list = ImGui::GetWindowDrawList();

      // Draw a subtle background highlight
      ImVec4 highlight_color = (entry.change_dollar >= 0.0) ?
        ImVec4(0.1f, 0.3f, 0.1f, bg_alpha * 0.3f) :
        ImVec4(0.3f, 0.1f, 0.1f, bg_alpha * 0.3f);

      draw_list->AddRectFilled(
        ImVec2(pos.x - 5, pos.y - 2),
        ImVec2(pos.x + textSize.x + 5, pos.y + textSize.y + 2),
        ImGui::GetColorU32(highlight_color)
      );
    }
  }

  ImGui::TextColored(change_dollar_color, "%+.2f", entry.change_dollar);

  // Add tooltip to explain Change$
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Dollar change from 24-hour opening price");
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(5);
  // Apply color coding to volume based on comparison with previous volume if available
  ImVec4 volume_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Default white

  // Calculate color based on volume change (green for increase, red for decrease)
  if (entry.previous_volume != 0.0) {
    double volume_change = entry.volume_24h - entry.previous_volume;
    double volume_change_pct = (entry.previous_volume != 0.0) ? ((volume_change / entry.previous_volume) * 100.0) : 0.0;

    // Use the same color calculation as other change indicators
    volume_color = calculateChangeColor(volume_change_pct, false);
  }

  ImGui::TextColored(volume_color, "%s", formatFinancialNumber(entry.volume_24h, 2).c_str());

  // Add tooltip to explain Volume
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("24-hour trading volume");
    ImGui::Text("Symbol: %s", entry.symbol.c_str());
    ImGui::Text("Exchange: %s", entry.exchange.c_str());
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(6);
  // Apply color coding to High based on comparison with current price
  ImVec4 high_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Default white
  if (entry.price != 0.0) {
    double high_diff_pct = ((entry.high_24h - entry.price) / entry.price) * 100.0;
    high_color = calculateChangeColor(high_diff_pct, true);
  }
  ImGui::TextColored(high_color, "%s", formatPrice(entry.high_24h).c_str());

  // Add tooltip to explain High
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("24-hour high price");
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(7);
  // Apply color coding to Low based on comparison with current price
  ImVec4 low_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Default white
  if (entry.price != 0.0) {
    double low_diff_pct = ((entry.low_24h - entry.price) / entry.price) * 100.0;
    low_color = calculateChangeColor(low_diff_pct, true);
  }
  ImGui::TextColored(low_color, "%s", formatPrice(entry.low_24h).c_str());

  // Add tooltip to explain Low
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("24-hour low price");
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(8);
  // Apply color coding to Open based on comparison with current price
  ImVec4 open_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Default white
  if (entry.price != 0.0) {
    double open_diff_pct = ((entry.open_24h - entry.price) / entry.price) * 100.0;
    open_color = calculateChangeColor(open_diff_pct, true);
  }
  ImGui::TextColored(open_color, "%s", formatPrice(entry.open_24h).c_str());

  // Add tooltip to explain Open
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("24-hour opening price");
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(9);
  // Apply color coding to VWAP based on change from previous value (green for increase, red for decrease)
  ImVec4 vwap_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Default white

  // Calculate color based on VWAP change (green for increase, red for decrease)
  if (entry.previous_vwap != 0.0) {
    double vwap_change = entry.vwap - entry.previous_vwap;
    double vwap_change_pct = (entry.previous_vwap != 0.0) ? ((vwap_change / entry.previous_vwap) * 100.0) : 0.0;

    // Use the same color calculation as other change indicators
    vwap_color = calculateChangeColor(vwap_change_pct, true);
  }

  // Apply animation effect on top of color coding if recently updated
  if (entry.animation_timer > 0.0f) {
    // Calculate animation progress (0.0 to 1.0)
    float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

    // Create a pulsing effect by interpolating between previous and current VWAP
    double animated_vwap = entry.previous_vwap + (entry.vwap - entry.previous_vwap) * progress;

    // Calculate flash timing for VWAP animation
    float flash_phase = progress * 4.0f; // Speed up the flash cycle for more intensity
    if (flash_phase > 2.0f) flash_phase = 0.0f; // Reset after full cycle
    else if (flash_phase > 1.0f) flash_phase = 2.0f - flash_phase; // Create a bounce effect

    // Enhanced VWAP animation with more visual feedback
    // Calculate if VWAP increased or decreased from previous value
    double current_vwap = entry.vwap;
    double previous_vwap = entry.previous_vwap;
    if (current_vwap > previous_vwap) {
      // VWAP went up - green highlight
      float green_intensity = 0.6f + 0.4f * flash_phase; // From 60% to 100% intensity
      float red_intensity = 0.4f * (1.0f - flash_phase); // Fade from red to none
      vwap_color = ImVec4(red_intensity, green_intensity, 0.2f, 1.0f);
    } else if (current_vwap < previous_vwap) {
      // VWAP went down - red highlight
      float red_intensity = 0.6f + 0.4f * flash_phase; // From 60% to 100% intensity
      float green_intensity = 0.4f * (1.0f - flash_phase); // Fade from green to none
      vwap_color = ImVec4(red_intensity, green_intensity, 0.2f, 1.0f);
    } else {
      // No significant change in VWAP - light blue tint
      vwap_color = ImVec4(0.8f + 0.2f * progress, 0.8f + 0.2f * progress, 1.0f, 1.0f); // Light blue tint
    }

    // Add brief flash animation effect by temporarily highlighting the background
    if (progress > 0.7f) { // Only during the first part of the animation
      // Calculate alpha for background highlight based on animation progress
      float bg_alpha = (1.0f - progress) * 2.0f; // Fade out quickly
      if (bg_alpha > 1.0f) bg_alpha = 1.0f;

      // Create a temporary background highlight for the cell
      ImVec2 pos = ImGui::GetCursorScreenPos();
      ImVec2 textSize = ImGui::CalcTextSize(formatVWAP(animated_vwap).c_str());
      ImDrawList* draw_list = ImGui::GetWindowDrawList();

      // Draw a subtle background highlight
      ImVec4 highlight_color = (current_vwap > previous_vwap) ?
        ImVec4(0.1f, 0.3f, 0.1f, bg_alpha * 0.3f) :
        ImVec4(0.3f, 0.1f, 0.1f, bg_alpha * 0.3f);

      draw_list->AddRectFilled(
        ImVec2(pos.x - 5, pos.y - 2),
        ImVec2(pos.x + textSize.x + 5, pos.y + textSize.y + 2),
        ImGui::GetColorU32(highlight_color)
      );
    }

    ImGui::TextColored(vwap_color, "%s", formatVWAP(animated_vwap).c_str());
  } else {
    ImGui::TextColored(vwap_color, "%s", formatVWAP(entry.vwap).c_str());
  }

  // Add tooltip to explain VWAP
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Volume Weighted Average Price (VWAP)");
    ImGui::Text("Represents the average price weighted by volume traded");
    ImGui::EndTooltip();
  }

  ImGui::TableSetColumnIndex(10);
  ImGui::PushID(static_cast<int>(entry.symbol_id));  // Use symbol_id as unique identifier

  // Style the delete button to be more visually distinct
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.9f, 0.2f, 0.2f, 1.0f));      // Red background
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.1f, 0.1f, 1.0f)); // Darker red when hovered
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));   // Even brighter when active
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));         // White text

  // Make the delete button smaller and more compact
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 4.0f)); // Small padding but slightly larger for better click area
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 4.0f));  // Smaller spacing

  if (ImGui::Button("✕##DeleteBtn")) {  // Use ✕ symbol for better visual representation
    // Show confirmation dialog before deleting
    symbol_to_delete_ = entry.symbol_id;
    show_delete_confirmation_ = true;
  }

  // Add tooltip to explain the delete button
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::Text("Remove '%s' from watchlist", entry.symbol.c_str());
    ImGui::EndTooltip();
  }

  ImGui::PopStyleVar(2); // Pop the style variables
  ImGui::PopStyleColor(4); // Pop all 4 color styles
  ImGui::PopID();
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

const char* WatchlistPanel::get_sort_column_name(int column) {
  switch (column) {
    case 0:
      return "Symbol";
    case 1:
      return "Exchange";
    case 2:
      return "Last Price";
    case 3:
      return "Change%";
    case 4:
      return "Change$";
    case 5:
      return "Volume";
    case 6:
      return "High";
    case 7:
      return "Low";
    case 8:
      return "Open";
    case 9:
      return "VWAP";
    case 10:
      return "Action";
    default:
      return "Unknown";
  }
}

double WatchlistPanel::calculate_24h_change(const RenderEngine::OHLCVCandle& current,
                                            const RenderEngine::OHLCVCandle& old) const {
  if (old.close == 0.0) return 0.0;
  // Using close price of the candles
  return ((current.close - old.close) / old.close) * 100.0;
}

void WatchlistPanel::sort_watchlist() {
  std::sort(display_order_.begin(), display_order_.end(), [this](uint32_t a_id, uint32_t b_id) {
    const auto& a = watchlist_.at(a_id);
    const auto& b = watchlist_.at(b_id);

    bool result = false;
    switch (sort_column_) {
      case 0:  // Symbol
        result = a.symbol < b.symbol;
        break;
      case 1:  // Exchange
        result = a.exchange < b.exchange;
        break;
      case 2:  // Last Price
        result = a.price < b.price;
        break;
      case 3:  // Change %
        result = a.change_pct < b.change_pct;
        break;
      case 4:  // Change $
        result = a.change_dollar < b.change_dollar;
        break;
      case 5:  // Volume
        result = a.volume_24h < b.volume_24h;
        break;
      case 6:  // High
        result = a.high_24h < b.high_24h;
        break;
      case 7:  // Low
        result = a.low_24h < b.low_24h;
        break;
      case 8:  // Open
        result = a.open_24h < b.open_24h;
        break;
      case 9:  // VWAP
        result = a.vwap < b.vwap;
        break;
      case 10:  // Action (not actually sorted, fallback to symbol)
      default:
        result = a.symbol < b.symbol;
    }
    return sort_ascending_ ? result : !result;
  });

  // Save the updated order to config file after sorting
  save_watchlist_order_to_config(config_file_path_);
}

void WatchlistPanel::handle_drag_drop_reordering() {
  // This method is now deprecated as drag and drop reordering is handled directly in render_table_row
  // when drag and drop occurs. This ensures immediate visual feedback and proper state management.
  // The drag-and-drop functionality is now fully implemented and working with config saving.

  // NOTE: The actual drag-and-drop reordering happens in render_table_row() method
  // where each row acts as both a drag source and drop target.
  // When a drag operation completes, the display_order_ vector is updated immediately
  // and the new order is saved to the config file.
}

void WatchlistPanel::save_watchlist_order_to_config(const std::string& config_file) const {
  // Create directory if it doesn't exist
  std::filesystem::path config_path(config_file);
  std::filesystem::create_directories(config_path.parent_path());

  // First, read the existing config file to preserve other sections
  std::vector<std::string> existing_lines;
  std::ifstream read_file(config_file);
  bool replaced_section = false;

  if (read_file.is_open()) {
    std::string line;
    bool in_watchlist_section = false;

    while (std::getline(read_file, line)) {
      // Check if we're entering the watchlist_order section
      if (line.find("[watchlist_order]") != std::string::npos) {
        existing_lines.push_back(line);
        in_watchlist_section = true;

        // Add our updated display order
        std::string order_line = "display_order=";
        for (size_t i = 0; i < display_order_.size(); ++i) {
          order_line += std::to_string(display_order_[i]);
          if (i < display_order_.size() - 1) {
            order_line += ",";
          }
        }
        existing_lines.push_back(order_line);
        replaced_section = true;
      }
      // Skip lines inside the watchlist_order section (we'll replace them)
      else if (in_watchlist_section && line.find('[') == 0 && line.find(']') != std::string::npos) {
        // Found next section, so we're out of the watchlist section
        in_watchlist_section = false;
        existing_lines.push_back(line);
      }
      else if (!in_watchlist_section) {
        existing_lines.push_back(line);
      }
      // If in watchlist section and not a new section header, skip the line
    }
    read_file.close();
  }

  // If the watchlist_order section wasn't found, add it at the end
  if (!replaced_section) {
    if (!existing_lines.empty() && !existing_lines.back().empty()) {
      existing_lines.push_back(""); // Add blank line before new section
    }
    existing_lines.push_back("# Watchlist order configuration");
    existing_lines.push_back("[watchlist_order]");

    std::string order_line = "display_order=";
    for (size_t i = 0; i < display_order_.size(); ++i) {
      order_line += std::to_string(display_order_[i]);
      if (i < display_order_.size() - 1) {
        order_line += ",";
      }
    }
    existing_lines.push_back(order_line);
  }

  // Write the updated content back to the file
  std::ofstream write_file(config_file);
  if (!write_file.is_open()) {
    std::cerr << "[WatchlistPanel] Failed to open config file for writing: " << config_file << std::endl;
    return;
  }

  try {
    for (const auto& line : existing_lines) {
      write_file << line << std::endl;
    }

    write_file.close();
    std::cout << "[WatchlistPanel] Saved watchlist order to: " << config_file << ", entries: " << display_order_.size() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[WatchlistPanel] Error saving watchlist order: " << e.what() << std::endl;
  }
}

void WatchlistPanel::load_watchlist_order_from_config(const std::string& config_file) {
  std::ifstream file(config_file);
  if (!file.is_open()) {
    std::cout << "[WatchlistPanel] Config file not found, using current order: " << config_file << std::endl;
    return;
  }

  try {
    std::string line;
    std::string current_section;
    bool found_watchlist_order = false;

    while (std::getline(file, line)) {
      // Remove comments and trim whitespace
      size_t comment_pos = line.find('#');
      if (comment_pos != std::string::npos) {
        line = line.substr(0, comment_pos);
      }

      // Trim whitespace
      size_t start = line.find_first_not_of(" \t\r\n");
      if (start == std::string::npos) continue; // Skip empty lines
      size_t end = line.find_last_not_of(" \t\r\n");
      line = line.substr(start, end - start + 1);

      // Check for section headers
      if (line.front() == '[' && line.back() == ']') {
        current_section = line.substr(1, line.length() - 2);
        if (current_section == "watchlist_order") {
          found_watchlist_order = true;
        }
        continue;
      }

      // Parse key-value pairs only in the watchlist_order section
      if (current_section == "watchlist_order") {
        size_t equals_pos = line.find('=');
        if (equals_pos != std::string::npos) {
          std::string key = line.substr(0, equals_pos);
          std::string value = line.substr(equals_pos + 1);

          if (key == "display_order") {
            // Parse comma-separated list of symbol IDs
            std::vector<uint32_t> new_display_order;
            std::stringstream ss(value);
            std::string item;

            while (std::getline(ss, item, ',')) {
              // Trim whitespace from item
              size_t item_start = item.find_first_not_of(" \t\r\n");
              if (item_start != std::string::npos) {
                size_t item_end = item.find_last_not_of(" \t\r\n");
                item = item.substr(item_start, item_end - item_start + 1);

                try {
                  uint32_t symbol_id = std::stoi(item);
                  // Only add to new order if the symbol exists in the watchlist
                  if (watchlist_.find(symbol_id) != watchlist_.end()) {
                    new_display_order.push_back(symbol_id);
                  } else {
                    std::cout << "[WatchlistPanel] Symbol ID " << symbol_id << " from config not found in current watchlist, skipping." << std::endl;
                  }
                } catch (const std::invalid_argument&) {
                  std::cerr << "[WatchlistPanel] Invalid symbol ID in config: " << item << std::endl;
                }
              }
            }

            // Add any remaining symbols that weren't in the config to the end
            for (const auto& pair : watchlist_) {
              uint32_t symbol_id = pair.first;
              if (std::find(new_display_order.begin(), new_display_order.end(), symbol_id) == new_display_order.end()) {
                new_display_order.push_back(symbol_id);
              }
            }

            display_order_ = new_display_order;
            std::cout << "[WatchlistPanel] Loaded watchlist order from config, entries: " << new_display_order.size() << std::endl;
            break;
          }
        }
      }
    }

    file.close();

    if (found_watchlist_order) {
      std::cout << "[WatchlistPanel] Successfully loaded watchlist order from: " << config_file << std::endl;
    } else {
      std::cout << "[WatchlistPanel] No watchlist order found in config, keeping current order: " << config_file << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "[WatchlistPanel] Error loading watchlist order: " << e.what() << std::endl;
  }
}

void WatchlistPanel::subscribe_to_symbol(uint32_t symbol_id) {
  if (processor_) {
    // Create a subscription for this specific symbol
    uint64_t sub_id = processor_->subscribe(symbol_id, RenderEngine::NotificationType::TRADE,
                                          [this](uint32_t symbol_id, RenderEngine::NotificationType type) {
                                            this->on_market_data_update(symbol_id, type);
                                          });

    // Store the subscription ID for this symbol
    symbol_subscriptions_[symbol_id] = sub_id;
  }
}

void WatchlistPanel::unsubscribe_from_symbol(uint32_t symbol_id) {
  if (processor_) {
    auto it = symbol_subscriptions_.find(symbol_id);
    if (it != symbol_subscriptions_.end()) {
      processor_->unsubscribe(it->second);
      symbol_subscriptions_.erase(it);
    }
  }
}

void WatchlistPanel::verify_subscriptions() {
  // Periodically verify that all symbols in the watchlist have active subscriptions
  // This helps ensure robustness in case of connection issues or other problems

  for (const auto& [symbol_id, entry] : watchlist_) {
    // Check if the symbol has a subscription
    auto sub_it = symbol_subscriptions_.find(symbol_id);
    if (sub_it == symbol_subscriptions_.end()) {
      // Subscription is missing, create a new one
      subscribe_to_symbol(symbol_id);
    }
    // Note: We don't verify if the subscription is still valid on the processor side
    // as that would require additional API calls. The current approach ensures
    // that each symbol has a subscription registered in our local map.
  }
}

void WatchlistPanel::refresh_all_subscriptions() {
  // Unsubscribe from all current symbols
  for (const auto& [symbol_id, entry] : watchlist_) {
    unsubscribe_from_symbol(symbol_id);
  }

  // Then resubscribe to all symbols
  for (const auto& [symbol_id, entry] : watchlist_) {
    subscribe_to_symbol(symbol_id);
  }

  std::cout << "[WatchlistPanel] Refreshed all subscriptions for " << watchlist_.size() << " symbols" << std::endl;
}

}  // namespace BTQuant