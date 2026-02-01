#include "../../include/components/watchlist_panel.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

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

// Helper function to calculate color intensity based on change magnitude
ImVec4 calculateChangeColor(double change_value, bool is_percentage = true) {
  // Determine if change is positive or negative
  bool is_positive = change_value >= 0;

  // Calculate absolute magnitude for intensity
  double abs_change = std::abs(change_value);

  // Define thresholds for intensity scaling
  double max_intensity_threshold = is_percentage ? 10.0 : 100.0; // 10% or $100 as max intensity
  double min_intensity = 0.2f;  // Minimum color intensity
  double max_intensity = 0.9f;  // Maximum color intensity

  // Calculate intensity factor (clamped between 0 and 1)
  double intensity_factor = std::min(1.0, abs_change / max_intensity_threshold);
  double color_intensity = min_intensity + (max_intensity - min_intensity) * intensity_factor;

  // Return appropriate color based on sign and intensity
  if (is_positive) {
    // Green for positive changes
    return ImVec4(color_intensity, std::min(1.0, 0.3f + intensity_factor * 0.7f), 0.2f, 1.0f);
  } else {
    // Red for negative changes
    return ImVec4(std::min(1.0, 0.3f + intensity_factor * 0.7f), color_intensity * 0.3f, color_intensity * 0.3f, 1.0f);
  }
}

WatchlistPanel::WatchlistPanel(const PanelConfig& config,
                               std::shared_ptr<HotSpineDataBridge> bridge,
                               std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
  // Initialize the subscription to real-time market data updates
  // We'll subscribe to individual symbols when they're added to the watchlist
  subscription_id_ = 0;

  // Load the saved watchlist order from config file
  load_watchlist_order_from_config("watchlist_config.ini");

  // Subscribe to all currently watched symbols
  for (const auto& [symbol_id, entry] : watchlist_) {
    subscribe_to_symbol(symbol_id);
  }
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
}

void WatchlistPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();

  // Section for adding new symbols - enhanced UI with better visual grouping
  ImGui::Text("Add Symbol:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(150);
  bool input_entered = ImGui::InputTextWithHint("##NewSymbolInput", "e.g., AAPL", new_symbol_buffer_, sizeof(new_symbol_buffer_), ImGuiInputTextFlags_EnterReturnsTrue);

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
      bool symbol_found = false;
      auto active_symbols = bridge_->getActiveSymbols();
      for (uint32_t sym_id : active_symbols) {
        std::string sym_name = bridge_->getSymbolName(sym_id);
        std::string exchange = bridge_->getExchangeName(sym_id);
        if (!sym_name.empty() && sym_name == symbol_to_add && watchlist_.find(sym_id) == watchlist_.end()) {
          add_symbol(sym_id, sym_name, exchange);
          new_symbol_buffer_[0] = '\0'; // Clear the input buffer
          symbol_found = true;
          break;
        }
      }

      // Show error message if symbol was not found
      if (!symbol_found) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.3f, 0.3f, 1.0f)); // Red text for error
        ImGui::Text("Symbol '%s' not found!", symbol_to_add.c_str());
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
  save_watchlist_order_to_config("watchlist_config.ini");
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
      double prev_change_pct = it->second.change_pct;
      double prev_change_dollar = it->second.change_dollar;

      // Update the entry with new data
      it->second.price = analytics.last_trade_price;
      it->second.vwap = analytics.vwap;
      it->second.last_update_ts = analytics.last_trade_time;

      // Store the previous price in the entry for animation purposes
      it->second.previous_price = prev_price;

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
  save_watchlist_order_to_config("watchlist_config.ini");
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
  save_watchlist_order_to_config("watchlist_config.ini");
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
      "Symbol", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 100.0f);
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

  // Drag and drop source
  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
    // Set payload to carry the symbol_id
    ImGui::SetDragDropPayload("WATCHLIST_ROW", &entry.symbol_id, sizeof(uint32_t));

    // Display preview of what is being dragged with enhanced visual
    ImGui::Text("Dragging: %s (%s)", entry.symbol.c_str(), entry.exchange.c_str());
    ImGui::Text("Release to drop at new position");

    ImGui::EndDragDropSource();
  }

  // Drag and drop target
  if (ImGui::BeginDragDropTarget()) {
    const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("WATCHLIST_ROW");
    if (payload) {
      IM_ASSERT(payload->DataSize == sizeof(uint32_t));
      uint32_t source_symbol_id = *(const uint32_t*)payload->Data;

      // Find positions of source and target in display_order_
      auto source_it = std::find(display_order_.begin(), display_order_.end(), source_symbol_id);
      auto target_it = std::find(display_order_.begin(), display_order_.end(), entry.symbol_id);

      if (source_it != display_order_.end() && target_it != display_order_.end()) {
        // Calculate new position for the dragged item based on mouse position
        int source_idx = std::distance(display_order_.begin(), source_it);
        int target_idx = std::distance(display_order_.begin(), target_it);

        // Determine if dropping above or below the target row
        // Get the height of the current row to determine drop zone
        float row_height = ImGui::GetTextLineHeightWithSpacing();
        float current_y = ImGui::GetCursorScreenPos().y;
        float mouse_y = ImGui::GetMousePos().y;

        // If mouse is in upper half of the row, insert above; lower half, insert below
        bool drop_above = (mouse_y - current_y) < (row_height / 2.0f);

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
          // If we removed an item before the target position, adjust the target index
          if (source_idx < target_idx) {
            target_idx--;
          }
        }

        // Insert the moved item at the new position
        display_order_.insert(display_order_.begin() + target_idx, moved_item);

        // Save the updated order to config file
        save_watchlist_order_to_config("watchlist_config.ini");
      }
    }

    // Visual feedback for drop target - draw a horizontal line to indicate drop zone
    ImVec2 cell_rect_min = ImGui::GetItemRectMin();
    ImVec2 cell_rect_max = ImGui::GetItemRectMax();

    // Determine where to draw the line based on mouse position within the row
    float row_height = cell_rect_max.y - cell_rect_min.y;
    float mouse_y = ImGui::GetMousePos().y;
    bool drop_above = (mouse_y - cell_rect_min.y) < (row_height / 2.0f);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    float line_y = drop_above ? cell_rect_min.y : cell_rect_max.y;

    // Draw a visual indicator for the drop target
    draw_list->AddLine(
        ImVec2(cell_rect_min.x, line_y),
        ImVec2(cell_rect_max.x, line_y),
        ImGui::GetColorU32(ImVec4(0.4f, 0.8f, 1.0f, 1.0f)), // Bright blue color
        2.0f // Line thickness
    );

    ImGui::EndDragDropTarget();
  }

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

  ImGui::TableSetColumnIndex(2);
  // Apply animation effect to price if recently updated
  if (entry.animation_timer > 0.0f) {
    // Calculate animation progress (0.0 to 1.0)
    float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

    // Create a pulsing effect by interpolating between previous and current price
    double animated_price = entry.previous_price + (entry.price - entry.previous_price) * progress;

    // Enhanced flash animation - more prominent flash effect
    ImVec4 flash_color;

    // Calculate flash timing for brief flash effect with more intensity
    float flash_phase = progress * 3.0f; // Speed up the flash cycle for more intensity
    if (flash_phase > 1.0f) flash_phase = 2.0f - flash_phase; // Create a bounce effect
    if (flash_phase > 1.0f) flash_phase = 0.0f; // Clamp to 0 after bounce

    // More pronounced flash colors with better contrast
    if (entry.price > entry.previous_price) {
      // Price went up - bright green flash then fade to normal
      float green_intensity = 0.5f + 0.5f * flash_phase;
      float red_intensity = 0.2f * (1.0f - flash_phase);
      flash_color = ImVec4(red_intensity, green_intensity, 0.2f * (1.0f - flash_phase), 1.0f);
    } else if (entry.price < entry.previous_price) {
      // Price went down - bright red flash then fade to normal
      float red_intensity = 0.5f + 0.5f * flash_phase;
      float green_intensity = 0.3f * (1.0f - flash_phase);
      flash_color = ImVec4(red_intensity, green_intensity, green_intensity, 1.0f);
    } else {
      // No change - use normal color
      flash_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    ImGui::TextColored(flash_color, "%s", formatPrice(animated_price).c_str());
  } else {
    ImGui::Text("%s", formatPrice(entry.price).c_str());
  }

  ImGui::TableSetColumnIndex(3);
  // Apply subtle animation to change percentage when it updates significantly
  ImVec4 change_pct_color = calculateChangeColor(entry.change_pct, true);
  ImGui::TextColored(change_pct_color, "%+.2f%%", entry.change_pct);

  ImGui::TableSetColumnIndex(4);
  // Apply subtle animation to change dollar when it updates significantly
  ImVec4 change_dollar_color = calculateChangeColor(entry.change_dollar, false);
  ImGui::TextColored(change_dollar_color, "%+.2f", entry.change_dollar);

  ImGui::TableSetColumnIndex(5);
  // Format volume with K/M/B suffix for readability
  ImGui::Text("%s", formatFinancialNumber(entry.volume_24h, 2).c_str());

  ImGui::TableSetColumnIndex(6);
  ImGui::Text("%s", formatPrice(entry.high_24h).c_str());

  ImGui::TableSetColumnIndex(7);
  ImGui::Text("%s", formatPrice(entry.low_24h).c_str());

  ImGui::TableSetColumnIndex(8);
  ImGui::Text("%s", formatPrice(entry.open_24h).c_str());

  ImGui::TableSetColumnIndex(9);
  // Apply subtle animation to VWAP when it updates significantly
  ImVec4 vwap_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Default white
  if (entry.animation_timer > 0.0f) {
    // Slightly highlight VWAP when there's an active animation
    float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);
    vwap_color = ImVec4(0.8f + 0.2f * progress, 0.8f + 0.2f * progress, 1.0f, 1.0f); // Light blue tint
  }
  ImGui::TextColored(vwap_color, "%s", formatPrice(entry.vwap).c_str());

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

  if (ImGui::Button("×##DeleteBtn")) {  // Use × symbol for better visual representation
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
  save_watchlist_order_to_config("watchlist_config.ini");
}

void WatchlistPanel::handle_drag_drop_reordering() {
  // This method handles the drag and drop reordering logic
  // The actual reordering is handled in render_table_row when drag and drop occurs
  // This method can be used for additional processing if needed

  // Currently, the drag and drop reordering is handled directly in render_table_row
  // This method is kept for potential future enhancements or cleanup operations
}

void WatchlistPanel::save_watchlist_order_to_config(const std::string& config_file) const {
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
    std::cout << "[WatchlistPanel] Saved watchlist order to: " << config_file << std::endl;
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
            break;
          }
        }
      }
    }

    file.close();

    if (found_watchlist_order) {
      std::cout << "[WatchlistPanel] Loaded watchlist order from: " << config_file << std::endl;
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

}  // namespace BTQuant