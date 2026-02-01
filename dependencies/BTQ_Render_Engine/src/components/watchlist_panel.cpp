#include "../../include/components/watchlist_panel.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cmath>

#include "imgui.h"

namespace BTQuant {

// Helper method implementations for group management
std::map<uint32_t, WatchlistEntry>& WatchlistPanel::get_current_watchlist() {
  // Make sure the group exists in the map before returning
  if (watchlist_groups_.find(current_group_name_) == watchlist_groups_.end()) {
    watchlist_groups_[current_group_name_] = std::map<uint32_t, WatchlistEntry>();
  }
  return watchlist_groups_[current_group_name_];
}

const std::map<uint32_t, WatchlistEntry>& WatchlistPanel::get_current_watchlist() const {
  auto it = watchlist_groups_.find(current_group_name_);
  if (it != watchlist_groups_.end()) {
    return it->second;
  }
  static std::map<uint32_t, WatchlistEntry> empty_map;
  return empty_map;
}

std::vector<uint32_t>& WatchlistPanel::get_current_display_order() {
  // Make sure the group exists in the map before returning
  if (group_display_orders_.find(current_group_name_) == group_display_orders_.end()) {
    group_display_orders_[current_group_name_] = std::vector<uint32_t>();
  }
  return group_display_orders_[current_group_name_];
}

const std::vector<uint32_t>& WatchlistPanel::get_current_display_order() const {
  auto it = group_display_orders_.find(current_group_name_);
  if (it != group_display_orders_.end()) {
    return it->second;
  }
  static std::vector<uint32_t> empty_vector;
  return empty_vector;
}

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

// Enhanced interpolation function for smoother transitions
double interpolateValue(double start, double end, float progress) {
  // Use quintic easing for even smoother animation with better acceleration/deceleration
  float t = progress * progress * progress * (progress * (progress * 6.0f - 15.0f) + 10.0f);
  return start + (end - start) * t;
}

// Enhanced flash animation function with smoother transitions
float calculateFlashIntensity(float progress) {
  // Use a smoother flash curve with more gradual fade-in and fade-out
  // Using sine-based easing for even smoother transitions
  float t = progress * 4.0f; // Speed up the flash cycle
  if (t > 2.0f) t = 4.0f - t; // Create a smooth bounce effect (triangle wave)

  // Apply sine-based easing for even smoother transitions
  t = (1.0f - cosf(t * M_PI)) * 0.5f; // Smooth easing function

  return t / 2.0f; // Normalize to 0-1 range
}

// Enhanced flash animation function specifically for price changes
float calculatePriceFlashIntensity(float progress) {
  // Use a more pronounced flash effect for price changes
  // This creates a more noticeable visual feedback when prices update

  // Use a cubic easing function for a more dramatic effect
  float t = progress;
  if (t < 0.5f) {
    // Accelerate quickly at the beginning
    t = 4.0f * t * t * t;
  } else {
    // Decelerate smoothly at the end
    t = (t - 1.0f) * (2.0f * t - 2.0f) * (2.0f * t - 2.0f) + 1.0f;
  }

  return t;
}

// Enhanced animation function for smooth value transitions with better visual feedback
double animateValueTransition(double start, double end, float progress, bool& is_animating) {
  // Use quintic easing for smooth acceleration and deceleration
  float t = progress * progress * progress * (progress * (progress * 6.0f - 15.0f) + 10.0f);
  double result = start + (end - start) * t;

  // Mark as animating if progress is less than 1.0
  is_animating = (progress < 1.0f);

  return result;
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
ImVec4 calculateChangeColor(double change_value, bool is_percentage) {
  // Determine if change is positive or negative
  bool is_positive = change_value >= 0;

  // Calculate absolute magnitude for intensity
  double abs_change = std::abs(change_value);

  // Define thresholds for intensity scaling - make more adaptive based on typical market movements
  double max_intensity_threshold = is_percentage ? 5.0 : 50.0; // Lower threshold for more sensitivity (5% or $50 as max intensity)

  // For extremely large changes, cap the intensity to prevent overly saturated colors
  double capped_change = std::min(abs_change, max_intensity_threshold * 2.0);

  // Use logarithmic scaling to make intensity increase more gradually with larger changes
  // This provides better visual distinction for smaller changes while preventing oversaturation
  double normalized_change = std::min(1.0, capped_change / max_intensity_threshold);

  // Apply a more balanced curve for intensity scaling - using a combination of linear and exponential
  double intensity_factor = normalized_change; // Base linear scaling
  double saturation_factor = normalized_change; // Use same factor for consistency

  // Return appropriate color based on sign and intensity
  if (is_positive) {
    // Bright green for positive changes - more intense greens for larger changes
    // Start with a bright green and make it more intense with larger changes
    float red_comp = 0.1f * (1.0f - saturation_factor); // Reduce red as intensity increases
    float green_comp = 0.4f + 0.6f * saturation_factor; // Increase green as intensity increases
    float blue_comp = 0.1f * (1.0f - saturation_factor); // Reduce blue as intensity increases
    return ImVec4(red_comp, green_comp, blue_comp, 1.0f);
  } else {
    // Bright red for negative changes - more intense reds for larger changes
    // Start with a bright red and make it more intense with larger changes
    float red_comp = 0.4f + 0.6f * saturation_factor; // Increase red as intensity increases
    float green_comp = 0.1f * (1.0f - saturation_factor); // Reduce green as intensity increases
    float blue_comp = 0.1f * (1.0f - saturation_factor); // Reduce blue as intensity increases
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
  std::string panel_name = "watchlist";
  config_file_path_ = panel_name + "_config.ini";

  // Initialize default watchlist groups with better organization
  create_group("Stocks");
  create_group("Crypto");
  create_group("Futures");

  // Ensure default groups are at the beginning of the group names list in the right order
  ensure_default_groups_order();

  // Set default group to "Stocks" initially
  current_group_name_ = "Stocks";

  // Initialize column settings
  initialize_column_settings();

  // Transfer any existing symbols from the old watchlist to the default group
  if (!watchlist_.empty()) {
    for (const auto& [symbol_id, entry] : watchlist_) {
      watchlist_groups_["Stocks"][symbol_id] = entry;
      group_display_orders_["Stocks"].push_back(symbol_id);
    }
    watchlist_.clear(); // Clear the old watchlist since we're moving to groups
  }

  // Load the saved watchlist order from config file
  load_watchlist_order_from_config(config_file_path_);

  // Load column settings from config file
  load_column_settings_from_config(config_file_path_);

  // Initialize the alert manager
  alert_manager_ = std::make_shared<WatchlistAlertManager>(bridge_, processor_, nullptr);

  // Subscribe to all currently watched symbols using the new efficient method
  subscribe_to_all_watchlist_symbols();

  // Verify all subscriptions are active
  verify_subscriptions();

  std::cout << "[WatchlistPanel] Initialized with " << get_current_watchlist().size() << " symbols in '"
            << current_group_name_ << "' group and subscriptions" << std::endl;
}

void WatchlistPanel::update(float dt) {
  // Update animation timers for all watchlist entries in the current group to ensure smooth transitions
  for (auto& [symbol_id, entry] : get_current_watchlist()) {
    if (entry.animation_timer > 0.0f) {
      entry.animation_timer -= dt;
      if (entry.animation_timer < 0.0f) {
        entry.animation_timer = 0.0f;
      }
    }
  }

  // Update alerts for all watchlist symbols
  if (alert_manager_) {
    alert_manager_->update_alerts();
  }

  // Ensure all symbols in the current watchlist group are subscribed to real-time price feed updates
  // This handles cases where subscriptions might have been lost or need to be refreshed
  ensure_all_symbols_subscribed();

  // Additionally, periodically verify all subscriptions are active
  // This ensures robustness in case of connection issues or other problems
  verify_subscriptions();

  // Log subscription status periodically for debugging (every 10 seconds)
  static float subscription_check_timer = 0.0f;
  subscription_check_timer += dt;
  if (subscription_check_timer > 10.0f) {
    subscription_check_timer = 0.0f;
    std::cout << "[WatchlistPanel] Active symbols in '" << current_group_name_ << "': " << get_current_watchlist().size()
              << ", Active subscriptions: " << symbol_subscriptions_.size() << std::endl;

    // Log any discrepancies between watchlist and subscriptions
    for (const auto& [symbol_id, entry] : get_current_watchlist()) {
      if (symbol_subscriptions_.find(symbol_id) == symbol_subscriptions_.end()) {
        std::cout << "[WatchlistPanel] Missing subscription for symbol ID: " << symbol_id
                  << " (" << entry.symbol << ")" << std::endl;
      }
    }
  }

  // Handle real-time price updates for all symbols in the current watchlist group
  // Process any pending market data updates
  process_pending_updates();
}

void WatchlistPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();

  // Render group tabs
  render_group_tabs();

  // Enhanced section for adding new symbols with better visual grouping
  ImGui::Spacing();
  ImGui::Text("Add Symbol to Watchlist (%s):", current_group_name_.c_str());
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

  // Show current count of symbols in current watchlist group
  ImGui::SameLine();
  ImGui::TextDisabled("(%zu symbols in %s)", get_current_watchlist().size(), current_group_name_.c_str());

  // Add a clear all button
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));      // Red background
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.1f, 0.1f, 1.0f)); // Darker red when hovered
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));   // Even brighter when active
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));         // White text

  if (ImGui::Button("Clear All")) {
    if (!get_current_watchlist().empty()) {
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

          // Check if already in current watchlist group
          bool already_added = (get_current_watchlist().find(sym_id) != get_current_watchlist().end());

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
      ImGui::Text("(%zu in %s, %zu available)", get_current_watchlist().size(), current_group_name_.c_str(), active_symbols.size());
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

        // Compare the symbol name and check if it's already in the current watchlist group
        if (!sym_name.empty() &&
            sym_name == symbol_to_add &&
            get_current_watchlist().find(sym_id) == get_current_watchlist().end()) {
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
        ImGui::Text("Added '%s' to %s watchlist!", symbol_to_add.c_str(), current_group_name_.c_str());
        ImGui::PopStyleColor();
      } else {
        // Show error message if symbol was not found
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.3f, 0.3f, 1.0f)); // Red text for error
        ImGui::Text("Symbol '%s' not found or already in %s watchlist!", symbol_to_add.c_str(), current_group_name_.c_str());
        ImGui::PopStyleColor();
      }
    }
  }


  // Filter input
  render_filter_input();
  ImGui::Separator();

  // Count visible columns for table setup
  int visible_columns = 0;
  for (const auto& col : column_info_) {
    if (col.visible) {
      visible_columns++;
    }
  }

  // Table
  if (visible_columns > 0) {
    if (ImGui::BeginTable("WatchlistTable", visible_columns,
                          ImGuiTableFlags_Resizable | ImGuiTableFlags_Sortable |
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
      render_table_header();

      auto filtered_symbols = get_filtered_symbols();
      for (uint32_t symbol_id : filtered_symbols) {
        auto it = get_current_watchlist().find(symbol_id);
        if (it != get_current_watchlist().end()) {
          render_table_row(it->second);
        }
      }

      ImGui::EndTable();
    }
  } else {
    // Show a message when no columns are visible
    ImGui::Text("No columns are currently visible. Right-click on the header area to show columns.");
  }

  // Render the column context menu if needed
  render_column_context_menu();

  // Context menu for adding/removing symbols
  if (ImGui::BeginPopupContextWindow()) {
    if (ImGui::MenuItem("Add All Symbols")) {
      if (bridge_) {
        for (uint32_t sym_id : bridge_->getActiveSymbols()) {
          std::string sym_name = bridge_->getSymbolName(sym_id);
          std::string exchange = bridge_->getExchangeName(sym_id);
          if (!sym_name.empty() && (get_current_watchlist().find(sym_id) == get_current_watchlist().end())) {
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
  if (get_current_watchlist().find(symbol_id) != get_current_watchlist().end()) {
    return;  // Already exists in current group
  }

  WatchlistEntry entry;
  entry.symbol_id = symbol_id;
  entry.symbol = symbol;
  entry.exchange = exchange;
  entry.is_active = true;

  get_current_watchlist()[symbol_id] = entry;

  // Add to display order for current group - if there's an existing order, append to the end
  // otherwise, just add to the vector
  get_current_display_order().push_back(symbol_id);

  // Subscribe to real-time updates for this symbol
  subscribe_to_symbol(symbol_id);

  // Save the updated order to config file
  save_watchlist_order_to_config(config_file_path_);

  std::cout << "[WatchlistPanel] Added symbol " << symbol << " (ID: " << symbol_id
            << ") to group '" << current_group_name_ << "' and subscribed to real-time updates" << std::endl;
}

void WatchlistPanel::on_market_data_update(uint32_t symbol_id, RenderEngine::NotificationType type) {
  // Only process trade updates for real-time price feed
  if (type != RenderEngine::NotificationType::TRADE) {
    return;
  }

  // Check if this symbol is in our current watchlist group
  auto it = get_current_watchlist().find(symbol_id);
  if (it != get_current_watchlist().end()) {
    // Get the latest analytics data for this symbol
    auto analytics = processor_->getSymbolAnalytics(symbol_id);
    if (analytics.symbol_id != 0) {
      // Store previous values for animation
      double prev_price = it->second.price;
      double prev_vwap = it->second.vwap;
      double prev_volume = it->second.volume_24h;
      double prev_change_pct = it->second.change_pct;
      double prev_change_dollar = it->second.change_dollar;
      double prev_high_24h = it->second.high_24h;
      double prev_low_24h = it->second.low_24h;
      double prev_open_24h = it->second.open_24h;

      // Update the entry with new data from real-time price feed
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

      // Check if high/low/open changed significantly
      if (std::abs(it->second.high_24h - prev_high_24h) > 0.001) {
        significant_change = true;
      }
      if (std::abs(it->second.low_24h - prev_low_24h) > 0.001) {
        significant_change = true;
      }
      if (std::abs(it->second.open_24h - prev_open_24h) > 0.001) {
        significant_change = true;
      }

      if (significant_change) {
        // Reset animation timer to start fresh animation with brief flash effect
        it->second.animation_timer = WatchlistEntry::ANIMATION_DURATION;

        // Log the animation trigger for debugging
        std::cout << "[WatchlistPanel] Animation triggered for " << it->second.symbol
                  << " (ID: " << symbol_id << "). Price: " << prev_price << " -> " << it->second.price
                  << ", Change: " << price_change_pct << "%" << std::endl;
      }

      // Log every market data update for monitoring
      std::cout << "[WatchlistPanel] Market data update received for " << it->second.symbol
                << " (ID: " << symbol_id << "). New price: " << it->second.price
                << ", Timestamp: " << it->second.last_update_ts << std::endl;
    }
  }

  // Also check other groups to update their entries if needed
  for (auto& [group_name, group] : watchlist_groups_) {
    auto it = group.find(symbol_id);
    if (it != group.end() && group_name != current_group_name_) {
      // Update the entry in other groups too
      auto analytics = processor_->getSymbolAnalytics(symbol_id);
      if (analytics.symbol_id != 0) {
        // Store previous values for animation
        double prev_price = it->second.price;
        double prev_vwap = it->second.vwap;
        double prev_volume = it->second.volume_24h;
        double prev_change_pct = it->second.change_pct;
        double prev_change_dollar = it->second.change_dollar;
        double prev_high_24h = it->second.high_24h;
        double prev_low_24h = it->second.low_24h;
        double prev_open_24h = it->second.open_24h;

        // Update the entry with new data from real-time price feed
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

        // Check if high/low/open changed significantly
        if (std::abs(it->second.high_24h - prev_high_24h) > 0.001) {
          significant_change = true;
        }
        if (std::abs(it->second.low_24h - prev_low_24h) > 0.001) {
          significant_change = true;
        }
        if (std::abs(it->second.open_24h - prev_open_24h) > 0.001) {
          significant_change = true;
        }

        if (significant_change) {
          // Reset animation timer to start fresh animation with brief flash effect
          it->second.animation_timer = WatchlistEntry::ANIMATION_DURATION;

          // Log the animation trigger for debugging
          std::cout << "[WatchlistPanel] Animation triggered for " << it->second.symbol
                    << " (ID: " << symbol_id << ") in group '" << group_name << "'. Price: " << prev_price << " -> " << it->second.price
                    << ", Change: " << price_change_pct << "%" << std::endl;
        }

        // Log every market data update for monitoring
        std::cout << "[WatchlistPanel] Market data update received for " << it->second.symbol
                  << " (ID: " << symbol_id << ") in group '" << group_name << "'. New price: " << it->second.price
                  << ", Timestamp: " << it->second.last_update_ts << std::endl;
      }
    }
  }
}

void WatchlistPanel::remove_symbol(uint32_t symbol_id) {
  get_current_watchlist().erase(symbol_id);
  get_current_display_order().erase(std::remove(get_current_display_order().begin(), get_current_display_order().end(), symbol_id),
                                   get_current_display_order().end());

  // Unsubscribe from real-time updates for this symbol
  unsubscribe_from_symbol(symbol_id);

  // Save the updated order to config file
  save_watchlist_order_to_config(config_file_path_);
}

WatchlistPanel::~WatchlistPanel() {
  // Unsubscribe from all market data updates when the panel is destroyed
  // Iterate through all groups to unsubscribe from all symbols
  for (const auto& [group_name, group] : watchlist_groups_) {
    for (const auto& [symbol_id, entry] : group) {
      unsubscribe_from_symbol(symbol_id);
    }
  }

  // Also unsubscribe from the default watchlist if needed
  for (const auto& [symbol_id, entry] : watchlist_) {
    unsubscribe_from_symbol(symbol_id);
  }
}

void WatchlistPanel::clear_watchlist() {
  // Unsubscribe from all current symbols in the current group before clearing
  for (const auto& [symbol_id, entry] : get_current_watchlist()) {
    unsubscribe_from_symbol(symbol_id);
  }

  get_current_watchlist().clear();
  get_current_display_order().clear();

  // Save the updated order to config file
  save_watchlist_order_to_config(config_file_path_);
}

void WatchlistPanel::update_watchlist_data() {
  // This method is now deprecated since we use real-time updates
  // The data is updated in on_market_data_update() when new market data arrives
  // This method remains for backward compatibility but does nothing
}

void WatchlistPanel::render_group_tabs() {
  // Create tabs for different watchlist groups with enhanced styling
  ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, 6.0f);  // Round the tab corners more
  ImGui::PushStyleVar(ImGuiStyleVar_TabBorderSize, 1.5f); // Add thicker border to tabs
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12.0f, 8.0f)); // Add more padding to tabs

  // Customize tab colors for better visual hierarchy
  ImGui::PushStyleColor(ImGuiCol_Tab, ImVec4(0.18f, 0.18f, 0.18f, 0.86f));           // Inactive tab background
  ImGui::PushStyleColor(ImGuiCol_TabHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.8f));       // Hovered tab background
  ImGui::PushStyleColor(ImGuiCol_TabActive, ImVec4(0.25f, 0.25f, 0.25f, 1.0f));     // Active tab background
  ImGui::PushStyleColor(ImGuiCol_TabUnfocused, ImVec4(0.15f, 0.15f, 0.15f, 0.97f)); // Inactive tab when window unfocused
  ImGui::PushStyleColor(ImGuiCol_TabUnfocusedActive, ImVec4(0.2f, 0.2f, 0.2f, 1.0f)); // Active tab when window unfocused

  if (ImGui::BeginTabBar("WatchlistGroups", ImGuiTabBarFlags_Reorderable)) {
    for (const auto& group_name : group_names_) {
      bool is_selected = (group_name == current_group_name_);

      // Count symbols in this group for display
      auto it = watchlist_groups_.find(group_name);
      size_t symbol_count = (it != watchlist_groups_.end()) ? it->second.size() : 0;

      // Format the tab label with symbol count
      std::string tab_label = group_name + " (" + std::to_string(symbol_count) + ")";

      // Set tab item flags for better appearance
      ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;

      // Create the tab item with enhanced styling
      if (ImGui::BeginTabItem(tab_label.c_str(), nullptr, tab_flags)) {
        if (!is_selected) {
          // Switch to this group
          switch_to_group(group_name);
        }
        ImGui::EndTabItem();
      }

      // Add tooltip to each tab
      if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("Switch to %s watchlist group", group_name.c_str());
        ImGui::Text("Symbols in this group: %zu", symbol_count);

        // Show additional info if group is not a default group
        if (group_name != "Futures" && group_name != "Crypto" && group_name != "Stocks") {
          ImGui::Separator();
          ImGui::Text("Right-click to manage group");

          // Add context menu for custom groups
          if (ImGui::BeginPopupContextItem("GroupContextMenu")) {
            if (ImGui::MenuItem("Rename Group")) {
              // Future enhancement: implement group renaming with input dialog
              std::cout << "[WatchlistPanel] Rename functionality would be implemented here for: " << group_name << std::endl;
            }

            if (group_name != "Futures" && group_name != "Crypto" && group_name != "Stocks") {
              if (ImGui::MenuItem("Delete Group")) {
                delete_group(group_name);
              }
            }

            ImGui::EndPopup();
          }
        }

        ImGui::EndTooltip();
      }
    }

    // Add a '+' button to create new groups with better styling
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.35f, 0.25f, 1.0f));      // Greenish background
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.45f, 0.35f, 1.0f)); // Lighter green when hovered
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.55f, 0.45f, 1.0f));  // Even lighter when active
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));         // White text

    if (ImGui::Button("+##AddGroup")) {
      // Create a new group with a default name
      static int new_group_counter = 1;
      std::string new_group_name = "Group " + std::to_string(new_group_counter++);
      create_group(new_group_name);
      switch_to_group(new_group_name);
    }

    // Add tooltip to the add group button
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("Create a new watchlist group");
      ImGui::EndTooltip();
    }

    ImGui::PopStyleColor(4); // Pop all 4 color styles

    ImGui::EndTabBar();
  }

  // Restore original styles
  ImGui::PopStyleColor(5); // Pop the 5 color styles
  ImGui::PopStyleVar(3); // Pop the 3 style variables
}

void WatchlistPanel::render_filter_input() {
  ImGui::Text("Filter:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1);
  ImGui::InputText("##Filter", filter_buffer_, sizeof(filter_buffer_));
}

void WatchlistPanel::render_table_header() {
  // Sort visible columns by their order value to determine the display sequence
  std::vector<std::pair<int, int>> ordered_columns; // (original_index, order_value)
  for (int i = 0; i < static_cast<int>(column_info_.size()); ++i) {
    if (column_info_[i].visible) {
      ordered_columns.emplace_back(i, column_info_[i].order);
    }
  }

  // Sort by order value to determine display sequence
  std::sort(ordered_columns.begin(), ordered_columns.end(),
           [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
             return a.second < b.second;
           });

  // Create a mapping from table column index to original column index
  std::vector<int> table_to_original_index(ordered_columns.size());
  for (size_t i = 0; i < ordered_columns.size(); ++i) {
    table_to_original_index[i] = ordered_columns[i].first;
  }

  // Setup table columns with appropriate widths for better readability, respecting visibility settings
  for (const auto& [orig_idx, order_val] : ordered_columns) {
    ImGuiTableColumnFlags flags = ImGuiTableColumnFlags_None;

    // Set appropriate flags based on original column index
    switch (orig_idx) {
      case 0: // Symbol
        flags = ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthStretch;
        break;
      case 2: // Last Price
      case 3: // Change%
      case 4: // Change$
      case 5: // Volume
        flags = ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthFixed;
        break;
      case 10: // Action
        flags = ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed;
        break;
      default:
        flags = ImGuiTableColumnFlags_WidthFixed;
        break;
    }

    // Set up the column with the appropriate width
    float width = (flags & ImGuiTableColumnFlags_WidthStretch) ? 0.0f : column_info_[orig_idx].width;
    ImGui::TableSetupColumn(column_info_[orig_idx].name.c_str(), flags, width);
  }

  // Render headers with right-click detection and drag-and-drop support
  // We need to render each header individually to support drag-and-drop
  for (size_t idx = 0; idx < ordered_columns.size(); ++idx) {
    int orig_idx = ordered_columns[idx].first;

    // Render the header cell with drag-and-drop support
    ImGui::TableNextColumn();

    // Prepare the header text with sort indicator if this is the sort column
    std::string header_text = column_info_[orig_idx].name;
    if (orig_idx == sort_column_) {
      // Add sort direction indicator
      header_text += sort_ascending_ ? " \u2191" : " \u2193"; // Up arrow for ascending, Down arrow for descending
    }

    // Render the header text with sort indicator
    ImGui::Text("%s", header_text.c_str());

    // Check if the current visible column is being hovered for right-click
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
      clicked_column_index_ = orig_idx;  // Store the actual column index
      column_context_menu_open_ = true;
      ImGui::OpenPopup("ColumnContextMenu");
    }

    // Implement drag-and-drop for column reordering
    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
      // Set payload to carry the column index
      ImGui::SetDragDropPayload("COLUMN_REORDER", &orig_idx, sizeof(int));

      // Enhanced visual preview of what is being dragged
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Moving: %s", column_info_[orig_idx].name.c_str());

      // Add a visual border around the preview
      ImVec2 pos = ImGui::GetCursorScreenPos();
      ImVec2 size = ImVec2(200, ImGui::GetTextLineHeightWithSpacing() * 2); // Fixed size for cleaner preview
      ImDrawList* draw_list = ImGui::GetWindowDrawList();
      draw_list->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                        ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.8f)), 4.0f, 0, 2.0f); // Rounded corners

      ImGui::EndDragDropSource();
    }

    // Make this header a drop target
    if (ImGui::BeginDragDropTarget()) {
      const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("COLUMN_REORDER");
      if (payload && payload->DataSize == sizeof(int)) {
        int source_column = *(const int*)payload->Data;

        // Reorder the columns by updating their order values
        if (source_column != orig_idx) {
          reorder_columns(source_column, orig_idx);
        }
      }

      // Enhanced visual feedback for drop target - draw a more prominent indicator
      ImVec2 cell_rect_min = ImGui::GetItemRectMin();
      ImVec2 cell_rect_max = ImGui::GetItemRectMax();

      ImDrawList* draw_list = ImGui::GetWindowDrawList();

      // Draw a more prominent visual indicator for the drop target
      draw_list->AddLine(
          ImVec2(cell_rect_min.x, cell_rect_max.y),
          ImVec2(cell_rect_max.x, cell_rect_max.y),
          ImGui::GetColorU32(ImVec4(0.2f, 0.8f, 0.2f, 1.0f)), // Green color for better visibility
          3.0f // Increased line thickness for better visibility
      );

      // Add a more distinctive triangle indicator to show insertion direction
      ImVec2 triangle_points[3] = {
          ImVec2(cell_rect_max.x - 20, cell_rect_max.y - 10),
          ImVec2(cell_rect_max.x - 10, cell_rect_max.y),
          ImVec2(cell_rect_max.x, cell_rect_max.y - 10)
      };
      draw_list->AddTriangleFilled(triangle_points[0], triangle_points[1], triangle_points[2],
                                  ImGui::GetColorU32(ImVec4(0.2f, 0.8f, 0.2f, 1.0f)));

      ImGui::EndDragDropTarget();
    }
  }

  // Also detect right-click on the header row area (not just individual columns)
  // This allows showing the context menu when clicking in the header area but not on a specific column
  if (ImGui::TableGetColumnIndex() == -1 && ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
    // If we clicked in the header area but not on a specific column, show the context menu
    // We'll use -1 to indicate that no specific column was clicked
    clicked_column_index_ = -1;
    column_context_menu_open_ = true;
    ImGui::OpenPopup("ColumnContextMenu");
  }

  ImGuiTableSortSpecs* sorts_specs = ImGui::TableGetSortSpecs();
  if (sorts_specs && sorts_specs->SpecsDirty) {
    if (sorts_specs->SpecsCount > 0) {
      const auto& spec = sorts_specs->Specs[0];

      // Map the table column index to the original column index
      if (spec.ColumnIndex >= 0 && spec.ColumnIndex < static_cast<int>(table_to_original_index.size())) {
        sort_column_ = table_to_original_index[spec.ColumnIndex];
      } else {
        sort_column_ = spec.ColumnIndex; // Fallback
      }

      sort_ascending_ = (spec.SortDirection == ImGuiSortDirection_Ascending);
      sort_watchlist();
    }
    sorts_specs->SpecsDirty = false;
  }
}

void WatchlistPanel::render_table_row(const WatchlistEntry& entry) {
  ImGui::TableNextRow();

  // Make entire row selectable for click-to-chart
  int current_column = 0;

  // Column 0: Symbol
  if (column_info_[0].visible) {
    ImGui::TableSetColumnIndex(current_column);
    bool is_selected = (entry.symbol_id == selected_symbol_id_);

    ImGui::PushID(static_cast<int>(entry.symbol_id));  // Fix ID conflict

    // Visually highlight the selected row
    if (is_selected) {
      ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImVec4(0.2f, 0.3f, 0.6f, 0.5f))); // Blueish highlight for selected row
    }

    // Use Selectable spanning all columns with drag and drop support
    if (ImGui::Selectable(
            entry.symbol.c_str(), is_selected,
            ImGuiSelectableFlags_SpanAllColumns)) {  // Removed AllowDoubleClick to ensure single click triggers
      selected_symbol_id_ = entry.symbol_id;

      // Trigger symbol selection callback to switch all panels to this symbol
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

        // Find positions of source and target in current display order
        auto& current_order = get_current_display_order();
        auto source_it = std::find(current_order.begin(), current_order.end(), source_symbol_id);
        auto target_it = std::find(current_order.begin(), current_order.end(), entry.symbol_id);

        // Prevent dropping on the same item (self-drag)
        if (source_it != current_order.end() && target_it != current_order.end() && source_symbol_id != entry.symbol_id) {
          // Calculate new position for the dragged item based on mouse position
          int source_idx = std::distance(current_order.begin(), source_it);
          int target_idx = std::distance(current_order.begin(), target_it);

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
          current_order.erase(source_it);

          // Adjust target index if source was before target (since we removed an element)
          if (source_idx < target_idx) {
            target_idx--;  // If we removed an item before the target position, adjust the target index
          }

          // Insert the moved item at the new position
          current_order.insert(current_order.begin() + target_idx, moved_item);

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

    // Ensure the config is saved after any reordering operation
    // This ensures that even if the application crashes, the user's order is preserved

    // Show tooltip on hover
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("Click to switch all panels to %s", entry.symbol.c_str());
      ImGui::Text("Exchange: %s", entry.exchange.c_str());
      ImGui::Text("24h Open/High/Low: %.4f / %.4f / %.4f", entry.open_24h, entry.high_24h, entry.low_24h);
      ImGui::EndTooltip();
    }
    current_column++;
  }

  // Column 1: Exchange
  if (column_info_[1].visible) {
    ImGui::TableSetColumnIndex(current_column);
    ImGui::Text("%s", entry.exchange.c_str());

    // Add tooltip to explain Exchange
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("Trading exchange: %s", entry.exchange.c_str());
      ImGui::EndTooltip();
    }
    current_column++;
  }

  // Column 2: Last Price
  if (column_info_[2].visible) {
    ImGui::TableSetColumnIndex(current_column);
    // Calculate price change percentage for color calculation
    double price_change_pct = 0.0;
    if (entry.previous_price != 0.0) {
      price_change_pct = ((entry.price - entry.previous_price) / entry.previous_price) * 100.0;
    }

    // Apply smooth animation effect to price if recently updated
    if (entry.animation_timer > 0.0f) {
      // Calculate animation progress (0.0 to 1.0)
      float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

      // Use enhanced interpolation for smoother transition
      double animated_price = interpolateValue(entry.previous_price, entry.price, progress);

      // Enhanced flash animation - more prominent flash effect with smoother transition
      ImVec4 flash_color = calculateChangeColor(price_change_pct, true);

      // Calculate flash intensity using the enhanced function for prices
      float flash_intensity = calculatePriceFlashIntensity(progress);

      // Enhance the color intensity during animation with more pronounced flash
      if (price_change_pct >= 0.0) {
        // Positive change - enhance green component during animation
        flash_color.x = flash_color.x * (0.3f + 0.7f * flash_intensity); // Red - reduced to allow green to dominate
        flash_color.y = std::min(1.0f, flash_color.y * (0.3f + 0.7f * flash_intensity)); // Green - enhanced
        flash_color.z = flash_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      } else {
        // Negative change - enhance red component during animation
        flash_color.x = std::min(1.0f, flash_color.x * (0.3f + 0.7f * flash_intensity)); // Red - enhanced
        flash_color.y = flash_color.y * (0.3f + 0.7f * flash_intensity); // Green - reduced
        flash_color.z = flash_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      }

      // Add brief flash animation effect by temporarily highlighting the background
      if (progress > 0.3f) { // Adjust flash timing for better visibility
        // Calculate alpha for background highlight based on animation progress
        float bg_alpha = (1.0f - progress) * 4.0f; // Increase intensity
        if (bg_alpha > 1.0f) bg_alpha = 1.0f;

        // Create a temporary background highlight for the cell
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 textSize = ImGui::CalcTextSize(formatPrice(animated_price).c_str());
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw a more prominent background highlight
        ImVec4 highlight_color = (price_change_pct >= 0.0) ?
          ImVec4(0.0f, 0.5f, 0.0f, bg_alpha * 0.8f) :  // More green for positive
          ImVec4(0.5f, 0.0f, 0.0f, bg_alpha * 0.8f);   // More red for negative

        draw_list->AddRectFilled(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(highlight_color)
        );

        // Add a subtle border to make the highlight more defined
        draw_list->AddRect(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(ImVec4(highlight_color.x * 0.7f, highlight_color.y * 0.7f, highlight_color.z * 0.7f, bg_alpha * 0.9f))
        );
      }

      ImGui::TextColored(flash_color, "%s", formatPrice(animated_price).c_str());
    } else {
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
    current_column++;
  }

  // Column 3: Change%
  if (column_info_[3].visible) {
    ImGui::TableSetColumnIndex(current_column);
    // Apply subtle animation to change percentage when it updates significantly
    ImVec4 change_pct_color = calculateChangeColor(entry.change_pct, true);

    // Apply smooth animation effect to change percentage if recently updated
    if (entry.animation_timer > 0.0f) {
      // Calculate animation progress (0.0 to 1.0)
      float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

      // Calculate flash intensity using the enhanced function
      float flash_intensity = calculateFlashIntensity(progress);

      // Enhance the color intensity during animation
      if (entry.change_pct >= 0.0) {
        // Positive change - enhance green component during animation
        change_pct_color.x = change_pct_color.x * (0.3f + 0.7f * flash_intensity); // Red - reduced to allow green to dominate
        change_pct_color.y = std::min(1.0f, change_pct_color.y * (0.3f + 0.7f * flash_intensity)); // Green - enhanced
        change_pct_color.z = change_pct_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      } else {
        // Negative change - enhance red component during animation
        change_pct_color.x = std::min(1.0f, change_pct_color.x * (0.3f + 0.7f * flash_intensity)); // Red - enhanced
        change_pct_color.y = change_pct_color.y * (0.3f + 0.7f * flash_intensity); // Green - reduced
        change_pct_color.z = change_pct_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      }

      // Add brief flash animation effect by temporarily highlighting the background
      if (progress > 0.4f) { // Adjust flash timing for better visibility
        // Calculate alpha for background highlight based on animation progress
        float bg_alpha = (1.0f - progress) * 3.0f; // Increase intensity
        if (bg_alpha > 1.0f) bg_alpha = 1.0f;

        // Create a temporary background highlight for the cell
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 textSize = ImGui::CalcTextSize((std::string(entry.change_pct >= 0 ? "+" : "") + std::to_string(entry.change_pct) + "%").c_str());
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw a more prominent background highlight
        ImVec4 highlight_color = (entry.change_pct >= 0.0) ?
          ImVec4(0.0f, 0.4f, 0.0f, bg_alpha * 0.7f) :  // More green for positive
          ImVec4(0.4f, 0.0f, 0.0f, bg_alpha * 0.7f);   // More red for negative

        draw_list->AddRectFilled(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(highlight_color)
        );

        // Add a subtle border to make the highlight more defined
        draw_list->AddRect(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(ImVec4(highlight_color.x * 0.7f, highlight_color.y * 0.7f, highlight_color.z * 0.7f, bg_alpha * 0.8f))
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
    current_column++;
  }

  // Column 4: Change$
  if (column_info_[4].visible) {
    ImGui::TableSetColumnIndex(current_column);
    // Apply subtle animation to change dollar when it updates significantly
    ImVec4 change_dollar_color = calculateChangeColor(entry.change_dollar, false);

    // Apply smooth animation effect to change dollar if recently updated
    if (entry.animation_timer > 0.0f) {
      // Calculate animation progress (0.0 to 1.0)
      float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

      // Calculate flash intensity using the enhanced function
      float flash_intensity = calculateFlashIntensity(progress);

      // Enhance the color intensity during animation
      if (entry.change_dollar >= 0.0) {
        // Positive change - enhance green component during animation
        change_dollar_color.x = change_dollar_color.x * (0.3f + 0.7f * flash_intensity); // Red - reduced to allow green to dominate
        change_dollar_color.y = std::min(1.0f, change_dollar_color.y * (0.3f + 0.7f * flash_intensity)); // Green - enhanced
        change_dollar_color.z = change_dollar_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      } else {
        // Negative change - enhance red component during animation
        change_dollar_color.x = std::min(1.0f, change_dollar_color.x * (0.3f + 0.7f * flash_intensity)); // Red - enhanced
        change_dollar_color.y = change_dollar_color.y * (0.3f + 0.7f * flash_intensity); // Green - reduced
        change_dollar_color.z = change_dollar_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      }

      // Add brief flash animation effect by temporarily highlighting the background
      if (progress > 0.4f) { // Adjust flash timing for better visibility
        // Calculate alpha for background highlight based on animation progress
        float bg_alpha = (1.0f - progress) * 3.0f; // Increase intensity
        if (bg_alpha > 1.0f) bg_alpha = 1.0f;

        // Create a temporary background highlight for the cell
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 textSize = ImGui::CalcTextSize((std::string(entry.change_dollar >= 0 ? "+" : "") + std::to_string(entry.change_dollar)).c_str());
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw a more prominent background highlight
        ImVec4 highlight_color = (entry.change_dollar >= 0.0) ?
          ImVec4(0.0f, 0.4f, 0.0f, bg_alpha * 0.7f) :  // More green for positive
          ImVec4(0.4f, 0.0f, 0.0f, bg_alpha * 0.7f);   // More red for negative

        draw_list->AddRectFilled(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(highlight_color)
        );

        // Add a subtle border to make the highlight more defined
        draw_list->AddRect(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(ImVec4(highlight_color.x * 0.7f, highlight_color.y * 0.7f, highlight_color.z * 0.7f, bg_alpha * 0.8f))
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
    current_column++;
  }

  // Column 5: Volume
  if (column_info_[5].visible) {
    ImGui::TableSetColumnIndex(current_column);
    // Apply color coding to volume based on comparison with previous volume if available
    double volume_change_pct = 0.0;
    if (entry.previous_volume != 0.0) {
      double volume_change = entry.volume_24h - entry.previous_volume;
      volume_change_pct = (entry.previous_volume != 0.0) ? ((volume_change / entry.previous_volume) * 100.0) : 0.0;
    }

    ImVec4 volume_color = calculateChangeColor(volume_change_pct, false);

    // Apply smooth animation effect to volume if recently updated
    if (entry.animation_timer > 0.0f) {
      // Calculate animation progress (0.0 to 1.0)
      float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

      // Calculate flash intensity using the enhanced function
      float flash_intensity = calculateFlashIntensity(progress);

      // Enhance the color intensity during animation
      if (volume_change_pct >= 0.0) {
        // Positive change - enhance green component during animation
        volume_color.x = volume_color.x * (0.3f + 0.7f * flash_intensity); // Red - reduced to allow green to dominate
        volume_color.y = std::min(1.0f, volume_color.y * (0.3f + 0.7f * flash_intensity)); // Green - enhanced
        volume_color.z = volume_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      } else {
        // Negative change - enhance red component during animation
        volume_color.x = std::min(1.0f, volume_color.x * (0.3f + 0.7f * flash_intensity)); // Red - enhanced
        volume_color.y = volume_color.y * (0.3f + 0.7f * flash_intensity); // Green - reduced
        volume_color.z = volume_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      }

      // Add brief flash animation effect by temporarily highlighting the background
      if (progress > 0.4f) { // Adjust flash timing for better visibility
        // Calculate alpha for background highlight based on animation progress
        float bg_alpha = (1.0f - progress) * 3.0f; // Increase intensity
        if (bg_alpha > 1.0f) bg_alpha = 1.0f;

        // Create a temporary background highlight for the cell
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 textSize = ImGui::CalcTextSize(formatFinancialNumber(entry.volume_24h, 2).c_str());
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw a more prominent background highlight
        ImVec4 highlight_color = (volume_change_pct >= 0.0) ?
          ImVec4(0.0f, 0.4f, 0.0f, bg_alpha * 0.7f) :  // More green for positive
          ImVec4(0.4f, 0.0f, 0.0f, bg_alpha * 0.7f);   // More red for negative

        draw_list->AddRectFilled(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(highlight_color)
        );

        // Add a subtle border to make the highlight more defined
        draw_list->AddRect(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(ImVec4(highlight_color.x * 0.7f, highlight_color.y * 0.7f, highlight_color.z * 0.7f, bg_alpha * 0.8f))
        );
      }
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
    current_column++;
  }

  // Column 6: High
  if (column_info_[6].visible) {
    ImGui::TableSetColumnIndex(current_column);
    // Apply color coding to High based on comparison with current price
    double high_diff_pct = 0.0; // Initialize here to avoid scope issues
    if (entry.price != 0.0) {
      high_diff_pct = ((entry.high_24h - entry.price) / entry.price) * 100.0;
    }

    ImVec4 high_color = calculateChangeColor(high_diff_pct, true);

    // Apply smooth animation effect to high if recently updated
    if (entry.animation_timer > 0.0f) {
      // Calculate animation progress (0.0 to 1.0)
      float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

      // Calculate flash intensity using the enhanced function
      float flash_intensity = calculateFlashIntensity(progress);

      // Enhance the color intensity during animation
      if (high_diff_pct >= 0.0) {
        // Positive change - enhance green component during animation
        high_color.x = high_color.x * (0.3f + 0.7f * flash_intensity); // Red - reduced to allow green to dominate
        high_color.y = std::min(1.0f, high_color.y * (0.3f + 0.7f * flash_intensity)); // Green - enhanced
        high_color.z = high_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      } else {
        // Negative change - enhance red component during animation
        high_color.x = std::min(1.0f, high_color.x * (0.3f + 0.7f * flash_intensity)); // Red - enhanced
        high_color.y = high_color.y * (0.3f + 0.7f * flash_intensity); // Green - reduced
        high_color.z = high_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      }

      // Add brief flash animation effect by temporarily highlighting the background
      if (progress > 0.4f) { // Adjust flash timing for better visibility
        // Calculate alpha for background highlight based on animation progress
        float bg_alpha = (1.0f - progress) * 3.0f; // Increase intensity
        if (bg_alpha > 1.0f) bg_alpha = 1.0f;

        // Create a temporary background highlight for the cell
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 textSize = ImGui::CalcTextSize(formatPrice(entry.high_24h).c_str());
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw a more prominent background highlight
        ImVec4 highlight_color = (high_diff_pct >= 0.0) ?
          ImVec4(0.0f, 0.4f, 0.0f, bg_alpha * 0.7f) :  // More green for positive
          ImVec4(0.4f, 0.0f, 0.0f, bg_alpha * 0.7f);   // More red for negative

        draw_list->AddRectFilled(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(highlight_color)
        );

        // Add a subtle border to make the highlight more defined
        draw_list->AddRect(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(ImVec4(highlight_color.x * 0.7f, highlight_color.y * 0.7f, highlight_color.z * 0.7f, bg_alpha * 0.8f))
        );
      }
    }

    ImGui::TextColored(high_color, "%s", formatPrice(entry.high_24h).c_str());

    // Add tooltip to explain High
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("24-hour high price");
      ImGui::EndTooltip();
    }
    current_column++;
  }

  // Column 7: Low
  if (column_info_[7].visible) {
    ImGui::TableSetColumnIndex(current_column);
    // Apply color coding to Low based on comparison with current price
    double low_diff_pct = 0.0; // Initialize here to avoid scope issues
    if (entry.price != 0.0) {
      low_diff_pct = ((entry.low_24h - entry.price) / entry.price) * 100.0;
    }

    ImVec4 low_color = calculateChangeColor(low_diff_pct, true);

    // Apply smooth animation effect to low if recently updated
    if (entry.animation_timer > 0.0f) {
      // Calculate animation progress (0.0 to 1.0)
      float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

      // Calculate flash intensity using the enhanced function
      float flash_intensity = calculateFlashIntensity(progress);

      // Enhance the color intensity during animation
      if (low_diff_pct >= 0.0) {
        // Positive change - enhance green component during animation
        low_color.x = low_color.x * (0.3f + 0.7f * flash_intensity); // Red - reduced to allow green to dominate
        low_color.y = std::min(1.0f, low_color.y * (0.3f + 0.7f * flash_intensity)); // Green - enhanced
        low_color.z = low_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      } else {
        // Negative change - enhance red component during animation
        low_color.x = std::min(1.0f, low_color.x * (0.3f + 0.7f * flash_intensity)); // Red - enhanced
        low_color.y = low_color.y * (0.3f + 0.7f * flash_intensity); // Green - reduced
        low_color.z = low_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      }

      // Add brief flash animation effect by temporarily highlighting the background
      if (progress > 0.4f) { // Adjust flash timing for better visibility
        // Calculate alpha for background highlight based on animation progress
        float bg_alpha = (1.0f - progress) * 3.0f; // Increase intensity
        if (bg_alpha > 1.0f) bg_alpha = 1.0f;

        // Create a temporary background highlight for the cell
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 textSize = ImGui::CalcTextSize(formatPrice(entry.low_24h).c_str());
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw a more prominent background highlight
        ImVec4 highlight_color = (low_diff_pct >= 0.0) ?
          ImVec4(0.0f, 0.4f, 0.0f, bg_alpha * 0.7f) :  // More green for positive
          ImVec4(0.4f, 0.0f, 0.0f, bg_alpha * 0.7f);   // More red for negative

        draw_list->AddRectFilled(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(highlight_color)
        );

        // Add a subtle border to make the highlight more defined
        draw_list->AddRect(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(ImVec4(highlight_color.x * 0.7f, highlight_color.y * 0.7f, highlight_color.z * 0.7f, bg_alpha * 0.8f))
        );
      }
    }

    ImGui::TextColored(low_color, "%s", formatPrice(entry.low_24h).c_str());

    // Add tooltip to explain Low
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("24-hour low price");
      ImGui::EndTooltip();
    }
    current_column++;
  }

  // Column 8: Open
  if (column_info_[8].visible) {
    ImGui::TableSetColumnIndex(current_column);
    // Apply color coding to Open based on comparison with current price
    double open_diff_pct = 0.0; // Initialize here to avoid scope issues
    if (entry.price != 0.0) {
      open_diff_pct = ((entry.open_24h - entry.price) / entry.price) * 100.0;
    }

    ImVec4 open_color = calculateChangeColor(open_diff_pct, true);

    // Apply smooth animation effect to open if recently updated
    if (entry.animation_timer > 0.0f) {
      // Calculate animation progress (0.0 to 1.0)
      float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

      // Calculate flash intensity using the enhanced function
      float flash_intensity = calculateFlashIntensity(progress);

      // Enhance the color intensity during animation
      if (open_diff_pct >= 0.0) {
        // Positive change - enhance green component during animation
        open_color.x = open_color.x * (0.3f + 0.7f * flash_intensity); // Red - reduced to allow green to dominate
        open_color.y = std::min(1.0f, open_color.y * (0.3f + 0.7f * flash_intensity)); // Green - enhanced
        open_color.z = open_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      } else {
        // Negative change - enhance red component during animation
        open_color.x = std::min(1.0f, open_color.x * (0.3f + 0.7f * flash_intensity)); // Red - enhanced
        open_color.y = open_color.y * (0.3f + 0.7f * flash_intensity); // Green - reduced
        open_color.z = open_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      }

      // Add brief flash animation effect by temporarily highlighting the background
      if (progress > 0.4f) { // Adjust flash timing for better visibility
        // Calculate alpha for background highlight based on animation progress
        float bg_alpha = (1.0f - progress) * 3.0f; // Increase intensity
        if (bg_alpha > 1.0f) bg_alpha = 1.0f;

        // Create a temporary background highlight for the cell
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 textSize = ImGui::CalcTextSize(formatPrice(entry.open_24h).c_str());
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw a more prominent background highlight
        ImVec4 highlight_color = (open_diff_pct >= 0.0) ?
          ImVec4(0.0f, 0.4f, 0.0f, bg_alpha * 0.7f) :  // More green for positive
          ImVec4(0.4f, 0.0f, 0.0f, bg_alpha * 0.7f);   // More red for negative

        draw_list->AddRectFilled(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(highlight_color)
        );

        // Add a subtle border to make the highlight more defined
        draw_list->AddRect(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(ImVec4(highlight_color.x * 0.7f, highlight_color.y * 0.7f, highlight_color.z * 0.7f, bg_alpha * 0.8f))
        );
      }
    }

    ImGui::TextColored(open_color, "%s", formatPrice(entry.open_24h).c_str());

    // Add tooltip to explain Open
    if (ImGui::IsItemHovered()) {
      ImGui::BeginTooltip();
      ImGui::Text("24-hour opening price");
      ImGui::EndTooltip();
    }
    current_column++;
  }

  // Column 9: VWAP
  if (column_info_[9].visible) {
    ImGui::TableSetColumnIndex(current_column);
    // Apply color coding to VWAP based on change from previous value (green for increase, red for decrease)
    double vwap_change_pct = 0.0;
    if (entry.previous_vwap != 0.0) {
      double vwap_change = entry.vwap - entry.previous_vwap;
      vwap_change_pct = (entry.previous_vwap != 0.0) ? ((vwap_change / entry.previous_vwap) * 100.0) : 0.0;
    }

    ImVec4 vwap_color = calculateChangeColor(vwap_change_pct, true);

    // Apply smooth animation effect on top of color coding if recently updated
    if (entry.animation_timer > 0.0f) {
      // Calculate animation progress (0.0 to 1.0)
      float progress = 1.0f - (entry.animation_timer / WatchlistEntry::ANIMATION_DURATION);

      // Use enhanced interpolation for smoother transition
      double animated_vwap = interpolateValue(entry.previous_vwap, entry.vwap, progress);

      // Calculate flash intensity using the enhanced function for prices
      float flash_intensity = calculatePriceFlashIntensity(progress);

      // Enhance the color intensity during animation
      if (vwap_change_pct >= 0.0) {
        // Positive change - enhance green component during animation
        vwap_color.x = vwap_color.x * (0.3f + 0.7f * flash_intensity); // Red - reduced to allow green to dominate
        vwap_color.y = std::min(1.0f, vwap_color.y * (0.3f + 0.7f * flash_intensity)); // Green - enhanced
        vwap_color.z = vwap_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      } else {
        // Negative change - enhance red component during animation
        vwap_color.x = std::min(1.0f, vwap_color.x * (0.3f + 0.7f * flash_intensity)); // Red - enhanced
        vwap_color.y = vwap_color.y * (0.3f + 0.7f * flash_intensity); // Green - reduced
        vwap_color.z = vwap_color.z * (0.3f + 0.7f * flash_intensity); // Blue - reduced
      }

      // Add brief flash animation effect by temporarily highlighting the background
      if (progress > 0.3f) { // Adjust flash timing for better visibility
        // Calculate alpha for background highlight based on animation progress
        float bg_alpha = (1.0f - progress) * 4.0f; // Increase intensity
        if (bg_alpha > 1.0f) bg_alpha = 1.0f;

        // Create a temporary background highlight for the cell
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 textSize = ImGui::CalcTextSize(formatVWAP(animated_vwap).c_str());
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Draw a more prominent background highlight
        ImVec4 highlight_color = (vwap_change_pct >= 0.0) ?
          ImVec4(0.0f, 0.5f, 0.0f, bg_alpha * 0.8f) :  // More green for positive
          ImVec4(0.5f, 0.0f, 0.0f, bg_alpha * 0.8f);   // More red for negative

        draw_list->AddRectFilled(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(highlight_color)
        );

        // Add a subtle border to make the highlight more defined
        draw_list->AddRect(
          ImVec2(pos.x - 8, pos.y - 3),
          ImVec2(pos.x + textSize.x + 8, pos.y + textSize.y + 3),
          ImGui::GetColorU32(ImVec4(highlight_color.x * 0.7f, highlight_color.y * 0.7f, highlight_color.z * 0.7f, bg_alpha * 0.9f))
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
    current_column++;
  }

  // Column 10: Action
  if (column_info_[10].visible) {
    ImGui::TableSetColumnIndex(current_column);
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

    // Add context menu for additional actions including alerts
    if (ImGui::BeginPopupContextItem("WatchlistItemContextMenu")) {
      if (ImGui::MenuItem("Add Price Alert")) {
        // Store the symbol for which to add an alert
        // This would typically open a modal dialog to set the alert parameters
        // For now, we'll just add a simple alert above current price + 5%
        double target_price = entry.price * 1.05; // 5% above current price
        if (alert_manager_) {
          alert_manager_->add_price_alert(entry.symbol_id, entry.symbol, target_price,
                                         WatchlistPriceAlert::Direction::ABOVE);
        }
      }

      if (ImGui::MenuItem("Add Below Alert")) {
        // Add an alert for when price goes below current price - 5%
        double target_price = entry.price * 0.95; // 5% below current price
        if (alert_manager_) {
          alert_manager_->add_price_alert(entry.symbol_id, entry.symbol, target_price,
                                         WatchlistPriceAlert::Direction::BELOW);
        }
      }

      if (ImGui::MenuItem("View Existing Alerts")) {
        // Show existing alerts for this symbol
        if (alert_manager_) {
          auto alerts = alert_manager_->get_alerts_for_symbol(entry.symbol_id);
          if (alerts.empty()) {
            std::cout << "[WatchlistPanel] No alerts for symbol: " << entry.symbol << std::endl;
          } else {
            std::cout << "[WatchlistPanel] Found " << alerts.size() << " alerts for symbol: " << entry.symbol << std::endl;
            for (const auto& alert : alerts) {
              std::cout << "  Alert: " << (alert.direction == WatchlistPriceAlert::Direction::ABOVE ? "Above" : "Below")
                        << " " << alert.target_price << " Status: " <<
                           (alert.status == AlertStatus::ACTIVE ? "Active" :
                            alert.status == AlertStatus::TRIGGERED ? "Triggered" : "Disabled") << std::endl;
            }
          }
        }
      }

      ImGui::EndPopup();
    }

    ImGui::PopID();
  }
}

std::vector<uint32_t> WatchlistPanel::get_filtered_symbols() const {
  std::vector<uint32_t> filtered;

  std::string filter(filter_buffer_);
  std::transform(filter.begin(), filter.end(), filter.begin(), ::tolower);

  for (uint32_t symbol_id : get_current_display_order()) {
    auto it = get_current_watchlist().find(symbol_id);
    if (it != get_current_watchlist().end()) {
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
  std::sort(get_current_display_order().begin(), get_current_display_order().end(), [this](uint32_t a_id, uint32_t b_id) {
    const auto& a = get_current_watchlist().at(a_id);
    const auto& b = get_current_watchlist().at(b_id);

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
        result = a.symbol < b.symbol;
        break;
      default:
        result = a.symbol < b.symbol;
    }

    // Handle equal values to ensure consistent sorting
    bool values_equal = false;
    switch (sort_column_) {
      case 0:  // Symbol
        values_equal = (a.symbol == b.symbol);
        break;
      case 1:  // Exchange
        values_equal = (a.exchange == b.exchange);
        break;
      case 2:  // Last Price
        values_equal = (a.price == b.price);
        break;
      case 3:  // Change %
        values_equal = (a.change_pct == b.change_pct);
        break;
      case 4:  // Change $
        values_equal = (a.change_dollar == b.change_dollar);
        break;
      case 5:  // Volume
        values_equal = (a.volume_24h == b.volume_24h);
        break;
      case 6:  // High
        values_equal = (a.high_24h == b.high_24h);
        break;
      case 7:  // Low
        values_equal = (a.low_24h == b.low_24h);
        break;
      case 8:  // Open
        values_equal = (a.open_24h == b.open_24h);
        break;
      case 9:  // VWAP
        values_equal = (a.vwap == b.vwap);
        break;
      case 10:  // Action
        values_equal = true; // Action column doesn't have comparable values
        break;
      default:
        values_equal = (a.symbol == b.symbol);
    }

    // If values are equal, sort by symbol as secondary criteria to ensure consistent ordering
    if (values_equal) {
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

        // Add our updated display order for the current group
        std::string order_line = "display_order=";
        const auto& current_order = get_current_display_order();
        for (size_t i = 0; i < current_order.size(); ++i) {
          order_line += std::to_string(current_order[i]);
          if (i < current_order.size() - 1) {
            order_line += ",";
          }
        }
        existing_lines.push_back(order_line);

        // Add group-specific orders
        for (const auto& [group_name, group_order] : group_display_orders_) {
          std::string group_order_line = "order_" + group_name + "=";
          for (size_t i = 0; i < group_order.size(); ++i) {
            group_order_line += std::to_string(group_order[i]);
            if (i < group_order.size() - 1) {
              group_order_line += ",";
            }
          }
          existing_lines.push_back(group_order_line);
        }

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
    const auto& current_order = get_current_display_order();
    for (size_t i = 0; i < current_order.size(); ++i) {
      order_line += std::to_string(current_order[i]);
      if (i < current_order.size() - 1) {
        order_line += ",";
      }
    }
    existing_lines.push_back(order_line);

    // Add group-specific orders
    for (const auto& [group_name, group_order] : group_display_orders_) {
      std::string group_order_line = "order_" + group_name + "=";
      for (size_t i = 0; i < group_order.size(); ++i) {
        group_order_line += std::to_string(group_order[i]);
        if (i < group_order.size() - 1) {
          group_order_line += ",";
        }
      }
      existing_lines.push_back(group_order_line);
    }
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
    std::cout << "[WatchlistPanel] Saved watchlist order to: " << config_file << ", entries in '"
              << current_group_name_ << "': " << get_current_display_order().size() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[WatchlistPanel] Error saving watchlist order: " << e.what() << std::endl;
  }
}

// Additional helper method to ensure proper cleanup of resources during drag operations
void WatchlistPanel::cleanup_drag_resources() {
  // Currently no specific cleanup needed for drag operations
  // This method is provided for future extensibility if needed
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
            // Parse comma-separated list of symbol IDs for the current group
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
                  // Only add to new order if the symbol exists in the current watchlist
                  if (get_current_watchlist().find(symbol_id) != get_current_watchlist().end()) {
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
            for (const auto& pair : get_current_watchlist()) {
              uint32_t symbol_id = pair.first;
              if (std::find(new_display_order.begin(), new_display_order.end(), symbol_id) == new_display_order.end()) {
                new_display_order.push_back(symbol_id);
              }
            }

            get_current_display_order() = new_display_order;
            std::cout << "[WatchlistPanel] Loaded watchlist order from config for group '"
                      << current_group_name_ << "', entries: " << new_display_order.size() << std::endl;
          } else if (key.substr(0, 6) == "order_") {
            // This is a group-specific order
            std::string group_name = key.substr(6); // Remove "order_" prefix
            std::vector<uint32_t> new_group_order;
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
                  // Only add to new order if the symbol exists in the group
                  if (watchlist_groups_[group_name].find(symbol_id) != watchlist_groups_[group_name].end()) {
                    new_group_order.push_back(symbol_id);
                  } else {
                    std::cout << "[WatchlistPanel] Symbol ID " << symbol_id << " from config not found in group '"
                              << group_name << "', skipping." << std::endl;
                  }
                } catch (const std::invalid_argument&) {
                  std::cerr << "[WatchlistPanel] Invalid symbol ID in config for group '" << group_name << "': " << item << std::endl;
                }
              }
            }

            // Add any remaining symbols that weren't in the config to the end
            for (const auto& pair : watchlist_groups_[group_name]) {
              uint32_t symbol_id = pair.first;
              if (std::find(new_group_order.begin(), new_group_order.end(), symbol_id) == new_group_order.end()) {
                new_group_order.push_back(symbol_id);
              }
            }

            group_display_orders_[group_name] = new_group_order;
            std::cout << "[WatchlistPanel] Loaded order for group '" << group_name << "', entries: " << new_group_order.size() << std::endl;

            // Add to group names if not already present
            if (std::find(group_names_.begin(), group_names_.end(), group_name) == group_names_.end()) {
              group_names_.push_back(group_name);
            }
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
    // Check if already subscribed to avoid duplicate subscriptions
    if (symbol_subscriptions_.find(symbol_id) != symbol_subscriptions_.end()) {
      std::cout << "[WatchlistPanel] Already subscribed to symbol ID: " << symbol_id << std::endl;
      return;
    }

    // Create a subscription for this specific symbol to receive real-time price updates
    uint64_t sub_id = processor_->subscribe(symbol_id, RenderEngine::NotificationType::TRADE,
                                          [this](uint32_t symbol_id, RenderEngine::NotificationType type) {
                                            this->on_market_data_update(symbol_id, type);
                                          });

    // Store the subscription ID for this symbol
    symbol_subscriptions_[symbol_id] = sub_id;

    std::cout << "[WatchlistPanel] Subscribed to symbol ID: " << symbol_id
              << " with subscription ID: " << sub_id << std::endl;
  }
}

void WatchlistPanel::unsubscribe_from_symbol(uint32_t symbol_id) {
  if (processor_) {
    auto it = symbol_subscriptions_.find(symbol_id);
    if (it != symbol_subscriptions_.end()) {
      processor_->unsubscribe(it->second);
      std::cout << "[WatchlistPanel] Unsubscribed from symbol ID: " << symbol_id
                << " with subscription ID: " << it->second << std::endl;
      symbol_subscriptions_.erase(it);
    } else {
      std::cout << "[WatchlistPanel] No active subscription found for symbol ID: " << symbol_id << std::endl;
    }
  }
}

void WatchlistPanel::verify_subscriptions() {
  // Periodically verify that all symbols in the current watchlist group have active subscriptions
  // This helps ensure robustness in case of connection issues or other problems

  for (const auto& [symbol_id, entry] : get_current_watchlist()) {
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
  // Unsubscribe from all current symbols in the current group
  for (const auto& [symbol_id, entry] : get_current_watchlist()) {
    unsubscribe_from_symbol(symbol_id);
  }

  // Then resubscribe to all symbols in the current group
  for (const auto& [symbol_id, entry] : get_current_watchlist()) {
    subscribe_to_symbol(symbol_id);
  }

  std::cout << "[WatchlistPanel] Refreshed all subscriptions for " << get_current_watchlist().size()
            << " symbols in group '" << current_group_name_ << "'" << std::endl;
}

void WatchlistPanel::ensure_all_symbols_subscribed() {
  // Ensure all symbols in the current watchlist group have active subscriptions
  size_t subscribed_count = 0;
  for (const auto& [symbol_id, entry] : get_current_watchlist()) {
    if (symbol_subscriptions_.find(symbol_id) == symbol_subscriptions_.end()) {
      subscribe_to_symbol(symbol_id);
      subscribed_count++;
    }
  }

  if (subscribed_count > 0) {
    std::cout << "[WatchlistPanel] Subscribed to " << subscribed_count
              << " additional symbols in group '" << current_group_name_
              << "' to ensure all watchlist symbols are covered" << std::endl;
  }
}

void WatchlistPanel::subscribe_to_all_watchlist_symbols() {
  // Subscribe to all symbols in the current watchlist group efficiently
  std::vector<uint32_t> symbols_to_subscribe;

  for (const auto& [symbol_id, entry] : get_current_watchlist()) {
    if (symbol_subscriptions_.find(symbol_id) == symbol_subscriptions_.end()) {
      symbols_to_subscribe.push_back(symbol_id);
    }
  }

  if (!symbols_to_subscribe.empty()) {
    std::cout << "[WatchlistPanel] Subscribing to " << symbols_to_subscribe.size()
              << " symbols in group '" << current_group_name_ << "' for real-time updates..." << std::endl;

    for (uint32_t symbol_id : symbols_to_subscribe) {
      subscribe_to_symbol(symbol_id);
    }

    std::cout << "[WatchlistPanel] Successfully subscribed to all "
              << symbols_to_subscribe.size() << " symbols in group '" << current_group_name_ << "'" << std::endl;
  }
}

void WatchlistPanel::initialize_column_settings() {
  // Initialize column info with default settings
  // The order of initialization corresponds to the table column indices
  column_info_.clear();

  // Column indices correspond to:
  // 0: Symbol
  // 1: Exchange
  // 2: Last Price
  // 3: Change%
  // 4: Change$
  // 5: Volume
  // 6: High
  // 7: Low
  // 8: Open
  // 9: VWAP
  // 10: Action

  column_info_.emplace_back("Symbol", true, 0.0f, 0);  // Stretch to fill
  column_info_.emplace_back("Exchange", true, 80.0f, 1);
  column_info_.emplace_back("Last Price", true, 100.0f, 2);
  column_info_.emplace_back("Change%", true, 90.0f, 3);
  column_info_.emplace_back("Change$", true, 90.0f, 4);
  column_info_.emplace_back("Volume", true, 100.0f, 5);
  column_info_.emplace_back("High", true, 90.0f, 6);
  column_info_.emplace_back("Low", true, 90.0f, 7);
  column_info_.emplace_back("Open", true, 90.0f, 8);
  column_info_.emplace_back("VWAP", true, 90.0f, 9);
  column_info_.emplace_back("Action", true, 70.0f, 10);
}

void WatchlistPanel::render_column_context_menu() {
  if (column_context_menu_open_) {
    if (ImGui::BeginPopup("ColumnContextMenu")) {
      ImGui::Text("Column Options:");
      ImGui::Separator();

      // Show/hide options for each column with current visibility status
      for (int i = 0; i < static_cast<int>(column_info_.size()); ++i) {
        bool is_visible = column_info_[i].visible;
        std::string label = column_info_[i].name + (is_visible ? " (Visible)" : " (Hidden)");

        if (ImGui::MenuItem(label.c_str(), nullptr, &is_visible)) {
          column_info_[i].visible = is_visible;

          // Save the updated settings to config
          save_column_settings_to_config(config_file_path_);
        }
      }

      ImGui::Separator();

      // Option to reset to default column layout
      if (ImGui::MenuItem("Reset to Default Layout")) {
        initialize_column_settings(); // Reset to default settings

        // Save the updated settings to config
        save_column_settings_to_config(config_file_path_);
      }

      ImGui::EndPopup();

      // Close the popup after processing
      if (!ImGui::IsPopupOpen("ColumnContextMenu")) {
        column_context_menu_open_ = false;
        clicked_column_index_ = -1;
      }
    } else {
      // Popup was closed, reset the state
      column_context_menu_open_ = false;
      clicked_column_index_ = -1;
    }
  }
}

void WatchlistPanel::toggle_column_visibility(int column_index) {
  if (column_index >= 0 && column_index < static_cast<int>(column_info_.size())) {
    column_info_[column_index].visible = !column_info_[column_index].visible;

    // Save the updated settings to config
    save_column_settings_to_config(config_file_path_);
  }
}

void WatchlistPanel::save_column_settings_to_config(const std::string& config_file) const {
  // Create directory if it doesn't exist
  std::filesystem::path config_path(config_file);
  std::filesystem::create_directories(config_path.parent_path());

  // Read the existing config file to preserve other sections
  std::vector<std::string> existing_lines;
  std::ifstream read_file(config_file);
  bool replaced_section = false;

  if (read_file.is_open()) {
    std::string line;
    bool in_column_settings_section = false;

    while (std::getline(read_file, line)) {
      // Check if we're entering the column_settings section
      if (line.find("[column_settings]") != std::string::npos) {
        existing_lines.push_back(line);
        in_column_settings_section = true;

        // Add our updated column settings
        for (size_t i = 0; i < column_info_.size(); ++i) {
          std::string visible_key = "col_" + std::to_string(i) + "_visible";
          std::string width_key = "col_" + std::to_string(i) + "_width";
          std::string order_key = "col_" + std::to_string(i) + "_order";

          existing_lines.push_back(visible_key + "=" + std::to_string(column_info_[i].visible ? 1 : 0));
          existing_lines.push_back(width_key + "=" + std::to_string(column_info_[i].width));
          existing_lines.push_back(order_key + "=" + std::to_string(column_info_[i].order));
        }

        replaced_section = true;
      }
      // Skip lines inside the column_settings section (we'll replace them)
      else if (in_column_settings_section && line.find('[') == 0 && line.find(']') != std::string::npos) {
        // Found next section, so we're out of the column_settings section
        in_column_settings_section = false;
        existing_lines.push_back(line);
      }
      else if (!in_column_settings_section) {
        existing_lines.push_back(line);
      }
      // If in column_settings section and not a new section header, skip the line
    }
    read_file.close();
  }

  // If the column_settings section wasn't found, add it at the end
  if (!replaced_section) {
    if (!existing_lines.empty() && !existing_lines.back().empty()) {
      existing_lines.push_back(""); // Add blank line before new section
    }
    existing_lines.push_back("# Column settings configuration");
    existing_lines.push_back("[column_settings]");

    for (size_t i = 0; i < column_info_.size(); ++i) {
      std::string visible_key = "col_" + std::to_string(i) + "_visible";
      std::string width_key = "col_" + std::to_string(i) + "_width";
      std::string order_key = "col_" + std::to_string(i) + "_order";

      existing_lines.push_back(visible_key + "=" + std::to_string(column_info_[i].visible ? 1 : 0));
      existing_lines.push_back(width_key + "=" + std::to_string(column_info_[i].width));
      existing_lines.push_back(order_key + "=" + std::to_string(column_info_[i].order));
    }
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
    std::cout << "[WatchlistPanel] Saved column settings to: " << config_file << ", columns: " << column_info_.size() << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[WatchlistPanel] Error saving column settings: " << e.what() << std::endl;
  }
}

void WatchlistPanel::load_column_settings_from_config(const std::string& config_file) {
  std::ifstream file(config_file);
  if (!file.is_open()) {
    std::cout << "[WatchlistPanel] Config file not found, using default column settings: " << config_file << std::endl;
    return;
  }

  try {
    std::string line;
    std::string current_section;
    bool found_column_settings = false;

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
        if (current_section == "column_settings") {
          found_column_settings = true;
        }
        continue;
      }

      // Parse key-value pairs only in the column_settings section
      if (current_section == "column_settings") {
        size_t equals_pos = line.find('=');
        if (equals_pos != std::string::npos) {
          std::string key = line.substr(0, equals_pos);
          std::string value = line.substr(equals_pos + 1);

          // Parse column visibility, width, and order settings
          if (key.find("col_") == 0) {
            size_t underscore1 = key.find('_');
            size_t underscore2 = key.find('_', underscore1 + 1);

            if (underscore1 != std::string::npos && underscore2 != std::string::npos) {
              std::string col_index_str = key.substr(underscore1 + 1, underscore2 - underscore1 - 1);
              std::string property = key.substr(underscore2 + 1);

              try {
                int col_index = std::stoi(col_index_str);

                if (col_index >= 0 && col_index < static_cast<int>(column_info_.size())) {
                  if (property == "visible") {
                    column_info_[col_index].visible = (std::stoi(value) != 0);
                  } else if (property == "width") {
                    column_info_[col_index].width = std::stof(value);
                  } else if (property == "order") {
                    column_info_[col_index].order = std::stoi(value);
                  }
                }
              } catch (const std::exception& e) {
                std::cerr << "[WatchlistPanel] Error parsing column setting: " << key << "=" << value << std::endl;
              }
            }
          }
        }
      }
    }

    file.close();

    if (found_column_settings) {
      std::cout << "[WatchlistPanel] Successfully loaded column settings from: " << config_file << std::endl;
    } else {
      std::cout << "[WatchlistPanel] No column settings found in config, keeping defaults: " << config_file << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "[WatchlistPanel] Error loading column settings: " << e.what() << std::endl;
  }
}

const char* WatchlistPanel::get_column_name_by_index(int column_index) {
  if (column_index >= 0 && column_index < static_cast<int>(column_info_.size())) {
    return column_info_[column_index].name.c_str();
  }
  return "Unknown";
}

void WatchlistPanel::swap_column_positions(int index1, int index2) {
  if (index1 >= 0 && index1 < static_cast<int>(column_info_.size()) &&
      index2 >= 0 && index2 < static_cast<int>(column_info_.size()) &&
      index1 != index2) {

    // Get the current order values
    int order1 = column_info_[index1].order;
    int order2 = column_info_[index2].order;

    // Swap the order values
    column_info_[index1].order = order2;
    column_info_[index2].order = order1;

    // Save the updated settings to config
    save_column_settings_to_config(config_file_path_);
  }
}

void WatchlistPanel::reorder_columns(int source_index, int target_index) {
  if (source_index < 0 || source_index >= static_cast<int>(column_info_.size()) ||
      target_index < 0 || target_index >= static_cast<int>(column_info_.size()) ||
      source_index == target_index) {
    return;
  }

  // Get the source column's current order value
  int source_order = column_info_[source_index].order;

  // Get the target column's current order value
  int target_order = column_info_[target_index].order;

  // If the source is being moved after the target, we need to shift other columns appropriately
  if (source_order < target_order) {
    // Moving right/down - shift columns between source and target left
    for (auto& col : column_info_) {
      if (col.order > source_order && col.order <= target_order) {
        col.order--;
      }
    }
    // Set the source column to the target position
    column_info_[source_index].order = target_order;
  } else {
    // Moving left/up - shift columns between target and source right
    for (auto& col : column_info_) {
      if (col.order >= target_order && col.order < source_order) {
        col.order++;
      }
    }
    // Set the source column to the target position
    column_info_[source_index].order = target_order;
  }

  // Save the updated settings to config
  save_column_settings_to_config(config_file_path_);
}

void WatchlistPanel::process_pending_updates() {
  // This method handles any pending market data updates
  // Currently, updates are processed directly in on_market_data_update
  // This method can be extended to handle batch updates or other processing
  // For now, it serves as a placeholder for future enhancements
}

void WatchlistPanel::create_group(const std::string& group_name) {
  // Check if group already exists
  if (watchlist_groups_.find(group_name) != watchlist_groups_.end()) {
    return; // Group already exists
  }

  // Create new empty watchlist for this group
  watchlist_groups_[group_name] = std::map<uint32_t, WatchlistEntry>();

  // Create new display order for this group
  group_display_orders_[group_name] = std::vector<uint32_t>();

  // Add to group names list if not already present
  if (std::find(group_names_.begin(), group_names_.end(), group_name) == group_names_.end()) {
    group_names_.push_back(group_name);
  }

  std::cout << "[WatchlistPanel] Created new watchlist group: " << group_name << std::endl;
}

void WatchlistPanel::rename_group(const std::string& old_name, const std::string& new_name) {
  // Check if old group exists
  if (watchlist_groups_.find(old_name) == watchlist_groups_.end()) {
    std::cout << "[WatchlistPanel] Cannot rename non-existent group: " << old_name << std::endl;
    return;
  }

  // Check if new name already exists
  if (watchlist_groups_.find(new_name) != watchlist_groups_.end()) {
    std::cout << "[WatchlistPanel] Group with name already exists: " << new_name << std::endl;
    return;
  }

  // Don't allow renaming default groups
  if (old_name == "Futures" || old_name == "Crypto" || old_name == "Stocks") {
    std::cout << "[WatchlistPanel] Cannot rename default group: " << old_name << std::endl;
    return;
  }

  // Move the group data to the new name
  watchlist_groups_[new_name] = watchlist_groups_[old_name];
  watchlist_groups_.erase(old_name);

  // Move the display order to the new name
  if (group_display_orders_.find(old_name) != group_display_orders_.end()) {
    group_display_orders_[new_name] = group_display_orders_[old_name];
    group_display_orders_.erase(old_name);
  }

  // Update the group names list
  auto it = std::find(group_names_.begin(), group_names_.end(), old_name);
  if (it != group_names_.end()) {
    *it = new_name;
  }

  // If we're renaming the current group, update the current group name
  if (current_group_name_ == old_name) {
    current_group_name_ = new_name;
  }

  std::cout << "[WatchlistPanel] Renamed watchlist group: " << old_name << " -> " << new_name << std::endl;
}

void WatchlistPanel::delete_group(const std::string& group_name) {
  // Don't delete if it's one of the default groups
  if (group_name == "Futures" || group_name == "Crypto" || group_name == "Stocks") {
    std::cout << "[WatchlistPanel] Cannot delete default group: " << group_name << std::endl;
    return;
  }

  // Remove from watchlist groups
  watchlist_groups_.erase(group_name);

  // Remove from display orders
  group_display_orders_.erase(group_name);

  // Remove from group names
  group_names_.erase(std::remove(group_names_.begin(), group_names_.end(), group_name), group_names_.end());

  // If we're deleting the current group, switch to the first available group
  if (current_group_name_ == group_name) {
    if (!group_names_.empty()) {
      current_group_name_ = group_names_[0];
    } else {
      current_group_name_ = "Default";
    }
  }

  std::cout << "[WatchlistPanel] Deleted watchlist group: " << group_name << std::endl;
}

void WatchlistPanel::switch_to_group(const std::string& group_name) {
  auto it = watchlist_groups_.find(group_name);
  if (it != watchlist_groups_.end()) {
    current_group_name_ = group_name;

    // Subscribe to all symbols in the new group
    subscribe_to_all_watchlist_symbols();

    std::cout << "[WatchlistPanel] Switched to watchlist group: " << group_name << std::endl;
  } else {
    std::cout << "[WatchlistPanel] Attempted to switch to non-existent group: " << group_name << std::endl;
  }
}

void WatchlistPanel::add_symbol_to_group(const std::string& group_name, uint32_t symbol_id, const std::string& symbol, const std::string& exchange) {
  auto& group = watchlist_groups_[group_name];
  if (group.find(symbol_id) != group.end()) {
    return;  // Already exists in this group
  }

  WatchlistEntry entry;
  entry.symbol_id = symbol_id;
  entry.symbol = symbol;
  entry.exchange = exchange;
  entry.is_active = true;

  group[symbol_id] = entry;

  // Add to display order for this group
  auto& display_order = group_display_orders_[group_name];
  display_order.push_back(symbol_id);

  // Subscribe to real-time updates for this symbol if this is the current group
  if (group_name == current_group_name_) {
    subscribe_to_symbol(symbol_id);
  }

  std::cout << "[WatchlistPanel] Added symbol " << symbol << " (ID: " << symbol_id
            << ") to group " << group_name << " and subscribed to real-time updates" << std::endl;
}

void WatchlistPanel::remove_symbol_from_group(const std::string& group_name, uint32_t symbol_id) {
  auto group_it = watchlist_groups_.find(group_name);
  if (group_it == watchlist_groups_.end()) {
    return; // Group doesn't exist
  }

  auto& group = group_it->second;
  group.erase(symbol_id);

  // Remove from display order for this group
  auto& display_order = group_display_orders_[group_name];
  display_order.erase(std::remove(display_order.begin(), display_order.end(), symbol_id), display_order.end());

  // Unsubscribe from real-time updates for this symbol if this is the current group
  if (group_name == current_group_name_) {
    unsubscribe_from_symbol(symbol_id);
  }
}

void WatchlistPanel::clear_group(const std::string& group_name) {
  auto group_it = watchlist_groups_.find(group_name);
  if (group_it == watchlist_groups_.end()) {
    return; // Group doesn't exist
  }

  auto& group = group_it->second;

  // Unsubscribe from all symbols in this group if it's the current group
  if (group_name == current_group_name_) {
    for (const auto& [symbol_id, entry] : group) {
      unsubscribe_from_symbol(symbol_id);
    }
  }

  // Remove alerts for all symbols in this group
  for (const auto& [symbol_id, entry] : group) {
    if (alert_manager_) {
      alert_manager_->remove_alerts_for_symbol(symbol_id);
    }
  }

  group.clear();
  group_display_orders_[group_name].clear();
}

void WatchlistPanel::set_alerts_panel(std::shared_ptr<AlertsPanel> alerts_panel) {
  if (alert_manager_) {
    // Update the alerts panel reference in the alert manager
    alert_manager_->set_alert_triggered_callback([alerts_panel](const WatchlistPriceAlert& alert, double current_price) {
      if (alerts_panel) {
        auto now = std::chrono::system_clock::now();
        std::string message = std::string("Price ") +
                             (alert.direction == WatchlistPriceAlert::Direction::ABOVE ? "above" : "below") +
                             std::string(" target: ") + std::to_string(alert.target_price);

        // Create a new alert log entry
        AlertLog log_entry;
        log_entry.time = now;
        log_entry.rule_name = alert.symbol_name + " Price Alert";
        log_entry.symbol = alert.symbol_name;
        log_entry.price = current_price;
        log_entry.message = message;

        // Note: Since we can't directly access the logs_ vector in AlertsPanel,
        // we would need to add a public method to AlertsPanel to add logs
        // For now, we'll just log to console
        std::cout << "[WatchlistAlert] Triggered: " << alert.symbol_name
                  << " price alert at " << current_price << " (target: " << alert.target_price << ")" << std::endl;
      }
    });
  }
}

void WatchlistPanel::add_price_alert(uint32_t symbol_id, const std::string& symbol_name,
                                    double target_price, WatchlistPriceAlert::Direction direction) {
  if (alert_manager_) {
    alert_manager_->add_price_alert(symbol_id, symbol_name, target_price, direction);
  }
}

void WatchlistPanel::remove_alerts_for_symbol(uint32_t symbol_id) {
  if (alert_manager_) {
    alert_manager_->remove_alerts_for_symbol(symbol_id);
  }
}

void WatchlistPanel::ensure_default_groups_order() {
  // Create a temporary vector to hold the reordered group names
  std::vector<std::string> reordered_groups;

  // Add the default groups in the desired order first
  std::vector<std::string> default_groups = {"Stocks", "Crypto", "Futures"};

  for (const auto& default_group : default_groups) {
    if (std::find(group_names_.begin(), group_names_.end(), default_group) != group_names_.end()) {
      reordered_groups.push_back(default_group);
    }
  }

  // Add any other groups after the default ones
  for (const auto& group_name : group_names_) {
    if (std::find(default_groups.begin(), default_groups.end(), group_name) == default_groups.end()) {
      reordered_groups.push_back(group_name);
    }
  }

  // Update the group_names_ vector with the reordered groups
  group_names_ = reordered_groups;
}

// Helper method to validate column settings and ensure consistency
void WatchlistPanel::validate_column_settings() {
  // Ensure all columns have unique order values
  std::vector<int> orders;
  for (const auto& col : column_info_) {
    orders.push_back(col.order);
  }

  // Sort the orders to check for duplicates
  std::sort(orders.begin(), orders.end());

  // Check for duplicates and fix if needed
  for (size_t i = 0; i < orders.size() - 1; ++i) {
    if (orders[i] == orders[i + 1]) {
      // If there are duplicates, reassign order values sequentially
      for (size_t j = 0; j < column_info_.size(); ++j) {
        column_info_[j].order = static_cast<int>(j);
      }
      break; // Exit after fixing duplicates
    }
  }
}

}  // namespace BTQuant