#include "../../include/ui/tooltips.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <string>

namespace BTQuant {
namespace UI {

// TooltipManager implementation
TooltipManager::TooltipManager() {
    initialize_default_tooltips();
}

void TooltipManager::initialize_default_tooltips() {
    // Dashboard Controls tooltips
    register_tooltip("exchange_selector_button", "Select which exchanges to include in your trading view");
    register_tooltip("symbol_search_top", "Search for trading symbols across all selected exchanges");
    register_tooltip("add_chart_panel", "Add a price chart panel for technical analysis");
    register_tooltip("add_footprint_panel", "Add a footprint chart showing trade volume at price levels");
    register_tooltip("add_volume_profile_panel", "Add a volume profile chart showing volume distribution by price");
    register_tooltip("add_order_book_panel", "Add an order book panel showing buy/sell orders at different price levels");
    register_tooltip("add_time_sales_panel", "Add a time and sales panel showing recent trades");
    register_tooltip("add_watchlist_panel", "Add a watchlist panel to monitor multiple symbols");
    register_tooltip("add_news_panel", "Add a news and alerts panel for market updates");
    
    // Chart Panel tooltips
    register_tooltip("chart_timeframe_selector", "Select the timeframe for the chart data (1m, 5m, 1h, etc.)");
    register_tooltip("chart_indicator_selector", "Choose technical indicators to display on the chart");
    register_tooltip("chart_zoom_in", "Zoom in on the chart to see more detail");
    register_tooltip("chart_zoom_out", "Zoom out on the chart to see more historical data");
    register_tooltip("chart_reset_view", "Reset the chart view to default zoom and position");
    register_tooltip("chart_crosshair_toggle", "Toggle crosshair cursor for precise price/time readings");
    register_tooltip("chart_grid_toggle", "Toggle visibility of grid lines on the chart");
    register_tooltip("chart_legend_toggle", "Toggle visibility of the chart legend");
    
    // Trading Controls tooltips
    register_tooltip("place_buy_order", "Place a market buy order for the selected symbol");
    register_tooltip("place_sell_order", "Place a market sell order for the selected symbol");
    register_tooltip("limit_order_price", "Set the price for a limit order");
    register_tooltip("order_quantity", "Set the quantity of shares/contracts for the order");
    register_tooltip("order_type_selector", "Choose order type: Market, Limit, Stop, Stop-Limit");
    register_tooltip("take_profit_level", "Set the price level for automatic profit taking");
    register_tooltip("stop_loss_level", "Set the price level for automatic loss prevention");
    
    // Indicator Settings tooltips
    register_tooltip("enable_sma_9", "Enable Simple Moving Average with 9-period length");
    register_tooltip("enable_sma_20", "Enable Simple Moving Average with 20-period length");
    register_tooltip("enable_sma_50", "Enable Simple Moving Average with 50-period length");
    register_tooltip("enable_sma_200", "Enable Simple Moving Average with 200-period length");
    register_tooltip("enable_ema_9", "Enable Exponential Moving Average with 9-period length");
    register_tooltip("enable_ema_21", "Enable Exponential Moving Average with 21-period length");
    register_tooltip("enable_ema_50", "Enable Exponential Moving Average with 50-period length");
    register_tooltip("enable_ema_200", "Enable Exponential Moving Average with 200-period length");
    register_tooltip("enable_rsi", "Enable Relative Strength Index oscillator");
    register_tooltip("enable_bollinger_bands", "Enable Bollinger Bands showing volatility channels");
    
    // Appearance Settings tooltips
    register_tooltip("theme_selector", "Choose between Dark, Light, or High Contrast themes");
    register_tooltip("font_family_selector", "Select the primary font family for the interface");
    register_tooltip("font_size_slider", "Adjust the base font size for all text elements");
    register_tooltip("header_font_scale", "Scale factor for header text sizes relative to base font");
    register_tooltip("panel_opacity_slider", "Adjust transparency level of panel backgrounds");
    register_tooltip("border_width_slider", "Set the thickness of panel borders in pixels");
    register_tooltip("border_style_selector", "Choose the visual style of panel borders");
    register_tooltip("corner_radius_slider", "Set the radius for rounded corners on panels");
    
    // Data Settings tooltips
    register_tooltip("refresh_interval", "Set how often market data is updated from the feed");
    register_tooltip("historical_depth", "Amount of historical data to load for analysis");
    register_tooltip("data_compression", "Level of data compression to reduce memory usage");
    register_tooltip("cache_retention", "How long to keep cached data before purging");
    
    // Performance Settings tooltips
    register_tooltip("target_fps", "Maximum frames per second to render (affects CPU usage)");
    register_tooltip("adaptive_sync", "Enable adaptive synchronization for smoother rendering");
    register_tooltip("frame_smoothing", "Smooth frame timing to reduce stuttering");
    register_tooltip("burst_reduction", "Reduce frame rate bursts for consistent performance");
    register_tooltip("lod_enabled", "Enable Level of Detail to improve performance with large datasets");
    register_tooltip("lod_distance", "Distance threshold for applying level of detail reductions");
    
    // Alert Settings tooltips
    register_tooltip("price_alert_threshold", "Price change percentage that triggers an alert");
    register_tooltip("volume_spike_multiplier", "Volume multiplier that triggers a spike alert");
    register_tooltip("news_notification_toggle", "Enable/disable news notifications");
    register_tooltip("email_alerts_toggle", "Send email notifications for important alerts");
    register_tooltip("sound_alerts_toggle", "Play sound notifications for alerts");
    
    // Keyboard Shortcuts tooltips
    register_tooltip("shortcut_key_binding", "Press keys to set a new keyboard shortcut");
    register_tooltip("reset_shortcut_button", "Reset this shortcut to its default binding");
    register_tooltip("disable_shortcut_button", "Disable this keyboard shortcut");
    
    // Panel Management tooltips
    register_tooltip("panel_minimize", "Minimize this panel to save screen space");
    register_tooltip("panel_maximize", "Maximize this panel to fill available space");
    register_tooltip("panel_close", "Close this panel (data will be preserved)");
    register_tooltip("panel_duplicate", "Create a copy of this panel with the same settings");
    register_tooltip("panel_export", "Export this panel's configuration to a file");
    register_tooltip("panel_import", "Import a panel configuration from a file");
    
    // Time & Sales tooltips
    register_tooltip("trade_direction_filter", "Filter trades by direction (buy/sell/initiator)");
    register_tooltip("volume_filter_min", "Show only trades with volume greater than this value");
    register_tooltip("aggressor_flag", "Show whether buyer or seller initiated the trade");

    // Order Book tooltips
    register_tooltip("order_book_depth", "Number of price levels to display in the order book");
    register_tooltip("imbalance_ratio", "Threshold for highlighting order book imbalances");
    register_tooltip("iceberg_detection", "Detect and highlight potential iceberg orders");

    // Footprint Chart tooltips
    register_tooltip("footprint_step_size", "Price step size for aggregating footprint data");
    register_tooltip("footprint_volume_colormap", "Color mapping for volume intensity display");
    register_tooltip("footprint_delta_display", "Show buy/sell delta alongside volume");

    // Volume Profile tooltips
    register_tooltip("vp_time_period", "Time period to aggregate volume for the profile");
    register_tooltip("vp_point_of_control", "Highlight the price level with highest volume");
    register_tooltip("vp_value_area", "Show the range containing 70% of total volume");

    // Watchlist tooltips
    register_tooltip("watchlist_add_symbol", "Add a new symbol to this watchlist");
    register_tooltip("watchlist_remove_symbol", "Remove selected symbol from watchlist");
    register_tooltip("watchlist_sort_column", "Sort watchlist by this column in ascending order");
    register_tooltip("watchlist_auto_refresh", "Automatically update watchlist data at intervals");

    // Chart Replay tooltips
    register_tooltip("chart_replay_speed", "Adjust the playback speed of the historical data replay");
    register_tooltip("chart_replay_loop", "Enable looping playback when reaching the end of data");
    register_tooltip("chart_replay_step_by_step", "Enable step-by-step playback mode for detailed analysis");
    register_tooltip("chart_replay_manual_control", "Enable manual control of playback position");
    register_tooltip("chart_replay_backtesting_mode", "Enable backtesting practice mode with manual controls");
    register_tooltip("chart_replay_play", "Start or resume the historical data replay");
    register_tooltip("chart_replay_pause", "Pause the historical data replay");
    register_tooltip("chart_replay_stop", "Stop the replay and return to the beginning");
    register_tooltip("chart_replay_reset", "Reset the replay position to the beginning");
    register_tooltip("chart_replay_prev", "Step backward to the previous data point");
    register_tooltip("chart_replay_next", "Step forward to the next data point");
}

void TooltipManager::register_tooltip(const std::string& control_id, const std::string& tooltip_text) {
    tooltips_[control_id] = tooltip_text;
}

std::string TooltipManager::get_tooltip(const std::string& control_id) const {
    auto it = tooltips_.find(control_id);
    if (it != tooltips_.end()) {
        return it->second;
    }
    return ""; // Return empty string if tooltip not found
}

void TooltipManager::show_tooltip(const std::string& control_id) const {
    std::string tooltip = get_tooltip(control_id);
    if (!tooltip.empty()) {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(tooltip.c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }
}

void TooltipManager::show_tooltip_for_last_item(const std::string& control_id) const {
    std::string tooltip = get_tooltip(control_id);
    if (!tooltip.empty()) {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(tooltip.c_str());
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }
}

void TooltipManager::show_simple_tooltip(const std::string& tooltip_text) const {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
        ImGui::SetTooltip("%s", tooltip_text.c_str());
    }
}

// Static instance for global access
static TooltipManager g_tooltip_manager;

TooltipManager& get_global_tooltip_manager() {
    return g_tooltip_manager;
}

void show_control_tooltip(const std::string& control_id) {
    get_global_tooltip_manager().show_tooltip_for_last_item(control_id);
}

void show_simple_tooltip(const std::string& tooltip_text) {
    get_global_tooltip_manager().show_simple_tooltip(tooltip_text);
}

} // namespace UI
} // namespace BTQuant