#include "../../include/ui/data_settings.hpp"
#include "../../include/ui/settings_manager.hpp"

#include <imgui.h>
#include <vector>
#include <string>

namespace BTQuant {
namespace UI {

// ============================================================================
// Data Settings Implementation
// ============================================================================

DataSettings::DataSettings(SettingsManager& settings_manager)
    : settings_manager_(settings_manager) {
    initialize_data_settings();
}

void DataSettings::initialize_data_settings() {
    // Initialize all data-related settings
    initialize_timeframe_settings();
    initialize_symbol_settings();
    initialize_workspace_settings();
    initialize_retention_settings();
    initialize_websocket_settings();
}

void DataSettings::initialize_timeframe_settings() {
    // Default timeframe setting
    SettingInfo default_timeframe_setting;
    default_timeframe_setting.key = "data.default_timeframe";
    default_timeframe_setting.display_name = "Default Timeframe";
    default_timeframe_setting.description = "Default timeframe for new charts and data views";
    default_timeframe_setting.type = SettingType::ENUM;
    default_timeframe_setting.category = SettingCategory::DATA;
    default_timeframe_setting.enum_options = {
        "1s", "5s", "15s", "30s", "1m", "3m", "5m", "15m", "30m", 
        "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"
    };
    default_timeframe_setting.enum_selected_index = 9; // Default to 1h
    
    settings_manager_.register_setting(default_timeframe_setting);
}

void DataSettings::initialize_symbol_settings() {
    // Default symbol setting
    SettingInfo default_symbol_setting;
    default_symbol_setting.key = "data.default_symbol";
    default_symbol_setting.display_name = "Default Symbol";
    default_symbol_setting.description = "Default symbol to load when opening new charts";
    default_symbol_setting.type = SettingType::STRING;
    default_symbol_setting.category = SettingCategory::DATA;
    default_symbol_setting.string_value = "BTCUSDT"; // Default to BTCUSDT
    
    settings_manager_.register_setting(default_symbol_setting);
}

void DataSettings::initialize_workspace_settings() {
    // Auto-load last used workspace setting
    SettingInfo auto_load_workspace_setting;
    auto_load_workspace_setting.key = "data.auto_load_last_workspace";
    auto_load_workspace_setting.display_name = "Auto-load Last Workspace";
    auto_load_workspace_setting.description = "Automatically load the last used workspace on startup";
    auto_load_workspace_setting.type = SettingType::BOOLEAN;
    auto_load_workspace_setting.category = SettingCategory::DATA;
    auto_load_workspace_setting.bool_value = true; // Default to enabled
    
    settings_manager_.register_setting(auto_load_workspace_setting);
}

void DataSettings::initialize_retention_settings() {
    // Data retention period setting
    SettingInfo retention_period_setting;
    retention_period_setting.key = "data.retention_period_hours";
    retention_period_setting.display_name = "Data Retention Period (Hours)";
    retention_period_setting.description = "How long to keep historical data in memory before purging (in hours)";
    retention_period_setting.type = SettingType::INTEGER;
    retention_period_setting.category = SettingCategory::DATA;
    retention_period_setting.int_value = 24; // Default to 24 hours
    retention_period_setting.min_int = 1;    // Minimum 1 hour
    retention_period_setting.max_int = 168;  // Maximum 1 week (168 hours)
    
    settings_manager_.register_setting(retention_period_setting);
}

void DataSettings::initialize_websocket_settings() {
    // WebSocket reconnection attempts
    SettingInfo ws_reconnect_attempts_setting;
    ws_reconnect_attempts_setting.key = "data.websocket.reconnect_attempts";
    ws_reconnect_attempts_setting.display_name = "WebSocket Reconnection Attempts";
    ws_reconnect_attempts_setting.description = "Number of times to attempt reconnection when WebSocket connection is lost";
    ws_reconnect_attempts_setting.type = SettingType::INTEGER;
    ws_reconnect_attempts_setting.category = SettingCategory::DATA;
    ws_reconnect_attempts_setting.int_value = 5; // Default to 5 attempts
    ws_reconnect_attempts_setting.min_int = 0;   // Minimum 0 (no reconnection)
    ws_reconnect_attempts_setting.max_int = 50;  // Maximum 50 attempts
    
    settings_manager_.register_setting(ws_reconnect_attempts_setting);

    // WebSocket reconnection delay
    SettingInfo ws_reconnect_delay_setting;
    ws_reconnect_delay_setting.key = "data.websocket.reconnect_delay_seconds";
    ws_reconnect_delay_setting.display_name = "WebSocket Reconnection Delay (Seconds)";
    ws_reconnect_delay_setting.description = "Delay between reconnection attempts in seconds";
    ws_reconnect_delay_setting.type = SettingType::INTEGER;
    ws_reconnect_delay_setting.category = SettingCategory::DATA;
    ws_reconnect_delay_setting.int_value = 5; // Default to 5 seconds
    ws_reconnect_delay_setting.min_int = 1;   // Minimum 1 second
    ws_reconnect_delay_setting.max_int = 60;  // Maximum 60 seconds
    
    settings_manager_.register_setting(ws_reconnect_delay_setting);

    // WebSocket heartbeat interval
    SettingInfo ws_heartbeat_interval_setting;
    ws_heartbeat_interval_setting.key = "data.websocket.heartbeat_interval_seconds";
    ws_heartbeat_interval_setting.display_name = "WebSocket Heartbeat Interval (Seconds)";
    ws_heartbeat_interval_setting.description = "Interval for sending heartbeat messages to maintain connection";
    ws_heartbeat_interval_setting.type = SettingType::INTEGER;
    ws_heartbeat_interval_setting.category = SettingCategory::DATA;
    ws_heartbeat_interval_setting.int_value = 30; // Default to 30 seconds
    ws_heartbeat_interval_setting.min_int = 10;   // Minimum 10 seconds
    ws_heartbeat_interval_setting.max_int = 300;  // Maximum 5 minutes
    
    settings_manager_.register_setting(ws_heartbeat_interval_setting);
}

void DataSettings::apply_data_settings() {
    // Apply data settings to the system
    // This would typically involve updating data providers, cache managers, etc.
    // with the current values from the settings manager
}

std::string DataSettings::get_default_timeframe() const {
    int timeframe_index = settings_manager_.get_enum("data.default_timeframe", 9); // Default to 1h
    std::vector<std::string> timeframes = {
        "1s", "5s", "15s", "30s", "1m", "3m", "5m", "15m", "30m", 
        "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"
    };
    
    if (timeframe_index >= 0 && timeframe_index < static_cast<int>(timeframes.size())) {
        return timeframes[timeframe_index];
    }
    return "1h"; // Fallback
}

std::string DataSettings::get_default_symbol() const {
    return settings_manager_.get_string("data.default_symbol", "BTCUSDT");
}

bool DataSettings::should_auto_load_workspace() const {
    return settings_manager_.get_bool("data.auto_load_last_workspace", true);
}

int DataSettings::get_retention_period_hours() const {
    return settings_manager_.get_int("data.retention_period_hours", 24);
}

int DataSettings::get_websocket_reconnect_attempts() const {
    return settings_manager_.get_int("data.websocket.reconnect_attempts", 5);
}

int DataSettings::get_websocket_reconnect_delay_seconds() const {
    return settings_manager_.get_int("data.websocket.reconnect_delay_seconds", 5);
}

int DataSettings::get_websocket_heartbeat_interval_seconds() const {
    return settings_manager_.get_int("data.websocket.heartbeat_interval_seconds", 30);
}

}  // namespace UI
}  // namespace BTQuant