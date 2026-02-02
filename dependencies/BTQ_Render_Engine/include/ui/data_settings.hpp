#pragma once

#include <string>
#include <vector>

#include "settings_manager.hpp"

namespace BTQuant {
namespace UI {

class DataSettings {
public:
    explicit DataSettings(SettingsManager& settings_manager);

    // Initialize all data settings
    void initialize_data_settings();

    // Apply current data settings to the system
    void apply_data_settings();

    // Getter methods for specific settings
    std::string get_default_timeframe() const;
    std::string get_default_symbol() const;
    bool should_auto_load_workspace() const;
    int get_retention_period_hours() const;
    int get_websocket_reconnect_attempts() const;
    int get_websocket_reconnect_delay_seconds() const;
    int get_websocket_heartbeat_interval_seconds() const;

private:
    SettingsManager& settings_manager_;

    // Initialize specific setting groups
    void initialize_timeframe_settings();
    void initialize_symbol_settings();
    void initialize_workspace_settings();
    void initialize_retention_settings();
    void initialize_websocket_settings();
};

}  // namespace UI
}  // namespace BTQuant