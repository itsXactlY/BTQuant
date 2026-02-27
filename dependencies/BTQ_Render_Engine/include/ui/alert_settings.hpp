#pragma once

#include <string>
#include <vector>

#include "settings_manager.hpp"

namespace BTQuant {
namespace UI {

// ============================================================================
// Alert Settings Class
// ============================================================================

class AlertSettings {
public:
    explicit AlertSettings(SettingsManager& settings_manager);

    // Initialize all alert settings
    void initialize_alert_settings();

    // Apply notification settings
    void apply_notification_settings();

    // Apply history settings
    void apply_history_settings();

    // Apply template settings
    void apply_template_settings();

private:
    SettingsManager& settings_manager_;

    // Initialize notification method settings
    void initialize_notification_method_settings();

    // Initialize alert history settings
    void initialize_alert_history_settings();

    // Initialize alert conditions template settings
    void initialize_alert_conditions_template_settings();
};

}  // namespace UI
}  // namespace BTQuant