#include "../../include/ui/alert_settings.hpp"
#include "../../include/ui/settings_manager.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <algorithm>
#include <vector>
#include <string>
#include <map>

namespace BTQuant {
namespace UI {

// ============================================================================
// Alert Settings Implementation
// ============================================================================

AlertSettings::AlertSettings(SettingsManager& settings_manager)
    : settings_manager_(settings_manager) {
    initialize_alert_settings();
}

void AlertSettings::initialize_alert_settings() {
    // Initialize notification method settings
    initialize_notification_method_settings();

    // Initialize alert history settings
    initialize_alert_history_settings();

    // Initialize alert conditions template settings
    initialize_alert_conditions_template_settings();
}

void AlertSettings::initialize_notification_method_settings() {
    // Notification method selection (Popup, Sound, System Tray, Email, Webhook)
    SettingInfo notification_method_setting;
    notification_method_setting.key = "alerts.notification_method";
    notification_method_setting.display_name = "Notification Method";
    notification_method_setting.description = "Select how you want to receive alerts (Popup, Sound, System Tray, Email, Webhook)";
    notification_method_setting.type = SettingType::ENUM;
    notification_method_setting.category = SettingCategory::ALERTS;
    notification_method_setting.enum_options = {"Popup Only", "Sound Only", "System Tray Only", 
                                               "Popup + Sound", "Popup + System Tray", 
                                               "Sound + System Tray", "All Methods", "Email", "Webhook"};
    notification_method_setting.enum_selected_index = 0;

    settings_manager_.register_setting(notification_method_setting);

    // Popup notification settings
    SettingInfo popup_enabled_setting;
    popup_enabled_setting.key = "alerts.popup_enabled";
    popup_enabled_setting.display_name = "Enable Popup Notifications";
    popup_enabled_setting.description = "Show popup notifications when alerts trigger";
    popup_enabled_setting.type = SettingType::BOOLEAN;
    popup_enabled_setting.category = SettingCategory::ALERTS;
    popup_enabled_setting.bool_value = true;
    
    settings_manager_.register_setting(popup_enabled_setting);

    SettingInfo popup_duration_setting;
    popup_duration_setting.key = "alerts.popup_duration";
    popup_duration_setting.display_name = "Popup Duration (seconds)";
    popup_duration_setting.description = "How long popup notifications remain visible";
    popup_duration_setting.type = SettingType::INTEGER;
    popup_duration_setting.category = SettingCategory::ALERTS;
    popup_duration_setting.int_value = 5;
    popup_duration_setting.min_int = 1;
    popup_duration_setting.max_int = 30;

    settings_manager_.register_setting(popup_duration_setting);

    // Sound notification settings
    SettingInfo sound_enabled_setting;
    sound_enabled_setting.key = "alerts.sound_enabled";
    sound_enabled_setting.display_name = "Enable Sound Notifications";
    sound_enabled_setting.description = "Play sound when alerts trigger";
    sound_enabled_setting.type = SettingType::BOOLEAN;
    sound_enabled_setting.category = SettingCategory::ALERTS;
    sound_enabled_setting.bool_value = true;

    settings_manager_.register_setting(sound_enabled_setting);

    SettingInfo sound_volume_setting;
    sound_volume_setting.key = "alerts.sound_volume";
    sound_volume_setting.display_name = "Sound Volume";
    sound_volume_setting.description = "Volume level for alert sounds";
    sound_volume_setting.type = SettingType::INTEGER;
    sound_volume_setting.category = SettingCategory::ALERTS;
    sound_volume_setting.int_value = 75;
    sound_volume_setting.min_int = 0;
    sound_volume_setting.max_int = 100;

    settings_manager_.register_setting(sound_volume_setting);

    SettingInfo sound_type_setting;
    sound_type_setting.key = "alerts.sound_type";
    sound_type_setting.display_name = "Sound Type";
    sound_type_setting.description = "Type of sound to play for alerts";
    sound_type_setting.type = SettingType::ENUM;
    sound_type_setting.category = SettingCategory::ALERTS;
    sound_type_setting.enum_options = {"Beep", "Chime", "Bell", "Custom", "Alarm", "Notification"};
    sound_type_setting.enum_selected_index = 0;

    settings_manager_.register_setting(sound_type_setting);

    // System tray notification settings
    SettingInfo system_tray_enabled_setting;
    system_tray_enabled_setting.key = "alerts.system_tray_enabled";
    system_tray_enabled_setting.display_name = "Enable System Tray Notifications";
    system_tray_enabled_setting.description = "Show notifications in system tray when alerts trigger";
    system_tray_enabled_setting.type = SettingType::BOOLEAN;
    system_tray_enabled_setting.category = SettingCategory::ALERTS;
    system_tray_enabled_setting.bool_value = true;

    settings_manager_.register_setting(system_tray_enabled_setting);

    SettingInfo system_tray_persistence_setting;
    system_tray_persistence_setting.key = "alerts.system_tray_persistence";
    system_tray_persistence_setting.display_name = "System Tray Persistence";
    system_tray_persistence_setting.description = "How long system tray notifications remain visible";
    system_tray_persistence_setting.type = SettingType::INTEGER;
    system_tray_persistence_setting.category = SettingCategory::ALERTS;
    system_tray_persistence_setting.int_value = 10;
    system_tray_persistence_setting.min_int = 5;
    system_tray_persistence_setting.max_int = 60;

    settings_manager_.register_setting(system_tray_persistence_setting);

    // Email notification settings
    SettingInfo email_enabled_setting;
    email_enabled_setting.key = "alerts.email_enabled";
    email_enabled_setting.display_name = "Enable Email Notifications";
    email_enabled_setting.description = "Send email notifications when alerts trigger";
    email_enabled_setting.type = SettingType::BOOLEAN;
    email_enabled_setting.category = SettingCategory::ALERTS;
    email_enabled_setting.bool_value = false;

    settings_manager_.register_setting(email_enabled_setting);

    SettingInfo email_address_setting;
    email_address_setting.key = "alerts.email_address";
    email_address_setting.display_name = "Email Address";
    email_address_setting.description = "Email address to send notifications to";
    email_address_setting.type = SettingType::STRING;
    email_address_setting.category = SettingCategory::ALERTS;
    email_address_setting.string_value = "";

    settings_manager_.register_setting(email_address_setting);

    // Webhook notification settings
    SettingInfo webhook_enabled_setting;
    webhook_enabled_setting.key = "alerts.webhook_enabled";
    webhook_enabled_setting.display_name = "Enable Webhook Notifications";
    webhook_enabled_setting.description = "Send webhook notifications when alerts trigger";
    webhook_enabled_setting.type = SettingType::BOOLEAN;
    webhook_enabled_setting.category = SettingCategory::ALERTS;
    webhook_enabled_setting.bool_value = false;

    settings_manager_.register_setting(webhook_enabled_setting);

    SettingInfo webhook_url_setting;
    webhook_url_setting.key = "alerts.webhook_url";
    webhook_url_setting.display_name = "Webhook URL";
    webhook_url_setting.description = "URL to send webhook notifications to";
    webhook_url_setting.type = SettingType::STRING;
    webhook_url_setting.category = SettingCategory::ALERTS;
    webhook_url_setting.string_value = "";

    settings_manager_.register_setting(webhook_url_setting);
}

void AlertSettings::initialize_alert_history_settings() {
    // Alert history size setting
    SettingInfo alert_history_size_setting;
    alert_history_size_setting.key = "alerts.history_size";
    alert_history_size_setting.display_name = "Alert History Size";
    alert_history_size_setting.description = "Maximum number of alerts to store in history";
    alert_history_size_setting.type = SettingType::INTEGER;
    alert_history_size_setting.category = SettingCategory::ALERTS;
    alert_history_size_setting.int_value = 100;
    alert_history_size_setting.min_int = 10;
    alert_history_size_setting.max_int = 10000;

    settings_manager_.register_setting(alert_history_size_setting);

    // Alert history retention days
    SettingInfo alert_history_retention_setting;
    alert_history_retention_setting.key = "alerts.history_retention_days";
    alert_history_retention_setting.display_name = "History Retention (days)";
    alert_history_retention_setting.description = "Number of days to retain alert history";
    alert_history_retention_setting.type = SettingType::INTEGER;
    alert_history_retention_setting.category = SettingCategory::ALERTS;
    alert_history_retention_setting.int_value = 30;
    alert_history_retention_setting.min_int = 1;
    alert_history_retention_setting.max_int = 365;

    settings_manager_.register_setting(alert_history_retention_setting);

    // Enable/disable alert history
    SettingInfo alert_history_enabled_setting;
    alert_history_enabled_setting.key = "alerts.history_enabled";
    alert_history_enabled_setting.display_name = "Enable Alert History";
    alert_history_enabled_setting.description = "Keep a record of triggered alerts";
    alert_history_enabled_setting.type = SettingType::BOOLEAN;
    alert_history_enabled_setting.category = SettingCategory::ALERTS;
    alert_history_enabled_setting.bool_value = true;

    settings_manager_.register_setting(alert_history_enabled_setting);

    // Export alert history format
    SettingInfo alert_history_export_format_setting;
    alert_history_export_format_setting.key = "alerts.history_export_format";
    alert_history_export_format_setting.display_name = "Export Format";
    alert_history_export_format_setting.description = "Format to use when exporting alert history";
    alert_history_export_format_setting.type = SettingType::ENUM;
    alert_history_export_format_setting.category = SettingCategory::ALERTS;
    alert_history_export_format_setting.enum_options = {"JSON", "CSV", "TXT", "XML"};
    alert_history_export_format_setting.enum_selected_index = 0;

    settings_manager_.register_setting(alert_history_export_format_setting);
}

void AlertSettings::initialize_alert_conditions_template_settings() {
    // Predefined alert condition templates
    SettingInfo alert_condition_templates_setting;
    alert_condition_templates_setting.key = "alerts.condition_templates";
    alert_condition_templates_setting.display_name = "Alert Condition Templates";
    alert_condition_templates_setting.description = "Predefined templates for common alert conditions";
    alert_condition_templates_setting.type = SettingType::ENUM;
    alert_condition_templates_setting.category = SettingCategory::ALERTS;
    alert_condition_templates_setting.enum_options = {
        "Price Crosses SMA", 
        "Price Crosses EMA", 
        "RSI Overbought/Oversold", 
        "Bollinger Band Touch/Breakout",
        "MACD Cross Signal", 
        "Stochastic Overbought/Oversold",
        "Volume Spike",
        "Price Action Patterns",
        "Custom Template"
    };
    alert_condition_templates_setting.enum_selected_index = 0;

    settings_manager_.register_setting(alert_condition_templates_setting);

    // Custom template name
    SettingInfo custom_template_name_setting;
    custom_template_name_setting.key = "alerts.custom_template_name";
    custom_template_name_setting.display_name = "Custom Template Name";
    custom_template_name_setting.description = "Name for your custom alert template";
    custom_template_name_setting.type = SettingType::STRING;
    custom_template_name_setting.category = SettingCategory::ALERTS;
    custom_template_name_setting.string_value = "My Custom Template";

    settings_manager_.register_setting(custom_template_name_setting);

    // Custom template content
    SettingInfo custom_template_content_setting;
    custom_template_content_setting.key = "alerts.custom_template_content";
    custom_template_content_setting.display_name = "Custom Template Content";
    custom_template_content_setting.description = "Content of your custom alert template";
    custom_template_content_setting.type = SettingType::STRING;
    custom_template_content_setting.category = SettingCategory::ALERTS;
    custom_template_content_setting.string_value = "";

    settings_manager_.register_setting(custom_template_content_setting);

    // Enable/disable specific templates
    SettingInfo enable_sma_template_setting;
    enable_sma_template_setting.key = "alerts.enable_sma_template";
    enable_sma_template_setting.display_name = "Enable SMA Template";
    enable_sma_template_setting.description = "Enable the Simple Moving Average cross template";
    enable_sma_template_setting.type = SettingType::BOOLEAN;
    enable_sma_template_setting.category = SettingCategory::ALERTS;
    enable_sma_template_setting.bool_value = true;

    settings_manager_.register_setting(enable_sma_template_setting);

    SettingInfo enable_ema_template_setting;
    enable_ema_template_setting.key = "alerts.enable_ema_template";
    enable_ema_template_setting.display_name = "Enable EMA Template";
    enable_ema_template_setting.description = "Enable the Exponential Moving Average cross template";
    enable_ema_template_setting.type = SettingType::BOOLEAN;
    enable_ema_template_setting.category = SettingCategory::ALERTS;
    enable_ema_template_setting.bool_value = true;

    settings_manager_.register_setting(enable_ema_template_setting);

    SettingInfo enable_rsi_template_setting;
    enable_rsi_template_setting.key = "alerts.enable_rsi_template";
    enable_rsi_template_setting.display_name = "Enable RSI Template";
    enable_rsi_template_setting.description = "Enable the RSI overbought/oversold template";
    enable_rsi_template_setting.type = SettingType::BOOLEAN;
    enable_rsi_template_setting.category = SettingCategory::ALERTS;
    enable_rsi_template_setting.bool_value = true;

    settings_manager_.register_setting(enable_rsi_template_setting);

    SettingInfo enable_bollinger_template_setting;
    enable_bollinger_template_setting.key = "alerts.enable_bollinger_template";
    enable_bollinger_template_setting.display_name = "Enable Bollinger Template";
    enable_bollinger_template_setting.description = "Enable the Bollinger Band touch/breakout template";
    enable_bollinger_template_setting.type = SettingType::BOOLEAN;
    enable_bollinger_template_setting.category = SettingCategory::ALERTS;
    enable_bollinger_template_setting.bool_value = true;

    settings_manager_.register_setting(enable_bollinger_template_setting);

    SettingInfo enable_macd_template_setting;
    enable_macd_template_setting.key = "alerts.enable_macd_template";
    enable_macd_template_setting.display_name = "Enable MACD Template";
    enable_macd_template_setting.description = "Enable the MACD cross signal template";
    enable_macd_template_setting.type = SettingType::BOOLEAN;
    enable_macd_template_setting.category = SettingCategory::ALERTS;
    enable_macd_template_setting.bool_value = true;

    settings_manager_.register_setting(enable_macd_template_setting);

    SettingInfo enable_stochastic_template_setting;
    enable_stochastic_template_setting.key = "alerts.enable_stochastic_template";
    enable_stochastic_template_setting.display_name = "Enable Stochastic Template";
    enable_stochastic_template_setting.description = "Enable the Stochastic overbought/oversold template";
    enable_stochastic_template_setting.type = SettingType::BOOLEAN;
    enable_stochastic_template_setting.category = SettingCategory::ALERTS;
    enable_stochastic_template_setting.bool_value = true;

    settings_manager_.register_setting(enable_stochastic_template_setting);

    SettingInfo enable_volume_template_setting;
    enable_volume_template_setting.key = "alerts.enable_volume_template";
    enable_volume_template_setting.display_name = "Enable Volume Template";
    enable_volume_template_setting.description = "Enable the Volume spike template";
    enable_volume_template_setting.type = SettingType::BOOLEAN;
    enable_volume_template_setting.category = SettingCategory::ALERTS;
    enable_volume_template_setting.bool_value = true;

    settings_manager_.register_setting(enable_volume_template_setting);
}

void AlertSettings::apply_notification_settings() {
    // Apply notification settings based on current configuration
    // This would typically involve updating the alert system with the current settings
    
    bool popup_enabled = settings_manager_.get_bool("alerts.popup_enabled", true);
    bool sound_enabled = settings_manager_.get_bool("alerts.sound_enabled", true);
    bool system_tray_enabled = settings_manager_.get_bool("alerts.system_tray_enabled", true);
    bool email_enabled = settings_manager_.get_bool("alerts.email_enabled", false);
    bool webhook_enabled = settings_manager_.get_bool("alerts.webhook_enabled", false);
    
    // In a real implementation, this would update the alert system with these settings
    // For now, we just have the settings registered
}

void AlertSettings::apply_history_settings() {
    // Apply history settings based on current configuration
    int history_size = settings_manager_.get_int("alerts.history_size", 100);
    int retention_days = settings_manager_.get_int("alerts.history_retention_days", 30);
    bool history_enabled = settings_manager_.get_bool("alerts.history_enabled", true);
    
    // In a real implementation, this would update the alert history system with these settings
    // For now, we just have the settings registered
}

void AlertSettings::apply_template_settings() {
    // Apply template settings based on current configuration
    int selected_template = settings_manager_.get_enum("alerts.condition_templates", 0);
    bool enable_sma = settings_manager_.get_bool("alerts.enable_sma_template", true);
    bool enable_ema = settings_manager_.get_bool("alerts.enable_ema_template", true);
    bool enable_rsi = settings_manager_.get_bool("alerts.enable_rsi_template", true);
    bool enable_bollinger = settings_manager_.get_bool("alerts.enable_bollinger_template", true);
    bool enable_macd = settings_manager_.get_bool("alerts.enable_macd_template", true);
    bool enable_stochastic = settings_manager_.get_bool("alerts.enable_stochastic_template", true);
    bool enable_volume = settings_manager_.get_bool("alerts.enable_volume_template", true);
    
    // In a real implementation, this would update the alert template system with these settings
    // For now, we just have the settings registered
}

}  // namespace UI
}  // namespace BTQuant