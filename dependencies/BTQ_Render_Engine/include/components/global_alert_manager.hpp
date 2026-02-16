#pragma once

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "alert_common.hpp"
#include "watchlist_alerts.hpp"

namespace BTQuant {

// Enum for different alert types
enum class GlobalAlertType {
  PRICE_ABOVE,
  PRICE_BELOW,
  VOLUME_ABOVE,
  VOLUME_BELOW,
  INDICATOR_CROSS,
  CUSTOM_EXPRESSION
};

// Structure to represent a global alert
struct GlobalAlert {
  std::string id;
  std::string name;
  uint32_t symbol_id;
  std::string symbol_name;
  GlobalAlertType type;
  double threshold_value;
  double current_value;
  AlertStatus status;
  std::chrono::system_clock::time_point created_at;
  std::chrono::system_clock::time_point triggered_at;
  std::vector<std::string> actions;  // e.g. "log", "sound", "email", "notification"

  // Default constructor for std::map compatibility
  GlobalAlert()
      : symbol_id(0),
        type(GlobalAlertType::PRICE_ABOVE),
        threshold_value(0.0),
        current_value(0.0),
        status(AlertStatus::ACTIVE),
        created_at(std::chrono::system_clock::now()),
        triggered_at(std::chrono::system_clock::time_point{}) {}

  // Constructor
  GlobalAlert(const std::string& alert_id, const std::string& alert_name, uint32_t sym_id,
              const std::string& sym_name, GlobalAlertType alert_type, double threshold)
      : id(alert_id),
        name(alert_name),
        symbol_id(sym_id),
        symbol_name(sym_name),
        type(alert_type),
        threshold_value(threshold),
        current_value(0.0),
        status(AlertStatus::ACTIVE),
        created_at(std::chrono::system_clock::now()),
        triggered_at(std::chrono::system_clock::time_point{}),
        actions({"log"}) {}  // Default to logging
};

// Callback type for when alerts are triggered
using GlobalAlertTriggeredCallback = std::function<void(const GlobalAlert&, double current_value)>;

class GlobalAlertManager {
 public:
  // DEPRECATED - Legacy hotspine
  GlobalAlertManager(std::shared_ptr<HotSpineDataBridge> bridge,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                     std::shared_ptr<AlertsPanel> alerts_panel);
  ~GlobalAlertManager() = default;

  // Add a price alert
  std::string add_price_alert(uint32_t symbol_id, const std::string& symbol_name,
                              double target_price, GlobalAlertType type,
                              const std::string& name = "");

  // Add a volume alert
  std::string add_volume_alert(uint32_t symbol_id, const std::string& symbol_name,
                               double target_volume, GlobalAlertType type,
                               const std::string& name = "");

  // Add a custom alert with expression
  std::string add_custom_alert(uint32_t symbol_id, const std::string& symbol_name,
                               const std::string& expression, const std::string& name = "");

  // Remove an alert by ID
  bool remove_alert(const std::string& alert_id);

  // Remove all alerts for a specific symbol
  bool remove_alerts_for_symbol(uint32_t symbol_id);

  // Enable/disable an alert
  bool enable_alert(const std::string& alert_id, bool enable);

  // Update and check all active alerts against current market data
  void update_alerts();

  // Get all alerts for a specific symbol
  std::vector<GlobalAlert> get_alerts_for_symbol(uint32_t symbol_id) const;

  // Get all alerts
  const std::map<std::string, GlobalAlert>& get_all_alerts() const { return alerts_; }

  // Set callback for when an alert is triggered
  void set_alert_triggered_callback(GlobalAlertTriggeredCallback callback) {
    on_alert_triggered_ = std::move(callback);
  }

  // Set the alerts panel to send notifications to
  void set_alerts_panel(std::shared_ptr<AlertsPanel> alerts_panel) { alerts_panel_ = alerts_panel; }

  // Get statistics about alerts
  size_t get_total_alerts_count() const { return alerts_.size(); }
  size_t get_active_alerts_count() const;
  size_t get_triggered_alerts_count() const;

 private:
  // DEPRECATED - Legacy hotspine
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<AlertsPanel> alerts_panel_;
  std::map<std::string, GlobalAlert> alerts_;
  GlobalAlertTriggeredCallback on_alert_triggered_;
  mutable std::mutex alerts_mutex_;  // Mutex for thread-safe access

  // Check if a specific alert should be triggered based on current data
  bool should_trigger_alert(const GlobalAlert& alert, double current_price,
                            double current_volume) const;

  // Trigger an alert and log it
  void trigger_alert(const GlobalAlert& alert, double current_value);

  // Generate unique alert ID
  std::string generate_alert_id() const;

  // Helper method to get current price and volume for a symbol
  std::pair<double, double> get_current_market_data(uint32_t symbol_id) const;
};

}  // namespace BTQuant