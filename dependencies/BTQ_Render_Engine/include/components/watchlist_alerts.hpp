#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../market_data_processor.hpp"
#include "alerts_panel.hpp"

namespace BTQuant {

// Structure to represent a price alert for a watchlist symbol
struct WatchlistPriceAlert {
    std::string id;
    uint32_t symbol_id;
    std::string symbol_name;
    double target_price;
    enum class Direction { ABOVE, BELOW } direction;
    AlertStatus status;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point triggered_at;

    // Default constructor
    WatchlistPriceAlert() : id(""), symbol_id(0), symbol_name(""), target_price(0.0),
                           direction(Direction::ABOVE), status(AlertStatus::ACTIVE),
                           created_at(std::chrono::system_clock::time_point{}),
                           triggered_at(std::chrono::system_clock::time_point{}) {}

    // Constructor
    WatchlistPriceAlert(const std::string& alert_id, uint32_t sym_id, const std::string& sym_name,
                       double price, Direction dir)
        : id(alert_id), symbol_id(sym_id), symbol_name(sym_name), target_price(price),
          direction(dir), status(AlertStatus::ACTIVE),
          created_at(std::chrono::system_clock::now()),
          triggered_at(std::chrono::system_clock::time_point{}) {}
};

class WatchlistAlertManager {
public:
    using AlertTriggeredCallback = std::function<void(const WatchlistPriceAlert&, double current_price)>;

    WatchlistAlertManager(std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                         std::shared_ptr<AlertsPanel> alerts_panel);
    ~WatchlistAlertManager() = default;

    // Add a price alert for a watchlist symbol
    std::string add_price_alert(uint32_t symbol_id, const std::string& symbol_name, 
                               double target_price, WatchlistPriceAlert::Direction direction);

    // Remove a specific alert by ID
    bool remove_alert(const std::string& alert_id);

    // Remove all alerts for a specific symbol
    bool remove_alerts_for_symbol(uint32_t symbol_id);

    // Enable/disable an alert
    bool enable_alert(const std::string& alert_id, bool enable);

    // Update and check all active alerts against current market data
    void update_alerts();

    // Get all alerts for a specific symbol
    std::vector<WatchlistPriceAlert> get_alerts_for_symbol(uint32_t symbol_id) const;

    // Get all alerts
    const std::map<std::string, WatchlistPriceAlert>& get_all_alerts() const { return alerts_; }

    // Set callback for when an alert is triggered
    void set_alert_triggered_callback(AlertTriggeredCallback callback) {
        on_alert_triggered_ = std::move(callback);
    }

    // Set the alerts panel to send notifications to
    void set_alerts_panel(std::shared_ptr<AlertsPanel> alerts_panel) {
        alerts_panel_ = alerts_panel;
        // When setting shared_ptr, also clear the raw pointer to avoid duplicate notifications
        alerts_panel_raw_ = nullptr;
    }

    // Set the alerts panel using a raw pointer (for internal use by panel manager)
    void set_alerts_panel_raw(AlertsPanel* alerts_panel) {
        alerts_panel_raw_ = alerts_panel;
        // When setting raw pointer, also clear the shared_ptr to avoid duplicate notifications
        alerts_panel_ = nullptr;
    }

private:
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    std::shared_ptr<AlertsPanel> alerts_panel_;
    AlertsPanel* alerts_panel_raw_ = nullptr;  // Raw pointer for panel manager connections
    std::map<std::string, WatchlistPriceAlert> alerts_;
    AlertTriggeredCallback on_alert_triggered_;

    // Check if a specific alert should be triggered based on current price
    bool should_trigger_alert(const WatchlistPriceAlert& alert, double current_price) const;

    // Trigger an alert and log it
    void trigger_alert(const WatchlistPriceAlert& alert, double current_price);

    // Generate unique alert ID
    std::string generate_alert_id() const;
};

}  // namespace BTQuant