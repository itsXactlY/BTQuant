#include "../../include/components/watchlist_alerts.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include "imgui.h"

namespace BTQuant {

WatchlistAlertManager::WatchlistAlertManager(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    std::shared_ptr<AlertsPanel> alerts_panel)
    : bridge_(bridge), processor_(processor), alerts_panel_(alerts_panel) {

    // Set up default callback to handle alerts even when alerts_panel_ is nullptr initially
    on_alert_triggered_ = [this](const WatchlistPriceAlert& alert, double current_price) {
        // Log the alert in the alerts panel if available (try shared_ptr first)
        if (alerts_panel_) {
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

            // Add the log entry to the alerts panel
            alerts_panel_->add_alert_log(log_entry);
        }
        // If shared_ptr is not available, try raw pointer
        else if (alerts_panel_raw_) {
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

            // Add the log entry to the alerts panel using raw pointer
            alerts_panel_raw_->add_alert_log(log_entry);
        }

        // Also output to console for debugging
        std::cout << "[WatchlistAlert] Triggered: " << alert.symbol_name
                  << " price alert at " << current_price << " (target: " << alert.target_price << ")" << std::endl;
    };
}

std::string WatchlistAlertManager::add_price_alert(
    uint32_t symbol_id, const std::string& symbol_name, 
    double target_price, WatchlistPriceAlert::Direction direction) {
    
    std::string alert_id = generate_alert_id();
    
    WatchlistPriceAlert alert(alert_id, symbol_id, symbol_name, target_price, direction);
    alerts_[alert_id] = std::move(alert);
    
    return alert_id;
}

bool WatchlistAlertManager::remove_alert(const std::string& alert_id) {
    auto it = alerts_.find(alert_id);
    if (it != alerts_.end()) {
        alerts_.erase(it);
        return true;
    }
    return false;
}

bool WatchlistAlertManager::remove_alerts_for_symbol(uint32_t symbol_id) {
    bool removed_any = false;
    for (auto it = alerts_.begin(); it != alerts_.end();) {
        if (it->second.symbol_id == symbol_id) {
            it = alerts_.erase(it);
            removed_any = true;
        } else {
            ++it;
        }
    }
    return removed_any;
}

bool WatchlistAlertManager::enable_alert(const std::string& alert_id, bool enable) {
    auto it = alerts_.find(alert_id);
    if (it != alerts_.end()) {
        it->second.status = enable ? AlertStatus::ACTIVE : AlertStatus::DISABLED;
        return true;
    }
    return false;
}

void WatchlistAlertManager::update_alerts() {
    // Iterate through all active alerts and check if they should be triggered
    for (auto& [alert_id, alert] : alerts_) {
        if (alert.status != AlertStatus::ACTIVE) {
            continue; // Skip disabled or triggered alerts
        }

        // Get current price for the symbol from the market data processor
        auto analytics = processor_->getSymbolAnalytics(alert.symbol_id);
        if (analytics.symbol_id != 0) {  // Check if valid data was returned
            double current_price = analytics.last_trade_price;

            if (should_trigger_alert(alert, current_price)) {
                // Update the alert status and timestamp
                alert.status = AlertStatus::TRIGGERED;
                alert.triggered_at = std::chrono::system_clock::now();

                // Trigger the alert
                trigger_alert(alert, current_price);
            }
        }
    }
}

std::vector<WatchlistPriceAlert> WatchlistAlertManager::get_alerts_for_symbol(uint32_t symbol_id) const {
    std::vector<WatchlistPriceAlert> result;
    for (const auto& [id, alert] : alerts_) {
        if (alert.symbol_id == symbol_id) {
            result.push_back(alert);
        }
    }
    return result;
}

bool WatchlistAlertManager::should_trigger_alert(const WatchlistPriceAlert& alert, double current_price) const {
    switch (alert.direction) {
        case WatchlistPriceAlert::Direction::ABOVE:
            return current_price >= alert.target_price;
        case WatchlistPriceAlert::Direction::BELOW:
            return current_price <= alert.target_price;
        default:
            return false;
    }
}

void WatchlistAlertManager::trigger_alert(const WatchlistPriceAlert& alert, double current_price) {
    // Call the callback which handles logging to the alerts panel
    if (on_alert_triggered_) {
        on_alert_triggered_(alert, current_price);
    }

    // Optionally play a sound or show a notification
    // ImGui::SetKeyboardFocusHere(-1); // Could trigger a notification
}

std::string WatchlistAlertManager::generate_alert_id() const {
    // Generate a random ID for the alert
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(1000, 9999);
    
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    return "alert_" + std::to_string(now) + "_" + std::to_string(dis(gen));
}

}  // namespace BTQuant