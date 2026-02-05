#pragma once

#include <string>
#include <chrono>

namespace BTQuant {

// Common alert status enum used across the application
enum class AlertStatus { ACTIVE, TRIGGERED, DISABLED, COOLDOWN };

// Common alert types
enum class AlertType {
    PRICE_ABOVE,
    PRICE_BELOW,
    VOLUME_ABOVE,
    VOLUME_BELOW,
    INDICATOR_CROSS,
    CUSTOM_EXPRESSION
};

// Structure for alert logs
struct AlertLog {
    std::chrono::system_clock::time_point time;
    std::string rule_name;
    std::string symbol;
    double price;
    std::string message;
};

}  // namespace BTQuant