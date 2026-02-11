#include "../../include/components/global_alert_manager.hpp"

#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>

#include "../../include/components/alerts_panel.hpp"

namespace BTQuant {

GlobalAlertManager::GlobalAlertManager(std::shared_ptr<HotSpineDataBridge> bridge,
                                       std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                                       std::shared_ptr<AlertsPanel> alerts_panel)
    : bridge_(bridge), processor_(processor), alerts_panel_(alerts_panel) {
  // Initialize with empty callback
  on_alert_triggered_ = [](const GlobalAlert&, double) {};
  
  // Initialize the atomic shared_ptr with an empty map
  alerts_ptr_.store(std::make_shared<std::map<std::string, GlobalAlert>>());
}

std::string GlobalAlertManager::add_price_alert(uint32_t symbol_id, const std::string& symbol_name,
                                                double target_price, GlobalAlertType type,
                                                const std::string& name) {
  std::string alert_id = generate_alert_id();
  std::string alert_name = name.empty() ? ("Price Alert: " + symbol_name) : name;

  GlobalAlert alert(alert_id, alert_name, symbol_id, symbol_name, type, target_price);
  alert.type = type;

  // Atomically update the alerts map
  auto current_alerts = alerts_ptr_.load();
  auto new_alerts = std::make_shared<std::map<std::string, GlobalAlert>>(*current_alerts);
  (*new_alerts)[alert_id] = alert;
  alerts_ptr_.store(new_alerts);

  return alert_id;
}

std::string GlobalAlertManager::add_volume_alert(uint32_t symbol_id, const std::string& symbol_name,
                                                 double target_volume, GlobalAlertType type,
                                                 const std::string& name) {
  std::string alert_id = generate_alert_id();
  std::string alert_name = name.empty() ? ("Volume Alert: " + symbol_name) : name;

  GlobalAlert alert(alert_id, alert_name, symbol_id, symbol_name, type, target_volume);
  alert.type = type;

  // Atomically update the alerts map
  auto current_alerts = alerts_ptr_.load();
  auto new_alerts = std::make_shared<std::map<std::string, GlobalAlert>>(*current_alerts);
  (*new_alerts)[alert_id] = alert;
  alerts_ptr_.store(new_alerts);

  return alert_id;
}

std::string GlobalAlertManager::add_custom_alert(uint32_t symbol_id, const std::string& symbol_name,
                                                 const std::string& expression,
                                                 const std::string& name) {
  std::string alert_id = generate_alert_id();
  std::string alert_name = name.empty() ? ("Custom Alert: " + symbol_name) : name;

  GlobalAlert alert(alert_id, alert_name, symbol_id, symbol_name,
                    GlobalAlertType::CUSTOM_EXPRESSION, 0.0);
  alert.type = GlobalAlertType::CUSTOM_EXPRESSION;

  // Atomically update the alerts map
  auto current_alerts = alerts_ptr_.load();
  auto new_alerts = std::make_shared<std::map<std::string, GlobalAlert>>(*current_alerts);
  (*new_alerts)[alert_id] = alert;
  alerts_ptr_.store(new_alerts);

  return alert_id;
}

bool GlobalAlertManager::remove_alert(const std::string& alert_id) {
  auto current_alerts = alerts_ptr_.load();
  auto new_alerts = std::make_shared<std::map<std::string, GlobalAlert>>(*current_alerts);
  
  auto it = new_alerts->find(alert_id);
  if (it != new_alerts->end()) {
    new_alerts->erase(it);
    alerts_ptr_.store(new_alerts);
    return true;
  }
  return false;
}

bool GlobalAlertManager::remove_alerts_for_symbol(uint32_t symbol_id) {
  auto current_alerts = alerts_ptr_.load();
  auto new_alerts = std::make_shared<std::map<std::string, GlobalAlert>>(*current_alerts);
  
  size_t initial_size = new_alerts->size();

  for (auto it = new_alerts->begin(); it != new_alerts->end();) {
    if (it->second.symbol_id == symbol_id) {
      it = new_alerts->erase(it);
    } else {
      ++it;
    }
  }

  bool changed = initial_size != new_alerts->size();
  if (changed) {
    alerts_ptr_.store(new_alerts);
  }
  
  return changed;
}

bool GlobalAlertManager::enable_alert(const std::string& alert_id, bool enable) {
  auto current_alerts = alerts_ptr_.load();
  auto new_alerts = std::make_shared<std::map<std::string, GlobalAlert>>(*current_alerts);
  
  auto it = new_alerts->find(alert_id);
  if (it != new_alerts->end()) {
    it->second.status = enable ? AlertStatus::ACTIVE : AlertStatus::DISABLED;
    alerts_ptr_.store(new_alerts);
    return true;
  }
  return false;
}

void GlobalAlertManager::update_alerts() {
  if (!bridge_) {
    return;
  }

  // Get all active symbols from the bridge
  auto active_symbols = bridge_->getActiveSymbols();
  
  // Load current alerts
  auto current_alerts = alerts_ptr_.load();

  for (auto symbol_id : active_symbols) {
    auto [current_price, current_volume] = get_current_market_data(symbol_id);

    // Check all alerts for this symbol
    for (const auto& [alert_id, alert] : *current_alerts) {
      if (alert.symbol_id == symbol_id && alert.status == AlertStatus::ACTIVE) {
        if (should_trigger_alert(alert, current_price, current_volume)) {
          trigger_alert(alert, alert.type == GlobalAlertType::PRICE_ABOVE ||
                                       alert.type == GlobalAlertType::PRICE_BELOW
                                   ? current_price
                                   : current_volume);
        }
      }
    }
  }
}

std::vector<GlobalAlert> GlobalAlertManager::get_alerts_for_symbol(uint32_t symbol_id) const {
  auto current_alerts = alerts_ptr_.load();
  std::vector<GlobalAlert> result;

  for (const auto& [alert_id, alert] : *current_alerts) {
    if (alert.symbol_id == symbol_id) {
      result.push_back(alert);
    }
  }

  return result;
}

bool GlobalAlertManager::should_trigger_alert(const GlobalAlert& alert, double current_price,
                                              double current_volume) const {
  switch (alert.type) {
    case GlobalAlertType::PRICE_ABOVE:
      return current_price > alert.threshold_value;
    case GlobalAlertType::PRICE_BELOW:
      return current_price < alert.threshold_value;
    case GlobalAlertType::VOLUME_ABOVE:
      return current_volume > alert.threshold_value;
    case GlobalAlertType::VOLUME_BELOW:
      return current_volume < alert.threshold_value;
    case GlobalAlertType::CUSTOM_EXPRESSION:
      // For now, we'll skip custom expressions - this would require a more complex expression
      // parser
      return false;
    default:
      return false;
  }
}

void GlobalAlertManager::trigger_alert(const GlobalAlert& alert, double current_value) {
  // Update the alert's status and trigger time
  auto current_alerts = alerts_ptr_.load();
  auto new_alerts = std::make_shared<std::map<std::string, GlobalAlert>>(*current_alerts);
  
  auto it = new_alerts->find(alert.id);
  if (it != new_alerts->end()) {
    it->second.current_value = current_value;
    it->second.triggered_at = std::chrono::system_clock::now();
    it->second.status = AlertStatus::TRIGGERED;
    alerts_ptr_.store(new_alerts);
  }

  // Call the callback if set
  if (on_alert_triggered_) {
    on_alert_triggered_(alert, current_value);
  }

  // Log the alert to the alerts panel if available
  if (alerts_panel_) {
    AlertLog log;
    log.time = std::chrono::system_clock::now();
    log.rule_name = alert.name;
    log.symbol = alert.symbol_name;
    log.price = current_value;
    log.message = "Alert triggered: " + alert.name + " for " + alert.symbol_name;

    alerts_panel_->add_alert_log(log);
  }
}

std::string GlobalAlertManager::generate_alert_id() const {
  // Generate a unique ID using timestamp and random numbers
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  auto seed = duration.count();

  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(1000, 9999);

  std::stringstream ss;
  ss << "ALERT_" << std::hex << seed << "_" << std::dec << dis(gen);

  return ss.str();
}

std::pair<double, double> GlobalAlertManager::get_current_market_data(uint32_t symbol_id) const {
  // Try to get market data from processor if available
  if (processor_) {
    auto analytics = processor_->getSymbolAnalytics(symbol_id);
    return {analytics.last_trade_price, analytics.volume_1m};
  }

  // Fallback: no data available
  return {0.0, 0.0};
}

size_t GlobalAlertManager::get_active_alerts_count() const {
  auto current_alerts = alerts_ptr_.load();
  size_t count = 0;
  for (const auto& [id, alert] : *current_alerts) {
    if (alert.status == AlertStatus::ACTIVE) {
      count++;
    }
  }
  return count;
}

size_t GlobalAlertManager::get_triggered_alerts_count() const {
  auto current_alerts = alerts_ptr_.load();
  size_t count = 0;
  for (const auto& [id, alert] : *current_alerts) {
    if (alert.status == AlertStatus::TRIGGERED) {
      count++;
    }
  }
  return count;
}

}  // namespace BTQuant