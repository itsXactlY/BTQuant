#include "trading/trading_interface.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

namespace BTQuant {

// ============================================================================
// TradingInterface Implementation
// ============================================================================

TradingInterface::TradingInterface()
    : order_manager_(std::make_unique<OrderManager>()),
      position_manager_(std::make_unique<PositionManager>()),
      risk_assessment_(std::make_unique<RiskAssessment>()) {
  // Set up internal callbacks
  order_manager_->set_order_update_callback(
      [this](const OrderManager::Order& order) { on_order_update(order); });

  order_manager_->set_execution_callback(
      [this](const OrderManager::OrderExecution& execution) { on_execution(execution); });

  position_manager_->set_position_update_callback(
      [this](const PositionManager::Position& position) { on_position_update(position); });

  // Set default risk limits
  RiskAssessment::RiskLimits default_limits;
  default_limits.max_position_size = 100000.0;
  default_limits.max_portfolio_value = 1000000.0;
  default_limits.max_daily_loss = 5000.0;
  default_limits.max_drawdown = 0.20;
  default_limits.max_leverage = 5.0;
  default_limits.max_concentration = 0.30;
  default_limits.var_limit = 0.05;
  risk_assessment_->set_risk_limits(default_limits);
}

TradingInterface::~TradingInterface() = default;

// Order Management
std::string TradingInterface::place_order(const OrderManager::Order& order) {
  // Validate order against risk limits
  if (!validate_order(order)) {
    std::cerr << "Order rejected by risk assessment" << std::endl;
    return "";
  }

  // Place order through order manager
  return order_manager_->place_order(order);
}

bool TradingInterface::modify_order(const std::string& order_id, double new_quantity,
                                    double new_price) {
  return order_manager_->modify_order(order_id, new_quantity, new_price);
}

bool TradingInterface::cancel_order(const std::string& order_id) {
  return order_manager_->cancel_order(order_id);
}

std::vector<OrderManager::Order> TradingInterface::get_active_orders() const {
  return order_manager_->get_active_orders();
}

std::vector<OrderManager::Order> TradingInterface::get_orders(const std::string& symbol) const {
  return order_manager_->get_orders(symbol);
}

// Position Management
std::vector<PositionManager::Position> TradingInterface::get_positions() const {
  return position_manager_->get_positions();
}

PositionManager::Position TradingInterface::get_position(const std::string& symbol) const {
  return position_manager_->get_position(symbol);
}

PositionManager::PortfolioSummary TradingInterface::get_portfolio_summary() const {
  return position_manager_->get_portfolio_summary();
}

// Risk Management
RiskAssessment::RiskMetrics TradingInterface::get_risk_metrics() const {
  return risk_assessment_->get_risk_metrics();
}

RiskAssessment::RiskReport TradingInterface::get_risk_report() const {
  auto summary = position_manager_->get_portfolio_summary();
  auto positions = position_manager_->get_positions();
  return risk_assessment_->generate_risk_report(summary, positions);
}

bool TradingInterface::validate_order(const OrderManager::Order& order) {
  // Get current position
  auto position = position_manager_->get_position(order.symbol);

  // Get current risk metrics
  auto metrics = risk_assessment_->get_risk_metrics();

  // Validate order
  return risk_assessment_->validate_order_risk(order, position_manager_->get_portfolio_summary(),
                                               position_manager_->get_positions());
}

void TradingInterface::set_risk_limits(const RiskAssessment::RiskLimits& limits) {
  risk_assessment_->set_risk_limits(limits);
}

// Callback Registration
void TradingInterface::set_order_update_callback(OrderUpdateCallback callback) {
  user_order_callback_ = std::move(callback);
}

void TradingInterface::set_execution_callback(ExecutionCallback callback) {
  user_execution_callback_ = std::move(callback);
}

void TradingInterface::set_position_update_callback(PositionUpdateCallback callback) {
  user_position_callback_ = std::move(callback);
}

void TradingInterface::set_risk_alert_callback(RiskAlertCallback callback) {
  user_risk_alert_callback_ = std::move(callback);
}

// State Queries
TradingInterface::TradingState TradingInterface::get_trading_state(
    const std::string& symbol) const {
  TradingState state;

  state.order_status = OrderManager::OrderStatus::Pending;
  state.position = position_manager_->get_position(symbol);
  state.risk_metrics = risk_assessment_->get_risk_metrics();
  state.is_risk_compliant = is_risk_compliant();
  state.available_margin = get_available_margin();
  state.buying_power = get_buying_power();

  return state;
}

bool TradingInterface::is_risk_compliant() const {
  auto metrics = risk_assessment_->get_risk_metrics();
  return risk_assessment_->is_risk_compliant(metrics);
}

double TradingInterface::get_available_margin() const {
  auto summary = position_manager_->get_portfolio_summary();
  auto limits = risk_assessment_->get_risk_limits();
  return limits.max_portfolio_value - summary.total_value;
}

double TradingInterface::get_buying_power() const {
  auto summary = position_manager_->get_portfolio_summary();
  auto limits = risk_assessment_->get_risk_limits();
  return (limits.max_portfolio_value - summary.total_value) * limits.max_leverage;
}

// Quick Actions
std::string TradingInterface::place_market_order(const std::string& symbol,
                                                 OrderManager::OrderSide side, double quantity) {
  OrderManager::Order order;
  order.order_id = generate_order_id();
  order.symbol = symbol;
  order.side = side;
  order.type = OrderManager::OrderType::Market;
  order.quantity = quantity;
  order.price = 0.0;  // Market order
  order.status = OrderManager::OrderStatus::Pending;
  auto now = std::chrono::system_clock::now();
  order.created_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  order.updated_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  return place_order(order);
}

std::string TradingInterface::place_limit_order(const std::string& symbol,
                                                OrderManager::OrderSide side, double quantity,
                                                double price) {
  OrderManager::Order order;
  order.order_id = generate_order_id();
  order.symbol = symbol;
  order.side = side;
  order.type = OrderManager::OrderType::Limit;
  order.quantity = quantity;
  order.price = price;
  order.status = OrderManager::OrderStatus::Pending;
  auto now = std::chrono::system_clock::now();
  order.created_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  order.updated_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  return place_order(order);
}

std::string TradingInterface::place_stop_order(const std::string& symbol,
                                               OrderManager::OrderSide side, double quantity,
                                               double stop_price) {
  OrderManager::Order order;
  order.order_id = generate_order_id();
  order.symbol = symbol;
  order.side = side;
  order.type = OrderManager::OrderType::Stop;
  order.quantity = quantity;
  order.price = stop_price;
  order.status = OrderManager::OrderStatus::Pending;
  auto now = std::chrono::system_clock::now();
  order.created_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  order.updated_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  return place_order(order);
}

bool TradingInterface::close_position(const std::string& symbol) {
  auto position = position_manager_->get_position(symbol);

  if (std::abs(position.quantity) < 0.0001) {
    return true;  // Already flat
  }

  // Place opposite order to close position
  OrderManager::OrderSide side =
      position.quantity > 0 ? OrderManager::OrderSide::Sell : OrderManager::OrderSide::Buy;

  auto order_id = place_market_order(symbol, side, std::abs(position.quantity));
  return !order_id.empty();
}

bool TradingInterface::close_all_positions() {
  auto positions = position_manager_->get_positions();
  bool all_closed = true;

  for (const auto& position : positions) {
    if (!close_position(position.symbol)) {
      all_closed = false;
    }
  }

  return all_closed;
}

// Market Data Integration
void TradingInterface::update_market_price(const std::string& symbol, double price) {
  position_manager_->update_market_price(symbol, price);
  monitor_risk();
}

void TradingInterface::update_market_prices(const std::unordered_map<std::string, double>& prices) {
  position_manager_->update_market_prices(prices);
  monitor_risk();
}

// Private Methods
void TradingInterface::on_order_update(const OrderManager::Order& order) {
  // Forward to user callback if set
  if (user_order_callback_) {
    user_order_callback_(order);
  }
}

void TradingInterface::on_execution(const OrderManager::OrderExecution& execution) {
  // Update position
  position_manager_->update_position(execution);

  // Forward to user callback if set
  if (user_execution_callback_) {
    user_execution_callback_(execution);
  }

  // Monitor risk after execution
  monitor_risk();
}

void TradingInterface::on_position_update(const PositionManager::Position& position) {
  // Forward to user callback if set
  if (user_position_callback_) {
    user_position_callback_(position);
  }
}

void TradingInterface::monitor_risk() {
  auto metrics = risk_assessment_->get_risk_metrics();
  check_risk_alerts(metrics);
}

void TradingInterface::check_risk_alerts(const RiskAssessment::RiskMetrics& metrics) {
  if (!user_risk_alert_callback_) {
    return;
  }

  // Check for various risk conditions
  if (metrics.daily_loss > risk_assessment_->get_risk_limits().max_daily_loss * 0.8) {
    RiskAssessment::RiskAlert alert;
    alert.severity = RiskAssessment::RiskAlert::Severity::Warning;
    alert.message = "Approaching daily loss limit";
    alert.metric_name = "daily_loss";
    alert.current_value = metrics.daily_loss;
    alert.threshold_value = risk_assessment_->get_risk_limits().max_daily_loss;
    user_risk_alert_callback_(alert);
  }

  if (metrics.drawdown > risk_assessment_->get_risk_limits().max_drawdown * 0.9) {
    RiskAssessment::RiskAlert alert;
    alert.severity = RiskAssessment::RiskAlert::Severity::Critical;
    alert.message = "Critical drawdown level reached";
    alert.metric_name = "drawdown";
    alert.current_value = metrics.drawdown;
    alert.threshold_value = risk_assessment_->get_risk_limits().max_drawdown;
    user_risk_alert_callback_(alert);
  }

  if (metrics.leverage > risk_assessment_->get_risk_limits().max_leverage) {
    RiskAssessment::RiskAlert alert;
    alert.severity = RiskAssessment::RiskAlert::Severity::Critical;
    alert.message = "Leverage limit exceeded";
    alert.metric_name = "leverage";
    alert.current_value = metrics.leverage;
    alert.threshold_value = risk_assessment_->get_risk_limits().max_leverage;
    user_risk_alert_callback_(alert);
  }
}

// Helper function to generate order IDs
std::string TradingInterface::generate_order_id() {
  static std::atomic<uint64_t> counter{0};
  auto now = std::chrono::system_clock::now();
  auto timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  return "ORD-" + std::to_string(timestamp) + "-" + std::to_string(counter.fetch_add(1));
}

}  // namespace BTQuant
