#pragma once

#include <imgui.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "order_manager.hpp"
#include "position_manager.hpp"
#include "risk_assessment.hpp"

namespace BTQuant {

// ============================================================================
// Trading Interface - Unified API for Trading Terminal
// ============================================================================

class TradingInterface {
 public:
  // Callback types for terminal integration
  using OrderUpdateCallback = std::function<void(const OrderManager::Order&)>;
  using ExecutionCallback = std::function<void(const OrderManager::OrderExecution&)>;
  using PositionUpdateCallback = std::function<void(const PositionManager::Position&)>;
  using RiskAlertCallback = std::function<void(const RiskAssessment::RiskAlert&)>;

  struct TradingState {
    OrderManager::OrderStatus order_status;
    PositionManager::Position position;
    RiskAssessment::RiskMetrics risk_metrics;
    bool is_risk_compliant;
    double available_margin;
    double buying_power;
  };

  TradingInterface();
  ~TradingInterface();

  // Order Management
  std::string place_order(const OrderManager::Order& order);
  bool modify_order(const std::string& order_id, double new_quantity, double new_price);
  bool cancel_order(const std::string& order_id);
  std::vector<OrderManager::Order> get_active_orders() const;
  std::vector<OrderManager::Order> get_orders(const std::string& symbol) const;

  // Position Management
  std::vector<PositionManager::Position> get_positions() const;
  PositionManager::Position get_position(const std::string& symbol) const;
  PositionManager::PortfolioSummary get_portfolio_summary() const;

  // Risk Management
  RiskAssessment::RiskMetrics get_risk_metrics() const;
  RiskAssessment::RiskReport get_risk_report() const;
  bool validate_order(const OrderManager::Order& order);
  void set_risk_limits(const RiskAssessment::RiskLimits& limits);

  // Callback Registration
  void set_order_update_callback(OrderUpdateCallback callback);
  void set_execution_callback(ExecutionCallback callback);
  void set_position_update_callback(PositionUpdateCallback callback);
  void set_risk_alert_callback(RiskAlertCallback callback);

  // State Queries
  TradingState get_trading_state(const std::string& symbol) const;
  bool is_risk_compliant() const;
  double get_available_margin() const;
  double get_buying_power() const;

  // Quick Actions
  std::string place_market_order(const std::string& symbol, OrderManager::OrderSide side,
                                 double quantity);
  std::string place_limit_order(const std::string& symbol, OrderManager::OrderSide side,
                                double quantity, double price);
  std::string place_stop_order(const std::string& symbol, OrderManager::OrderSide side,
                               double quantity, double stop_price);
  bool close_position(const std::string& symbol);
  bool close_all_positions();

  // Market Data Integration
  void update_market_price(const std::string& symbol, double price);
  void update_market_prices(const std::unordered_map<std::string, double>& prices);

 private:
  std::unique_ptr<OrderManager> order_manager_;
  std::unique_ptr<PositionManager> position_manager_;
  std::unique_ptr<RiskAssessment> risk_assessment_;

  // Internal callbacks
  void on_order_update(const OrderManager::Order& order);
  void on_execution(const OrderManager::OrderExecution& execution);
  void on_position_update(const PositionManager::Position& position);

  // User callbacks
  OrderUpdateCallback user_order_callback_;
  ExecutionCallback user_execution_callback_;
  PositionUpdateCallback user_position_callback_;
  RiskAlertCallback user_risk_alert_callback_;

  // Risk monitoring
  void monitor_risk();
  void check_risk_alerts(const RiskAssessment::RiskMetrics& metrics);

  // Order ID generation
  std::string generate_order_id();
};

// ============================================================================
// Trading Terminal Integration Helpers
// ============================================================================

namespace TerminalHelpers {

// Format order status for display
inline std::string format_order_status(OrderManager::OrderStatus status) {
  switch (status) {
    case OrderManager::OrderStatus::Pending:
      return "Pending";
    case OrderManager::OrderStatus::PartiallyFilled:
      return "Partially Filled";
    case OrderManager::OrderStatus::Filled:
      return "Filled";
    case OrderManager::OrderStatus::Cancelled:
      return "Cancelled";
    case OrderManager::OrderStatus::Rejected:
      return "Rejected";
    case OrderManager::OrderStatus::Expired:
      return "Expired";
    default:
      return "Unknown";
  }
}

// Format order side for display
inline std::string format_order_side(OrderManager::OrderSide side) {
  return side == OrderManager::OrderSide::Buy ? "BUY" : "SELL";
}

// Format order type for display
inline std::string format_order_type(OrderManager::OrderType type) {
  switch (type) {
    case OrderManager::OrderType::Market:
      return "Market";
    case OrderManager::OrderType::Limit:
      return "Limit";
    case OrderManager::OrderType::Stop:
      return "Stop";
    case OrderManager::OrderType::StopLimit:
      return "Stop Limit";
    case OrderManager::OrderType::TrailingStop:
      return "Trailing Stop";
    case OrderManager::OrderType::Iceberg:
      return "Iceberg";
    case OrderManager::OrderType::TWAP:
      return "TWAP";
    case OrderManager::OrderType::VWAP:
      return "VWAP";
    default:
      return "Unknown";
  }
}

// Format risk alert severity for display
inline std::string format_risk_severity(RiskAssessment::RiskAlert::Severity severity) {
  switch (severity) {
    case RiskAssessment::RiskAlert::Severity::Info:
      return "INFO";
    case RiskAssessment::RiskAlert::Severity::Warning:
      return "WARNING";
    case RiskAssessment::RiskAlert::Severity::Critical:
      return "CRITICAL";
    default:
      return "UNKNOWN";
  }
}

// Get color for order side (for UI rendering)
inline ImVec4 get_order_side_color(OrderManager::OrderSide side) {
  return side == OrderManager::OrderSide::Buy ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)
                                              : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
}

// Get color for risk severity (for UI rendering)
inline ImVec4 get_risk_severity_color(RiskAssessment::RiskAlert::Severity severity) {
  switch (severity) {
    case RiskAssessment::RiskAlert::Severity::Info:
      return ImVec4(0.2f, 0.6f, 1.0f, 1.0f);
    case RiskAssessment::RiskAlert::Severity::Warning:
      return ImVec4(1.0f, 0.8f, 0.2f, 1.0f);
    case RiskAssessment::RiskAlert::Severity::Critical:
      return ImVec4(1.0f, 0.2f, 0.2f, 1.0f);
    default:
      return ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
  }
}

// Calculate position PnL percentage
inline double calculate_pnl_percentage(const PositionManager::Position& position) {
  if (position.cost_basis == 0) return 0.0;
  return (position.unrealized_pnl / position.cost_basis) * 100.0;
}

// Check if position is profitable
inline bool is_position_profitable(const PositionManager::Position& position) {
  return position.unrealized_pnl > 0;
}

// Get position direction string
inline std::string get_position_direction(const PositionManager::Position& position) {
  if (position.quantity > 0) return "LONG";
  if (position.quantity < 0) return "SHORT";
  return "FLAT";
}

}  // namespace TerminalHelpers

}  // namespace BTQuant
