#include "trading/position_manager.hpp"
#include "trading/order_manager.hpp"
#include <algorithm>
#include <cmath>

namespace BTQuant {

PositionManager::PositionManager() : cash_balance_(100000.0) {}

void PositionManager::update_position(const OrderManager::OrderExecution &execution) {
  std::string symbol = get_symbol_from_order(execution.order_id);
  if (symbol.empty()) {
    return;
  }

  Position &position = positions_[symbol];
  bool is_buy = is_buy_execution(execution);

  if (position.quantity == 0) {
    // New position
    position.symbol = symbol;
    position.quantity = is_buy ? execution.quantity : -execution.quantity;
    position.average_price = execution.price;
    position.first_trade_time = execution.timestamp;
    position.trade_count = 1;
  } else {
    // Update existing position
    double total_cost = position.quantity * position.average_price;
    double new_cost = (is_buy ? execution.quantity : -execution.quantity) * execution.price;
    double new_quantity = position.quantity + (is_buy ? execution.quantity : -execution.quantity);

    if (new_quantity != 0) {
      position.average_price = (total_cost + new_cost) / new_quantity;
    }
    position.quantity = new_quantity;
    position.trade_count++;
  }

  position.last_trade_time = execution.timestamp;
  position.total_commission += execution.commission;

  // Update market values
  auto price_it = market_prices_.find(symbol);
  if (price_it != market_prices_.end()) {
    double current_price = price_it->second;
    position.market_value = std::abs(position.quantity) * current_price;
    position.cost_basis = std::abs(position.quantity) * position.average_price;

    if (position.quantity > 0) {
      position.unrealized_pnl = position.market_value - position.cost_basis;
    } else {
      position.unrealized_pnl = position.cost_basis - position.market_value;
    }
  }

  // Calculate risk metrics
  calculate_risk_metrics(position);

  // Notify update
  notify_position_update(position);
}

void PositionManager::update_market_price(const std::string &symbol, double price) {
  market_prices_[symbol] = price;
  update_market_values();
}

void PositionManager::update_market_prices(const std::unordered_map<std::string, double> &prices) {
  for (const auto &pair : prices) {
    market_prices_[pair.first] = pair.second;
  }
  update_market_values();
}

std::vector<PositionManager::Position> PositionManager::get_positions() const {
  std::vector<Position> result;

  for (const auto &pair : positions_) {
    if (pair.second.quantity != 0) {
      result.push_back(pair.second);
    }
  }

  return result;
}

std::vector<PositionManager::Position> PositionManager::get_all_positions() const {
  std::vector<Position> result;

  for (const auto &pair : positions_) {
    result.push_back(pair.second);
  }

  return result;
}

PositionManager::Position PositionManager::get_position(const std::string &symbol) const {
  auto it = positions_.find(symbol);
  if (it != positions_.end()) {
    return it->second;
  }
  return Position();
}

PositionManager::PortfolioSummary PositionManager::get_portfolio_summary() const {
  PortfolioSummary summary;

  summary.cash_balance = cash_balance_;
  summary.position_count = 0;
  summary.trade_count = 0;

  for (const auto &pair : positions_) {
    const Position &pos = pair.second;

    if (pos.quantity != 0) {
      summary.total_value += pos.market_value;
      summary.total_unrealized_pnl += pos.unrealized_pnl;
      summary.total_realized_pnl += pos.realized_pnl;
      summary.total_commission += pos.total_commission;
      summary.position_count++;
      summary.trade_count += pos.trade_count;
    }
  }

  summary.portfolio_beta = calculate_portfolio_beta();
  summary.portfolio_var = calculate_portfolio_var();
  summary.sharpe_ratio = calculate_portfolio_sharpe();
  summary.buying_power = calculate_buying_power();
  summary.margin_used = calculate_margin_used();

  return summary;
}

void PositionManager::set_cash_balance(double balance) {
  cash_balance_ = balance;
}

void PositionManager::set_position_update_callback(PositionUpdateCallback callback) {
  position_update_callback_ = std::move(callback);
}

std::string PositionManager::get_symbol_from_order(const std::string &order_id) {
  auto it = order_symbols_.find(order_id);
  if (it != order_symbols_.end()) {
    return it->second;
  }
  return "";
}

bool PositionManager::is_buy_execution(const OrderManager::OrderExecution &execution) {
  // For simulation, assume buy by default if we can't determine
  // In a real implementation, this would be stored with the execution
  (void)execution; // Suppress unused parameter warning
  return true;
}

void PositionManager::update_market_values() {
  for (auto &pair : positions_) {
    Position &position = pair.second;
    auto price_it = market_prices_.find(position.symbol);

    if (price_it != market_prices_.end()) {
      double current_price = price_it->second;
      position.market_value = std::abs(position.quantity) * current_price;
      position.cost_basis = std::abs(position.quantity) * position.average_price;

      if (position.quantity > 0) {
        position.unrealized_pnl = position.market_value - position.cost_basis;
      } else {
        position.unrealized_pnl = position.cost_basis - position.market_value;
      }

      calculate_risk_metrics(position);
    }
  }
}

void PositionManager::calculate_risk_metrics(Position &position) {
  if (position.quantity == 0) {
    return;
  }

  // Calculate max drawdown and max profit
  if (position.unrealized_pnl > position.max_profit) {
    position.max_profit = position.unrealized_pnl;
  }
  if (position.unrealized_pnl < position.max_drawdown) {
    position.max_drawdown = position.unrealized_pnl;
  }

  // Calculate VaR (simplified)
  position.var_95 = position.market_value * 0.02; // 2% daily VaR

  // Calculate Sharpe ratio (simplified)
  if (position.market_value > 0) {
    position.sharpe_ratio = position.unrealized_pnl / position.market_value * 252; // Annualized
  }

  // Beta relative to market (simplified - assume market beta is 1.0)
  position.beta = 1.0;
}

double PositionManager::calculate_buying_power() const {
  // Standard buying power calculation: 2x available cash for day trading
  double available = cash_balance_ * 2.0;

  // Subtract margin used
  double margin_used = calculate_margin_used();

  return std::max(0.0, available - margin_used);
}

double PositionManager::calculate_margin_used() const {
  double margin = 0.0;

  for (const auto &pair : positions_) {
    const Position &pos = pair.second;
    if (pos.quantity < 0) {
      // Short position requires margin
      margin += pos.market_value * 0.5; // 50% initial margin
    }
  }

  return margin;
}

double PositionManager::calculate_portfolio_beta() const {
  if (positions_.empty()) {
    return 1.0;
  }

  double weighted_beta = 0.0;
  double total_value = 0.0;

  for (const auto &pair : positions_) {
    const Position &pos = pair.second;
    if (pos.quantity != 0) {
      weighted_beta += pos.beta * std::abs(pos.market_value);
      total_value += std::abs(pos.market_value);
    }
  }

  return total_value > 0 ? weighted_beta / total_value : 1.0;
}

double PositionManager::calculate_portfolio_var() const {
  if (positions_.empty()) {
    return 0.0;
  }

  double total_var = 0.0;

  for (const auto &pair : positions_) {
    const Position &pos = pair.second;
    if (pos.quantity != 0) {
      total_var += pos.var_95 * pos.var_95; // Simplified VaR calculation
    }
  }

  return std::sqrt(total_var);
}

double PositionManager::calculate_portfolio_sharpe() const {
  PortfolioSummary summary = get_portfolio_summary();
  return summary.sharpe_ratio;
}

void PositionManager::notify_position_update(const Position &position) {
  if (position_update_callback_) {
    position_update_callback_(position);
  }
}

} // namespace BTQuant
