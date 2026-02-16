#include "trading/position_manager.hpp"

#include <algorithm>
#include <cmath>

#include "trading/order_manager.hpp"
#include "market_data_processor.hpp"

namespace BTQuant {

PositionManager::PositionManager() : cash_balance_(100000.0) {}

void PositionManager::setMarketDataProcessor(
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor) {
  market_data_processor_ = processor;
}

void PositionManager::updateMarkToMarketPnL() {
  // Get the MarketDataProcessor (lock-free atomic access)
  auto processor = market_data_processor_.lock();
  if (!processor) {
    // Fallback to traditional market_prices_ if processor not available
    update_market_values();
    return;
  }

  // Update each position using atomic best_bid/best_ask from MarketDataProcessor
  for (auto& pair : positions_) {
    Position& position = pair.second;
    if (position.quantity == 0) {
      continue;  // Skip closed positions
    }

    // Get atomic snapshot (lock-free, uses acquire semantics)
    // Symbol ID mapping: for now use hash of symbol string or lookup table
    // In production, this would use a proper symbol_id mapping
    uint32_t symbol_id = static_cast<uint32_t>(
        std::hash<std::string>{}(position.symbol) & 0xFFFFFFFF);

    auto snapshot_opt = processor->get_atomic_snapshot(symbol_id);
    if (!snapshot_opt.has_value()) {
      // Fallback to cached market price if atomic snapshot not available
      auto price_it = market_prices_.find(position.symbol);
      if (price_it != market_prices_.end()) {
        double current_price = price_it->second;
        position.mtm_bid_price = current_price;
        position.mtm_ask_price = current_price;
        position.mtm_mid_price = current_price;
      }
      continue;
    }

    const auto& snapshot = snapshot_opt.value();

    // Store atomic prices for zero-latency access
    position.mtm_bid_price = snapshot.best_bid;
    position.mtm_ask_price = snapshot.best_ask;
    position.mtm_mid_price = snapshot.mid_price;
    position.mtm_timestamp = snapshot.timestamp;

    // Calculate Mark-to-Market PnL using atomic prices
    // For long positions: use best_bid (exit price)
    // For short positions: use best_ask (exit price)
    double exit_price = (position.quantity > 0) ? snapshot.best_bid : snapshot.best_ask;

    position.market_value = std::abs(position.quantity) * exit_price;
    position.cost_basis = std::abs(position.quantity) * position.average_price;

    // Calculate unrealized PnL based on position side
    if (position.quantity > 0) {
      // Long position: profit when price goes up
      position.unrealized_pnl = position.market_value - position.cost_basis;
    } else {
      // Short position: profit when price goes down
      position.unrealized_pnl = position.cost_basis - position.market_value;
    }

    calculate_risk_metrics(position);
  }

  // Notify update for UI refresh
  for (const auto& pair : positions_) {
    if (pair.second.quantity != 0 && position_update_callback_) {
      position_update_callback_(pair.second);
    }
  }
}

void PositionManager::update_position(const OrderManager::OrderExecution& execution) {
  std::string symbol = get_symbol_from_order(execution.order_id);
  if (symbol.empty()) {
    return;
  }

  Position& position = positions_[symbol];
  bool is_buy = is_buy_execution(execution);

  // Determine if this execution is reducing the position size (closing)
  double execution_qty = is_buy ? execution.quantity : -execution.quantity;
  bool is_closing =
      (position.quantity > 0 && execution_qty < 0) || (position.quantity < 0 && execution_qty > 0);

  if (position.quantity == 0) {
    // New position
    position.symbol = symbol;
    position.quantity = execution_qty;
    position.average_price = execution.price;
    position.first_trade_time = execution.timestamp;
    position.trade_count = 1;
    position.contributing_orders.push_back(execution.order_id);
  } else if (is_closing) {
    // Closing or reducing position - calculate realized P&L
    double qty_to_close = std::min(std::abs(position.quantity), std::abs(execution_qty));
    double realized_pnl_per_unit = execution.price - position.average_price;
    double realized_pnl = qty_to_close * realized_pnl_per_unit * (position.quantity > 0 ? 1 : -1);

    position.realized_pnl += realized_pnl;

    // Update quantity
    position.quantity += execution_qty;  // execution_qty already has the correct sign

    // If position is fully closed, reset average price
    if (position.quantity == 0) {
      position.average_price = 0;
    } else {
      // Average price remains the same for remaining position
      // The average price doesn't change when closing part of position
      // Only the quantity changes
    }

    position.trade_count++;
    position.contributing_orders.push_back(execution.order_id);
  } else {
    // Adding to existing position
    double total_cost = std::abs(position.quantity) * position.average_price;
    double new_cost = std::abs(execution_qty) * execution.price;
    double new_quantity = position.quantity + execution_qty;

    if (new_quantity != 0) {
      position.average_price = (total_cost + new_cost) / std::abs(new_quantity);
    }
    position.quantity = new_quantity;
    position.trade_count++;
    position.contributing_orders.push_back(execution.order_id);
  }

  position.last_trade_time = execution.timestamp;
  position.total_commission += execution.commission;

  // Update market values using atomic BBO if available, otherwise cached price
  auto price_it = market_prices_.find(symbol);
  double current_price = (price_it != market_prices_.end()) ? price_it->second : execution.price;

  // Initialize mtm fields with available price (will be updated by atomic BBO on next tick)
  position.mtm_bid_price = current_price;
  position.mtm_ask_price = current_price;
  position.mtm_mid_price = current_price;
  position.mtm_timestamp = execution.timestamp;

  position.market_value = std::abs(position.quantity) * current_price;
  position.cost_basis = std::abs(position.quantity) * position.average_price;

  if (position.quantity > 0) {
    position.unrealized_pnl = position.market_value - position.cost_basis;
  } else if (position.quantity < 0) {
    position.unrealized_pnl = position.cost_basis - position.market_value;
  } else {
    position.unrealized_pnl = 0;  // No position
  }

  // Calculate risk metrics
  calculate_risk_metrics(position);

  // Notify update
  notify_position_update(position);
}

void PositionManager::update_market_price(const std::string& symbol, double price) {
  market_prices_[symbol] = price;
  update_market_values();
}

void PositionManager::update_market_prices(const std::unordered_map<std::string, double>& prices) {
  for (const auto& pair : prices) {
    market_prices_[pair.first] = pair.second;
  }
  update_market_values();
}

std::vector<PositionManager::Position> PositionManager::get_positions() const {
  std::vector<Position> result;

  for (const auto& pair : positions_) {
    if (pair.second.quantity != 0) {
      result.push_back(pair.second);
    }
  }

  return result;
}

std::vector<PositionManager::Position> PositionManager::get_all_positions() const {
  std::vector<Position> result;

  for (const auto& pair : positions_) {
    result.push_back(pair.second);
  }

  return result;
}

PositionManager::Position PositionManager::get_position(const std::string& symbol) const {
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

  for (const auto& pair : positions_) {
    const Position& pos = pair.second;

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

void PositionManager::set_cash_balance(double balance) { cash_balance_ = balance; }

void PositionManager::set_position_update_callback(PositionUpdateCallback callback) {
  position_update_callback_ = std::move(callback);
}

std::string PositionManager::get_symbol_from_order(const std::string& order_id) {
  auto it = order_symbols_.find(order_id);
  if (it != order_symbols_.end()) {
    return it->second;
  }
  return "";
}

bool PositionManager::is_buy_execution(const OrderManager::OrderExecution& execution) {
  // For simulation, assume buy by default if we can't determine
  // In a real implementation, this would be stored with the execution
  (void)execution;  // Suppress unused parameter warning
  return true;
}

void PositionManager::update_market_values() {
  // Try to use atomic BBO from MarketDataProcessor if available
  auto processor = market_data_processor_.lock();
  if (processor) {
    // Use atomic Mark-to-Market update
    updateMarkToMarketPnL();
    return;
  }

  // Fallback to traditional market_prices_ based update
  for (auto& pair : positions_) {
    Position& position = pair.second;
    auto price_it = market_prices_.find(position.symbol);

    if (price_it != market_prices_.end()) {
      double current_price = price_it->second;
      position.market_value = std::abs(position.quantity) * current_price;
      position.cost_basis = std::abs(position.quantity) * position.average_price;
      position.mtm_bid_price = current_price;
      position.mtm_ask_price = current_price;
      position.mtm_mid_price = current_price;

      if (position.quantity > 0) {
        position.unrealized_pnl = position.market_value - position.cost_basis;
      } else {
        position.unrealized_pnl = position.cost_basis - position.market_value;
      }

      calculate_risk_metrics(position);
    }
  }
}

void PositionManager::calculate_risk_metrics(Position& position) {
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
  position.var_95 = position.market_value * 0.02;  // 2% daily VaR

  // Calculate Sharpe ratio (simplified)
  if (position.market_value > 0) {
    position.sharpe_ratio = position.unrealized_pnl / position.market_value * 252;  // Annualized
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

  for (const auto& pair : positions_) {
    const Position& pos = pair.second;
    if (pos.quantity < 0) {
      // Short position requires margin
      margin += pos.market_value * 0.5;  // 50% initial margin
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

  for (const auto& pair : positions_) {
    const Position& pos = pair.second;
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

  for (const auto& pair : positions_) {
    const Position& pos = pair.second;
    if (pos.quantity != 0) {
      total_var += pos.var_95 * pos.var_95;  // Simplified VaR calculation
    }
  }

  return std::sqrt(total_var);
}

double PositionManager::calculate_portfolio_sharpe() const {
  PortfolioSummary summary = get_portfolio_summary();
  return summary.sharpe_ratio;
}

void PositionManager::notify_position_update(const Position& position) {
  if (position_update_callback_) {
    position_update_callback_(position);
  }
}

PositionManager::TradeRecord PositionManager::create_trade_record(
    const OrderManager::OrderExecution& execution, double realized_pnl) {
  TradeRecord record;
  record.trade_id = execution.execution_id;
  record.symbol = get_symbol_from_order(execution.order_id);
  record.order_id = execution.order_id;
  record.quantity = execution.quantity;
  record.price = execution.price;
  record.commission = execution.commission;
  record.pnl = realized_pnl;  // Use the calculated P&L
  record.timestamp = execution.timestamp;
  record.is_buy = is_buy_execution(execution);

  return record;
}

}  // namespace BTQuant
