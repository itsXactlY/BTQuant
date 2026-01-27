#pragma once

#include "order_manager.hpp"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace BTQuant {

class PositionManager {
public:
  struct Position {
    std::string symbol;
    double quantity = 0;
    double average_price = 0;
    double unrealized_pnl = 0;
    double realized_pnl = 0;
    double market_value = 0;
    double cost_basis = 0;
    uint64_t first_trade_time = 0;
    uint64_t last_trade_time = 0;
    double max_drawdown = 0;
    double max_profit = 0;
    double var_95 = 0;
    double beta = 1.0;
    double sharpe_ratio = 0;
    std::vector<std::string> contributing_orders;
    double total_commission = 0;
    int trade_count = 0;
  };

  struct PortfolioSummary {
    double total_value = 0;
    double total_unrealized_pnl = 0;
    double total_realized_pnl = 0;
    double total_commission = 0;
    double cash_balance = 0;
    double buying_power = 0;
    double margin_used = 0;
    double portfolio_beta = 0;
    double portfolio_var = 0;
    double sharpe_ratio = 0;
    int position_count = 0;
    int trade_count = 0;
  };

  PositionManager();
  void update_position(const OrderManager::OrderExecution &execution);
  void update_market_price(const std::string &symbol, double price);
  void
  update_market_prices(const std::unordered_map<std::string, double> &prices);
  std::vector<Position> get_positions() const;
  std::vector<Position> get_all_positions() const;
  Position get_position(const std::string &symbol) const;
  PortfolioSummary get_portfolio_summary() const;
  void set_cash_balance(double balance);

  using PositionUpdateCallback = std::function<void(const Position &)>;
  void set_position_update_callback(PositionUpdateCallback callback);

private:
  std::unordered_map<std::string, Position> positions_;
  std::unordered_map<std::string, double> market_prices_;
  std::unordered_map<std::string, std::string> order_symbols_;
  PositionUpdateCallback position_update_callback_;
  double cash_balance_ = 100000.0;

  std::string get_symbol_from_order(const std::string &order_id);
  bool is_buy_execution(const OrderManager::OrderExecution &execution);
  void update_market_values();
  void calculate_risk_metrics(Position &position);
  double calculate_buying_power() const;
  double calculate_margin_used() const;
  double calculate_portfolio_beta() const;
  double calculate_portfolio_var() const;
  double calculate_portfolio_sharpe() const;
  void notify_position_update(const Position &position);
};

} // namespace BTQuant
