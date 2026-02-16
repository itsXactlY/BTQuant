#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "order_manager.hpp"

// Forward declaration to avoid circular dependency
namespace BTQuant {
namespace RenderEngine {
class MarketDataProcessor;
}
}  // namespace BTQuant

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
    // Zero-latency Mark-to-Market fields (atomic best_bid/best_ask)
    double mtm_bid_price = 0.0;   // Atomic best bid for PnL calc
    double mtm_ask_price = 0.0;   // Atomic best ask for PnL calc
    double mtm_mid_price = 0.0;   // Mid price for reference
    uint64_t mtm_timestamp = 0;   // Timestamp of last atomic update
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
  ~PositionManager() = default;

  /**
   * Set the MarketDataProcessor for zero-latency Mark-to-Market PnL calculation
   * Uses atomic best_bid/best_ask pointers for lock-free access
   */
  void setMarketDataProcessor(std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  /**
   * Update Mark-to-Market PnL using atomic best_bid/best_ask from MarketDataProcessor
   * This provides zero-latency PnL calculation without locks
   */
  void updateMarkToMarketPnL();

  void update_position(const OrderManager::OrderExecution& execution);
  void update_market_price(const std::string& symbol, double price);
  void update_market_prices(const std::unordered_map<std::string, double>& prices);
  std::vector<Position> get_positions() const;
  std::vector<Position> get_all_positions() const;
  Position get_position(const std::string& symbol) const;
  PortfolioSummary get_portfolio_summary() const;
  void set_cash_balance(double balance);

  using PositionUpdateCallback = std::function<void(const Position&)>;
  void set_position_update_callback(PositionUpdateCallback callback);

  // Analytics integration
  struct TradeRecord {
    std::string trade_id;
    std::string symbol;
    std::string order_id;
    double quantity;
    double price;
    double commission;
    double pnl;  // Calculated P&L
    uint64_t timestamp;
    bool is_buy;
  };

  TradeRecord create_trade_record(const OrderManager::OrderExecution& execution,
                                  double realized_pnl = 0.0);

 private:
  std::unordered_map<std::string, Position> positions_;
  std::unordered_map<std::string, double> market_prices_;
  std::unordered_map<std::string, std::string> order_symbols_;
  PositionUpdateCallback position_update_callback_;
  double cash_balance_ = 100000.0;

  // MarketDataProcessor for zero-latency atomic best_bid/best_ask access
  std::weak_ptr<RenderEngine::MarketDataProcessor> market_data_processor_;

  std::string get_symbol_from_order(const std::string& order_id);
  bool is_buy_execution(const OrderManager::OrderExecution& execution);
  void update_market_values();
  void calculate_risk_metrics(Position& position);
  double calculate_buying_power() const;
  double calculate_margin_used() const;
  double calculate_portfolio_beta() const;
  double calculate_portfolio_var() const;
  double calculate_portfolio_sharpe() const;
  void notify_position_update(const Position& position);
};

}  // namespace BTQuant
