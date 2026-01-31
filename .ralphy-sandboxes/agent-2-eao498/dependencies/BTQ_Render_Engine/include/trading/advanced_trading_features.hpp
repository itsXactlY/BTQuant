#pragma once

#include "trading_interface.hpp"
#include "order_manager.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace BTQuant {

// ============================================================================
// Advanced Algorithmic Trading Features
// ============================================================================

struct TWAPParams {
    std::string symbol;
    OrderManager::OrderSide side;
    double total_quantity;
    std::chrono::seconds execution_window;
    int num_slices = 10;
    double slippage_tolerance = 0.001; // 0.1%
};

struct VWAPParams {
    std::string symbol;
    OrderManager::OrderSide side;
    double total_quantity;
    std::chrono::minutes lookback_period = std::chrono::minutes(15);
    double slippage_tolerance = 0.001; // 0.1%
};

struct OCOOrder {  // One-Cancels-Other
    OrderManager::Order first_order;
    OrderManager::Order second_order;
    std::string oco_group_id;
};

struct BracketOrder {
    std::string parent_order_id;
    std::string take_profit_order_id;
    std::string stop_loss_order_id;
    double take_profit_price;
    double stop_loss_price;
};

class AdvancedTradingFeatures {
public:
    explicit AdvancedTradingFeatures(TradingInterface* trading_interface);
    
    // Time-Weighted Average Price (TWAP) Orders
    std::string place_twap_order(const TWAPParams& params);
    
    // Volume-Weighted Average Price (VWAP) Orders
    std::string place_vwap_order(const VWAPParams& params);
    
    // One-Cancels-Other (OCO) Orders
    std::pair<std::string, std::string> place_oco_order(const OCOOrder& oco);
    
    // Bracket Orders (Entry + Take Profit + Stop Loss)
    BracketOrder place_bracket_order(const std::string& symbol,
                                   OrderManager::OrderSide side,
                                   double quantity,
                                   double entry_price,
                                   double take_profit_price,
                                   double stop_loss_price);
    
    // Trailing Stop Orders
    std::string place_trailing_stop_order(const std::string& symbol,
                                        OrderManager::OrderSide side,
                                        double quantity,
                                        double trail_amount,
                                        double trail_percent = 0.0);
    
    // Iceberg Orders
    std::string place_iceberg_order(const std::string& symbol,
                                  OrderManager::OrderSide side,
                                  double total_quantity,
                                  double displayed_quantity,
                                  double price);
    
    // Scale Orders (Pyramiding)
    std::vector<std::string> place_scale_orders(const std::string& symbol,
                                              OrderManager::OrderSide side,
                                              const std::vector<std::pair<double, double>>& price_qty_pairs);
    
    // OTO (One-Triggers-Other) Orders
    std::pair<std::string, std::string> place_oto_order(const OrderManager::Order& primary_order,
                                                       const OrderManager::Order& secondary_order);
    
    // Risk-Adjusted Position Sizing
    double calculate_position_size(const std::string& symbol,
                                 double risk_percentage,
                                 double stop_loss_distance);
    
    // Portfolio-Level Risk Management
    void set_portfolio_risk_limits(double max_portfolio_risk_percentage,
                                  double max_correlation_threshold,
                                  double max_sector_exposure);
    
    // Correlation-Based Risk Checks
    bool is_correlation_safe(const std::string& new_symbol,
                           double max_correlation_threshold = 0.7);
    
    // Multi-Leg Options Strategies
    std::vector<std::string> place_options_strategy(const std::string& underlying_symbol,
                                                  const std::vector<OrderManager::Order>& legs);
    
private:
    TradingInterface* trading_interface_;
    
    // Internal helper methods
    void schedule_twap_slice(const TWAPParams& params, int slice_num, double slice_qty);
    double calculate_vwap_price(const std::string& symbol, std::chrono::minutes lookback);
    void monitor_bracket_order(const BracketOrder& bracket);
    void update_trailing_stop(const std::string& order_id, double current_price, double trail_amount);
};

} // namespace BTQuant