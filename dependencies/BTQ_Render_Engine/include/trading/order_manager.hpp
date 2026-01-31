#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace BTQuant {

class OrderManager {
 public:
  enum class OrderType { Market, Limit, Stop, StopLimit, TrailingStop, Iceberg, TWAP, VWAP };
  enum class OrderSide { Buy, Sell };
  enum class OrderStatus { Pending, PartiallyFilled, Filled, Cancelled, Rejected, Expired };
  enum class TimeInForce { GTC, IOC, FOK, DAY, GTD };

  struct Order {
    std::string order_id;
    std::string symbol;
    OrderType type;
    OrderSide side;
    double quantity;
    double price;
    double stop_price = 0;
    double filled_quantity = 0;
    double average_fill_price = 0;
    OrderStatus status = OrderStatus::Pending;
    TimeInForce time_in_force = TimeInForce::GTC;
    uint64_t created_time = 0;
    uint64_t updated_time = 0;
    uint64_t expiry_time = 0;
    double trailing_amount = 0;
    double iceberg_visible_quantity = 0;
    double twap_duration_minutes = 0;
    std::string parent_order_id;
    std::vector<std::string> child_order_ids;
    double max_position_size = 0;
    double max_loss_amount = 0;
    bool reduce_only = false;
    std::string execution_venue;
    double slippage_tolerance = 0;
    bool post_only = false;
    std::unordered_map<std::string, std::string> custom_fields;
  };

  struct OrderExecution {
    std::string execution_id;
    std::string order_id;
    double quantity;
    double price;
    double commission;
    uint64_t timestamp;
    std::string venue;
    std::string liquidity_flag;
  };

  OrderManager();
  std::string place_order(const Order& order);
  bool modify_order(const std::string& order_id, double new_quantity, double new_price);
  bool cancel_order(const std::string& order_id);
  std::vector<Order> get_orders(const std::string& symbol = "") const;
  std::vector<Order> get_active_orders(const std::string& symbol = "") const;
  void add_execution(const OrderExecution& execution);
  std::vector<OrderExecution> get_executions(const std::string& order_id) const;

  // Event callbacks
  using OrderUpdateCallback = std::function<void(const Order&)>;
  using ExecutionCallback = std::function<void(const OrderExecution&)>;
  void set_order_update_callback(OrderUpdateCallback callback);
  void set_execution_callback(ExecutionCallback callback);

 private:
  std::unordered_map<std::string, Order> orders_;
  std::unordered_map<std::string, OrderExecution> executions_;
  std::unordered_map<std::string, std::vector<std::string>> order_executions_;
  std::unordered_map<std::string, std::unordered_set<std::string>> symbol_orders_;
  std::unordered_map<std::string, std::string> order_symbols_;
  OrderUpdateCallback order_update_callback_;
  ExecutionCallback execution_callback_;
  std::atomic<uint64_t> order_counter_{0};

  bool validate_order(const Order& order);
  std::string generate_order_id();
  uint64_t get_current_timestamp();
  void process_order(const Order& order);
  void simulate_market_order_execution(const Order& order);
  void add_to_order_book(const Order& order);
  void add_to_stop_orders(const Order& order);
  void add_to_trailing_stops(const Order& order);
  void process_iceberg_order(const Order& order);
  void process_algorithmic_order(const Order& order);
  std::string generate_execution_id();
  double get_current_market_price(const std::string& symbol, OrderSide side);
  double calculate_commission(double quantity, double price);
  void notify_order_update(const Order& order);
  void notify_execution(const OrderExecution& execution);
};

}  // namespace BTQuant
