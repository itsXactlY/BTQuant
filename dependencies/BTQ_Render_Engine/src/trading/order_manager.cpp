#include "trading/order_manager.hpp"

#include <chrono>
#include <random>
#include <sstream>

namespace BTQuant {

OrderManager::OrderManager() : order_counter_(0) {}

std::string OrderManager::place_order(const Order& order) {
  if (!validate_order(order)) {
    return "";
  }

  Order new_order = order;
  new_order.order_id = generate_order_id();
  new_order.created_time = get_current_timestamp();
  new_order.updated_time = new_order.created_time;

  orders_[new_order.order_id] = new_order;
  symbol_orders_[new_order.symbol].insert(new_order.order_id);
  order_symbols_[new_order.order_id] = new_order.symbol;

  process_order(new_order);

  notify_order_update(new_order);

  return new_order.order_id;
}

bool OrderManager::modify_order(const std::string& order_id, double new_quantity,
                                double new_price) {
  auto it = orders_.find(order_id);
  if (it == orders_.end()) {
    return false;
  }

  Order& order = it->second;

  if (order.status != OrderStatus::Pending && order.status != OrderStatus::PartiallyFilled) {
    return false;
  }

  order.quantity = new_quantity;
  order.price = new_price;
  order.updated_time = get_current_timestamp();

  notify_order_update(order);

  return true;
}

bool OrderManager::cancel_order(const std::string& order_id) {
  auto it = orders_.find(order_id);
  if (it == orders_.end()) {
    return false;
  }

  Order& order = it->second;

  if (order.status == OrderStatus::Filled || order.status == OrderStatus::Cancelled) {
    return false;
  }

  order.status = OrderStatus::Cancelled;
  order.updated_time = get_current_timestamp();

  notify_order_update(order);

  return true;
}

std::vector<OrderManager::Order> OrderManager::get_orders(const std::string& symbol) const {
  std::vector<Order> result;

  if (symbol.empty()) {
    result.reserve(orders_.size());
    for (const auto& pair : orders_) {
      result.push_back(pair.second);
    }
  } else {
    auto it = symbol_orders_.find(symbol);
    if (it != symbol_orders_.end()) {
      result.reserve(it->second.size());
      for (const auto& order_id : it->second) {
        auto order_it = orders_.find(order_id);
        if (order_it != orders_.end()) {
          result.push_back(order_it->second);
        }
      }
    }
  }

  return result;
}

std::vector<OrderManager::Order> OrderManager::get_active_orders(const std::string& symbol) const {
  std::vector<Order> result;

  std::vector<Order> all_orders = get_orders(symbol);

  for (const auto& order : all_orders) {
    if (order.status == OrderStatus::Pending || order.status == OrderStatus::PartiallyFilled) {
      result.push_back(order);
    }
  }

  return result;
}

void OrderManager::add_execution(const OrderExecution& execution) {
  executions_[execution.execution_id] = execution;
  order_executions_[execution.order_id].push_back(execution.execution_id);

  auto order_it = orders_.find(execution.order_id);
  if (order_it != orders_.end()) {
    Order& order = order_it->second;
    order.filled_quantity += execution.quantity;
    order.updated_time = get_current_timestamp();

    if (order.filled_quantity >= order.quantity) {
      order.status = OrderStatus::Filled;
      order.average_fill_price =
          (order.average_fill_price * (order.filled_quantity - execution.quantity) +
           execution.price * execution.quantity) /
          order.filled_quantity;
    } else {
      order.status = OrderStatus::PartiallyFilled;
      order.average_fill_price =
          (order.average_fill_price * (order.filled_quantity - execution.quantity) +
           execution.price * execution.quantity) /
          order.filled_quantity;
    }

    notify_order_update(order);
  }

  notify_execution(execution);
}

std::vector<OrderManager::OrderExecution> OrderManager::get_executions(
    const std::string& order_id) const {
  std::vector<OrderExecution> result;

  auto it = order_executions_.find(order_id);
  if (it != order_executions_.end()) {
    result.reserve(it->second.size());
    for (const auto& execution_id : it->second) {
      auto exec_it = executions_.find(execution_id);
      if (exec_it != executions_.end()) {
        result.push_back(exec_it->second);
      }
    }
  }

  return result;
}

void OrderManager::set_order_update_callback(OrderUpdateCallback callback) {
  order_update_callback_ = std::move(callback);
}

void OrderManager::set_execution_callback(ExecutionCallback callback) {
  execution_callback_ = std::move(callback);
}

bool OrderManager::validate_order(const Order& order) {
  if (order.symbol.empty()) {
    return false;
  }

  if (order.quantity <= 0) {
    return false;
  }

  if (order.type == OrderType::Limit || order.type == OrderType::StopLimit) {
    if (order.price <= 0) {
      return false;
    }
  }

  if (order.type == OrderType::Stop || order.type == OrderType::StopLimit) {
    if (order.stop_price <= 0) {
      return false;
    }
  }

  return true;
}

std::string OrderManager::generate_order_id() {
  std::stringstream ss;
  ss << "ORD-" << std::hex << order_counter_++;
  return ss.str();
}

uint64_t OrderManager::get_current_timestamp() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

void OrderManager::process_order(const Order& order) {
  switch (order.type) {
    case OrderType::Market:
      simulate_market_order_execution(order);
      break;
    case OrderType::Iceberg:
      process_iceberg_order(order);
      break;
    case OrderType::TWAP:
    case OrderType::VWAP:
      process_algorithmic_order(order);
      break;
    case OrderType::Stop:
      add_to_stop_orders(order);
      break;
    case OrderType::TrailingStop:
      add_to_trailing_stops(order);
      break;
    default:
      add_to_order_book(order);
      break;
  }
}

void OrderManager::simulate_market_order_execution(const Order& order) {
  double market_price = get_current_market_price(order.symbol, order.side);

  OrderExecution execution;
  execution.execution_id = generate_execution_id();
  execution.order_id = order.order_id;
  execution.quantity = order.quantity;
  execution.price = market_price;
  execution.commission = calculate_commission(order.quantity, market_price);
  execution.timestamp = get_current_timestamp();
  execution.venue = "SIMULATED";
  execution.liquidity_flag = "M";

  add_execution(execution);
}

void OrderManager::add_to_order_book(const Order& /*order*/) {
  // In a real implementation, this would add to an order book
  // For simulation, we just mark it as pending
}

void OrderManager::add_to_stop_orders(const Order& /*order*/) {
  // In a real implementation, this would add to a stop order list
  // For simulation, we mark it as pending
}

void OrderManager::add_to_trailing_stops(const Order& /*order*/) {
  // In a real implementation, this would add to a trailing stop list
  // For simulation, we mark it as pending
}

void OrderManager::process_iceberg_order(const Order& order) {
  // Simulate iceberg order by splitting into visible and hidden portions
  double visible_quantity =
      order.iceberg_visible_quantity > 0 ? order.iceberg_visible_quantity : order.quantity / 10.0;

  Order visible_order = order;
  visible_order.quantity = std::min(visible_quantity, order.quantity - order.filled_quantity);

  add_to_order_book(visible_order);
}

void OrderManager::process_algorithmic_order(const Order& order) {
  // Simulate TWAP/VWAP order by splitting into smaller chunks
  int num_chunks = 10;
  double chunk_quantity = order.quantity / num_chunks;

  for (int i = 0; i < num_chunks; ++i) {
    Order chunk = order;
    chunk.quantity = chunk_quantity;
    chunk.parent_order_id = order.order_id;
    chunk.order_id = generate_order_id();
    chunk.created_time = get_current_timestamp();

    orders_[chunk.order_id] = chunk;
    symbol_orders_[chunk.symbol].insert(chunk.order_id);
    order_symbols_[chunk.order_id] = chunk.symbol;

    add_to_order_book(chunk);
  }
}

std::string OrderManager::generate_execution_id() {
  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::stringstream ss;
  ss << "EXEC-" << std::hex << ms << "-" << order_counter_++;
  return ss.str();
}

double OrderManager::get_current_market_price(const std::string& symbol, OrderSide /*side*/) {
  // Simulated market price - in a real implementation, this would fetch from market data
  static std::unordered_map<std::string, double> mock_prices;
  auto it = mock_prices.find(symbol);
  if (it == mock_prices.end()) {
    mock_prices[symbol] = 100.0;
    return 100.0;
  }
  return it->second;
}

double OrderManager::calculate_commission(double quantity, double price) {
  // Simulated commission - 0.1% of trade value
  return quantity * price * 0.001;
}

void OrderManager::notify_order_update(const Order& order) {
  if (order_update_callback_) {
    order_update_callback_(order);
  }
}

void OrderManager::notify_execution(const OrderExecution& execution) {
  if (execution_callback_) {
    execution_callback_(execution);
  }
}

}  // namespace BTQuant
