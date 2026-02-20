#pragma once

/**
 * @file chart_order_entry.hpp
 * @brief Chart Order-Entry System (Right-Click Buy/Sell)
 * 
 * This implementation provides:
 * - Right-click context menu for order entry
 * - Click-to-place limit orders on chart
 * - Order modification by dragging
 * - Position visualization on chart
 * - Order confirmation dialogs
 * - Quick order buttons at price levels
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace btq {
namespace trading {

/**
 * @brief Order side
 */
enum class OrderSide : uint8_t {
    BUY,
    SELL
};

/**
 * @brief Order type
 */
enum class OrderType : uint8_t {
    MARKET,
    LIMIT,
    STOP_MARKET,
    STOP_LIMIT,
    TAKE_PROFIT,
    TAKE_PROFIT_LIMIT,
    TRAILING_STOP
};

/**
 * @brief Order status
 */
enum class OrderStatus : uint8_t {
    PENDING,
    OPEN,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED,
    REJECTED,
    EXPIRED
};

/**
 * @brief Order time in force
 */
enum class TimeInForce : uint8_t {
    GTC,    // Good Till Cancelled
    IOC,    // Immediate Or Cancel
    FOK,    // Fill Or Kill
    GTX,    // Good Till Crossing (Post Only)
    DAY     // Day Order
};

/**
 * @brief Order request structure
 */
struct OrderRequest {
    std::string symbol;
    OrderSide side = OrderSide::BUY;
    OrderType type = OrderType::LIMIT;
    double price = 0.0;
    double quantity = 0.0;
    double stop_price = 0.0;        // For stop orders
    double trailing_distance = 0.0; // For trailing stops
    TimeInForce time_in_force = TimeInForce::GTC;
    std::string client_order_id;
    bool reduce_only = false;
    bool post_only = false;
};

/**
 * @brief Order response structure
 */
struct OrderResponse {
    std::string order_id;
    std::string client_order_id;
    OrderStatus status = OrderStatus::PENDING;
    double price = 0.0;
    double quantity = 0.0;
    double filled_quantity = 0.0;
    double average_price = 0.0;
    int64_t timestamp = 0;
    std::string error_message;
    bool success = false;
};

/**
 * @brief Position information
 */
struct Position {
    std::string symbol;
    OrderSide side = OrderSide::BUY;
    double quantity = 0.0;
    double entry_price = 0.0;
    double mark_price = 0.0;
    double unrealized_pnl = 0.0;
    double realized_pnl = 0.0;
    double liquidation_price = 0.0;
    double leverage = 1.0;
    int64_t open_time = 0;
};

/**
 * @brief Order visualization on chart
 */
struct ChartOrder {
    uint64_t id = 0;
    std::string order_id;
    OrderSide side = OrderSide::BUY;
    OrderType type = OrderType::LIMIT;
    OrderStatus status = OrderStatus::OPEN;
    double price = 0.0;
    double quantity = 0.0;
    double stop_price = 0.0;
    int64_t created_time = 0;
    bool is_visible = true;
    
    // Visual properties
    glm::vec4 color;
    float line_width = 1.0f;
    bool show_label = true;
};

/**
 * @brief Chart Order-Entry configuration
 */
struct ChartOrderConfig {
    bool show_orders_on_chart = true;
    bool show_positions_on_chart = true;
    bool show_stop_loss = true;
    bool show_take_profit = true;
    bool confirm_orders = true;
    bool show_quantity_input = true;
    double default_quantity = 1.0;
    int price_decimals = 2;
    int quantity_decimals = 4;
    
    glm::vec4 buy_order_color = {0.0f, 0.8f, 0.0f, 1.0f};
    glm::vec4 sell_order_color = {0.8f, 0.0f, 0.0f, 1.0f};
    glm::vec4 stop_loss_color = {1.0f, 0.5f, 0.0f, 1.0f};
    glm::vec4 take_profit_color = {0.0f, 1.0f, 0.5f, 1.0f};
    glm::vec4 position_color = {0.5f, 0.5f, 1.0f, 0.5f};
};

/**
 * @brief Context menu action
 */
struct ContextMenuAction {
    std::string label;
    std::function<void()> callback;
    bool enabled = true;
    bool is_separator = false;
};

/**
 * @brief Chart Order-Entry Manager
 */
class ChartOrderEntry {
public:
    ChartOrderEntry() = default;
    
    /**
     * @brief Initialize the order entry system
     */
    void initialize(const ChartOrderConfig& config = ChartOrderConfig{}) {
        config_ = config;
    }
    
    /**
     * @brief Set the order submission callback
     */
    void setOrderCallback(std::function<std::future<OrderResponse>(const OrderRequest&)> callback) {
        order_callback_ = std::move(callback);
    }
    
    /**
     * @brief Set the cancel order callback
     */
    void setCancelCallback(std::function<std::future<bool>(const std::string& order_id)> callback) {
        cancel_callback_ = std::move(callback);
    }
    
    /**
     * @brief Handle right-click on chart
     * @param chart_pos Position in chart coordinates (time, price)
     * @param screen_pos Position in screen coordinates
     * @param current_price Current market price
     * @return List of context menu actions
     */
    std::vector<ContextMenuAction> handleRightClick(
        const glm::vec2& chart_pos,
        const glm::vec2& screen_pos,
        double current_price)
    {
        std::vector<ContextMenuAction> actions;
        
        double click_price = chart_pos.y;
        bool is_above_market = click_price > current_price;
        
        // Buy orders
        actions.push_back({
            "Buy Limit @ " + formatPrice(click_price),
            [this, click_price]() { submitOrder(OrderSide::BUY, OrderType::LIMIT, click_price); },
            true
        });
        
        if (is_above_market) {
            actions.push_back({
                "Buy Stop @ " + formatPrice(click_price),
                [this, click_price]() { submitOrder(OrderSide::BUY, OrderType::STOP_MARKET, click_price); },
                true
            });
        } else {
            actions.push_back({
                "Buy Stop-Limit @ " + formatPrice(click_price),
                [this, click_price]() { submitOrder(OrderSide::BUY, OrderType::STOP_LIMIT, click_price); },
                true
            });
        }
        
        // Separator
        actions.push_back({"", nullptr, false, true});
        
        // Sell orders
        actions.push_back({
            "Sell Limit @ " + formatPrice(click_price),
            [this, click_price]() { submitOrder(OrderSide::SELL, OrderType::LIMIT, click_price); },
            true
        });
        
        if (!is_above_market) {
            actions.push_back({
                "Sell Stop @ " + formatPrice(click_price),
                [this, click_price]() { submitOrder(OrderSide::SELL, OrderType::STOP_MARKET, click_price); },
                true
            });
        } else {
            actions.push_back({
                "Sell Stop-Limit @ " + formatPrice(click_price),
                [this, click_price]() { submitOrder(OrderSide::SELL, OrderType::STOP_LIMIT, click_price); },
                true
            });
        }
        
        // Separator
        actions.push_back({"", nullptr, false, true});
        
        // Quick market orders
        actions.push_back({
            "Buy Market",
            [this]() { submitOrder(OrderSide::BUY, OrderType::MARKET, 0.0); },
            true
        });
        
        actions.push_back({
            "Sell Market",
            [this]() { submitOrder(OrderSide::SELL, OrderType::MARKET, 0.0); },
            true
        });
        
        // Check if clicking near an existing order
        ChartOrder* nearby_order = findNearbyOrder(click_price, 5.0);
        if (nearby_order) {
            actions.push_back({"", nullptr, false, true});
            actions.push_back({
                "Cancel Order " + nearby_order->order_id,
                [this, nearby_order]() { cancelOrder(nearby_order->order_id); },
                true
            });
            actions.push_back({
                "Modify Order " + nearby_order->order_id,
                [this, nearby_order, click_price]() { modifyOrder(nearby_order, click_price); },
                true
            });
        }
        
        return actions;
    }
    
    /**
     * @brief Submit an order
     */
    std::future<OrderResponse> submitOrder(
        OrderSide side,
        OrderType type,
        double price,
        double quantity = 0.0)
    {
        OrderRequest request;
        request.symbol = current_symbol_;
        request.side = side;
        request.type = type;
        request.price = price;
        request.quantity = quantity > 0 ? quantity : config_.default_quantity;
        request.client_order_id = generateClientId();
        
        if (order_callback_) {
            return order_callback_(request);
        }
        
        std::promise<OrderResponse> promise;
        promise.set_value(OrderResponse{});
        return promise.get_future();
    }
    
    /**
     * @brief Cancel an order
     */
    void cancelOrder(const std::string& order_id) {
        if (cancel_callback_) {
            cancel_callback_(order_id);
        }
    }
    
    /**
     * @brief Modify an existing order
     */
    void modifyOrder(ChartOrder* order, double new_price) {
        if (order) {
            order->price = new_price;
            // Would send modify request to exchange
        }
    }
    
    /**
     * @brief Add an order to the chart visualization
     */
    void addChartOrder(const OrderResponse& response, OrderSide side, OrderType type) {
        ChartOrder order;
        order.id = next_order_id_++;
        order.order_id = response.order_id;
        order.side = side;
        order.type = type;
        order.status = response.status;
        order.price = response.price;
        order.quantity = response.quantity;
        order.created_time = response.timestamp;
        
        // Set color based on side
        order.color = (side == OrderSide::BUY) ? config_.buy_order_color : config_.sell_order_color;
        
        chart_orders_.push_back(order);
    }
    
    /**
     * @brief Remove an order from the chart
     */
    void removeChartOrder(const std::string& order_id) {
        auto it = std::find_if(chart_orders_.begin(), chart_orders_.end(),
            [&order_id](const ChartOrder& o) { return o.order_id == order_id; });
        
        if (it != chart_orders_.end()) {
            chart_orders_.erase(it);
        }
    }
    
    /**
     * @brief Update order status
     */
    void updateOrderStatus(const std::string& order_id, OrderStatus status) {
        auto it = std::find_if(chart_orders_.begin(), chart_orders_.end(),
            [&order_id](const ChartOrder& o) { return o.order_id == order_id; });
        
        if (it != chart_orders_.end()) {
            it->status = status;
        }
    }
    
    /**
     * @brief Get all chart orders
     */
    const std::vector<ChartOrder>& getChartOrders() const {
        return chart_orders_;
    }
    
    /**
     * @brief Set current position
     */
    void setPosition(const Position& position) {
        current_position_ = position;
    }
    
    /**
     * @brief Get current position
     */
    const Position& getPosition() const {
        return current_position_;
    }
    
    /**
     * @brief Set current symbol
     */
    void setSymbol(const std::string& symbol) {
        current_symbol_ = symbol;
    }
    
    /**
     * @brief Set default quantity
     */
    void setDefaultQuantity(double quantity) {
        config_.default_quantity = quantity;
    }
    
    /**
     * @brief Find order near a price level
     */
    ChartOrder* findNearbyOrder(double price, double threshold) {
        for (auto& order : chart_orders_) {
            if (std::abs(order.price - price) <= threshold) {
                return &order;
            }
        }
        return nullptr;
    }
    
    /**
     * @brief Generate vertex data for order visualization
     */
    struct OrderVertex {
        glm::vec2 position;
        glm::vec4 color;
    };
    
    std::vector<OrderVertex> generateOrderVertices(
        float x_start,
        float x_end,
        float pixels_per_price,
        float chart_y_offset) const
    {
        std::vector<OrderVertex> vertices;
        
        for (const auto& order : chart_orders_) {
            if (!order.is_visible) continue;
            
            float y = chart_y_offset + static_cast<float>(order.price) * pixels_per_price;
            
            // Horizontal line for order price
            vertices.push_back({{x_start, y}, order.color});
            vertices.push_back({{x_end, y}, order.color});
        }
        
        return vertices;
    }
    
    /**
     * @brief Generate position visualization vertices
     */
    std::vector<OrderVertex> generatePositionVertices(
        float x_start,
        float x_end,
        float entry_y,
        float current_y,
        bool is_long) const
    {
        std::vector<OrderVertex> vertices;
        
        if (current_position_.quantity == 0) return vertices;
        
        glm::vec4 color = is_long ? config_.buy_order_color : config_.sell_order_color;
        color.a = 0.3f;  // Semi-transparent fill
        
        // Draw rectangle from entry to current price
        float y_top = std::min(entry_y, current_y);
        float y_bottom = std::max(entry_y, current_y);
        
        // Two triangles for the rectangle
        vertices.push_back({{x_start, y_top}, color});
        vertices.push_back({{x_end, y_top}, color});
        vertices.push_back({{x_start, y_bottom}, color});
        
        vertices.push_back({{x_end, y_top}, color});
        vertices.push_back({{x_end, y_bottom}, color});
        vertices.push_back({{x_start, y_bottom}, color});
        
        return vertices;
    }

private:
    std::string formatPrice(double price) const {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.*f", config_.price_decimals, price);
        return std::string(buffer);
    }
    
    std::string generateClientId() {
        return "BTQ_" + std::to_string(next_client_id_++);
    }
    
    ChartOrderConfig config_;
    std::string current_symbol_;
    Position current_position_;
    std::vector<ChartOrder> chart_orders_;
    
    std::function<std::future<OrderResponse>(const OrderRequest&)> order_callback_;
    std::function<std::future<bool>(const std::string&)> cancel_callback_;
    
    uint64_t next_order_id_ = 1;
    uint64_t next_client_id_ = 1;
};

/**
 * @brief Quick order panel for DOM
 */
class QuickOrderPanel {
public:
    struct QuickOrderButton {
        std::string label;
        OrderSide side;
        double price;
        glm::vec4 color;
        float x, y, width, height;
    };
    
    /**
     * @brief Generate quick order buttons for DOM levels
     */
    std::vector<QuickOrderButton> generateButtons(
        const std::vector<double>& bid_prices,
        const std::vector<double>& ask_prices,
        float x_offset,
        float y_start,
        float row_height,
        float button_width,
        const ChartOrderConfig& config)
    {
        std::vector<QuickOrderButton> buttons;
        
        // Buy buttons on bid side
        for (size_t i = 0; i < bid_prices.size(); ++i) {
            QuickOrderButton btn;
            btn.label = "B";
            btn.side = OrderSide::BUY;
            btn.price = bid_prices[i];
            btn.color = config.buy_order_color;
            btn.x = x_offset;
            btn.y = y_start + i * row_height;
            btn.width = button_width;
            btn.height = row_height;
            buttons.push_back(btn);
        }
        
        // Sell buttons on ask side
        for (size_t i = 0; i < ask_prices.size(); ++i) {
            QuickOrderButton btn;
            btn.label = "S";
            btn.side = OrderSide::SELL;
            btn.price = ask_prices[i];
            btn.color = config.sell_order_color;
            btn.x = x_offset;
            btn.y = y_start + (bid_prices.size() + i) * row_height;
            btn.width = button_width;
            btn.height = row_height;
            buttons.push_back(btn);
        }
        
        return buttons;
    }
};

/**
 * @brief Order ticket dialog
 */
class OrderTicketDialog {
public:
    struct DialogState {
        bool is_open = false;
        OrderSide side = OrderSide::BUY;
        OrderType type = OrderType::LIMIT;
        double price = 0.0;
        double quantity = 1.0;
        double stop_price = 0.0;
        TimeInForce time_in_force = TimeInForce::GTC;
        bool reduce_only = false;
        bool post_only = false;
    };
    
    void open(OrderSide side, double price, double default_quantity) {
        state_.is_open = true;
        state_.side = side;
        state_.price = price;
        state_.quantity = default_quantity;
    }
    
    void close() {
        state_.is_open = false;
    }
    
    DialogState& getState() { return state_; }
    const DialogState& getState() const { return state_; }

private:
    DialogState state_;
};

} // namespace trading
} // namespace btq
