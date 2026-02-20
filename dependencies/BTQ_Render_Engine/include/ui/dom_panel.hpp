#pragma once

/**
 * @file dom_panel.hpp
 * @brief Depth of Market (DOM) Panel with Atomic OrderBook State
 * 
 * This implementation provides:
 * - Real-time order book visualization
 * - Lock-free reading from double-buffered atomic orderbook
 * - Liquidity histogram rendering
 * - Price level aggregation
 * - Spread calculation and display
 * - Trade execution buttons at each price level
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

// Forward declaration
namespace btq {
namespace data {
    struct OrderBookSnapshot;
    struct OrderBookLevel;
}
}

namespace btq {
namespace ui {

/**
 * @brief DOM display configuration
 */
struct DOMConfig {
    int max_levels = 20;                    // Maximum price levels to display
    float row_height = 20.0f;               // Height of each price row
    float price_column_width = 100.0f;      // Width of price column
    float size_column_width = 80.0f;        // Width of size column
    float total_column_width = 80.0f;       // Width of total column
    float histogram_width = 150.0f;         // Width of liquidity histogram
    bool show_totals = true;                // Show cumulative totals
    bool show_histogram = true;             // Show liquidity histogram
    bool show_trade_buttons = true;         // Show buy/sell buttons
    int price_decimals = 2;                 // Decimal places for price
    int size_decimals = 4;                  // Decimal places for size
    
    glm::vec4 bid_color = {0.0f, 0.6f, 0.0f, 1.0f};
    glm::vec4 ask_color = {0.8f, 0.0f, 0.0f, 1.0f};
    glm::vec4 spread_color = {0.8f, 0.8f, 0.0f, 1.0f};
    glm::vec4 background_color = {0.1f, 0.1f, 0.1f, 1.0f};
    glm::vec4 text_color = {1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 grid_color = {0.3f, 0.3f, 0.3f, 0.5f};
};

/**
 * @brief Single price level in the DOM
 */
struct DOMLevel {
    double price = 0.0;
    double size = 0.0;
    double total = 0.0;         // Cumulative total
    int num_orders = 0;
    bool is_bid = true;
    float histogram_ratio = 0.0f;  // Ratio for histogram bar (0-1)
};

/**
 * @brief DOM state for rendering
 */
struct DOMState {
    std::vector<DOMLevel> bids;
    std::vector<DOMLevel> asks;
    double best_bid = 0.0;
    double best_ask = 0.0;
    double spread = 0.0;
    double spread_percent = 0.0;
    double mid_price = 0.0;
    int64_t timestamp = 0;
    bool is_valid = false;
};

/**
 * @brief DOM Panel implementation
 */
class DOMPanel {
public:
    DOMPanel() = default;
    
    /**
     * @brief Initialize the DOM panel
     */
    void initialize(const DOMConfig& config = DOMConfig{}) {
        config_ = config;
        dom_state_.bids.reserve(config.max_levels);
        dom_state_.asks.reserve(config.max_levels);
    }
    
    /**
     * @brief Update DOM state from atomic orderbook snapshot
     * This is the main entry point for lock-free reading
     */
    void updateFromOrderBook(const data::OrderBookSnapshot& snapshot);
    
    /**
     * @brief Get the current DOM state (for rendering)
     */
    const DOMState& getState() const { return dom_state_; }
    
    /**
     * @brief Calculate the total width needed
     */
    float getTotalWidth() const {
        float width = config_.price_column_width + config_.size_column_width;
        if (config_.show_totals) {
            width += config_.total_column_width;
        }
        if (config_.show_histogram) {
            width += config_.histogram_width;
        }
        return width;
    }
    
    /**
     * @brief Calculate the total height needed
     */
    float getTotalHeight() const {
        return config_.row_height * (config_.max_levels * 2 + 1);  // Bids + asks + spread
    }
    
    /**
     * @brief Set callback for trade execution
     */
    void setTradeCallback(std::function<void(bool is_buy, double price, double size)> callback) {
        trade_callback_ = std::move(callback);
    }
    
    /**
     * @brief Handle click on a price level
     */
    void onPriceLevelClick(float y_position, bool is_buy_side) {
        int level_index = static_cast<int>(y_position / config_.row_height);
        
        if (level_index >= 0 && level_index < config_.max_levels) {
            const auto& levels = is_buy_side ? dom_state_.bids : dom_state_.asks;
            if (level_index < static_cast<int>(levels.size())) {
                if (trade_callback_) {
                    trade_callback_(!is_buy_side, levels[level_index].price, 0.0);
                }
            }
        }
    }
    
    /**
     * @brief Get the configuration
     */
    const DOMConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set the configuration
     */
    void setConfig(const DOMConfig& config) { config_ = config; }
    
    /**
     * @brief Format price for display
     */
    std::string formatPrice(double price) const {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.*f", config_.price_decimals, price);
        return std::string(buffer);
    }
    
    /**
     * @brief Format size for display
     */
    std::string formatSize(double size) const {
        char buffer[32];
        if (size >= 1000000.0) {
            std::snprintf(buffer, sizeof(buffer), "%.2fM", size / 1000000.0);
        } else if (size >= 1000.0) {
            std::snprintf(buffer, sizeof(buffer), "%.2fK", size / 1000.0);
        } else {
            std::snprintf(buffer, sizeof(buffer), "%.*f", config_.size_decimals, size);
        }
        return std::string(buffer);
    }
    
    /**
     * @brief Calculate histogram ratios
     */
    void calculateHistogramRatios() {
        // Find maximum size for scaling
        double max_size = 0.0;
        
        for (const auto& level : dom_state_.bids) {
            max_size = std::max(max_size, level.size);
        }
        for (const auto& level : dom_state_.asks) {
            max_size = std::max(max_size, level.size);
        }
        
        if (max_size == 0.0) return;
        
        // Calculate ratios
        for (auto& level : dom_state_.bids) {
            level.histogram_ratio = static_cast<float>(level.size / max_size);
        }
        for (auto& level : dom_state_.asks) {
            level.histogram_ratio = static_cast<float>(level.size / max_size);
        }
    }
    
    /**
     * @brief Calculate cumulative totals
     */
    void calculateTotals() {
        double bid_total = 0.0;
        for (auto& level : dom_state_.bids) {
            bid_total += level.size;
            level.total = bid_total;
        }
        
        double ask_total = 0.0;
        for (auto& level : dom_state_.asks) {
            ask_total += level.size;
            level.total = ask_total;
        }
    }

private:
    DOMConfig config_;
    DOMState dom_state_;
    std::function<void(bool is_buy, double price, double size)> trade_callback_;
};

/**
 * @brief Volume-at-Price visualization for DOM
 */
class VolumeAtPrice {
public:
    struct VAPLevel {
        double price;
        double volume;
        float y_position;
        float bar_width;
        bool is_bid;
    };
    
    /**
     * @brief Calculate volume at price levels
     */
    void calculate(
        const data::OrderBookSnapshot& snapshot,
        double min_price,
        double max_price,
        float chart_height,
        float pixels_per_price)
    {
        levels_.clear();
        
        if (snapshot.bids.empty() && snapshot.asks.empty()) {
            return;
        }
        
        // Aggregate bids
        for (const auto& [price, size, orders] : snapshot.bids) {
            if (price >= min_price && price <= max_price) {
                VAPLevel level;
                level.price = price;
                level.volume = size;
                level.y_position = chart_height - static_cast<float>((price - min_price) * pixels_per_price);
                level.is_bid = true;
                levels_.push_back(level);
            }
        }
        
        // Aggregate asks
        for (const auto& [price, size, orders] : snapshot.asks) {
            if (price >= min_price && price <= max_price) {
                VAPLevel level;
                level.price = price;
                level.volume = size;
                level.y_position = chart_height - static_cast<float>((price - min_price) * pixels_per_price);
                level.is_bid = false;
                levels_.push_back(level);
            }
        }
        
        // Calculate bar widths
        double max_volume = 0.0;
        for (const auto& level : levels_) {
            max_volume = std::max(max_volume, level.volume);
        }
        
        if (max_volume > 0.0) {
            for (auto& level : levels_) {
                level.bar_width = static_cast<float>(level.volume / max_volume);
            }
        }
    }
    
    const std::vector<VAPLevel>& getLevels() const { return levels_; }

private:
    std::vector<VAPLevel> levels_;
};

/**
 * @brief DOM statistics calculator
 */
class DOMStatistics {
public:
    struct Stats {
        double bid_liquidity = 0.0;         // Total bid liquidity
        double ask_liquidity = 0.0;         // Total ask liquidity
        double bid_ask_ratio = 0.0;         // Bid/Ask ratio
        double imbalance = 0.0;             // Order book imbalance (-1 to 1)
        double weighted_bid_price = 0.0;    // Volume-weighted bid price
        double weighted_ask_price = 0.0;    // Volume-weighted ask price
        double spread = 0.0;
        double mid_price = 0.0;
        int bid_levels = 0;
        int ask_levels = 0;
    };
    
    static Stats calculate(const DOMState& dom) {
        Stats stats;
        
        if (!dom.is_valid) {
            return stats;
        }
        
        // Calculate total liquidity
        for (const auto& level : dom.bids) {
            stats.bid_liquidity += level.size;
            stats.weighted_bid_price += level.price * level.size;
        }
        
        for (const auto& level : dom.asks) {
            stats.ask_liquidity += level.size;
            stats.weighted_ask_price += level.price * level.size;
        }
        
        // Calculate weighted prices
        if (stats.bid_liquidity > 0.0) {
            stats.weighted_bid_price /= stats.bid_liquidity;
        }
        if (stats.ask_liquidity > 0.0) {
            stats.weighted_ask_price /= stats.ask_liquidity;
        }
        
        // Calculate ratios
        double total_liquidity = stats.bid_liquidity + stats.ask_liquidity;
        if (total_liquidity > 0.0) {
            stats.bid_ask_ratio = stats.bid_liquidity / stats.ask_liquidity;
            stats.imbalance = (stats.bid_liquidity - stats.ask_liquidity) / total_liquidity;
        }
        
        stats.spread = dom.spread;
        stats.mid_price = dom.mid_price;
        stats.bid_levels = static_cast<int>(dom.bids.size());
        stats.ask_levels = static_cast<int>(dom.asks.size());
        
        return stats;
    }
};

/**
 * @brief Price level aggregator for DOM
 */
class PriceLevelAggregator {
public:
    /**
     * @brief Aggregate order book levels by price tick size
     */
    static std::vector<DOMLevel> aggregateLevels(
        const std::vector<std::tuple<double, double, int>>& raw_levels,
        double tick_size,
        int max_levels,
        bool is_bid)
    {
        std::vector<DOMLevel> result;
        result.reserve(max_levels);
        
        if (raw_levels.empty()) {
            return result;
        }
        
        // Group by tick
        std::unordered_map<int64_t, double> size_by_tick;
        std::unordered_map<int64_t, int> orders_by_tick;
        
        for (const auto& [price, size, orders] : raw_levels) {
            int64_t tick = static_cast<int64_t>(std::round(price / tick_size));
            size_by_tick[tick] += size;
            orders_by_tick[tick] += orders;
        }
        
        // Convert to sorted levels
        std::vector<std::pair<int64_t, double>> sorted_levels;
        for (const auto& [tick, size] : size_by_tick) {
            sorted_levels.emplace_back(tick, size);
        }
        
        // Sort by price (descending for bids, ascending for asks)
        if (is_bid) {
            std::sort(sorted_levels.begin(), sorted_levels.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
        } else {
            std::sort(sorted_levels.begin(), sorted_levels.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        }
        
        // Convert to DOM levels
        int count = 0;
        for (const auto& [tick, size] : sorted_levels) {
            if (count >= max_levels) break;
            
            DOMLevel level;
            level.price = tick * tick_size;
            level.size = size;
            level.num_orders = orders_by_tick[tick];
            level.is_bid = is_bid;
            
            result.push_back(level);
            ++count;
        }
        
        return result;
    }
};

} // namespace ui
} // namespace btq
