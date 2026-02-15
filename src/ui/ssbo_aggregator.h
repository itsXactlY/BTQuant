/**
 * SSBO Order Book Aggregator Header
 *
 * Header file for aggregating multiple exchange order books into a single SSBO feed
 * for GPU-based visualization and analysis in professional trading applications.
 */

#pragma once

#include "analytics/orderbook_snapshot_100level.h"
#include "imgui.h"
#include <vector>
#include <string>
#include <memory>

namespace BTQuant {
namespace UI {

/**
 * @brief Structure representing an aggregated order book for SSBO
 */
struct AggregatedOrderBookSSBO {
    uint32_t num_exchanges;
    uint32_t current_time_index;
    float base_price;
    float price_range;
    uint32_t price_levels_count;
    
    // Fixed-size arrays for aggregated price levels (Standard Layout POD)
    static constexpr size_t MAX_LEVELS = 1024;  // Maximum price levels per exchange
    
    struct Level {
        float price;
        uint32_t ask_quantity;
        uint32_t bid_quantity;
        uint32_t num_orders;
    };
    
    Level levels[MAX_LEVELS];
    
    AggregatedOrderBookSSBO() : num_exchanges(0), current_time_index(0), 
                               base_price(0.0f), price_range(0.0f), 
                               price_levels_count(0) {
        for (size_t i = 0; i < MAX_LEVELS; ++i) {
            levels[i] = Level{0.0f, 0, 0, 0};
        }
    }
};

/**
 * @brief Class for aggregating multiple exchange order books into a single SSBO feed
 */
class SSBOAggregator {
public:
    /**
     * @brief Constructor
     */
    SSBOAggregator();
    
    /**
     * @brief Destructor
     */
    ~SSBOAggregator();
    
    /**
     * @brief Add an exchange's order book snapshot to the aggregator
     * @param exchange_name Name of the exchange
     * @param snapshot Order book snapshot from the exchange
     */
    void addExchangeSnapshot(const std::string& exchange_name, 
                           const OrderBookSnapshot100Level& snapshot);
    
    /**
     * @brief Aggregate all exchange order books into a single SSBO structure
     * @return Aggregated order book SSBO structure
     */
    AggregatedOrderBookSSBO aggregateToSSBO() const;
    
    /**
     * @brief Render the right-click context menu for SSBO aggregation
     * @param position Position where the context menu should appear
     * @return True if aggregation was triggered, false otherwise
     */
    bool renderContextMenu(const ImVec2& position);
    
    /**
     * @brief Get the list of available exchanges
     * @return Vector of exchange names
     */
    std::vector<std::string> getAvailableExchanges() const;
    
    /**
     * @brief Clear all stored exchange snapshots
     */
    void clearSnapshots();
    
private:
    struct ExchangeSnapshot {
        std::string name;
        OrderBookSnapshot100Level snapshot;
        bool enabled;
    };
    
    std::vector<ExchangeSnapshot> exchange_snapshots_;
    
    // State for context menu
    bool show_context_menu_;
    ImVec2 context_menu_position_;
};

/**
 * @brief Helper function to render the SSBO aggregation context menu
 * @param aggregator Pointer to the SSBO aggregator instance
 * @param position Position where the context menu should appear
 * @return True if aggregation was triggered, false otherwise
 */
bool renderSSBOAggregationContextMenu(SSBOAggregator* aggregator, const ImVec2& position);

} // namespace UI
} // namespace BTQuant