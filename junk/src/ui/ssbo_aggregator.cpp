/**
 * SSBO Order Book Aggregator Implementation
 *
 * Implementation file for aggregating multiple exchange order books into a single SSBO feed
 * for GPU-based visualization and analysis in professional trading applications.
 */

#include "ui/ssbo_aggregator.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <limits>

namespace BTQuant {
namespace UI {

SSBOAggregator::SSBOAggregator() : show_context_menu_(false) {
}

SSBOAggregator::~SSBOAggregator() {
}

void SSBOAggregator::addExchangeSnapshot(const std::string& exchange_name, 
                                       const OrderBookSnapshot100Level& snapshot) {
    // Check if exchange already exists
    for (auto& existing_snapshot : exchange_snapshots_) {
        if (existing_snapshot.name == exchange_name) {
            existing_snapshot.snapshot = snapshot;
            return;
        }
    }
    
    // Add new exchange
    ExchangeSnapshot new_snapshot;
    new_snapshot.name = exchange_name;
    new_snapshot.snapshot = snapshot;
    new_snapshot.enabled = true; // Enable by default
    
    exchange_snapshots_.push_back(new_snapshot);
}

AggregatedOrderBookSSBO SSBOAggregator::aggregateToSSBO() const {
    AggregatedOrderBookSSBO ssbo_data;
    
    if (exchange_snapshots_.empty()) {
        return ssbo_data;
    }
    
    // Count enabled exchanges
    uint32_t enabled_count = 0;
    for (const auto& snapshot : exchange_snapshots_) {
        if (snapshot.enabled) {
            enabled_count++;
        }
    }
    
    ssbo_data.num_exchanges = enabled_count;
    
    if (enabled_count == 0) {
        return ssbo_data;
    }
    
    // Calculate aggregated price range and base price
    double min_price = std::numeric_limits<double>::max();
    double max_price = std::numeric_limits<double>::lowest();
    double total_bid_volume = 0.0;
    double total_ask_volume = 0.0;
    
    // Find the overall price range across all enabled exchanges
    for (const auto& snapshot : exchange_snapshots_) {
        if (!snapshot.enabled) continue;
        
        // Check bid levels
        for (uint32_t i = 0; i < snapshot.snapshot.bid_levels_count && i < OrderBookSnapshot100Level::MAX_LEVELS; ++i) {
            if (snapshot.snapshot.bids[i].price > 0) {
                min_price = std::min(min_price, snapshot.snapshot.bids[i].price);
                max_price = std::max(max_price, snapshot.snapshot.bids[i].price);
                total_bid_volume += snapshot.snapshot.bids[i].size;
            }
        }
        
        // Check ask levels
        for (uint32_t i = 0; i < snapshot.snapshot.ask_levels_count && i < OrderBookSnapshot100Level::MAX_LEVELS; ++i) {
            if (snapshot.snapshot.asks[i].price > 0) {
                min_price = std::min(min_price, snapshot.snapshot.asks[i].price);
                max_price = std::max(max_price, snapshot.snapshot.asks[i].price);
                total_ask_volume += snapshot.snapshot.asks[i].size;
            }
        }
    }
    
    if (min_price == std::numeric_limits<double>::max() || max_price == std::numeric_limits<double>::lowest()) {
        // No valid price data found
        return ssbo_data;
    }
    
    ssbo_data.base_price = static_cast<float>(min_price);
    ssbo_data.price_range = static_cast<float>(max_price - min_price);
    
    // Aggregate price levels from all enabled exchanges
    std::map<double, std::pair<uint32_t, uint32_t>> aggregated_levels; // price -> (bid_qty, ask_qty)
    
    for (const auto& snapshot : exchange_snapshots_) {
        if (!snapshot.enabled) continue;
        
        // Aggregate bid levels
        for (uint32_t i = 0; i < snapshot.snapshot.bid_levels_count && i < OrderBookSnapshot100Level::MAX_LEVELS; ++i) {
            if (snapshot.snapshot.bids[i].price > 0) {
                auto& level = aggregated_levels[snapshot.snapshot.bids[i].price];
                level.first += static_cast<uint32_t>(snapshot.snapshot.bids[i].size);
            }
        }
        
        // Aggregate ask levels
        for (uint32_t i = 0; i < snapshot.snapshot.ask_levels_count && i < OrderBookSnapshot100Level::MAX_LEVELS; ++i) {
            if (snapshot.snapshot.asks[i].price > 0) {
                auto& level = aggregated_levels[snapshot.snapshot.asks[i].price];
                level.second += static_cast<uint32_t>(snapshot.snapshot.asks[i].size);
            }
        }
    }
    
    // Convert aggregated levels to SSBO format
    uint32_t level_count = 0;
    for (const auto& [price, quantities] : aggregated_levels) {
        if (level_count >= AggregatedOrderBookSSBO::MAX_LEVELS) {
            break; // Prevent overflow
        }
        
        ssbo_data.levels[level_count].price = static_cast<float>(price);
        ssbo_data.levels[level_count].bid_quantity = quantities.first;
        ssbo_data.levels[level_count].ask_quantity = quantities.second;
        ssbo_data.levels[level_count].num_orders = quantities.first + quantities.second; // Approximation
        
        level_count++;
    }
    
    ssbo_data.price_levels_count = level_count;
    
    // Use current time as the time index (in microseconds)
    auto now = std::chrono::high_resolution_clock::now();
    ssbo_data.current_time_index = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 10000
    );
    
    return ssbo_data;
}

bool SSBOAggregator::renderContextMenu(const ImVec2& position) {
    bool aggregation_triggered = false;
    
    // Create context menu window
    ImGui::SetNextWindowPos(position);
    ImGui::OpenPopup("##SSBO_Aggregation_Menu");
    
    if (ImGui::BeginPopup("##SSBO_Aggregation_Menu")) {
        ImGui::Text("SSBO Order Book Aggregation");
        ImGui::Separator();
        
        if (exchange_snapshots_.empty()) {
            ImGui::Text("No exchange data available");
        } else {
            ImGui::Text("Select exchanges to aggregate:");
            
            for (auto& snapshot : exchange_snapshots_) {
                ImGui::Checkbox(snapshot.name.c_str(), &snapshot.enabled);
            }
            
            ImGui::Separator();
            
            if (ImGui::Button("Aggregate to SSBO")) {
                aggregation_triggered = true;
                ImGui::CloseCurrentPopup();
            }
            
            ImGui::SameLine();
            
            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }
        }
        
        ImGui::EndPopup();
    }
    
    return aggregation_triggered;
}

std::vector<std::string> SSBOAggregator::getAvailableExchanges() const {
    std::vector<std::string> exchanges;
    for (const auto& snapshot : exchange_snapshots_) {
        exchanges.push_back(snapshot.name);
    }
    return exchanges;
}

void SSBOAggregator::clearSnapshots() {
    exchange_snapshots_.clear();
}

bool renderSSBOAggregationContextMenu(SSBOAggregator* aggregator, const ImVec2& position) {
    if (!aggregator) {
        return false;
    }
    
    return aggregator->renderContextMenu(position);
}

} // namespace UI
} // namespace BTQuant