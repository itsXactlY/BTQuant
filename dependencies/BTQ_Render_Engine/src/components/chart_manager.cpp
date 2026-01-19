#include "../../include/components/chart_manager.hpp"
#include "../../include/market_data_processor.hpp"
#include "../../include/hotspine_data_bridge.hpp"
#include <iostream>

namespace BTQuant {

ChartManager::ChartManager(std::shared_ptr<HotSpineDataBridge> bridge,
                           std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : bridge_(bridge), processor_(processor), next_chart_id_(0) {
    std::cout << "[ChartManager] Initialized" << std::endl;
}

uint32_t ChartManager::create_chart(const std::string &symbol, RenderEngine::TimeFrame timeframe) {
    ChartInstance chart;
    chart.symbol = symbol;
    chart.timeframe = timeframe;
    chart.chart_id = next_chart_id_++;
    chart.visible = true;
    chart.minimized = false;
    
    // Calculate initial position based on existing charts
    int chart_count = static_cast<int>(charts_.size());
    int grid_cols = 2;
    float x_offset = (chart_count % grid_cols) * 650.0f + 10.0f;
    float y_offset = (chart_count / grid_cols) * 450.0f + 10.0f;
    chart.position = {x_offset, y_offset};
    
    charts_[chart.chart_id] = chart;
    
    std::cout << "[ChartManager] Created chart " << chart.chart_id 
              << " for " << symbol << " (" << static_cast<int>(timeframe) << ")" << std::endl;
    
    return chart.chart_id;
}

void ChartManager::destroy_chart(uint32_t chart_id) {
    auto it = charts_.find(chart_id);
    if (it != charts_.end()) {
        std::cout << "[ChartManager] Destroyed chart " << chart_id 
                  << " for " << it->second.symbol << std::endl;
        charts_.erase(it);
    }
}

void ChartManager::toggle_chart_visibility(uint32_t chart_id) {
    auto it = charts_.find(chart_id);
    if (it != charts_.end()) {
        it->second.visible = !it->second.visible;
        std::cout << "[ChartManager] Chart " << chart_id << " " 
                  << (it->second.visible ? "visible" : "hidden") << std::endl;
    }
}

void ChartManager::toggle_chart_minimization(uint32_t chart_id) {
    auto it = charts_.find(chart_id);
    if (it != charts_.end()) {
        it->second.minimized = !it->second.minimized;
        std::cout << "[ChartManager] Chart " << chart_id << " " 
                  << (it->second.minimized ? "minimized" : "maximized") << std::endl;
    }
}

void ChartManager::update_chart_position(uint32_t chart_id, const ImVec2 &position) {
    auto it = charts_.find(chart_id);
    if (it != charts_.end()) {
        it->second.position = position;
    }
}

void ChartManager::update_chart_size(uint32_t chart_id, const ImVec2 &size) {
    auto it = charts_.find(chart_id);
    if (it != charts_.end()) {
        it->second.size = size;
    }
}

const std::unordered_map<uint32_t, ChartInstance> &ChartManager::get_charts() const {
    return charts_;
}

std::vector<ChartInstance> ChartManager::get_visible_charts() const {
    std::vector<ChartInstance> visible_charts;
    for (const auto &[id, chart] : charts_) {
        if (chart.visible) {
            visible_charts.push_back(chart);
        }
    }
    return visible_charts;
}

std::vector<ChartInstance> ChartManager::get_charts_for_symbol(const std::string &symbol) const {
    std::vector<ChartInstance> symbol_charts;
    for (const auto &[id, chart] : charts_) {
        if (chart.symbol == symbol) {
            symbol_charts.push_back(chart);
        }
    }
    return symbol_charts;
}

void ChartManager::update() {
    // Currently, just ensure all active instruments have at least one chart
    auto &instruments = bridge_->GetAllInstruments();
    std::lock_guard<std::mutex> lock(bridge_->GetMapMutex());
    
    for (const auto &[symbol, inst] : instruments) {
        // Check if we already have a chart for this symbol
        bool has_chart = false;
        for (const auto &[id, chart] : charts_) {
            if (chart.symbol == symbol) {
                has_chart = true;
                break;
            }
        }
        
        // Create a chart if none exists (default to 1-minute timeframe)
        if (!has_chart) {
            create_chart(symbol, RenderEngine::TimeFrame::TF_1MIN);
        }
    }
}

} // namespace BTQuant
