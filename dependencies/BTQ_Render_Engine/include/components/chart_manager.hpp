#pragma once

#include "market_data_processor.hpp"
#include "hotspine_data_bridge.hpp"
#include <unordered_map>
#include <string>
#include <memory>
#include "imgui.h"

namespace BTQuant {

struct ChartInstance {
    std::string symbol;
    RenderEngine::TimeFrame timeframe;
    uint32_t chart_id;
    bool visible = true;
    bool minimized = false;
    ImVec2 position = {0, 0};
    ImVec2 size = {600, 400};
};

class ChartManager {
public:
    ChartManager(std::shared_ptr<HotSpineDataBridge> bridge,
                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
    
    uint32_t create_chart(const std::string &symbol, RenderEngine::TimeFrame timeframe);
    void destroy_chart(uint32_t chart_id);
    void toggle_chart_visibility(uint32_t chart_id);
    void toggle_chart_minimization(uint32_t chart_id);
    void update_chart_position(uint32_t chart_id, const ImVec2 &position);
    void update_chart_size(uint32_t chart_id, const ImVec2 &size);
    
    const std::unordered_map<uint32_t, ChartInstance> &get_charts() const;
    std::vector<ChartInstance> get_visible_charts() const;
    std::vector<ChartInstance> get_charts_for_symbol(const std::string &symbol) const;
    
    void update();
    
private:
    std::shared_ptr<HotSpineDataBridge> bridge_;
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    std::unordered_map<uint32_t, ChartInstance> charts_;
    uint32_t next_chart_id_ = 0;
};

} // namespace BTQuant
