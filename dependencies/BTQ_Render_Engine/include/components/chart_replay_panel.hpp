#pragma once

#include <memory>
#include <string>

#include "panel_base.hpp"
#include "chart_replay.hpp"
#include "chart_manager.hpp"
#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"

namespace BTQuant {

class ChartReplayPanel : public PanelBase {
public:
    // DEPRECATED - Legacy hotspine
    ChartReplayPanel(const PanelConfig& config,
                     std::shared_ptr<HotSpineDataBridge> bridge,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                     ChartManager* chart_manager);

    void update(float dt) override;
    void render() override;
    void initialize() override;

    // Method to update the replay timeframe
    void set_timeframe(RenderEngine::TimeFrame timeframe);

private:
    // DEPRECATED - Legacy hotspine
    std::shared_ptr<HotSpineDataBridge> bridge_;
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    ChartManager* chart_manager_;

    std::unique_ptr<ChartReplay> chart_replay_;
};

} // namespace BTQuant