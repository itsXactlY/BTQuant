#include "../../include/components/chart_replay_panel.hpp"

#include "imgui.h"

namespace BTQuant {

// DEPRECATED - Legacy hotspine
ChartReplayPanel::ChartReplayPanel(const PanelConfig& config, 
                                   std::shared_ptr<HotSpineDataBridge> bridge,
                                   std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                                   ChartManager* chart_manager)
    : PanelBase(config), bridge_(bridge), processor_(processor), chart_manager_(chart_manager) {
    // Initialize the chart replay component
    chart_replay_ = std::make_unique<ChartReplay>(bridge_, processor_, chart_manager_);
}

void ChartReplayPanel::update(float dt) {
    // Nothing to update in this panel - the chart replay handles its own timing
    (void)dt;
}

void ChartReplayPanel::render() {
    begin_panel_window();

    if (!is_visible()) {
        end_panel_window();
        return;
    }

    ImGui::Text("Chart Replay Mode");
    ImGui::Separator();

    // Render the replay controls
    if (chart_replay_) {
        chart_replay_->render_replay_controls();
    }

    end_panel_window();
}

void ChartReplayPanel::initialize() {
    // Initialization is handled in constructor
}

void ChartReplayPanel::set_timeframe(RenderEngine::TimeFrame timeframe) {
    if (chart_replay_) {
        // Get the current config
        auto config = chart_replay_->get_replay_config();

        // Update the timeframe in the config
        config.timeframe = timeframe;

        // Set the updated config back
        chart_replay_->set_replay_config(config);
    }
}

} // namespace BTQuant