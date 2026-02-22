#pragma once

#include <memory>

#include "components/panel_base.hpp"
#include "data/core_types.hpp"

namespace BTQuant {
class ClusterEngine;
}

namespace BTQuant {

/**
 * @brief Footprint Panel - Visualizes trade clusters and volume imbalances
 * 
 * Displays a heatmap showing buy/sell volume distribution across price levels,
 * with special highlighting for diagonal imbalances and key metrics like CVD.
 */
class FootprintPanel : public PanelBase {
public:
    explicit FootprintPanel(const PanelConfig& config,
                         std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                         std::shared_ptr<ClusterEngine> cluster_engine);

    void render_content() override;

    void set_cluster_engine(std::shared_ptr<ClusterEngine> engine);

private:
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    std::shared_ptr<ClusterEngine> cluster_engine_;
};

}  // namespace BTQuant