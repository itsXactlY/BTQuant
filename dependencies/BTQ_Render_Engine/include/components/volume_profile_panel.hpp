#pragma once

#include <memory>

#include "components/panel_base.hpp"
#include "data/core_types.hpp"

namespace BTQuant {
class ClusterEngine;
class CacheManager;
}

namespace BTQuant {

/**
 * @brief Volume Profile Panel - Visualizes volume distribution and CVD
 * 
 * Displays the volume profile showing buy/sell volume distribution across price levels,
 * along with Cumulative Volume Delta (CVD) metrics.
 */
class VolumeProfilePanel : public PanelBase {
public:
    explicit VolumeProfilePanel(const PanelConfig& config,
                             std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                             std::shared_ptr<ClusterEngine> cluster_engine,
                             std::shared_ptr<CacheManager> cache_manager);

    void render_content() override;

    void set_cluster_engine(std::shared_ptr<ClusterEngine> engine);
    void set_cache_manager(std::shared_ptr<CacheManager> cache);

private:
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    std::shared_ptr<ClusterEngine> cluster_engine_;
    std::shared_ptr<CacheManager> cache_manager_;
};

}  // namespace BTQuant