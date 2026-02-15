#pragma once

#include "panel_base.hpp"
#include "lockfreesnapshotpipeline.h"
#include "ui/compute_to_imgui_bind.h"
#include <memory>

namespace BTQuant {

class MarketDepthTablePanel : public PanelBase {
public:
    explicit MarketDepthTablePanel(const PanelConfig& config);

    void render() override;

    // Method to set the snapshot pipeline for market data
    void setSnapshotPipeline(const std::shared_ptr<BTQuant::RenderEngine::LockFreeSnapshotPipeline>& pipeline);

    // Method to set the symbol index to display
    void setSymbolIndex(uint32_t symbol_index);

private:
    std::unique_ptr<BTQuant::UI::ComputeToImGuiBind> compute_binding_;
    std::shared_ptr<BTQuant::RenderEngine::LockFreeSnapshotPipeline> snapshot_pipeline_;
    uint32_t symbol_index_ = 0;
};

} // namespace BTQuant