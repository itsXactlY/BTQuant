#pragma once

#include "panel_base.hpp"
#include <vector>
#include <string>

namespace BTQuant {

/**
 * CorrelationHeatmapPanel - Displays correlation matrix between multiple symbols
 * 
 * Features (Stub):
 * - Symbol selection for correlation analysis
 * - Timeframe selection
 * - Correlation method selection (Pearson, Spearman, Kendall)
 * - Heatmap visualization with color gradient
 */
class CorrelationHeatmapPanel : public PanelBase {
public:
    explicit CorrelationHeatmapPanel(const PanelConfig& config);
    ~CorrelationHeatmapPanel() override = default;

    void initialize() override;
    void render() override;

    // Symbol management
    void add_symbol(const std::string& symbol);
    void remove_symbol(const std::string& symbol);
    void clear_symbols();

private:
    std::vector<std::string> symbols_;
    int selected_timeframe_ = 0;  // 0: 1D, 1: 1W, 2: 1M
    int correlation_method_ = 0;  // 0: Pearson, 1: Spearman, 2: Kendall
    
    void render_symbol_selector();
    void render_heatmap();
    void render_controls();
};

}  // namespace BTQuant
