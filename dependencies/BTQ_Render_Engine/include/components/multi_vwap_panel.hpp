#pragma once

#include "panel_base.hpp"
#include <vector>
#include <string>

namespace BTQuant {

/**
 * MultiVWAPPanel - Displays multiple VWAP lines on a chart
 * 
 * Features (Stub):
 * - Multiple VWAP periods (Session, Weekly, Monthly, Custom)
 * - Standard deviation bands
 * - Anchor point selection
 * - Color customization per VWAP
 */
class MultiVWAPPanel : public PanelBase {
public:
    explicit MultiVWAPPanel(const PanelConfig& config);
    ~MultiVWAPPanel() override = default;

    void initialize() override;
    void render_content() override;

    // VWAP configuration
    void add_vwap(const std::string& period, const ImVec4& color);
    void remove_vwap(int index);
    void clear_vwaps();

private:
    struct VWAPConfig {
        std::string period;
        ImVec4 color;
        bool show_sd_bands = true;
        int sd_levels = 3;
    };
    
    std::vector<VWAPConfig> vwaps_;
    int selected_period_ = 0;  // 0: Session, 1: Weekly, 2: Monthly, 3: Custom
    
    void render_vwap_list();
    void render_vwap_settings();
    void render_chart_overlay();
};

}  // namespace BTQuant
