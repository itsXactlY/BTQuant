#pragma once

#include <memory>

#include "components/panel_base.hpp"
#include "data/core_types.hpp"

// Forward declaration
struct ImDrawList;
struct ImVec2;

namespace BTQuant {
class TPOEngine;
struct TPOBar;
}

namespace BTQuant {

/**
 * @brief TPO Panel - Visualizes market profile and time-price opportunities
 *
 * Displays TPO (Time Price Opportunity) characters for each price level,
 * showing the distribution of trading activity across time brackets (A-P).
 */
class TPOPanel : public PanelBase {
public:
    explicit TPOPanel(const PanelConfig& config,
                      std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                      std::shared_ptr<TPOEngine> tpo_engine);

    void render_content() override;

    void set_tpo_engine(std::shared_ptr<TPOEngine> engine);

private:
    /**
     * @brief Render detailed view with TPO characters
     */
    void render_detailed_view(ImDrawList* draw_list, ImVec2 canvas_pos, ImVec2 canvas_size,
                             const std::vector<TPOBar>& profile, size_t va_low, size_t va_high);

    /**
     * @brief Render condensed view when cells are too small
     */
    void render_condensed_view(ImDrawList* draw_list, ImVec2 canvas_pos, ImVec2 canvas_size,
                              const std::vector<TPOBar>& profile, size_t va_low, size_t va_high);

    /**
     * @brief Convert bracket index to character (A-P)
     */
    char bracket_to_char(uint8_t bracket);

private:
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    std::shared_ptr<TPOEngine> tpo_engine_;
};

}  // namespace BTQuant