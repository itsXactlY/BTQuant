#pragma once

#include "panel_base.hpp"
#include "market_data_processor.hpp"
#include <imgui.h>
#include <vector>
#include <string>
#include <map>

namespace BTQuant {

class TimeStatisticsPanel : public PanelBase {
public:
    TimeStatisticsPanel(const PanelConfig& config);
    ~TimeStatisticsPanel() override = default;

    void render() override;
    void updateData(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& data);

private:
    void renderColumnSelectionPopup();
    void setupTableColumns();
    void sortDataByColumn(int columnIndex);

    // Data storage
    std::vector<BTQuant::RenderEngine::OHLCVCandle> m_data;

    // Column visibility flags
    struct ColumnVisibility {
        bool time = true;
        bool open = true;
        bool high = true;
        bool low = true;
        bool close = true;
        bool volume = true;
        bool buyVolume = false;  // OHLCVCandle doesn't have buyVolume directly
        bool sellVolume = false; // OHLCVCandle doesn't have sellVolume directly
        bool delta = false;      // OHLCVCandle doesn't have delta directly
        bool trades = true;
        bool avgSize = false;    // OHLCVCandle doesn't have avgSize directly
        bool maxTrade = false;   // OHLCVCandle doesn't have maxTrade directly
    } m_columnVisibility;

    // Column ordering and visibility
    std::vector<std::pair<std::string, bool>> m_columns;

    // UI state
    bool m_showColumnSelector = false;
    ImVec2 m_columnSelectorPos = ImVec2(0, 0);

    // Sorting state
    int m_sortColumnIndex = -1;
    bool m_isSortAscending = true;
};

} // namespace BTQuant