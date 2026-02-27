#pragma once

#include "panel_base.hpp"
#include "market_data_processor.hpp"
#include <imgui.h>
#include <vector>
#include <string>
#include <map>
#include <functional>

namespace BTQuant {

class ChartPanel; // Forward declaration

class TimeStatisticsPanel : public PanelBase {
public:
    using RowDoubleClickedCallback = std::function<void(uint64_t timestamp)>;
    using ScrollSyncCallback = std::function<void(uint64_t start_timestamp, uint64_t end_timestamp)>;

    TimeStatisticsPanel(const PanelConfig& config);
    ~TimeStatisticsPanel() override = default;

    void render() override;
    void updateData(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& data);

    // Set callback for when a row is double-clicked
    void set_row_double_clicked_callback(RowDoubleClickedCallback callback) {
        on_row_double_clicked_ = std::move(callback);
    }

    // Set callback for scroll synchronization
    void set_scroll_sync_callback(ScrollSyncCallback callback) {
        on_scroll_sync_ = std::move(callback);
    }

    // Set associated chart panel for synchronization
    void set_associated_chart_panel(ChartPanel* chart_panel) {
        associated_chart_panel_ = chart_panel;
    }

    // Get visible time range
    std::pair<uint64_t, uint64_t> get_visible_time_range() const;

    // Scroll to show the specified time range
    void scroll_to_time_range(uint64_t start_timestamp, uint64_t end_timestamp);

private:
    void renderColumnSelectionPopup();
    void setupTableColumns();
    void sortDataByColumn(int columnIndex);
    void calculateDeltas();
    void calculateDeltaStatistics();

    // Data storage
    std::vector<BTQuant::RenderEngine::OHLCVCandle> m_data;

    // Delta values for each candle (calculated as close - previous_close)
    std::vector<double> m_deltas;

    // Statistics for delta values
    double m_deltaMean = 0.0;
    double m_deltaStdDev = 0.0;

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
        bool delta = true;       // Delta is now calculated as close - previous close
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

    // Callback for double-clicked row
    RowDoubleClickedCallback on_row_double_clicked_;

    // Callback for scroll synchronization
    ScrollSyncCallback on_scroll_sync_;

    // Associated chart panel for synchronization
    ChartPanel* associated_chart_panel_ = nullptr;

    // Track scroll position for synchronization
    float last_table_scroll_y_ = 0.0f;

    // Variables for synchronized scrolling
    size_t target_scroll_row_for_sync_ = 0;
    bool need_to_scroll_for_sync_ = false;
};

} // namespace BTQuant