#include "components/time_statistics_panel.hpp"
#include <imgui.h>
#include <imgui_internal.h>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <cmath>

namespace BTQuant {

TimeStatisticsPanel::TimeStatisticsPanel(const PanelConfig& config) : PanelBase(config) {
    // Initialize column list with default visibility
    m_columns.push_back({"Time", true});
    m_columns.push_back({"Open", true});
    m_columns.push_back({"High", true});
    m_columns.push_back({"Low", true});
    m_columns.push_back({"Close", true});
    m_columns.push_back({"Volume", true});
    m_columns.push_back({"BuyVolume", false});  // Not available in OHLCVCandle
    m_columns.push_back({"SellVolume", false}); // Not available in OHLCVCandle
    m_columns.push_back({"Delta", true});       // Now available - calculated as close - previous close
    m_columns.push_back({"Trades", true});
    m_columns.push_back({"AvgSize", false});    // Not available in OHLCVCandle
    m_columns.push_back({"MaxTrade", false});   // Not available in OHLCVCandle
}

void TimeStatisticsPanel::sortDataByColumn(int columnIndex) {
    if (columnIndex < 0 || columnIndex >= static_cast<int>(m_columns.size())) {
        return;
    }

    // Toggle sort direction if clicking the same column
    if (m_sortColumnIndex == columnIndex) {
        m_isSortAscending = !m_isSortAscending;
    } else {
        m_sortColumnIndex = columnIndex;
        m_isSortAscending = true;
    }

    // For delta column, we'll use a special approach
    // Deltas should be calculated based on the original temporal sequence
    // But for display purposes, we can sort by the calculated delta values
    if (columnIndex == 8) { // Delta column
        // Create pairs of (index, delta_value) to track original delta values
        std::vector<std::pair<size_t, double>> index_delta_pairs;
        for (size_t i = 0; i < m_data.size(); ++i) {
            double delta_value = (i < m_deltas.size()) ? m_deltas[i] : 0.0;
            index_delta_pairs.emplace_back(i, delta_value);
        }

        // Sort by delta value
        std::sort(index_delta_pairs.begin(), index_delta_pairs.end(),
                  [this](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                      bool result = a.second < b.second; // Compare delta values
                      return m_isSortAscending ? result : !result;
                  });

        // Reorder m_data based on sorted indices
        std::vector<BTQuant::RenderEngine::OHLCVCandle> sorted_data;
        std::vector<double> sorted_deltas;
        for (const auto& pair : index_delta_pairs) {
            size_t original_index = pair.first;
            sorted_data.push_back(m_data[original_index]);
            if (original_index < m_deltas.size()) {
                sorted_deltas.push_back(m_deltas[original_index]);
            } else {
                sorted_deltas.push_back(0.0);
            }
        }

        m_data = sorted_data;
        m_deltas = sorted_deltas;
    } else {
        // Original sorting logic for other columns
        std::sort(m_data.begin(), m_data.end(), [this, columnIndex](const BTQuant::RenderEngine::OHLCVCandle& a, const BTQuant::RenderEngine::OHLCVCandle& b) {
            bool result = false;

            switch (columnIndex) {
                case 0: // Time
                    result = a.timestamp < b.timestamp;
                    break;
                case 1: // Open
                    result = a.open < b.open;
                    break;
                case 2: // High
                    result = a.high < b.high;
                    break;
                case 3: // Low
                    result = a.low < b.low;
                    break;
                case 4: // Close
                    result = a.close < b.close;
                    break;
                case 5: // Volume
                    result = a.volume < b.volume;
                    break;
                case 6: // BuyVolume (placeholder)
                    result = false; // Always false since it's a placeholder
                    break;
                case 7: // SellVolume (placeholder)
                    result = false; // Always false since it's a placeholder
                    break;
                case 8: // Delta - Handled above
                    result = false;
                    break;
                case 9: // Trades
                    result = a.trade_count < b.trade_count;
                    break;
                case 10: // AvgSize (placeholder)
                    result = false; // Always false since it's a placeholder
                    break;
                case 11: // MaxTrade (placeholder)
                    result = false; // Always false since it's a placeholder
                    break;
                default:
                    result = false;
                    break;
            }

            return m_isSortAscending ? result : !result;
        });

        // After sorting by other columns, we need to recalculate the delta statistics
        // but not the deltas themselves since they depend on the original sequence
        calculateDeltaStatistics();
    }
}

void TimeStatisticsPanel::render_content() {
    if (!config_.visible) return;

    ImGui::SetNextWindowSize(ImVec2(1000, 600), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(config_.title.c_str(), &config_.visible)) {

        // Column selector button
        if (ImGui::Button("Select Columns")) {
            ImGui::OpenPopup("Column Selector");
        }

        ImGui::SameLine();

        // Reset to defaults button
        if (ImGui::Button("Reset Columns")) {
            for (size_t i = 0; i < m_columns.size(); ++i) {
                if (i == 0 || i == 1 || i == 2 || i == 3 || i == 4 || i == 5 || i == 8 || i == 9) {
                    // Time, Open, High, Low, Close, Volume, Delta, Trades - show by default
                    m_columns[i].second = true;
                } else {
                    // BuyVolume, SellVolume, AvgSize, MaxTrade - hide by default
                    m_columns[i].second = false;
                }
            }
        }

        // Render column selection popup
        renderColumnSelectionPopup();

        // Count visible columns
        int visibleColCount = 0;
        for (const auto& col : m_columns) {
            if (col.second) visibleColCount++;
        }

        // Begin table only if there are visible columns
        if (visibleColCount > 0 && ImGui::BeginTable("TimeStatisticsTable", visibleColCount,
                             ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY |
                             ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame |
                             ImGuiTableFlags_Sortable)) {

            // Store the scroll position before rendering to detect changes
            float current_scroll_y = ImGui::GetScrollY();
            float scroll_max_y = ImGui::GetScrollMaxY();

            // Setup only visible columns
            int currentIndex = 0;
            int visibleIndex = 0; // Track the visible column index for mapping back to original
            for (size_t i = 0; i < m_columns.size(); ++i) {
                const auto& col = m_columns[i];
                if (col.second) { // If column is visible
                    ImGuiTableColumnFlags flags = ImGuiTableColumnFlags_None;

                    // Check if this is the currently sorted column
                    if (static_cast<int>(i) == m_sortColumnIndex) {
                        flags |= ImGuiTableColumnFlags_DefaultSort;
                    }

                    ImGui::TableSetupColumn(col.first.c_str(), flags, 0.0f);
                    currentIndex++;
                }
            }

            ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
            ImGui::TableHeadersRow();

            // Handle header clicks for sorting
            ImGuiTableSortSpecs* sorts_specs = ImGui::TableGetSortSpecs();
            if (sorts_specs && sorts_specs->SpecsDirty) {
                if (sorts_specs->SpecsCount > 0) {
                    const ImGuiTableColumnSortSpecs* sort_spec = &sorts_specs->Specs[0];

                    // Find which original column this corresponds to
                    int original_col_index = -1;
                    int visible_col_index = 0;
                    for (size_t i = 0; i < m_columns.size(); ++i) {
                        if (m_columns[i].second) { // If column is visible
                            if (visible_col_index == sort_spec->ColumnIndex) {
                                original_col_index = static_cast<int>(i);
                                break;
                            }
                            visible_col_index++;
                        }
                    }

                    if (original_col_index != -1) {
                        sortDataByColumn(original_col_index);
                    }
                }
                sorts_specs->SpecsDirty = false;
            }

            // Calculate visible range based on scroll position
            int visible_start_row = 0;
            int visible_end_row = static_cast<int>(m_data.size()) - 1;

            if (!m_data.empty() && scroll_max_y > 0) {
                // Estimate which rows are visible based on scroll position
                float row_height = ImGui::GetTextLineHeightWithSpacing();
                int total_rows = static_cast<int>(m_data.size());

                if (row_height > 0) {
                    int rows_in_view = static_cast<int>(ImGui::GetContentRegionAvail().y / row_height);

                    // Calculate the approximate start row based on scroll position
                    visible_start_row = static_cast<int>((current_scroll_y / scroll_max_y) * (total_rows - rows_in_view));
                    if (visible_start_row < 0) visible_start_row = 0;

                    visible_end_row = visible_start_row + rows_in_view;
                    if (visible_end_row >= total_rows) visible_end_row = total_rows - 1;
                }
            }

            // Detect scroll changes to synchronize with chart
            if (last_table_scroll_y_ != current_scroll_y && on_scroll_sync_ && !m_data.empty()) {
                // Calculate the time range of the visible rows
                if (visible_start_row < static_cast<int>(m_data.size()) &&
                    visible_end_row < static_cast<int>(m_data.size())) {

                    uint64_t start_timestamp = m_data[visible_start_row].timestamp;
                    uint64_t end_timestamp = m_data[visible_end_row].timestamp;

                    // Ensure correct order (earlier timestamp first)
                    if (start_timestamp > end_timestamp) {
                        std::swap(start_timestamp, end_timestamp);
                    }

                    // Call the scroll sync callback
                    on_scroll_sync_(start_timestamp, end_timestamp);
                }
            }

            // Update the stored scroll position
            last_table_scroll_y_ = current_scroll_y;

            // Perform synchronized scrolling if needed
            if (need_to_scroll_for_sync_ && m_data.size() > 0) {
                // Calculate the scroll position to bring the target row into view
                float row_height = ImGui::GetTextLineHeightWithSpacing();
                float table_height = ImGui::GetContentRegionAvail().y;
                int rows_in_view = static_cast<int>(table_height / row_height);

                // Calculate the scroll offset needed to center the target row
                float target_offset = static_cast<float>(target_scroll_row_for_sync_) * row_height;

                // Set the scroll position to center the target row
                ImGui::SetScrollFromPosY(ImGui::GetCursorStartPos().y + target_offset - (table_height / 2.0f), 0.5f);

                // Reset the sync flag
                need_to_scroll_for_sync_ = false;
            }

            // Render data rows
            for (size_t rowIndex = 0; rowIndex < m_data.size(); ++rowIndex) {
                const auto& entry = m_data[rowIndex];

                ImGui::TableNextRow();

                // Create a unique ID for this row to properly handle selection
                ImGui::PushID(static_cast<int>(rowIndex));

                int tableColIndex = 0; // Index for visible columns in the table

                // Time column - we'll use a selectable to enable hover highlighting and double-click detection
                if (m_columns[0].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);

                    // Convert timestamp from microseconds to time
                    time_t time_sec = entry.timestamp / 1000000; // Convert microseconds to seconds
                    std::tm* tm_info = std::localtime(&time_sec);
                    char buffer[20]; // Buffer for HH:MM:SS format
                    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", tm_info);

                    // Use Selectable to enable hover effects and double-click detection
                    bool is_selected = false; // We don't actually maintain selection state
                    if (ImGui::Selectable(buffer, is_selected,
                                          ImGuiSelectableFlags_SpanAllColumns |
                                          ImGuiSelectableFlags_AllowDoubleClick)) {
                        // Check if this was a double-click
                        if (ImGui::IsMouseDoubleClicked(0) && on_row_double_clicked_) {
                            // Call the callback with the timestamp
                            on_row_double_clicked_(entry.timestamp);
                        }
                    }
                } else {
                    // If time column is hidden, we still need to set the column index for other columns
                    ImGui::TableSetColumnIndex(tableColIndex++);

                    // Still need to handle row selection for other columns
                    // Use an invisible selectable to capture row clicks
                    if (ImGui::Selectable("##invisible", false,
                                          ImGuiSelectableFlags_SpanAllColumns |
                                          ImGuiSelectableFlags_AllowDoubleClick)) {
                        // Check if this was a double-click
                        if (ImGui::IsMouseDoubleClicked(0) && on_row_double_clicked_) {
                            // Call the callback with the timestamp
                            on_row_double_clicked_(entry.timestamp);
                        }
                    }
                }

                // Open column
                if (m_columns[1].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("%.2f", entry.open);
                }

                // High column
                if (m_columns[2].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("%.2f", entry.high);
                }

                // Low column
                if (m_columns[3].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("%.2f", entry.low);
                }

                // Close column
                if (m_columns[4].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("%.2f", entry.close);
                }

                // Volume column
                if (m_columns[5].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("%.0f", entry.volume);
                }

                // BuyVolume column - not available in OHLCVCandle
                if (m_columns[6].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("--"); // Placeholder since not available
                }

                // SellVolume column - not available in OHLCVCandle
                if (m_columns[7].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("--"); // Placeholder since not available
                }

                // Delta column - calculated as close - previous close
                if (m_columns[8].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);

                    if (rowIndex < m_deltas.size()) {
                        double delta = m_deltas[rowIndex];

                        // Determine color based on delta value and standard deviation
                        ImVec4 textColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // Default white

                        // Check if this is an extreme value (> 3 standard deviations)
                        bool isExtreme = (std::abs(delta - m_deltaMean) > 3.0 * m_deltaStdDev && m_deltaStdDev > 0);

                        if (isExtreme) {
                            // Yellow for extreme values
                            textColor = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow
                        } else if (delta > 0) {
                            // Green for positive delta
                            textColor = ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green
                        } else if (delta < 0) {
                            // Red for negative delta
                            textColor = ImVec4(1.0f, 0.0f, 0.0f, 1.0f); // Red
                        }

                        ImGui::TextColored(textColor, "%.2f", delta);
                    } else {
                        ImGui::Text("--"); // Placeholder if no delta data
                    }
                }

                // Trades column
                if (m_columns[9].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("%llu", static_cast<unsigned long long>(entry.trade_count));
                }

                // AvgSize column - not available in OHLCVCandle
                if (m_columns[10].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("--"); // Placeholder since not available
                }

                // MaxTrade column - not available in OHLCVCandle
                if (m_columns[11].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("--"); // Placeholder since not available
                }

                // Pop the ID we pushed earlier
                ImGui::PopID();
            }

            ImGui::EndTable();
        } else if (visibleColCount == 0) {
            ImGui::Text("No columns visible. Please select at least one column.");
        }
    }
    ImGui::End();
}

void TimeStatisticsPanel::updateData(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& data) {
    m_data = data;

    // Calculate delta values and statistics first
    calculateDeltas();
    calculateDeltaStatistics();

    // Then re-sort data if we had a previous sort applied
    if (m_sortColumnIndex != -1) {
        sortDataByColumn(m_sortColumnIndex);
    }
}

void TimeStatisticsPanel::calculateDeltas() {
    if (m_data.empty()) {
        m_deltas.clear();
        return;
    }

    m_deltas.resize(m_data.size());

    // First element has no previous value, so delta is 0
    m_deltas[0] = 0.0;

    // Calculate delta as current close - previous close
    for (size_t i = 1; i < m_data.size(); ++i) {
        m_deltas[i] = m_data[i].close - m_data[i-1].close;
    }
}

void TimeStatisticsPanel::calculateDeltaStatistics() {
    if (m_deltas.empty()) {
        m_deltaMean = 0.0;
        m_deltaStdDev = 0.0;
        return;
    }

    // Calculate mean
    double sum = 0.0;
    for (double delta : m_deltas) {
        sum += delta;
    }
    m_deltaMean = sum / m_deltas.size();

    // Calculate standard deviation
    double sumSquaredDiff = 0.0;
    for (double delta : m_deltas) {
        double diff = delta - m_deltaMean;
        sumSquaredDiff += diff * diff;
    }
    double variance = sumSquaredDiff / m_deltas.size();
    m_deltaStdDev = std::sqrt(variance);
}

void TimeStatisticsPanel::renderColumnSelectionPopup() {
    if (ImGui::BeginPopup("Column Selector")) {
        ImGui::Text("Toggle column visibility:");
        ImGui::Separator();

        for (auto& col : m_columns) {
            ImGui::Checkbox(col.first.c_str(), &col.second);
        }

        ImGui::Separator();
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
}

std::pair<uint64_t, uint64_t> TimeStatisticsPanel::get_visible_time_range() const {
    if (m_data.empty()) {
        return {0, 0};
    }

    // Calculate which rows are currently visible in the table based on scroll position
    // This method is called externally, so we can't access the current scroll position directly
    // We'll return the full range of data for now, but in a real implementation
    // this would need to be calculated differently

    uint64_t start_time = m_data.front().timestamp;
    uint64_t end_time = m_data.back().timestamp;

    // If sorted in descending order, swap them
    if (start_time > end_time) {
        std::swap(start_time, end_time);
    }

    return {start_time, end_time};
}

void TimeStatisticsPanel::scroll_to_time_range(uint64_t start_timestamp, uint64_t end_timestamp) {
    if (m_data.empty()) {
        return;
    }

    // Find the indices that correspond to the requested time range
    // We'll use binary search to find the appropriate rows

    // First, ensure the timestamps are in the correct order
    uint64_t min_timestamp = std::min(start_timestamp, end_timestamp);
    uint64_t max_timestamp = std::max(start_timestamp, end_timestamp);

    // Find the first row with timestamp >= min_timestamp
    auto lower_it = std::lower_bound(m_data.begin(), m_data.end(), min_timestamp,
        [](const BTQuant::RenderEngine::OHLCVCandle& candle, uint64_t ts) {
            return candle.timestamp < ts;
        });

    // Find the first row with timestamp > max_timestamp
    auto upper_it = std::upper_bound(m_data.begin(), m_data.end(), max_timestamp,
        [](uint64_t ts, const BTQuant::RenderEngine::OHLCVCandle& candle) {
            return ts < candle.timestamp;
        });

    // Calculate the middle row to scroll to
    size_t start_idx = std::distance(m_data.begin(), lower_it);
    size_t end_idx = std::distance(m_data.begin(), upper_it);

    if (start_idx >= m_data.size()) start_idx = m_data.size() - 1;
    if (end_idx > m_data.size()) end_idx = m_data.size();

    // Calculate the middle row to center the view on
    size_t target_row = (start_idx + end_idx) / 2;
    if (target_row >= m_data.size()) target_row = m_data.size() - 1;

    // Store the target row to scroll to during the next render
    target_scroll_row_for_sync_ = target_row;
    need_to_scroll_for_sync_ = true;
}

} // namespace BTQuant