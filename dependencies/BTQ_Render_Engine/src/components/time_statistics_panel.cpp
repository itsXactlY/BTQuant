#include "components/time_statistics_panel.hpp"
#include <imgui.h>
#include <imgui_internal.h>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>

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
    m_columns.push_back({"Delta", false});      // Not available in OHLCVCandle
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
            case 8: // Delta (placeholder)
                result = false; // Always false since it's a placeholder
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
}

void TimeStatisticsPanel::render() {
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
                if (i == 0 || i == 1 || i == 2 || i == 3 || i == 4 || i == 5 || i == 9) {
                    // Time, Open, High, Low, Close, Volume, Trades - show by default
                    m_columns[i].second = true;
                } else {
                    // BuyVolume, SellVolume, Delta, AvgSize, MaxTrade - hide by default
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

                // Delta column - not available in OHLCVCandle
                if (m_columns[8].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    ImGui::Text("--"); // Placeholder since not available
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

    // Re-sort data if we had a previous sort applied
    if (m_sortColumnIndex != -1) {
        sortDataByColumn(m_sortColumnIndex);
    }
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

} // namespace BTQuant