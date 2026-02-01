#include "components/time_statistics_panel.hpp"
#include <imgui.h>
#include <imgui_internal.h>
#include <sstream>
#include <iomanip>
#include <ctime>

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
                             ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame)) {

            // Setup only visible columns
            int currentIndex = 0;
            for (const auto& col : m_columns) {
                if (col.second) { // If column is visible
                    ImGui::TableSetupColumn(col.first.c_str(), ImGuiTableColumnFlags_None, 0.0f);
                    currentIndex++;
                }
            }

            ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
            ImGui::TableHeadersRow();

            // Render data rows
            for (const auto& entry : m_data) {
                ImGui::TableNextRow();

                int tableColIndex = 0; // Index for visible columns in the table

                // Time column
                if (m_columns[0].second) {
                    ImGui::TableSetColumnIndex(tableColIndex++);
                    // Convert timestamp from microseconds to time
                    time_t time_sec = entry.timestamp / 1000000; // Convert microseconds to seconds
                    std::tm* tm_info = std::localtime(&time_sec);
                    char buffer[20]; // Buffer for HH:MM:SS format
                    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", tm_info);
                    ImGui::Text("%s", buffer);
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