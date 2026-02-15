#include "components/market_depth_table_panel.hpp"
#include <imgui.h>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <cmath>

namespace BTQuant {

MarketDepthTablePanel::MarketDepthTablePanel(const PanelConfig& config) : PanelBase(config) {
    // Initialize the compute-to-imgui binding
    compute_binding_ = std::make_unique<BTQuant::UI::ComputeToImGuiBind>();
}

void MarketDepthTablePanel::render() {
    if (!config_.visible) return;

    ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(config_.title.c_str(), &config_.visible)) {
        
        // Add instructions or title
        ImGui::Text("Market Depth Table: Buys | Asks | Price | Bids | Sells");
        ImGui::Separator();
        
        // Check if we have access to the snapshot pipeline to render the table
        if (snapshot_pipeline_) {
            // Render the market depth table using the function we added to the binding
            BTQuant::UI::visualizeMarketDepthTable(*snapshot_pipeline_, symbol_index_, 800.0f, 500.0f);
        } else {
            ImGui::Text("No market data pipeline available");
            ImGui::Text("Waiting for data connection...");
            
            // Show a simple placeholder table with the column headers
            if (ImGui::BeginTable("MarketDepthPlaceholder", 5, ImGuiTableFlags_Borders)) {
                ImGui::TableSetupColumn("Buys", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Asks", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Bids", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Sells", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableHeadersRow();

                ImGui::TableNextRow();
                
                // Placeholder values
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("--");
                
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("--");
                
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("--");
                
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("--");
                
                ImGui::TableSetColumnIndex(4);
                ImGui::Text("--");

                ImGui::EndTable();
            }
        }
        
    }
    ImGui::End();
}

void MarketDepthTablePanel::setSnapshotPipeline(const std::shared_ptr<BTQuant::RenderEngine::LockFreeSnapshotPipeline>& pipeline) {
    snapshot_pipeline_ = pipeline;
}

void MarketDepthTablePanel::setSymbolIndex(uint32_t symbol_index) {
    symbol_index_ = symbol_index;
}

} // namespace BTQuant