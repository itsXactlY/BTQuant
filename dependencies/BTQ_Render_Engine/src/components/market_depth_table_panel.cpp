#include "components/market_depth_table_panel.hpp"
#include <imgui.h>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <cmath>

namespace BTQuant {

MarketDepthTablePanel::MarketDepthTablePanel(const PanelConfig& config) : PanelBase(config) {
    // Simplified implementation without compute_binding
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
            // Simplified market depth visualization
            renderMarketDepthTable();
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

void MarketDepthTablePanel::renderMarketDepthTable() {
    if (!snapshot_pipeline_) return;
    
    // Get the latest snapshot using the correct API
    RenderEngine::AtomicMarketData data;
    if (!snapshot_pipeline_->read_market_data_snapshot(symbol_index_, data)) {
        ImGui::Text("No data available for symbol %u", symbol_index_);
        return;
    }
    
    // Read values from atomic storage
    double price = data.price.load();
    double volume = data.volume.load();
    double bid_price = data.bid_price.load();
    double ask_price = data.ask_price.load();
    double bid_volume = data.bid_volume.load();
    double ask_volume = data.ask_volume.load();
    double mid_price = (bid_price + ask_price) / 2.0;
    double spread = ask_price - bid_price;
    double spread_percent = (bid_price > 0) ? (spread / bid_price) * 100.0 : 0.0;
    
    // Display basic market info
    ImGui::Text("Symbol: %u | Mid Price: %.2f", symbol_index_, mid_price);
    ImGui::Text("Spread: %.4f (%.4f%%)", spread, spread_percent);
    ImGui::Separator();
    
    // Create a table for market depth
    if (ImGui::BeginTable("MarketDepth", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Buys", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Asks", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Bids", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Sells", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        
        // Display top of book
        ImGui::TableNextRow();
        
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%.2f", bid_volume);
        
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.2f", ask_volume);
        
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.2f", mid_price);
        
        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.2f", bid_price);
        
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%.2f", ask_price);
        
        ImGui::EndTable();
    }
    
    // Display additional info
    ImGui::Text("Last Price: %.2f | Volume: %.2f", price, volume);
}

void MarketDepthTablePanel::setSnapshotPipeline(const std::shared_ptr<BTQuant::RenderEngine::LockFreeSnapshotPipeline>& pipeline) {
    snapshot_pipeline_ = pipeline;
}

void MarketDepthTablePanel::setSymbolIndex(uint32_t symbol_index) {
    symbol_index_ = symbol_index;
}

} // namespace BTQuant