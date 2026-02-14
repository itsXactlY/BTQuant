#include "../../include/components/correlation_heatmap_panel.hpp"
#include "imgui.h"
#include <cmath>

namespace BTQuant {

CorrelationHeatmapPanel::CorrelationHeatmapPanel(const PanelConfig& config)
    : PanelBase(config) {
    // Initialize with some default symbols
    symbols_ = {"BTC-USDT", "ETH-USDT", "SOL-USDT"};
}

void CorrelationHeatmapPanel::initialize() {
    // Stub implementation - no initialization needed
}

void CorrelationHeatmapPanel::render() {
    begin_panel_window();
    
    if (!is_visible()) {
        end_panel_window();
        return;
    }
    
    render_controls();
    ImGui::Separator();
    render_symbol_selector();
    ImGui::Separator();
    render_heatmap();
    
    end_panel_window();
}

void CorrelationHeatmapPanel::render_controls() {
    ImGui::Text("Correlation Analysis");
    
    // Timeframe selector
    const char* timeframes[] = {"1 Day", "1 Week", "1 Month"};
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("Timeframe", &selected_timeframe_, timeframes, IM_ARRAYSIZE(timeframes));
    
    ImGui::SameLine();
    
    // Correlation method selector
    const char* methods[] = {"Pearson", "Spearman", "Kendall"};
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("Method", &correlation_method_, methods, IM_ARRAYSIZE(methods));
}

void CorrelationHeatmapPanel::render_symbol_selector() {
    ImGui::Text("Symbols (%zu)", symbols_.size());
    
    // Add symbol input
    static char new_symbol[32] = "";
    ImGui::SetNextItemWidth(150);
    ImGui::InputText("##NewSymbol", new_symbol, sizeof(new_symbol));
    ImGui::SameLine();
    if (ImGui::Button("Add")) {
        if (new_symbol[0] != '\0') {
            add_symbol(std::string(new_symbol));
            new_symbol[0] = '\0';
        }
    }
    
    // Symbol list
    if (ImGui::BeginChild("SymbolList", ImVec2(0, 100), true)) {
        for (size_t i = 0; i < symbols_.size(); ++i) {
            ImGui::Text("%s", symbols_[i].c_str());
            ImGui::SameLine(ImGui::GetWindowWidth() - 50);
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::SmallButton("X")) {
                remove_symbol(symbols_[i]);
            }
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
}

void CorrelationHeatmapPanel::render_heatmap() {
    ImGui::Text("Correlation Matrix");
    
    // Stub: Draw a placeholder heatmap
    if (symbols_.size() < 2) {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Add at least 2 symbols to view correlation");
        return;
    }
    
    // Create a simple table as placeholder
    if (ImGui::BeginTable("CorrelationTable", static_cast<int>(symbols_.size() + 1), 
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        // Header row
        ImGui::TableSetupColumn("");
        for (const auto& symbol : symbols_) {
            ImGui::TableSetupColumn(symbol.c_str());
        }
        ImGui::TableHeadersRow();
        
        // Data rows (stub: random correlation values)
        for (size_t i = 0; i < symbols_.size(); ++i) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", symbols_[i].c_str());
            
            for (size_t j = 0; j < symbols_.size(); ++j) {
                ImGui::TableSetColumnIndex(static_cast<int>(j + 1));
                
                if (i == j) {
                    ImGui::TextColored(ImVec4(1, 1, 1, 1), "1.00");
                } else {
                    // Stub: Generate pseudo-random correlation
                    float corr = 0.5f + 0.3f * sinf(static_cast<float>(i * j));
                    ImVec4 color = corr > 0.5f ? ImVec4(0, 1, 0, 1) : ImVec4(1, 0, 0, 1);
                    ImGui::TextColored(color, "%.2f", corr);
                }
            }
        }
        ImGui::EndTable();
    }
    
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "[Stub] Real correlation data will be calculated from market data");
}

void CorrelationHeatmapPanel::add_symbol(const std::string& symbol) {
    if (std::find(symbols_.begin(), symbols_.end(), symbol) == symbols_.end()) {
        symbols_.push_back(symbol);
    }
}

void CorrelationHeatmapPanel::remove_symbol(const std::string& symbol) {
    symbols_.erase(std::remove(symbols_.begin(), symbols_.end(), symbol), symbols_.end());
}

void CorrelationHeatmapPanel::clear_symbols() {
    symbols_.clear();
}

}  // namespace BTQuant
