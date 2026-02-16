#include "../../include/components/technical_indicators_panel.hpp"
#include "imgui.h"
#include <algorithm>

namespace BTQuant {

TechnicalIndicatorsPanel::TechnicalIndicatorsPanel(const PanelConfig& config)
    : PanelBase(config) {
    // Initialize with some default indicators
    indicators_ = {
        {"RSI", "Relative Strength Index", true, ImVec4(0.8f, 0.2f, 0.2f, 1.0f)},
        {"MACD", "Moving Average Convergence Divergence", true, ImVec4(0.2f, 0.8f, 0.2f, 1.0f)},
        {"EMA", "Exponential Moving Average", true, ImVec4(0.2f, 0.2f, 0.8f, 1.0f)},
        {"BB", "Bollinger Bands", false, ImVec4(0.8f, 0.8f, 0.2f, 1.0f)}
    };
}

void TechnicalIndicatorsPanel::initialize() {
    // Stub implementation
}

void TechnicalIndicatorsPanel::render_content() {
    begin_panel_window();
    
    if (!is_visible()) {
        end_panel_window();
        return;
    }
    
    render_indicator_list();
    ImGui::Separator();
    render_indicator_settings();
    ImGui::Separator();
    render_available_indicators();
    
    end_panel_window();
}

void TechnicalIndicatorsPanel::render_indicator_list() {
    ImGui::Text("Active Indicators (%zu)", indicators_.size());
    
    if (ImGui::BeginChild("IndicatorList", ImVec2(0, 150), true)) {
        for (size_t i = 0; i < indicators_.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            
            // Visibility toggle
            ImGui::Checkbox("##visible", &indicators_[i].visible);
            ImGui::SameLine();
            
            // Color indicator
            ImGui::ColorButton("##color", indicators_[i].color, ImGuiColorEditFlags_NoInputs, ImVec2(20, 20));
            ImGui::SameLine();
            
            // Indicator name and description
            ImGui::Text("%s", indicators_[i].name.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1), "- %s", indicators_[i].description.c_str());
            
            // Remove button
            ImGui::SameLine(ImGui::GetWindowWidth() - 50);
            if (ImGui::SmallButton("X")) {
                remove_indicator(static_cast<int>(i));
            }
            
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
}

void TechnicalIndicatorsPanel::render_indicator_settings() {
    ImGui::Text("Indicator Settings");
    
    if (selected_indicator_ >= 0 && selected_indicator_ < static_cast<int>(indicators_.size())) {
        auto& ind = indicators_[selected_indicator_];
        
        ImGui::Text("Editing: %s", ind.name.c_str());
        
        // Color picker
        ImGui::ColorEdit4("Color", (float*)&ind.color, ImGuiColorEditFlags_NoInputs);
        
        // Visibility toggle
        ImGui::Checkbox("Visible", &ind.visible);
        
        // Stub: Add indicator-specific parameters
        if (ind.name == "RSI") {
            static int rsi_period = 14;
            ImGui::SliderInt("Period", &rsi_period, 5, 30);
            static float rsi_overbought = 70.0f;
            static float rsi_oversold = 30.0f;
            ImGui::SliderFloat("Overbought", &rsi_overbought, 60.0f, 90.0f);
            ImGui::SliderFloat("Oversold", &rsi_oversold, 10.0f, 40.0f);
        } else if (ind.name == "MACD") {
            static int fast = 12;
            static int slow = 26;
            static int signal = 9;
            ImGui::SliderInt("Fast Period", &fast, 5, 20);
            ImGui::SliderInt("Slow Period", &slow, 15, 40);
            ImGui::SliderInt("Signal Period", &signal, 5, 15);
        } else if (ind.name == "EMA") {
            static int ema_period = 20;
            ImGui::SliderInt("Period", &ema_period, 5, 100);
        } else if (ind.name == "BB") {
            static int bb_period = 20;
            static float bb_std = 2.0f;
            ImGui::SliderInt("Period", &bb_period, 5, 50);
            ImGui::SliderFloat("Std Dev", &bb_std, 0.5f, 3.0f);
        }
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "Select an indicator to edit settings");
    }
}

void TechnicalIndicatorsPanel::render_available_indicators() {
    ImGui::Text("Add Indicator");
    
    // Search filter
    static char search_buffer[64] = "";
    ImGui::SetNextItemWidth(200);
    ImGui::InputText("##Search", search_buffer, sizeof(search_buffer));
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        search_buffer[0] = '\0';
    }
    
    // Available indicators list
    const char* available[] = {
        "RSI", "MACD", "EMA", "SMA", "BB", "ATR", "ADX", "CCI", 
        "Stochastic", "Williams %R", "MFI", "OBV", "VWAP", "Ichimoku"
    };
    
    if (ImGui::BeginChild("AvailableIndicators", ImVec2(0, 100), true)) {
        for (const char* ind : available) {
            std::string ind_str = ind;
            std::string search = search_buffer;
            
            // Filter by search
            if (search.empty() || 
                ind_str.find(search) != std::string::npos) {
                
                // Check if already added
                bool already_added = false;
                for (const auto& existing : indicators_) {
                    if (existing.name == ind_str) {
                        already_added = true;
                        break;
                    }
                }
                
                if (!already_added) {
                    if (ImGui::Selectable(ind)) {
                        add_indicator(ind_str, "");
                    }
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "%s (added)", ind);
                }
            }
        }
    }
    ImGui::EndChild();
    
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "[Stub] Real indicator calculations will be implemented");
}

void TechnicalIndicatorsPanel::add_indicator(const std::string& name, const std::string& description) {
    // Generate a random-ish color based on name hash
    size_t hash = std::hash<std::string>{}(name);
    float r = (hash & 0xFF) / 255.0f;
    float g = ((hash >> 8) & 0xFF) / 255.0f;
    float b = ((hash >> 16) & 0xFF) / 255.0f;
    
    indicators_.push_back({name, description.empty() ? name : description, true, ImVec4(r, g, b, 1.0f)});
}

void TechnicalIndicatorsPanel::remove_indicator(int index) {
    if (index >= 0 && index < static_cast<int>(indicators_.size())) {
        indicators_.erase(indicators_.begin() + index);
        if (selected_indicator_ >= static_cast<int>(indicators_.size())) {
            selected_indicator_ = static_cast<int>(indicators_.size()) - 1;
        }
    }
}

void TechnicalIndicatorsPanel::clear_indicators() {
    indicators_.clear();
    selected_indicator_ = -1;
}

}  // namespace BTQuant
