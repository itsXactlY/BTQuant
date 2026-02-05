#include "components/optionanalyticspanel.hpp"

#include <iomanip>
#include <sstream>
#include <cmath>
#include "imgui.h"

namespace BTQuant {
namespace RenderEngine {

OptionAnalyticsPanel::OptionAnalyticsPanel()
    : PanelBase(PanelConfig{.title = "Option Analytics", .type = PanelType::OPTION_ANALYTICS})
    , activeTab(0)
{
    // Initialize the three tabs: Desk, Analyzer, Smile
    tabs = {"[Desk]", "[Analyzer]", "[Smile]"};
}

void OptionAnalyticsPanel::switchTab(int tabIndex) {
    if (tabIndex >= 0 && tabIndex < static_cast<int>(tabs.size())) {
        activeTab = tabIndex;
    }
}

std::string OptionAnalyticsPanel::getActiveTabName() const {
    if (activeTab >= 0 && activeTab < static_cast<int>(tabs.size())) {
        return tabs[activeTab];
    }
    return "";
}

void OptionAnalyticsPanel::render() {
    begin_panel_window();

    // Render tab navigation header
    if (ImGui::BeginTabBar("OptionAnalyticsTabs", ImGuiTabBarFlags_None)) {
        for (size_t i = 0; i < tabs.size(); ++i) {
            if (ImGui::BeginTabItem(tabs[i].c_str())) {
                activeTab = static_cast<int>(i);
                renderContent();
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }

    end_panel_window();
}

void OptionAnalyticsPanel::renderContent() {
    // Content for each tab
    switch(activeTab) {
        case 0: // Desk tab
            renderDeskTab();
            break;
        case 1: // Analyzer tab
            renderAnalyzerTab();
            break;
        case 2: // Smile tab
            renderSmileTab();
            break;
        default:
            ImGui::Text("Unknown tab");
            break;
    }
}

void OptionAnalyticsPanel::renderDeskTab() {
    ImGui::Text("Options Desk Content");
    ImGui::Separator();
    
    // Basic layout for the desk view
    ImGui::Text("Strike");
    ImGui::SameLine(150);
    ImGui::Text("Call");
    ImGui::SameLine(300);
    ImGui::Text("Put");
    
    // Example option data display
    for (int i = 0; i < 10; ++i) {
        double strike = 100.0 + (i * 5.0);
        
        ImGui::Text("%.2f", strike);
        ImGui::SameLine(150);
        ImGui::Text("%.2f / %.2f", 5.25 + i*0.5, 5.30 + i*0.5); // Bid / Ask for Call
        ImGui::SameLine(300);
        ImGui::Text("%.2f / %.2f", 4.75 - i*0.3, 4.80 - i*0.3); // Bid / Ask for Put
    }
}

void OptionAnalyticsPanel::renderAnalyzerTab() {
    ImGui::Text("Options Analyzer Content");
    ImGui::Separator();
    
    ImGui::Text("Greek Analysis:");
    ImGui::BulletText("Delta: Sensitivity to underlying price");
    ImGui::BulletText("Gamma: Rate of change of Delta");
    ImGui::BulletText("Theta: Time decay");
    ImGui::BulletText("Vega: Sensitivity to volatility");
    ImGui::BulletText("Rho: Sensitivity to interest rates");
}

void OptionAnalyticsPanel::renderSmileTab() {
    ImGui::Text("Volatility Smile Content");
    ImGui::Separator();
    
    ImGui::Text("Implied Volatility vs Strike Price:");
    ImGui::Text("ATM (At-The-Money) options typically have lower IV");
    ImGui::Text("OTM (Out-of-The-Money) & ITM (In-The-Money) options have higher IV");
    
    // Simple visualization placeholder
    if (ImGui::Button("Generate Volatility Smile Chart")) {
        ImGui::Text("Chart would appear here");
    }
}

} // namespace RenderEngine
} // namespace BTQuant