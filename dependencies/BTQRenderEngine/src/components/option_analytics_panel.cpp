#include "components/option_analytics_panel.hpp"

#include <iomanip>
#include <sstream>
#include <cmath>
#include "imgui.h"
#include "implot.h"

namespace BTQuant {
namespace RenderEngine {

OptionAnalyticsPanel::OptionAnalyticsPanel(StrategyBuilder* strategy_builder)
    : PanelBase(PanelConfig{.title = "Option Analytics", .type = PanelType::SCREENER})
    , activeTab(0)
    , strategy_builder_(strategy_builder)
{
    // Initialize the three tabs: Desk, Analyzer, Smile
    tabs.push_back("Desk");
    tabs.push_back("Analyzer");
    tabs.push_back("Smile");

    // Initialize sample option data for demonstration
    initializeSampleData();
}

void OptionAnalyticsPanel::initializeSampleData() {
    // Clear existing data
    optionsGrid.clear();
    expirationData.clear();

    // Define multiple expiration dates
    std::vector<std::string> exp_dates = {"2024-03-15", "2024-04-19", "2024-05-17", "2024-06-21"};

    for (const auto& exp_date : exp_dates) {
        ExpirationData exp_data(exp_date);

        // Generate sample option data for strikes from 100 to 200 in increments of 5
        for (double strike = 100.0; strike <= 200.0; strike += 5.0) {
            OptionData opt(strike);

            // Generate sample values for calls and puts
            opt.call_bid = 15.0 + (strike - 150.0) * 0.1;  // Sample bid price
            opt.call_ask = opt.call_bid + 0.1;              // Ask is slightly higher than bid
            opt.call_delta = 0.1 + (strike - 100.0) * 0.01; // Delta increases with strike for calls
            opt.call_gamma = 0.02 - abs(strike - 150.0) * 0.0002; // Gamma peaks near ATM

            opt.put_bid = 15.0 - (strike - 150.0) * 0.1;   // Sample bid price for puts
            opt.put_ask = opt.put_bid + 0.1;                // Ask is slightly higher than bid
            opt.put_delta = -0.9 + (strike - 100.0) * 0.01; // Delta decreases with strike for puts
            opt.put_gamma = 0.02 - abs(strike - 150.0) * 0.0002; // Same gamma for puts

            // Calculate implied volatility for volatility smile
            // Using a simplified model where IV is highest for ATM options and decreases for ITM/OTM
            double atm_strike = 150.0;
            double moneyness = abs(strike - atm_strike) / atm_strike;
            opt.implied_volatility_call = 0.20 + 0.30 * exp(-pow(moneyness * 2, 2)); // Peak at ATM
            opt.implied_volatility_put = 0.22 + 0.28 * exp(-pow(moneyness * 2, 2));  // Slightly different for puts

            // Adjust IV based on expiration (shorter dated options might have higher IV)
            if (exp_date == "2024-03-15") {
                opt.implied_volatility_call *= 1.1; // Higher IV for near-term options
                opt.implied_volatility_put *= 1.1;
            } else if (exp_date == "2024-06-21") {
                opt.implied_volatility_call *= 0.9; // Lower IV for longer-term options
                opt.implied_volatility_put *= 0.9;
            }

            opt.expiration_date = exp_date;

            exp_data.options.push_back(opt);
            optionsGrid.push_back(opt); // Also add to the main grid for backward compatibility
        }

        expirationData.push_back(exp_data);
    }
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

    // Render tab navigation
    if (ImGui::BeginTabBar("OptionTabs")) {
        for (size_t i = 0; i < tabs.size(); ++i) {
            std::string tabName = "[" + tabs[i] + "]";  // Add brackets as requested
            if (ImGui::BeginTabItem(tabName.c_str())) {
                if (static_cast<int>(i) == activeTab) {
                    renderContent();
                }
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }

    end_panel_window();
}

void OptionAnalyticsPanel::renderContent() {
    // Content for each tab
    if (getActiveTabName() == "Desk") {
        renderDeskTab();
    } else if (getActiveTabName() == "Analyzer") {
        renderAnalyzerTab();
    } else if (getActiveTabName() == "Smile") {
        renderSmileTab();
    }
}

void OptionAnalyticsPanel::renderDeskTab() {
    ImGui::Text("OPTIONS DESK");
    ImGui::Separator();

    // Create a table for the options grid: Left(Calls) - Center(Strike) - Right(Puts)
    if (ImGui::BeginTable("OptionsGrid", 9, ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg)) {
        // Left side - Calls
        ImGui::TableSetupColumn("Call Bid", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Call Ask", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Call Delta", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Call Gamma", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        
        // Center - Strike (highlighted)
        ImGui::TableSetupColumn("Strike", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        
        // Right side - Puts
        ImGui::TableSetupColumn("Put Delta", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Put Gamma", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Put Bid", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Put Ask", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        
        ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
        ImGui::TableHeadersRow();

        for (const auto& opt : optionsGrid) {
            ImGui::TableNextRow();

            // Left side - Calls
            // Call Bid - clickable to add to strategy
            ImGui::TableSetColumnIndex(0);
            std::string call_bid_button_id = "CB##" + std::to_string(static_cast<int>(opt.strike * 100));
            if (ImGui::Button(call_bid_button_id.c_str())) {
                // Callback to add call to strategy with buy order at bid
                onStrikeClick(opt.strike, "Call", "Buy");
            }
            ImGui::SameLine();
            ImGui::Text("%.2f", opt.call_bid);

            // Call Ask - clickable to add to strategy
            ImGui::TableSetColumnIndex(1);
            std::string call_ask_button_id = "CA##" + std::to_string(static_cast<int>(opt.strike * 100));
            if (ImGui::Button(call_ask_button_id.c_str())) {
                // Callback to add call to strategy with sell order at ask
                onStrikeClick(opt.strike, "Call", "Sell");
            }
            ImGui::SameLine();
            ImGui::Text("%.2f", opt.call_ask);

            // Call Delta
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.4f", opt.call_delta);

            // Call Gamma
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%.4f", opt.call_gamma);

            // Center - Strike (highlighted column)
            ImGui::TableSetColumnIndex(4);
            // Highlight the strike column
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.5f, 1.0f)); // Light yellow
            std::string strike_button_id = "S##" + std::to_string(static_cast<int>(opt.strike * 100));
            if (ImGui::Button(strike_button_id.c_str())) {
                // Show context menu for call/put selection
                ImGui::OpenPopup(strike_button_id.c_str());
            }
            ImGui::SameLine();
            ImGui::Text("%.2f", opt.strike);
            ImGui::PopStyleColor(); // Reset text color

            // Popup menu for strike selection
            if (ImGui::BeginPopup(strike_button_id.c_str())) {
                ImGui::Text("Add Strike: %.2f", opt.strike);
                ImGui::Separator();

                if (ImGui::MenuItem("Add Call")) {
                    onStrikeClick(opt.strike, "Call", "Buy");
                }
                if (ImGui::MenuItem("Add Put")) {
                    onStrikeClick(opt.strike, "Put", "Buy");
                }
                if (ImGui::MenuItem("Add Both")) {
                    onStrikeClick(opt.strike, "Call", "Buy");
                    onStrikeClick(opt.strike, "Put", "Buy");
                }
                if (ImGui::MenuItem("Add Short Call")) {
                    onStrikeClick(opt.strike, "Call", "Sell");
                }
                if (ImGui::MenuItem("Add Short Put")) {
                    onStrikeClick(opt.strike, "Put", "Sell");
                }

                ImGui::EndPopup();
            }

            // Right side - Puts
            // Put Delta
            ImGui::TableSetColumnIndex(5);
            ImGui::Text("%.4f", opt.put_delta);

            // Put Gamma
            ImGui::TableSetColumnIndex(6);
            ImGui::Text("%.4f", opt.put_gamma);

            // Put Bid - clickable to add to strategy
            ImGui::TableSetColumnIndex(7);
            std::string put_bid_button_id = "PB##" + std::to_string(static_cast<int>(opt.strike * 100));
            if (ImGui::Button(put_bid_button_id.c_str())) {
                // Callback to add put to strategy with buy order at bid
                onStrikeClick(opt.strike, "Put", "Buy");
            }
            ImGui::SameLine();
            ImGui::Text("%.2f", opt.put_bid);

            // Put Ask - clickable to add to strategy
            ImGui::TableSetColumnIndex(8);
            std::string put_ask_button_id = "PA##" + std::to_string(static_cast<int>(opt.strike * 100));
            if (ImGui::Button(put_ask_button_id.c_str())) {
                // Callback to add put to strategy with sell order at ask
                onStrikeClick(opt.strike, "Put", "Sell");
            }
            ImGui::SameLine();
            ImGui::Text("%.2f", opt.put_ask);
        }

        ImGui::EndTable();
    }
}

void OptionAnalyticsPanel::renderAnalyzerTab() {
    ImGui::Text("OPTIONS ANALYZER");
    ImGui::Separator();

    // Placeholder for analyzer functionality
    ImGui::Text("Advanced option analytics and greeks visualization would go here.");
    ImGui::Text("This could include:");
    ImGui::BulletText("Greeks heatmaps");
    ImGui::BulletText("Profit/Loss scenarios");
    ImGui::BulletText("Strategy payoffs");
    ImGui::BulletText("Risk analytics");
}

void OptionAnalyticsPanel::renderSmileTab() {
    ImGui::Text("VOLATILITY SMILE");
    ImGui::Separator();

    // Check if ImPlot is available and initialized
    if (!ImPlot::GetCurrentContext()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "ImPlot context not initialized!");
        return;
    }

    // Create the plot for volatility smile
    if (ImPlot::BeginPlot("Implied Volatility Smile")) {
        ImPlot::SetupAxis(ImAxis_X1, "Strike Price");
        ImPlot::SetupAxis(ImAxis_Y1, "Implied Volatility (%)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 90, 210);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 15, 50);
        ImPlot::SetupLegend(ImPlotLocation_NorthEast);

        // Define colors for different expiration dates
        ImVec4 colors[] = {
            ImVec4(1.0f, 0.0f, 0.0f, 1.0f),  // Red
            ImVec4(0.0f, 1.0f, 0.0f, 1.0f),  // Green
            ImVec4(0.0f, 0.0f, 1.0f, 1.0f),  // Blue
            ImVec4(1.0f, 1.0f, 0.0f, 1.0f)   // Yellow
        };

        int color_idx = 0;

        // Plot IV vs strike for each expiration
        for (const auto& exp_data : expirationData) {
            std::vector<double> strikes;
            std::vector<double> iv_calls;
            std::vector<double> iv_puts;

            for (const auto& opt : exp_data.options) {
                strikes.push_back(opt.strike);
                iv_calls.push_back(opt.implied_volatility_call * 100); // Convert to percentage
                iv_puts.push_back(opt.implied_volatility_put * 100);   // Convert to percentage
            }

            // Plot calls
            if (!strikes.empty()) {
                ImPlot::SetNextLineStyle(colors[color_idx % 4], 2.0f);
                ImPlot::PlotLine(("Calls " + exp_data.date).c_str(),
                                strikes.data(), iv_calls.data(), static_cast<int>(strikes.size()));

                // Plot puts
                ImPlot::SetNextLineStyle(ImColor(colors[color_idx % 4].x * 0.7f,
                                                colors[color_idx % 4].y * 0.7f,
                                                colors[color_idx % 4].z * 0.7f,
                                                colors[color_idx % 4].w), 1.0f);
                ImPlot::PlotLine(("Puts " + exp_data.date).c_str(),
                                strikes.data(), iv_puts.data(), static_cast<int>(strikes.size()));
            }

            color_idx++;
        }

        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
        ImPlot::EndPlot();
    }

    // Add some explanatory text
    ImGui::Spacing();
    ImGui::TextWrapped("The Volatility Smile shows how implied volatility varies with strike price for different expiration dates.");
    ImGui::TextWrapped("Typically, out-of-the-money and in-the-money options have higher implied volatility than at-the-money options.");
}

void OptionAnalyticsPanel::onStrikeClick(double strike, const std::string& optionType, const std::string& action) {
    // If we have a strategy builder, add the strike to it
    if (strategy_builder_) {
        strategy_builder_->addStrike(strike, optionType, action);
    }
    // Also print for debugging
    printf("Strike clicked: %.2f %s %s\n", strike, optionType.c_str(), action.c_str());
}

} // namespace RenderEngine
} // namespace BTQuant