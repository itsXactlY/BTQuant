#include "components/option_analytics_panel.hpp"

#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include "imgui.h"
#include "implot.h"

namespace BTQuant {
namespace RenderEngine {

OptionAnalyticsPanel::OptionAnalyticsPanel(std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(PanelConfig{.title = "Option Analytics", .type = PanelType::OPTION_ANALYTICS})
    , processor_(processor)
    , subscription_id_(0)
    , activeTab(0)
{
    // Initialize the three tabs: Desk, Analyzer, Smile
    tabs.push_back("Desk");
    tabs.push_back("Analyzer");
    tabs.push_back("Smile");
    
    // Subscribe to market data updates if processor is available
    if (processor_) {
        subscription_id_ = processor_->subscribe(0, RenderEngine::NotificationType::TRADE, 
            [this](uint32_t symbol_id, RenderEngine::NotificationType type) {
                // Handle market data updates for options analytics
                // This callback will be called when new trade data arrives
                markDirty(); // Mark panel as needing refresh
            });
    }
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

    if (!processor_) {
        ImGui::Text("No MarketDataProcessor available");
        return;
    }

    // Get active symbols from the processor
    auto active_symbols = processor_->getActiveSymbols();
    
    if (active_symbols.empty()) {
        ImGui::Text("No active symbols available");
        return;
    }

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
        ImGui::TableSetupColumn("Put Bid", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Put Ask", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Put Delta", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn("Put Gamma", ImGuiTableColumnFlags_WidthFixed, 70.0f);

        ImGui::TableSetupScrollFreeze(0, 1); // Make top row always visible
        ImGui::TableHeadersRow();

        // Display options data for the first few active symbols
        size_t display_count = 0;
        for (uint32_t symbol_id : active_symbols) {
            if (display_count >= 10) break; // Limit display
            
            auto analytics = processor_->getSymbolAnalytics(symbol_id);
            if (analytics.symbol_id != 0) {
                // Calculate theoretical option prices based on underlying price
                double underlying_price = analytics.last_trade_price > 0 ? analytics.last_trade_price : 100.0;
                
                // Generate sample strikes around the current price
                for (double strike = underlying_price - 20.0; strike <= underlying_price + 20.0; strike += 5.0) {
                    if (display_count >= 10) break; // Limit display
                    
                    // Calculate approximate option prices and Greeks based on market data
                    double call_bid = std::max(0.0, underlying_price - strike) * 0.8; // Simplified pricing
                    double call_ask = std::max(0.0, underlying_price - strike) * 1.0;
                    double put_bid = std::max(0.0, strike - underlying_price) * 0.7;
                    double put_ask = std::max(0.0, strike - underlying_price) * 0.9;
                    
                    // Calculate approximate Greeks based on market data
                    double delta_call = 0.5 + (underlying_price - strike) / 200.0; // Simplified delta
                    delta_call = std::max(0.0, std::min(1.0, delta_call)); // Clamp between 0 and 1
                    double delta_put = delta_call - 1.0; // Put delta = Call delta - 1
                    
                    double volatility = analytics.volatility > 0 ? analytics.volatility : 0.25; // 25% volatility
                    double gamma = volatility / (underlying_price * std::sqrt(0.25)); // Simplified gamma
                    gamma = std::max(0.0, gamma);
                    
                    ImGui::TableNextRow();

                    // Left side - Calls
                    // Call Bid - clickable to add to strategy
                    ImGui::TableSetColumnIndex(0);
                    std::string call_bid_button_id = "CB##" + std::to_string(static_cast<int>(strike * 100));
                    if (ImGui::Button(call_bid_button_id.c_str())) {
                        // Callback to add call to strategy with buy order at bid
                        onStrikeClick(strike, "Call", "Buy");
                    }
                    ImGui::SameLine();
                    ImGui::Text("%.2f", call_bid);

                    // Call Ask - clickable to add to strategy
                    ImGui::TableSetColumnIndex(1);
                    std::string call_ask_button_id = "CA##" + std::to_string(static_cast<int>(strike * 100));
                    if (ImGui::Button(call_ask_button_id.c_str())) {
                        // Callback to add call to strategy with sell order at ask
                        onStrikeClick(strike, "Call", "Sell");
                    }
                    ImGui::SameLine();
                    ImGui::Text("%.2f", call_ask);

                    // Call Delta
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.4f", delta_call);

                    // Call Gamma
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.4f", gamma);

                    // Center - Strike (highlighted column)
                    ImGui::TableSetColumnIndex(4);
                    // Highlight the strike column
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.5f, 1.0f)); // Light yellow
                    std::string strike_button_id = "S##" + std::to_string(static_cast<int>(strike * 100));
                    if (ImGui::Button(strike_button_id.c_str())) {
                        // Show context menu for call/put selection
                        ImGui::OpenPopup(strike_button_id.c_str());
                    }
                    ImGui::SameLine();
                    ImGui::Text("%.2f", strike);
                    ImGui::PopStyleColor(); // Reset text color

                    // Popup menu for strike selection
                    if (ImGui::BeginPopup(strike_button_id.c_str())) {
                        ImGui::Text("Add Strike: %.2f", strike);
                        ImGui::Separator();

                        if (ImGui::MenuItem("Add Call")) {
                            onStrikeClick(strike, "Call", "Buy");
                        }
                        if (ImGui::MenuItem("Add Put")) {
                            onStrikeClick(strike, "Put", "Buy");
                        }
                        if (ImGui::MenuItem("Add Both")) {
                            onStrikeClick(strike, "Call", "Buy");
                            onStrikeClick(strike, "Put", "Buy");
                        }
                        if (ImGui::MenuItem("Add Short Call")) {
                            onStrikeClick(strike, "Call", "Sell");
                        }
                        if (ImGui::MenuItem("Add Short Put")) {
                            onStrikeClick(strike, "Put", "Sell");
                        }

                        ImGui::EndPopup();
                    }

                    // Right side - Puts
                    // Put Bid - clickable to add to strategy
                    ImGui::TableSetColumnIndex(5);
                    std::string put_bid_button_id = "PB##" + std::to_string(static_cast<int>(strike * 100));
                    if (ImGui::Button(put_bid_button_id.c_str())) {
                        // Callback to add put to strategy with buy order at bid
                        onStrikeClick(strike, "Put", "Buy");
                    }
                    ImGui::SameLine();
                    ImGui::Text("%.2f", put_bid);

                    // Put Ask - clickable to add to strategy
                    ImGui::TableSetColumnIndex(6);
                    std::string put_ask_button_id = "PA##" + std::to_string(static_cast<int>(strike * 100));
                    if (ImGui::Button(put_ask_button_id.c_str())) {
                        // Callback to add put to strategy with sell order at ask
                        onStrikeClick(strike, "Put", "Sell");
                    }
                    ImGui::SameLine();
                    ImGui::Text("%.2f", put_ask);

                    // Put Delta
                    ImGui::TableSetColumnIndex(7);
                    ImGui::Text("%.4f", delta_put);

                    // Put Gamma
                    ImGui::TableSetColumnIndex(8);
                    ImGui::Text("%.4f", gamma);
                    
                    display_count++;
                }
            }
        }

        ImGui::EndTable();
    }
}

void OptionAnalyticsPanel::renderAnalyzerTab() {
    ImGui::Text("OPTIONS ANALYZER");
    ImGui::Separator();

    if (!processor_) {
        ImGui::Text("No MarketDataProcessor available");
        return;
    }

    // Get active symbols from the processor
    auto active_symbols = processor_->getActiveSymbols();
    
    if (active_symbols.empty()) {
        ImGui::Text("No active symbols available");
        return;
    }

    // Display greek analysis for the first active symbol
    uint32_t symbol_id = active_symbols[0];
    auto analytics = processor_->getSymbolAnalytics(symbol_id);
    
    if (analytics.symbol_id != 0) {
        ImGui::Text("Greek Analysis for Symbol ID: %u", symbol_id);
        ImGui::Separator();

        // Calculate Greeks based on market data
        double underlying_price = analytics.last_trade_price > 0 ? analytics.last_trade_price : 100.0;
        double volatility = analytics.volatility > 0 ? analytics.volatility : 0.25; // 25% volatility
        double volume = analytics.volume_1m > 0 ? analytics.volume_1m : 1000.0;
        
        // Calculate approximate Greeks (these are simplified calculations)
        double delta_call = 0.5 + (underlying_price - 100.0) / 200.0; // Simplified delta
        delta_call = std::max(0.0, std::min(1.0, delta_call)); // Clamp between 0 and 1
        
        double delta_put = delta_call - 1.0; // Put delta = Call delta - 1
        
        double gamma = volatility / (underlying_price * std::sqrt(0.25)); // Simplified gamma
        gamma = std::max(0.0, gamma);
        
        double theta = -(volatility * underlying_price) / (2 * std::sqrt(0.25)); // Simplified theta
        double vega = underlying_price * std::sqrt(0.25) * 0.4; // Simplified vega
        
        ImGui::Text("Underlying Price: $%.2f", underlying_price);
        ImGui::Text("Implied Volatility: %.2f%%", volatility * 100.0);
        ImGui::Text("Volume: %.0f", volume);
        ImGui::Separator();
        
        ImGui::Text("Greek Values:");
        ImGui::BulletText("Delta (Call): %.3f", delta_call);
        ImGui::BulletText("Delta (Put): %.3f", delta_put);
        ImGui::BulletText("Gamma: %.4f", gamma);
        ImGui::BulletText("Theta: %.3f", theta);
        ImGui::BulletText("Vega: %.3f", vega);
        
        // Add a simple visualization of greeks
        ImGui::Separator();
        ImGui::Text("Greeks Visualization:");
        
        // Delta visualization
        ImGui::Text("Delta: ");
        ImGui::SameLine();
        ImGui::ProgressBar(delta_call, ImVec2(200, 0), "");
        
        // Gamma visualization  
        ImGui::Text("Gamma: ");
        ImGui::SameLine();
        ImGui::ProgressBar(std::min(gamma * 10.0, 1.0), ImVec2(200, 0), ""); // Scale gamma for display
        
        // Add more advanced analytics
        ImGui::Separator();
        ImGui::Text("Advanced Analytics:");
        
        // Show volatility surface concept
        ImGui::Text("Volatility Surface:");
        ImGui::BulletText("ATM Volatility: %.2f%%", volatility * 100.0);
        ImGui::BulletText("Skew: %.3f", analytics.momentum); // Using momentum as a proxy for skew
        ImGui::BulletText("Kurtosis: %.3f", analytics.volatility * 2.0); // Simplified kurtosis
        
    } else {
        ImGui::Text("No analytics data available for symbol: %u", symbol_id);
    }
}

void OptionAnalyticsPanel::renderSmileTab() {
    ImGui::Text("VOLATILITY SMILE");
    ImGui::Separator();

    if (!processor_) {
        ImGui::Text("No MarketDataProcessor available");
        return;
    }

    // Get active symbols from the processor
    auto active_symbols = processor_->getActiveSymbols();
    
    if (active_symbols.empty()) {
        ImGui::Text("No active symbols available");
        return;
    }

    // Check if ImPlot is available and initialized
    if (!ImPlot::GetCurrentContext()) {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "ImPlot context not initialized!");
        return;
    }

    // Create the plot for volatility smile
    if (ImPlot::BeginPlot("Implied Volatility Smile", ImVec2(-1, 400))) {
        ImPlot::SetupAxis(ImAxis_X1, "Strike Price ($)");
        ImPlot::SetupAxis(ImAxis_Y1, "Implied Volatility (%)");

        // Get the first active symbol to generate volatility smile data
        uint32_t symbol_id = active_symbols[0];
        auto analytics = processor_->getSymbolAnalytics(symbol_id);
        
        if (analytics.symbol_id != 0) {
            double underlying_price = analytics.last_trade_price > 0 ? analytics.last_trade_price : 100.0;
            double base_volatility = analytics.volatility > 0 ? analytics.volatility * 100.0 : 25.0;
            
            // Generate sample strikes around the current price
            std::vector<double> strikes;
            std::vector<double> iv_calls;
            std::vector<double> iv_puts;
            
            for (int i = -10; i <= 10; i++) {
                double strike = underlying_price + (i * 5.0); // Strikes from -50 to +50 from current price
                double distance_from_atm = std::abs(strike - underlying_price) / underlying_price;
                
                // Calculate volatility smile effect - ATM has lowest IV, OTM/ITM have higher IV
                double smile_effect = 0.5 * distance_from_atm; // Smile effect increases with distance from ATM
                double iv = base_volatility + (smile_effect * 10.0); // Scale the smile effect
                
                strikes.push_back(strike);
                iv_calls.push_back(iv);
                iv_puts.push_back(iv); // For simplicity, using same IV for calls and puts
            }

            // Define colors for different expiration dates (using a single series for now)
            ImVec4 colors[] = {
                ImVec4(1.0f, 0.0f, 0.0f, 1.0f),  // Red
                ImVec4(0.0f, 1.0f, 0.0f, 1.0f),  // Green
                ImVec4(0.0f, 0.0f, 1.0f, 1.0f),  // Blue
                ImVec4(1.0f, 1.0f, 0.0f, 1.0f),  // Yellow
                ImVec4(1.0f, 0.0f, 1.0f, 1.0f),  // Magenta
                ImVec4(0.0f, 1.0f, 1.0f, 1.0f)   // Cyan
            };

            // Plot calls
            ImPlot::SetNextLineStyle(colors[0], 2.0f);
            ImPlot::PlotLine("Calls", strikes.data(), iv_calls.data(), static_cast<int>(strikes.size()));

            // Plot puts with different line style to distinguish from calls
            ImPlot::SetNextLineStyle(ImColor(colors[1].x * 0.7f,
                                            colors[1].y * 0.7f,
                                            colors[1].z * 0.7f,
                                            colors[1].w), 1.5f);
            ImPlot::PlotLine("Puts", strikes.data(), iv_puts.data(), static_cast<int>(strikes.size()));

            // Add ATM reference line
            ImPlot::SetNextLineStyle(ImVec4(0.5f, 0.5f, 0.5f, 0.5f), 1.0f, ImPlotLineFlags_SkipMissing);
            ImPlot::PlotLine("ATM Reference", &underlying_price, &base_volatility, 1, ImPlotLineFlags_Vertical);
        }

        ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
        ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
        ImPlot::EndPlot();
        ImPlot::PopStyleVar();
    }

    // Add some explanatory text
    ImGui::Spacing();
    ImGui::TextWrapped("The Volatility Smile shows how implied volatility varies with strike price for different expiration dates.");
    ImGui::TextWrapped("Typically, out-of-the-money and in-the-money options have higher implied volatility than at-the-money options.");

    // Add information about the current data
    ImGui::Spacing();
    ImGui::Text("Current Data:");
    if (active_symbols.size() > 0) {
        ImGui::BulletText("Underlying Price: $%.2f", processor_->getSymbolAnalytics(active_symbols[0]).last_trade_price);
        ImGui::BulletText("Base Volatility: %.2f%%", processor_->getSymbolAnalytics(active_symbols[0]).volatility * 100.0);
    }
    ImGui::BulletText("Active Symbols: %zu", active_symbols.size());
}

void OptionAnalyticsPanel::onStrikeClick(double strike, const std::string& optionType, const std::string& action) {
    // If we have a strategy builder, add the strike to it
    // Note: strategy_builder_ is no longer available since we switched to MarketDataProcessor
    // This functionality would need to be reconnected if needed
    // For now, just print for debugging
    printf("Strike clicked: %.2f %s %s\n", strike, optionType.c_str(), action.c_str());
}

void OptionAnalyticsPanel::update(float dt) {
    // Process any market data updates or analytics calculations
    // This method can be used to update options analytics in real-time
    // The panel will be marked as dirty by the subscription callback when new data arrives
}

// Destructor to clean up subscription
BTQuant::RenderEngine::OptionAnalyticsPanel::~OptionAnalyticsPanel() {
    if (processor_ && subscription_id_ > 0) {
        processor_->unsubscribe(subscription_id_);
    }
}

} // namespace RenderEngine
} // namespace BTQuant