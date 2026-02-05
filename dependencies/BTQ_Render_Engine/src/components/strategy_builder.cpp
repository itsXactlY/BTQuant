#include "components/strategy_builder.hpp"

#include <iomanip>
#include <sstream>

namespace BTQuant {
namespace RenderEngine {

StrategyBuilder::StrategyBuilder(const PanelConfig& config)
    : PanelBase(config.type == PanelType::STRATEGY_BUILDER ? config : PanelConfig{.title = "Strategy Builder", .type = PanelType::STRATEGY_BUILDER})
    , strategy_name_("New Strategy")
    , underlying_price_(0.0)
    , show_add_dialog_(false)
    , new_strike_input_(0.0)
    , new_option_type_("Call")
    , new_action_("Buy")
    , new_quantity_(1)
{
}

StrategyBuilder::StrategyBuilder()
    : PanelBase(PanelConfig{.title = "Strategy Builder", .type = PanelType::STRATEGY_BUILDER})
    , strategy_name_("New Strategy")
    , underlying_price_(0.0)
    , show_add_dialog_(false)
    , new_strike_input_(0.0)
    , new_option_type_("Call")
    , new_action_("Buy")
    , new_quantity_(1)
{
}

void StrategyBuilder::render() {
    begin_panel_window();

    // Compact footer-style UI
    ImGui::Text("Current Strategy: ");
    ImGui::SameLine();
    
    // Show strategy summary inline
    if (!strategy_legs_.empty()) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "%zu legs", strategy_legs_.size()); // Green for active strategy
        ImGui::SameLine();
        
        // Show Greeks inline in a compact form
        ImGui::Text("| Greeks: D:%.3f G:%.3f T:%.3f V:%.3f", 
                   getStrategyDelta(), getStrategyGamma(), getStrategyTheta(), getStrategyVega());
    } else {
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Empty (click strikes to build strategy)"); // Gray for empty
    }
    
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) {
        clearStrategy();
    }
    
    // Only show detailed view if there are legs in the strategy
    if (!strategy_legs_.empty()) {
        ImGui::Separator();
        
        // Render legs in a compact horizontal format
        renderLegsTable();
        
        // Render strategy metrics in a compact form
        renderStrategyMetrics();
    }

    // Add leg dialog (for manual addition if needed)
    renderAddLegDialog();

    end_panel_window();
}

void StrategyBuilder::addStrike(double strike, const std::string& option_type, 
                               const std::string& action, int quantity) {
    strategy_legs_.emplace_back(strike, option_type, action, quantity);
    markDirty();
}

void StrategyBuilder::removeLeg(size_t index) {
    if (index < strategy_legs_.size()) {
        strategy_legs_.erase(strategy_legs_.begin() + index);
        markDirty();
    }
}

void StrategyBuilder::clearStrategy() {
    strategy_legs_.clear();
    markDirty();
}

void StrategyBuilder::renderAddLegDialog() {
    if (show_add_dialog_) {
        ImGui::OpenPopup("Add Strategy Leg");
    }
    
    if (ImGui::BeginPopupModal("Add Strategy Leg", &show_add_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Add a new leg to your strategy:");
        ImGui::Separator();
        
        ImGui::InputDouble("Strike Price", &new_strike_input_);
        ImGui::InputInt("Quantity", &new_quantity_);
        new_quantity_ = std::max(1, new_quantity_); // Ensure at least 1
        
        // Option type selection
        if (ImGui::BeginCombo("Option Type", new_option_type_.c_str())) {
            const char* options[] = {"Call", "Put"};
            for (int i = 0; i < 2; i++) {
                bool is_selected = (new_option_type_ == options[i]);
                if (ImGui::Selectable(options[i], is_selected)) {
                    new_option_type_ = options[i];
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        
        // Action selection
        if (ImGui::BeginCombo("Action", new_action_.c_str())) {
            const char* actions[] = {"Buy", "Sell"};
            for (int i = 0; i < 2; i++) {
                bool is_selected = (new_action_ == actions[i]);
                if (ImGui::Selectable(actions[i], is_selected)) {
                    new_action_ = actions[i];
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        
        ImGui::Separator();
        
        if (ImGui::Button("Add Leg", ImVec2(120, 0))) {
            addStrike(new_strike_input_, new_option_type_, new_action_, new_quantity_);
            // Reset inputs
            new_strike_input_ = 0.0;
            new_quantity_ = 1;
            new_option_type_ = "Call";
            new_action_ = "Buy";
            show_add_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_add_dialog_ = false;
            ImGui::CloseCurrentPopup();
        }
        
        ImGui::EndPopup();
    }
}

void StrategyBuilder::renderStrategySummary() {
    if (!strategy_legs_.empty()) {
        ImGui::Text("Legs: %zu", strategy_legs_.size());
        ImGui::SameLine();
        ImGui::Text("Total Quantity: ");
        int total_qty = 0;
        for (const auto& leg : strategy_legs_) {
            total_qty += leg.quantity;
        }
        ImGui::SameLine();
        ImGui::Text("%d", total_qty);
    } else {
        ImGui::Text("No legs in strategy. Click 'Add Leg' to begin.");
    }
}

void StrategyBuilder::renderLegsTable() {
    if (strategy_legs_.empty()) {
        return;
    }

    // Compact representation of legs
    ImGui::Text("Legs: ");
    ImGui::SameLine();
    
    // Display all legs in a single line for compact view
    for (size_t i = 0; i < strategy_legs_.size(); ++i) {
        const auto& leg = strategy_legs_[i];
        // Color code based on action and type
        ImVec4 color = (leg.action == "Buy") ? 
                      (leg.option_type == "Call" ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(0.0f, 0.8f, 0.0f, 1.0f)) :  // Green tones for buys
                      (leg.option_type == "Call" ? ImVec4(1.0f, 0.0f, 0.0f, 1.0f) : ImVec4(0.8f, 0.0f, 0.0f, 1.0f));  // Red tones for sells
        
        ImGui::TextColored(color, "%s%dx%s@%.1f", 
                          leg.action.c_str(), leg.quantity, 
                          leg.option_type.c_str(), leg.strike);
        
        if (i < strategy_legs_.size() - 1) {
            ImGui::SameLine();
            ImGui::Text(" | ");
            ImGui::SameLine();
        }
    }
    
    // Add a small button to show detailed view if needed
    ImGui::SameLine();
    if (ImGui::SmallButton("Details...")) {
        // For now, we'll just show the detailed table below when expanded
        // This could be expanded to show a popup or expandable section
    }
}

void StrategyBuilder::renderStrategyMetrics() {
    if (strategy_legs_.empty()) {
        return;
    }

    // Compact metrics display
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Greeks: ");
    ImGui::SameLine();
    
    // Color code Greeks for better readability
    float delta = getStrategyDelta();
    float gamma = getStrategyGamma();
    float theta = getStrategyTheta();
    float vega = getStrategyVega();
    
    ImGui::Text("D:");
    ImGui::SameLine();
    ImGui::TextColored(delta >= 0 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 
                      "%.3f ", delta);
    
    ImGui::SameLine();
    ImGui::Text("G:");
    ImGui::SameLine();
    ImGui::TextColored(gamma >= 0 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 
                      "%.3f ", gamma);
    
    ImGui::SameLine();
    ImGui::Text("T:");
    ImGui::SameLine();
    ImGui::TextColored(theta >= 0 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 
                      "%.3f ", theta);
    
    ImGui::SameLine();
    ImGui::Text("V:");
    ImGui::SameLine();
    ImGui::TextColored(vega >= 0 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                      "%.3f", vega);
}

// Placeholder implementations for Greek calculations
// In a real implementation, these would use actual option pricing models
double StrategyBuilder::getStrategyDelta() const {
    double delta = 0.0;
    for (const auto& leg : strategy_legs_) {
        // Simplified calculation - in reality would depend on moneyness, time, vol, etc.
        double leg_delta = (leg.option_type == "Call") ? 0.5 : -0.5; // Placeholder
        if (leg.action == "Sell") {
            leg_delta *= -1;
        }
        delta += leg_delta * leg.quantity;
    }
    return delta;
}

double StrategyBuilder::getStrategyGamma() const {
    double gamma = 0.0;
    for (const auto& leg : strategy_legs_) {
        // Simplified calculation
        double leg_gamma = 0.05; // Placeholder
        if (leg.action == "Sell") {
            leg_gamma *= -1;
        }
        gamma += leg_gamma * leg.quantity;
    }
    return gamma;
}

double StrategyBuilder::getStrategyTheta() const {
    double theta = 0.0;
    for (const auto& leg : strategy_legs_) {
        // Simplified calculation
        double leg_theta = -0.02; // Placeholder (typically negative for long options)
        if (leg.action == "Sell") {
            leg_theta *= -1; // Positive for short options
        }
        theta += leg_theta * leg.quantity;
    }
    return theta;
}

double StrategyBuilder::getStrategyVega() const {
    double vega = 0.0;
    for (const auto& leg : strategy_legs_) {
        // Simplified calculation
        double leg_vega = 0.10; // Placeholder
        if (leg.action == "Sell") {
            leg_vega *= -1;
        }
        vega += leg_vega * leg.quantity;
    }
    return vega;
}

} // namespace RenderEngine
} // namespace BTQuant