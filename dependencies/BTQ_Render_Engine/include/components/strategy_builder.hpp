#pragma once

#include <vector>
#include <string>
#include <memory>

#include "imgui.h"
#include "panel_base.hpp"

namespace BTQuant {
namespace RenderEngine {

// Structure to represent a leg in an options strategy
struct StrategyLeg {
    double strike;
    std::string option_type;  // "Call" or "Put"
    std::string action;       // "Buy" or "Sell"
    int quantity;
    
    StrategyLeg(double s = 0.0, const std::string& type = "Call", const std::string& act = "Buy", int qty = 1)
        : strike(s), option_type(type), action(act), quantity(qty) {}
};

// Strategy Builder Panel - displays and manages multi-leg option strategies
class StrategyBuilder : public PanelBase {
public:
    explicit StrategyBuilder(const PanelConfig& config = PanelConfig{});
    StrategyBuilder();  // Default constructor for backward compatibility
    ~StrategyBuilder() override = default;

    void render_content() override;
    
    // Add a strike to the current strategy
    void addStrike(double strike, const std::string& option_type = "Call", 
                   const std::string& action = "Buy", int quantity = 1);
    
    // Remove a leg from the strategy
    void removeLeg(size_t index);
    
    // Clear the entire strategy
    void clearStrategy();
    
    // Get the current strategy legs
    const std::vector<StrategyLeg>& getStrategyLegs() const { return strategy_legs_; }
    
    // Calculate strategy metrics
    double getStrategyDelta() const;
    double getStrategyGamma() const;
    double getStrategyTheta() const;
    double getStrategyVega() const;

private:
    std::vector<StrategyLeg> strategy_legs_;
    std::string strategy_name_;
    double underlying_price_;
    
    // UI state
    bool show_add_dialog_;
    double new_strike_input_;
    std::string new_option_type_;
    std::string new_action_;
    int new_quantity_;
    
    void renderAddLegDialog();
    void renderStrategySummary();
    void renderLegsTable();
    void renderStrategyMetrics();
};

} // namespace RenderEngine
} // namespace BTQuant