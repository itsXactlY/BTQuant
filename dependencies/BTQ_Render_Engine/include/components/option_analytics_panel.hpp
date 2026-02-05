#pragma once

#include <vector>
#include <string>
#include <functional>

#include "panel_base.hpp"
#include "strategy_builder.hpp"

namespace BTQuant {
namespace RenderEngine {

struct OptionData {
    double strike;
    double call_bid;
    double call_ask;
    double call_delta;
    double call_gamma;
    double put_bid;
    double put_ask;
    double put_delta;
    double put_gamma;
    double implied_volatility_call = 0.0;  // Added for volatility smile
    double implied_volatility_put = 0.0;   // Added for volatility smile
    std::string expiration_date;           // Added for multiple expirations

    OptionData(double s) : strike(s) {}
};

struct ExpirationData {
    std::string date;
    std::vector<OptionData> options;
    
    ExpirationData(const std::string& exp_date) : date(exp_date) {}
};

class OptionAnalyticsPanel : public PanelBase {
public:
    OptionAnalyticsPanel(StrategyBuilder* strategy_builder);

    void render() override;
    void update(float dt) override {}

    void switchTab(int tabIndex);
    std::string getActiveTabName() const;

private:
    void initializeSampleData();
    void renderContent();
    void renderDeskTab();
    void renderAnalyzerTab();
    void renderSmileTab();  // New method for volatility smile
    void onStrikeClick(double strike, const std::string& optionType, const std::string& action);

    int activeTab;
    std::vector<std::string> tabs;
    std::vector<OptionData> optionsGrid;
    std::vector<ExpirationData> expirationData;  // For multiple expirations in volatility smile
    
    StrategyBuilder* strategy_builder_;
};

} // namespace RenderEngine
} // namespace BTQuant