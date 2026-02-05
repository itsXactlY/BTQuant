#pragma once

#include <vector>
#include <string>
#include "panel_base.hpp"

namespace BTQuant {
namespace RenderEngine {

class OptionAnalyticsPanel : public PanelBase {
public:
    OptionAnalyticsPanel();

    void render() override;
    void update(float dt) override {}

    void switchTab(int tabIndex);
    std::string getActiveTabName() const;

private:
    void renderContent();
    void renderDeskTab();
    void renderAnalyzerTab();
    void renderSmileTab();

    int activeTab;
    std::vector<std::string> tabs;
};

} // namespace RenderEngine
} // namespace BTQuant