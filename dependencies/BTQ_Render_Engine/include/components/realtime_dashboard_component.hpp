#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../vulkan_dashboard_advanced.hpp"
#include "imgui.h"
#include "implot.h"
#include <memory>
#include <vector>
#include <string>

namespace BTQuant {

enum class DashboardPanelType {
    PRICE_CHART,
    VOLUME_CHART,
    INDICATORS,
    ORDER_BOOK,
    RECENT_TRADES,
    MARKET_STATS,
    PERFORMANCE_METRICS,
    MULTI_SYMBOL_OVERVIEW
};

struct DashboardPanel {
    DashboardPanelType type;
    std::string title;
    ImVec2 position;
    ImVec2 size;
    bool visible = true;
    std::string symbol = "BTC-USDT";
    RenderEngine::TimeFrame timeframe = RenderEngine::TimeFrame::TF_1MIN;
};

class RealtimeDashboardComponent : public UIComponent {
public:
    explicit RealtimeDashboardComponent(
        std::shared_ptr<HotSpineDataBridge> bridge,
        std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
    virtual ~RealtimeDashboardComponent() = default;

    void update(float dt) override;
    void render_gui() override;

    void initialize_vulkan_resources(VulkanCore *core) override;
    void clear_data() override;

    void add_panel(DashboardPanelType type, const std::string& title, ImVec2 pos, ImVec2 size);
    void remove_panel(int index);

private:
    // Data sources
    std::shared_ptr<HotSpineDataBridge> bridge_;
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

    // Dashboard panels
    std::vector<DashboardPanel> panels_;

    // UI state
    bool show_panel_config_ = false;
    int selected_panel_ = -1;

    // Rendering methods
    void render_panel(const DashboardPanel& panel);
    void render_price_chart_panel(const DashboardPanel& panel);
    void render_volume_chart_panel(const DashboardPanel& panel);
    void render_indicators_panel(const DashboardPanel& panel);
    void render_order_book_panel(const DashboardPanel& panel);
    void render_recent_trades_panel(const DashboardPanel& panel);
    void render_market_stats_panel(const DashboardPanel& panel);
    void render_performance_metrics_panel(const DashboardPanel& panel);
    void render_multi_symbol_overview_panel(const DashboardPanel& panel);

    void render_panel_config_window();
    void render_dashboard_menu();

    // Helper methods
    std::vector<TechnicalIndicators::OHLCV> get_ohlcv_data(const std::string& symbol, RenderEngine::TimeFrame timeframe, int limit = 1000);
    RenderEngine::OrderbookData get_orderbook_data(const std::string& symbol);
    std::vector<RenderEngine::TradeData> get_recent_trades(const std::string& symbol, int limit = 50);
};

} // namespace BTQuant