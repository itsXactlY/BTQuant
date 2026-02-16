/**
 * BTQuant Compute-to-ImGui Binding Header
 *
 * Header file for the binding system that connects compute-intensive
 * analytics modules to ImGui UI elements for real-time visualization
 * in professional trading dashboard applications.
 */

#pragma once

#include "analytics/tpoengine.h"
#include "analytics/liquiditysweepdetector.h"
#include "analytics/lockfreesnapshotpipeline.h"
#include "analytics/rawtradetable.h"
#include "analytics/orderbook_snapshot_100level.h"
#include "trading/trade_command_queue.hpp"  // Include TradeCommand queue functionality
#include <memory>
#include <functional>
#include <vector>
#include <string>

// Need to include imgui.h to get ImU32 definition
#include "imgui.h"

// Forward declaration to avoid including imgui.h in header
struct ImGuiContext;
typedef struct ImGuiContext ImGuiContext;

namespace BTQuant {
namespace UI {

/**
 * @brief Interface for binding compute results to ImGui UI elements
 */
class ComputeToImGuiBind {
public:
    /**
     * @brief Constructor
     */
    ComputeToImGuiBind();

    /**
     * @brief Destructor
     */
    ~ComputeToImGuiBind();

    /**
     * @brief Bind TPO Engine data to ImGui visualization
     * @param engine Reference to the TPO Engine instance
     * @param window_name Name of the ImGui window to render in
     */
    void bindTPOEngine(const TPOEngine& engine, const char* window_name = "TPO Visualization");

    /**
     * @brief Bind Liquidity Sweep Detector data to ImGui visualization
     * @param detector Reference to the Liquidity Sweep Detector instance
     * @param window_name Name of the ImGui window to render in
     */
    void bindLiquiditySweepDetector(const LiquiditySweepDetector& detector, const char* window_name = "Liquidity Sweeps");

    /**
     * @brief Bind Lock-Free Snapshot Pipeline data to ImGui visualization
     * @param pipeline Reference to the Lock-Free Snapshot Pipeline instance
     * @param window_name Name of the ImGui window to render in
     */
    void bindLockFreeSnapshotPipeline(const LockFreeSnapshotPipeline& pipeline, const char* window_name = "Market Data");

    /**
     * @brief Bind market table data to ImGui visualization
     * @param bid_volume Volume at the best bid price
     * @param ask_volume Volume at the best ask price
     * @param last_price Last traded price
     * @param bid_price Best bid price
     * @param ask_price Best ask price
     * @param window_name Name of the ImGui window to render in
     */
    void bindMarketTable(double bid_volume, double ask_volume, double last_price,
                        double bid_price, double ask_price, const char* window_name = "Market Table");

    /**
     * @brief Bind order book data with USD/COIN toggle functionality
     * @param bid_volume Volume at the best bid price
     * @param ask_volume Volume at the best ask price
     * @param last_price Last traded price
     * @param bid_price Best bid price
     * @param ask_price Best ask price
     * @param window_name Name of the ImGui window to render in
     */
    void bindOrderBookWithToggle(double bid_volume, double ask_volume, double last_price,
                                double bid_price, double ask_price, const char* window_name = "Order Book");

    /**
     * @brief Render all bound visualizations
     */
    void render();

    /**
     * @brief Add custom visualization callback
     * @param callback Function to call during render
     * @param window_name Name of the window to render in
     */
    void addCustomVisualization(std::function<void()> callback, const char* window_name);

    /**
     * @brief Update the bound data from compute modules
     */
    void update();

    /**
     * @brief Bind horizontal bars visualization to ImGui
     * @param values Vector of values to represent as horizontal bars
     * @param colors Vector of colors for each bar
     * @param window_name Name of the ImGui window to render in
     */
    void bindHorizontalBars(const std::vector<float>& values, const std::vector<ImU32>& colors,
                          const char* window_name = "Horizontal Bars");

    /**
     * @brief Bind Raw Trade Table data to ImGui visualization
     * @param trade_table Reference to the Raw Trade Table instance
     * @param window_name Name of the ImGui window to render in
     */
    void bindRawTradeTable(const RawTradeTable& trade_table, const char* window_name = "Raw Trade Table");

    /**
     * @brief Bind live best bid/ask display button to ImGui
     * @param pipeline Reference to the Lock-Free Snapshot Pipeline instance
     * @param symbol_index Index of the symbol to visualize
     * @param window_name Name of the ImGui window to render in
     */
    void bindLiveBidAskButton(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index = 0, const char* window_name = "Live Bid/Ask Button");

    /**
     * @brief Render a button with live best bid/ask from atomic data
     * @param pipeline Reference to the Lock-Free Snapshot Pipeline instance
     * @param symbol_index Index of the symbol to visualize
     * @param button_label Custom label for the button (will be overridden with bid/ask data)
     * @return True if button was clicked, false otherwise
     */
    bool renderLiveBidAskButton(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index = 0, const char* button_label = "Best Bid/Ask");

    /**
     * @brief Bind mouse trading interface with massive BUY MKT / SELL MKT buttons
     * @param pipeline Reference to the Lock-Free Snapshot Pipeline instance
     * @param symbol_index Index of the symbol to visualize
     * @param window_name Name of the ImGui window to render in
     */
    void bindMouseTradingInterface(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index = 0, const char* window_name = "Mouse Trading Interface");

    /**
     * @brief Bind SSBO aggregator for multiple exchange order book aggregation
     * @param window_name Name of the ImGui window to render in
     */
    void bindSSBOAggregator(const char* window_name = "SSBO Order Book Aggregator");

    /**
     * @brief Add an exchange's order book snapshot to the SSBO aggregator
     * @param exchange_name Name of the exchange
     * @param snapshot Order book snapshot from the exchange
     */
    void addExchangeSnapshotToSSBOAggregator(const std::string& exchange_name,
                                           const OrderBookSnapshot100Level& snapshot);

    /**
     * @brief Bind exchange trade table data to ImGui visualization
     * @param trades Vector of raw trades to display
     * @param exchange_names Vector of exchange names corresponding to the trades
     * @param window_name Name of the ImGui window to render in
     */
    void bindExchangeTradeTable(const std::vector<RawTrade>& trades,
                               const std::vector<std::string>& exchange_names,
                               const char* window_name = "Exchange Trade Table");

    /**
     * @brief Render a header with USD/COIN toggle switch
     * @param usd_display_mode Reference to boolean controlling display mode
     * @param window_name Name of the ImGui window to render in
     */
    void renderHeaderWithToggle(bool& usd_display_mode, const char* window_name = "Header");

    /**
     * @brief Add a header with USD/COIN toggle to the visualization system
     * @param usd_display_mode Reference to boolean controlling display mode
     * @param window_name Name of the ImGui window to render in
     */
    void addHeaderWithToggle(bool& usd_display_mode, const char* window_name = "Header");

private:
    struct BoundVisualization {
        std::string window_name;
        std::function<void()> render_callback;
        bool is_visible;
    };

    struct RawTradeTableVisualization {
        std::string window_name;
        const RawTradeTable* trade_table;
        double volume_filter;
        bool is_visible;
    };

    struct MouseTradingData {
        bool mouse_trading_enabled;
        double bid_price;
        double ask_price;
        double order_quantity;
    };

    struct OrderBookData {
        double bid_volume;
        double ask_volume;
        double last_price;
        double bid_price;
        double ask_price;
        bool usd_display_mode;
    };

    struct OrderBookVisualization {
        std::string window_name;
        OrderBookData data;
        bool is_visible;
        bool usd_display_mode;
    };

    struct HeaderVisualization {
        std::string window_name;
        bool* display_mode_ref;  // Pointer to external boolean to control display mode
        bool is_visible;
    };

    std::vector<BoundVisualization> m_visualizations;
    std::vector<OrderBookVisualization> m_order_book_visualizations;
    std::vector<RawTradeTableVisualization> m_raw_trade_table_visualizations;
    std::vector<HeaderVisualization> m_header_visualizations;

    // References to compute modules (stored as weak references)
    const TPOEngine* m_tpo_engine;
    const LiquiditySweepDetector* m_liquidity_detector;
    const LockFreeSnapshotPipeline* m_snapshot_pipeline;
    const RawTradeTable* m_raw_trade_table;

    // Internal state
    bool m_initialized;

    // SSBO Aggregator
    std::unique_ptr<class SSBOAggregator> m_ssbo_aggregator;
};

/**
 * @brief Helper function to visualize TPO data in ImGui
 * @param engine Reference to the TPO Engine instance
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void visualizeTPOData(const TPOEngine& engine, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to visualize Liquidity Sweep data in ImGui
 * @param detector Reference to the Liquidity Sweep Detector instance
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void visualizeLiquiditySweeps(const LiquiditySweepDetector& detector, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to visualize Market Data from Snapshot Pipeline in ImGui
 * @param pipeline Reference to the Lock-Free Snapshot Pipeline instance
 * @param symbol_index Index of the symbol to visualize
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void visualizeMarketData(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index = 0, 
                        float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to render TPO profile as a histogram in ImGui
 * @param engine Reference to the TPO Engine instance
 * @param width Width of the histogram
 * @param height Height of the histogram
 */
void renderTPOProfileHistogram(const TPOEngine& engine, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to render Value Area and POC in ImGui
 * @param engine Reference to the TPO Engine instance
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void renderValueAreaAndPOC(const TPOEngine& engine, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to render sweep detection markers in ImGui
 * @param detector Reference to the Liquidity Sweep Detector instance
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void renderSweepMarkers(const LiquiditySweepDetector& detector, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to visualize market depth table in ImGui
 * @param pipeline Reference to the Lock-Free Snapshot Pipeline instance
 * @param symbol_index Index of the symbol to visualize
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void visualizeMarketDepthTable(const LockFreeSnapshotPipeline& pipeline, uint32_t symbol_index = 0,
                              float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to render a specific market table with [Buys | Asks | Price | Bids | Sells] format
 * @param bid_volume Volume at the best bid price
 * @param ask_volume Volume at the best ask price
 * @param last_price Last traded price
 * @param bid_price Best bid price
 * @param ask_price Best ask price
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void renderMarketTable(double bid_volume, double ask_volume, double last_price,
                      double bid_price, double ask_price, float width = 400.0f, float height = 300.0f);

/**
 * @brief Helper function to render horizontal bars using DrawList->AddRectFilled
 * @param values Vector of values to represent as horizontal bars
 * @param colors Vector of colors for each bar
 * @param width Width of the visualization
 * @param height Height of the visualization
 * @param label Label for the visualization
 */
void renderHorizontalBars(const std::vector<float>& values, const std::vector<ImU32>& colors,
                         float width = 400.0f, float height = 300.0f, const char* label = "Horizontal Bars");

/**
 * @brief Helper function to visualize Raw Trade Table in ImGui
 * @param trade_table Reference to the Raw Trade Table instance
 * @param volume_filter Minimum volume threshold for displaying trades
 */
void visualizeRawTradeTable(const RawTradeTable& trade_table, double volume_filter = 0.0);

/**
 * @brief Helper function to render exchange trade table with [Exchange Logo] | Price | Qty | Time format
 * @param trades Vector of raw trades to display
 * @param exchange_names Vector of exchange names corresponding to the trades
 * @param width Width of the visualization
 * @param height Height of the visualization
 */
void renderExchangeTradeTable(const std::vector<RawTrade>& trades, const std::vector<std::string>& exchange_names, float width = 600.0f, float height = 400.0f);

} // namespace UI
} // namespace BTQuant